import io
import os
import json
from PIL import Image
from google import genai
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pypdf import PdfReader

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_drive_service():
    if os.path.exists("credentials.json"):
        creds = service_account.Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
    else:
        creds_info = json.loads(os.environ["GOOGLE_CREDENTIALS_JSON"])
        creds = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def download_bytes(service, file_id, is_google_doc=False):
    if is_google_doc:
        request = service.files().export_media(fileId=file_id, mimeType='text/plain')
    else:
        request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh

def fetch_and_chunk_files(folder_id, gemini_client, chunk_size=1000, overlap=150):
    service = get_drive_service()
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    files = results.get('files', [])

    chunks = []
    for file in files:
        file_id = file['id']
        name = file['name']
        mime = file['mimeType']

        try:
            # 1. Google Docs
            if mime == 'application/vnd.google-apps.document':
                text = download_bytes(service, file_id, is_google_doc=True).getvalue().decode('utf-8')
                for i in range(0, len(text), chunk_size - overlap):
                    c = text[i:i + chunk_size].strip()
                    if c:
                        chunks.append(f"\n{c}")

            # 2. PDFs (SOPs, Manuals, Drawings with page numbers)
            elif mime == 'application/pdf':
                fh = download_bytes(service, file_id)
                reader = PdfReader(fh)
                for page_num, page in enumerate(reader.pages, start=1):
                    page_text = (page.extract_text() or "").strip()
                    if page_text:
                        for i in range(0, len(page_text), chunk_size - overlap):
                            c = page_text[i:i + chunk_size].strip()
                            if c:
                                chunks.append(f"\n{c}")

            # 3. Drawings / Schematics saved as Images (PNG/JPG)
            elif mime.startswith('image/'):
                fh = download_bytes(service, file_id)
                img = Image.open(fh)
                vision_res = gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        img,
                        "Extract all technical text, component labels, relay ANSI codes, terminal block numbers, "
                        "wiring connections, voltage ratings, and notes from this engineering drawing in structured detail."
                    ]
                )
                desc = (vision_res.text or "").strip()
                if desc:
                    for i in range(0, len(desc), chunk_size - overlap):
                        c = desc[i:i + chunk_size].strip()
                        if c:
                            chunks.append(f"[Source Drawing: {name}]\n{c}")

            # 4. Plain Text / CSV
            elif mime.startswith('text/'):
                text = download_bytes(service, file_id).getvalue().decode('utf-8')
                for i in range(0, len(text), chunk_size - overlap):
                    c = text[i:i + chunk_size].strip()
                    if c:
                        chunks.append(f"\n{c}")

        except Exception as e:
            print(f"Skipping {name}: {e}")

    return chunks