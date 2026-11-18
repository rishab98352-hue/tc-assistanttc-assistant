import io
import os
import json
import time
import fitz  # PyMuPDF
import openpyxl
from PIL import Image
from google import genai
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CACHE_FILE = "drive_chunks_cache.json"

def get_drive_service():
    if os.path.exists("credentials.json"):
        creds = service_account.Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
    else:
        creds_info = json.loads(os.environ["GOOGLE_CREDENTIALS_JSON"])
        creds = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def download_bytes(service, file_id, export_mime=None):
    if export_mime:
        request = service.files().export_media(fileId=file_id, mimeType=export_mime)
    else:
        request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh

def extract_with_vision(gemini_client, contents_list):
    while True:
        try:
            res = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents_list
            )
            return (res.text or "").strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(30)
            else:
                raise e

def chunk_text(text, source_label, chunk_size=2500, overlap=250):
    out = []
    for i in range(0, len(text), chunk_size - overlap):
        c = text[i:i + chunk_size].strip()
        if c:
            out.append(f"[{source_label}]\n{c}")
    return out

def fetch_and_chunk_files(folder_id, gemini_client, progress_callback=None, chunk_size=2500, overlap=250):
    service = get_drive_service()
    
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    files = []
    page_token = None
    while True:
        query = f"'{folder_id}' in parents and trashed = false"
        results = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
            pageSize=100,
            pageToken=page_token
        ).execute()
        files.extend(results.get('files', []))
        page_token = results.get('nextPageToken')
        if not page_token:
            break

    all_chunks = []
    updated_cache = {}
    total_files = len(files)

    for idx, file in enumerate(files, start=1):
        file_id = file['id']
        name = file['name']
        mime = file['mimeType']
        mod_time = file.get('modifiedTime', '')

        if progress_callback:
            progress_callback(idx / max(total_files, 1), f"Reading ({idx}/{total_files}): {name}")

        if file_id in cache and cache[file_id].get("modifiedTime") == mod_time:
            file_chunks = cache[file_id]["chunks"]
            updated_cache[file_id] = cache[file_id]
            all_chunks.extend(file_chunks)
            continue

        file_chunks = []
        try:
            # 1. Google Docs
            if mime == 'application/vnd.google-apps.document':
                text = download_bytes(service, file_id, export_mime='text/plain').getvalue().decode('utf-8')
                file_chunks.extend(chunk_text(text, f"Source: {name}", chunk_size, overlap))

            # 2. Google Sheets
            elif mime == 'application/vnd.google-apps.spreadsheet':
                csv_text = download_bytes(service, file_id, export_mime='text/csv').getvalue().decode('utf-8')
                file_chunks.extend(chunk_text(csv_text, f"Source Sheet: {name}", chunk_size, overlap))

            # 3. Excel Files (.xlsx)
            elif mime == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                fh = download_bytes(service, file_id)
                wb = openpyxl.load_workbook(fh, data_only=True)
                for sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    rows = []
                    for row in ws.iter_rows(values_only=True):
                        row_vals = [str(cell) if cell is not None else "" for cell in row]
                        if any(row_vals):
                            rows.append(" | ".join(row_vals))
                    sheet_text = "\n".join(rows)
                    if sheet_text:
                        file_chunks.extend(chunk_text(sheet_text, f"Source Excel: {name} | Tab: {sheet_name}", chunk_size, overlap))

            # 4. PDFs (Fast text extraction + Batched 5-page Vision OCR for scanned pages)
            elif mime == 'application/pdf':
                fh = download_bytes(service, file_id)
                doc = fitz.open(stream=fh.read(), filetype="pdf")
                scanned_batch = []
                scanned_page_nums = []

                for page_num, page in enumerate(doc, start=1):
                    page_text = page.get_text().strip()
                    if len(page_text) >= 40:
                        file_chunks.extend(chunk_text(page_text, f"Source: {name} | Page {page_num}", chunk_size, overlap))
                    else:
                        pix = page.get_pixmap(dpi=120)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        scanned_batch.append(img)
                        scanned_page_nums.append(page_num)

                        # Process 5 scanned pages in ONE API call
                        if len(scanned_batch) == 5 or page_num == len(doc):
                            if progress_callback:
                                progress_callback(idx / max(total_files, 1), f"OCR scanning {name} (Pages {scanned_page_nums[0]}-{scanned_page_nums[-1]}/{len(doc)})...")
                            prompt = (
                                f"Transcribe all text, tables, SOP steps, relay ANSI codes, and terminal numbers from these "
                                f"scanned engineering pages (Pages {scanned_page_nums}) in clean structured text."
                            )
                            ocr_text = extract_with_vision(gemini_client, scanned_batch + [prompt])
                            if ocr_text:
                                page_label = f"Pages {scanned_page_nums[0]}-{scanned_page_nums[-1]}" if len(scanned_page_nums) > 1 else f"Page {scanned_page_nums[0]}"
                                file_chunks.extend(chunk_text(ocr_text, f"Source: {name} | {page_label}", chunk_size, overlap))
                            scanned_batch = []
                            scanned_page_nums = []

                # Flush any remaining pages at end of PDF
                if scanned_batch:
                    prompt = (
                        f"Transcribe all text, tables, SOP steps, relay ANSI codes, and terminal numbers from these "
                        f"scanned engineering pages (Pages {scanned_page_nums}) in clean structured text."
                    )
                    ocr_text = extract_with_vision(gemini_client, scanned_batch + [prompt])
                    if ocr_text:
                        page_label = f"Pages {scanned_page_nums[0]}-{scanned_page_nums[-1]}" if len(scanned_page_nums) > 1 else f"Page {scanned_page_nums[0]}"
                        file_chunks.extend(chunk_text(ocr_text, f"Source: {name} | {page_label}", chunk_size, overlap))

            # 5. Standalone Drawings / Schematics saved as Images (PNG/JPG)
            elif mime.startswith('image/'):
                fh = download_bytes(service, file_id)
                img = Image.open(fh)
                desc = extract_with_vision(
                    gemini_client,
                    [img, "Extract all technical text, component labels, relay ANSI codes, terminal block numbers, wiring connections, and notes from this drawing."]
                )
                if desc:
                    file_chunks.extend(chunk_text(desc, f"Source Drawing: {name}", chunk_size, overlap))

            # 6. Plain Text / CSV
            elif mime.startswith('text/'):
                text = download_bytes(service, file_id).getvalue().decode('utf-8')
                file_chunks.extend(chunk_text(text, f"Source: {name}", chunk_size, overlap))

            updated_cache[file_id] = {"modifiedTime": mod_time, "chunks": file_chunks}
            all_chunks.extend(file_chunks)

        except Exception as e:
            print(f"Skipping {name}: {e}")

    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(updated_cache, f)
    except Exception:
        pass

    return all_chunks
