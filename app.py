# Try current active models, then auto-discover any available Flash model on your key
        models_to_try = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.0-flash", "gemini-2.5-flash"]
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=full_prompt
                )
                answer_text = response.text
                break
            except Exception as e:
                last_error = e
                continue

        # Automatic future-proof fallback: query your API key's live model list
        if not answer_text:
            try:
                for m in client.models.list():
                    if "flash" in m.name.lower() and "image" not in m.name.lower() and "audio" not in m.name.lower():
                        clean_name = m.name.replace("models/", "")
                        response = client.models.generate_content(
                            model=clean_name,
                            contents=full_prompt
                        )
                        answer_text = response.text
                        break
            except Exception as e:
                last_error = e
