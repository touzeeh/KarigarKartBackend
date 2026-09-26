# KarigarKart FastAPI backend

This backend is the local inference/API layer used by the Flutter prototype.

## Run on the demo laptop

```powershell
cd backend
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Keep the terminal running during the demo.

## Phone connection

The Flutter client currently uses `http://10.70.33.153:8000`. If the laptop's tethering adapter receives a different IPv4 address, update `kAiApiBaseUrl` in `lib/services/ai_background_service.dart`, `lib/services/voice_catalog_service.dart`, and the pricing screen to that current laptop IPv4 address.

## AI endpoints

- `GET /health` — connectivity/health check.
- `POST /ai/transcribe` — multipart audio transcription through Gemini.
- `POST /ai/catalog` — JSON catalog generation through Gemini.
- `POST /ai/pricing` — JSON pricing benchmark through Gemini with a deterministic cost-floor fallback.
- `POST /ai/enhance` — multipart product image processing; returns **actual `image/jpeg` bytes**, which is the contract expected by the Flutter background service.

The enhancement endpoint accepts the Flutter field name `file` (and also `image`) and never attempts to UTF-8 decode uploaded JPEG bytes as a validation string.
