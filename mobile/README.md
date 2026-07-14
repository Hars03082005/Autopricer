# PriceRef Mobile (Flutter)

The mobile app is a **Flutter shell** that wraps the existing React web UI in a WebView. The UI, screens, and ML workflow stay the same — only the deployment layer changes so you can ship Android and iOS builds.

## Architecture

```text
Flutter App (mobile/)
  └── WebView
        └── React UI (bundled from Vite build)
              └── FastAPI ML backend (Python)
```

## Prerequisites

- Flutter SDK 3.44+
- Node.js (for building the web UI)
- Python backend running for ML valuations

## 1. Bundle the web UI

From the project root:

```powershell
cd PriceRef-v2
npm install
npm run build:mobile
```

This builds the React app and copies it to `mobile/assets/web/`.

## 2. Start the ML backend

```powershell
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Use `--host 0.0.0.0` so a phone or emulator on your network can reach the API.

## 3. Run on Android emulator

```powershell
cd mobile
flutter pub get
flutter run
```

Default API URL on Android emulator: `http://10.0.2.2:8000`

## 4. Run on a physical device

Find your PC's local IP (e.g. `192.168.1.10`) and pass it to Flutter:

```powershell
flutter run --dart-define=API_URL=http://192.168.1.10:8000
```

Make sure the backend is running with `--host 0.0.0.0` and your firewall allows port 8000.

## 5. Live development (optional)

Run the Vite dev server and point the Flutter shell at it instead of bundled assets:

```powershell
# Terminal 1
npm run dev

# Terminal 2
cd mobile
flutter run --dart-define=WEB_URL=http://10.0.2.2:5173 --dart-define=API_URL=http://10.0.2.2:8000
```

## 6. Release builds

```powershell
npm run build:mobile
cd mobile
flutter build apk
flutter build appbundle
flutter build ios
```

## Configuration

| Dart define | Purpose | Default |
|-------------|---------|---------|
| `API_URL` | FastAPI backend base URL | `http://10.0.2.2:8000` (Android) / `http://localhost:8000` (iOS) |
| `WEB_URL` | Dev server URL; leave empty to use bundled assets | bundled `assets/web/` |

The Flutter shell injects `window.PriceRef_API_URL` into the WebView so the React app can call your ML backend from mobile.

## Demo login

```text
dealer@PriceRef.ai / dealer123
```
