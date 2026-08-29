# SENTINEL-Vision Browser Extension

A thin Chrome/Edge (Manifest V3) client for the SENTINEL-Vision local
gateway. It periodically screenshots your active tab and sends it to the
gateway for a risk verdict (ALLOW / PAUSE / HARD_BLOCK), showing a
color-coded badge, a desktop notification, and an in-page overlay on any
PAUSE/HARD_BLOCK.

**This extension has no model and does nothing on its own.** It requires
the SENTINEL-Vision gateway running locally:

```bash
sentinel-wall start
# or
python -m uvicorn src.integration.intercept_api:app --host 127.0.0.1 --port 8000
```

## Install (sideload / "Load unpacked")

This is not published to the Chrome Web Store (that requires a paid
developer account and a review cycle -- out of scope here). Install it as
an unpacked extension instead:

1. Start the gateway (see above). On first run it writes an API token to
   `~/.sentinel_vision/api_token.txt`.
2. Open `chrome://extensions` (or `edge://extensions`).
3. Enable **Developer mode** (top-right toggle).
4. Click **Load unpacked** and select this `extension/` folder.
5. Click the SENTINEL-Vision toolbar icon → **Settings**, and paste the
   token from step 1 into **API Token**. Adjust **Gateway URL** if the
   gateway isn't on the default `http://127.0.0.1:8000`.

## What it sends

Each poll tick sends a base64 PNG screenshot of the active tab (via
`chrome.tabs.captureVisibleTab`) plus the tab's URL/title to
`POST /intercept` on the gateway, authenticated with the `X-Sentinel-Token`
header. No other browsing data leaves the machine -- the gateway is
localhost-only by design (see its CORS config in
`src/integration/intercept_api.py`).

## Packaging a distributable zip

```bash
cd sentinel-vision/extension
zip -r ../sentinel-vision-extension.zip . -x '*.DS_Store'
```

Users can drag the resulting `.zip` into `chrome://extensions` (with
Developer mode on) after extracting it, or you can point them at the
"Load unpacked" flow above directly.
