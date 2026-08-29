/**
 * SENTINEL-Vision extension background service worker.
 *
 * This extension is deliberately a thin client: it has no model, no risk
 * logic, and cannot make a decision on its own. It periodically captures
 * the active tab as a screenshot and POSTs it to the local SENTINEL-Vision
 * gateway (src/integration/intercept_api.py, normally on
 * http://127.0.0.1:8000), which does the actual inference. Without the
 * gateway running, this extension does nothing useful -- see
 * docs/PUBLIC_DEPLOYMENT_GUIDE.md "Known Limitations".
 */

const DEFAULT_SETTINGS = {
  gatewayUrl: "http://127.0.0.1:8000",
  apiToken: "",
  pollIntervalMs: 2000,
  enabled: true,
};

let settings = { ...DEFAULT_SETTINGS };
let pollTimer = null;
let lastDecision = null;

async function loadSettings() {
  const stored = await chrome.storage.local.get(Object.keys(DEFAULT_SETTINGS));
  settings = { ...DEFAULT_SETTINGS, ...stored };
}

chrome.storage.onChanged.addListener((changes) => {
  for (const [key, { newValue }] of Object.entries(changes)) {
    if (key in settings) settings[key] = newValue;
  }
  restartPolling();
});

function setBadge(decisionAction) {
  const colors = {
    ALLOW: "#16a34a",
    PAUSE: "#d97706",
    HARD_BLOCK: "#dc2626",
  };
  const text = { ALLOW: "OK", PAUSE: "!!", HARD_BLOCK: "X" }[decisionAction] || "";
  chrome.action.setBadgeText({ text });
  chrome.action.setBadgeBackgroundColor({ color: colors[decisionAction] || "#64748b" });
}

async function captureAndSend() {
  if (!settings.enabled || !settings.apiToken) return;

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !tab.id) return;

    // captureVisibleTab requires the tab to be in the currently focused
    // window; it silently fails/throws on background windows, which is why
    // this is best-effort and swallows errors rather than treating a
    // capture failure as "no risk detected."
    const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, { format: "png" });
    const base64 = dataUrl.split(",")[1];

    const resp = await fetch(`${settings.gatewayUrl}/intercept`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Sentinel-Token": settings.apiToken,
      },
      body: JSON.stringify({
        action_type: "monitor",
        frame_b64: base64,
        metadata: { url: tab.url, title: tab.title, source: "browser_extension" },
      }),
    });

    if (!resp.ok) {
      lastDecision = { error: `Gateway returned ${resp.status}`, action: "UNKNOWN" };
      chrome.action.setBadgeText({ text: "ERR" });
      chrome.action.setBadgeBackgroundColor({ color: "#64748b" });
      return;
    }

    const decision = await resp.json();
    lastDecision = decision;
    setBadge(decision.decision);

    if (decision.decision === "PAUSE" || decision.decision === "HARD_BLOCK") {
      chrome.notifications.create({
        type: "basic",
        iconUrl: "icons/icon128.png",
        title: `SENTINEL-Vision: ${decision.decision}`,
        message: `${decision.category?.toUpperCase() || "risk"} detected (${Math.round(
          (decision.risk_score || 0) * 100
        )}%)`,
      });

      // Ask the content script to render a blocking/warning overlay.
      chrome.tabs.sendMessage(tab.id, { type: "SENTINEL_DECISION", decision }).catch(() => {
        // No content script listening on this page (e.g. chrome:// URLs) -- ignore.
      });
    }
  } catch (err) {
    lastDecision = { error: String(err), action: "UNKNOWN" };
  }
}

function restartPolling() {
  if (pollTimer) clearInterval(pollTimer);
  if (settings.enabled) {
    pollTimer = setInterval(captureAndSend, settings.pollIntervalMs);
  }
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "GET_LAST_DECISION") {
    sendResponse({ decision: lastDecision, settings });
    return true;
  }
});

loadSettings().then(restartPolling);
