const DEFAULT_SETTINGS = {
  gatewayUrl: "http://127.0.0.1:8000",
  apiToken: "",
  pollIntervalMs: 2000,
  enabled: true,
};

const fields = ["gatewayUrl", "apiToken", "pollIntervalMs", "enabled"];

async function load() {
  const stored = await chrome.storage.local.get(fields);
  const settings = { ...DEFAULT_SETTINGS, ...stored };
  document.getElementById("gatewayUrl").value = settings.gatewayUrl;
  document.getElementById("apiToken").value = settings.apiToken;
  document.getElementById("pollIntervalMs").value = settings.pollIntervalMs;
  document.getElementById("enabled").checked = settings.enabled;
}

document.getElementById("save").addEventListener("click", async () => {
  const gatewayUrl = document.getElementById("gatewayUrl").value.trim() || DEFAULT_SETTINGS.gatewayUrl;
  const apiToken = document.getElementById("apiToken").value.trim();
  const pollIntervalMs = Math.max(500, parseInt(document.getElementById("pollIntervalMs").value, 10) || DEFAULT_SETTINGS.pollIntervalMs);
  const enabled = document.getElementById("enabled").checked;

  await chrome.storage.local.set({ gatewayUrl, apiToken, pollIntervalMs, enabled });

  const savedEl = document.getElementById("saved");
  savedEl.style.display = "inline";
  setTimeout(() => (savedEl.style.display = "none"), 1500);
});

load();
