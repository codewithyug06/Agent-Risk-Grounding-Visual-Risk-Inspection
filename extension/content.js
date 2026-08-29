/**
 * Content script: renders the PAUSE/HARD_BLOCK overlay when the background
 * worker forwards a decision. All text values are inserted via textContent
 * (never innerHTML) since risk category/reasoning can echo back
 * attacker-influenced page content from the gateway response.
 */

const OVERLAY_ID = "__sentinel_vision_overlay__";

function removeOverlay() {
  const existing = document.getElementById(OVERLAY_ID);
  if (existing) existing.remove();
}

function renderOverlay(decision) {
  removeOverlay();

  const isBlock = decision.decision === "HARD_BLOCK";
  const overlay = document.createElement("div");
  overlay.id = OVERLAY_ID;
  overlay.style.cssText = `
    position: fixed; top: 16px; right: 16px; z-index: 2147483647;
    max-width: 360px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: ${isBlock ? "#7f1d1d" : "#78350f"}; color: #fff;
    border: 2px solid ${isBlock ? "#ef4444" : "#f59e0b"};
    border-radius: 10px; padding: 14px 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.4);
  `;

  const title = document.createElement("div");
  title.style.cssText = "font-weight: 700; font-size: 14px; margin-bottom: 6px;";
  title.textContent = `SENTINEL-Vision: ${decision.decision}`;

  const body = document.createElement("div");
  body.style.cssText = "font-size: 12px; line-height: 1.4; opacity: 0.95;";
  const category = String(decision.category || "unknown").toUpperCase();
  const riskPct = Math.round((decision.risk_score || 0) * 100);
  body.textContent = `${category} risk detected (${riskPct}%). ${decision.reasoning || ""}`;

  const closeBtn = document.createElement("button");
  closeBtn.textContent = "Dismiss";
  closeBtn.style.cssText = `
    margin-top: 10px; background: rgba(255,255,255,0.15); color: #fff; border: none;
    border-radius: 6px; padding: 6px 10px; font-size: 12px; cursor: pointer;
  `;
  closeBtn.addEventListener("click", removeOverlay);

  overlay.appendChild(title);
  overlay.appendChild(body);
  overlay.appendChild(closeBtn);
  document.documentElement.appendChild(overlay);

  if (!isBlock) {
    setTimeout(removeOverlay, 10000);
  }
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "SENTINEL_DECISION") {
    renderOverlay(msg.decision);
  }
});
