/**
 * OpticWall Browser Agent Visual Safety Firewall
 * Content Script: Intercepts high-risk clicks and actions executed by
 * autonomous browser agents (Browser-Use, Playwright, Puppeteer, OpenAI Operator, Claude).
 */

(function () {
  'use strict';

  let isArmed = true;
  let riskThreshold = 0.50;
  const LOCAL_API_URL = 'http://localhost:3000/api/inspect';

  // Dangerous keywords that warrant pre-action inspection
  const HIGH_RISK_KEYWORDS = [
    'confirm payment', 'pay now', 'buy now', 'purchase', 'checkout', 'transfer funds',
    'delete database', 'drop table', 'wipe all', 'destroy', 'format disk', 'permanently delete',
    'export pii', 'download customer data', 'reveal secret key', 'terminate instance',
    'remove user', 'revoke access', 'empty trash'
  ];

  function evaluateElementRisk(element) {
    if (!element) return { isRisky: false, score: 0, reason: '' };

    const text = (element.innerText || element.value || element.getAttribute('aria-label') || '').toLowerCase().trim();
    const idAndClass = `${element.id} ${element.className}`.toLowerCase();

    for (const kw of HIGH_RISK_KEYWORDS) {
      if (text.includes(kw) || idAndClass.includes(kw.replace(/\s+/g, '-'))) {
        return {
          isRisky: true,
          score: 0.94,
          keyword: kw,
          reason: `Action matches critical pattern: "${kw.toUpperCase()}"`
        };
      }
    }
    return { isRisky: false, score: 0.05, reason: 'Benign interaction' };
  }

  function showInterceptionModal(targetElement, riskData, originalEvent) {
    const existing = document.getElementById('opticwall-guard-overlay');
    if (existing) existing.remove();

    const overlay = document.createElement('div');
    overlay.id = 'opticwall-guard-overlay';
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100vw';
    overlay.style.height = '100vh';
    overlay.style.backgroundColor = 'rgba(6, 9, 19, 0.85)';
    overlay.style.backdropFilter = 'blur(10px)';
    overlay.style.zIndex = '2147483647';
    overlay.style.display = 'flex';
    overlay.style.alignItems = 'center';
    overlay.style.justifyContent = 'center';
    overlay.style.fontFamily = 'system-ui, -apple-system, sans-serif';

    const card = document.createElement('div');
    card.style.background = '#0f172a';
    card.style.border = '2px solid #ef4444';
    card.style.borderRadius = '16px';
    card.style.padding = '28px';
    card.style.maxWidth = '460px';
    card.style.width = '90%';
    card.style.boxShadow = '0 0 50px rgba(239, 68, 68, 0.35)';
    card.style.color = '#ffffff';

    card.innerHTML = `
      <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
        <div style="background:rgba(239, 68, 68, 0.2); border:1px solid #ef4444; border-radius:10px; padding:8px; display:flex;">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
        </div>
        <div>
          <div style="font-size:18px; font-weight:800; letter-spacing:-0.5px;">OPTICWALL CIRCUIT BREAKER</div>
          <div style="font-size:12px; color:#94a3b8;">High-Risk Agent Action Intercepted</div>
        </div>
      </div>

      <div style="background:rgba(15, 23, 42, 0.9); border:1px solid #334155; border-radius:10px; padding:14px; margin-bottom:18px; font-size:13px; line-height:1.5;">
        <div style="color:#f87171; font-weight:bold; margin-bottom:4px;">🚨 Warning: Destructive Action Detected</div>
        <div style="color:#cbd5e1;">${riskData.reason}</div>
        <div style="margin-top:8px; font-size:11px; color:#64748b; font-family:monospace;">Target: &lt;${targetElement.tagName.toLowerCase()} id="${targetElement.id || 'none'}"&gt;</div>
      </div>

      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; font-size:12px;">
        <span style="color:#94a3b8;">Visual Risk Score:</span>
        <span style="background:#ef4444; color:#fff; font-weight:bold; padding:3px 8px; border-radius:6px; font-family:monospace;">${(riskData.score * 100).toFixed(1)}% (HARD_BLOCK)</span>
      </div>

      <div style="display:flex; gap:12px;">
        <button id="opticwall-cancel-btn" style="flex:1; padding:10px 14px; background:#1e293b; border:1px solid #475569; color:#fff; border-radius:8px; font-weight:600; cursor:pointer;">
          Hard Block (Abort Action)
        </button>
        <button id="opticwall-approve-btn" style="flex:1; padding:10px 14px; background:#0284c7; border:none; color:#fff; border-radius:8px; font-weight:600; cursor:pointer;">
          Human Override & Proceed
        </button>
      </div>
    `;

    overlay.appendChild(card);
    document.body.appendChild(overlay);

    document.getElementById('opticwall-cancel-btn').onclick = function () {
      overlay.remove();
      console.warn('[OpticWall] Destructive agent action aborted by user.');
    };

    document.getElementById('opticwall-approve-btn').onclick = function () {
      overlay.remove();
      targetElement.dataset.opticwallBypass = 'true';
      targetElement.click();
    };
  }

  // Intercept click events in the capture phase (before other listeners or page scripts)
  document.addEventListener('click', function (e) {
    if (!isArmed) return;

    const target = e.target.closest('button, a, input[type="submit"], input[type="button"], [role="button"]') || e.target;
    if (target.dataset && target.dataset.opticwallBypass === 'true') {
      delete target.dataset.opticwallBypass;
      return;
    }

    const assessment = evaluateElementRisk(target);
    if (assessment.isRisky && assessment.score >= riskThreshold) {
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
      showInterceptionModal(target, assessment, e);
    }
  }, true);

  console.log('[OpticWall] Visual Security Guard armed and monitoring browser agent actions.');
})();
