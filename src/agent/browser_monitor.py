from playwright.sync_api import sync_playwright
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.agent.inference_bridge import SentinelInferenceBridge

def inject_monitor_script(page):
    """
    Injects JS to intercept clicks before they bubble down to the target.
    Uses `{capture: true}`.
    """
    script = """
    window.interceptedEvent = null;
    
    document.addEventListener('click', (e) => {
        // If it's a programmatic click from us allowing it, let it pass
        if (e.isTrusted === false && e.detail === 999) return;
        
        // Pause the click
        e.preventDefault();
        e.stopPropagation();
        
        // Save the event so we can dispatch it later if ALLOWED
        window.interceptedEvent = e;
        
        console.log("Click intercepted at", e.clientX, e.clientY);
        
        // Notify Python bridge
        window.notifyPython({x: e.clientX, y: e.clientY});
    }, {capture: true});
    """
    page.add_init_script(script)

def draw_red_box(page, box, is_pause=False):
    """
    Draws a bounding box overlay on the page (red for block, orange for pause).
    """
    border_color = 'orange' if is_pause else 'red'
    bg_color = 'rgba(255,165,0,0.3)' if is_pause else 'rgba(255,0,0,0.3)'
    
    script = f"""
    let div = document.createElement('div');
    div.id = 'sentinel-overlay';
    div.style.position = 'absolute';
    div.style.left = '{box["x"]}px';
    div.style.top = '{box["y"]}px';
    div.style.width = '{box["w"]}px';
    div.style.height = '{box["h"]}px';
    div.style.border = '3px solid {border_color}';
    div.style.backgroundColor = '{bg_color}';
    div.style.zIndex = '999999';
    div.style.pointerEvents = 'none'; // click through
    document.body.appendChild(div);
    
    // Flash message
    let msg = document.createElement('div');
    msg.innerText = is_pause ? '⏸️ SENTINEL PAUSED ACTION' : '⚠️ SENTINEL BLOCKED ACTION';
    msg.style.position = 'absolute';
    msg.style.left = '{box["x"]}px';
    msg.style.top = '{box["y"] - 30}px';
    msg.style.color = 'white';
    msg.style.backgroundColor = is_pause ? 'orange' : 'red';
    msg.style.padding = '5px';
    msg.style.fontWeight = 'bold';
    msg.style.zIndex = '999999';
    document.body.appendChild(msg);
    
    // Remove after 3 seconds
    setTimeout(() => {{
        div.remove();
        msg.remove();
    }}, 3000);
    """
    page.evaluate(script)

def run_agent():
    # Initialize in mock mode so we can test the UI interception logic
    # without needing a fully trained semantic model.
    bridge = SentinelInferenceBridge(mock_mode=True)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        
        inject_monitor_script(page)
        
        def handle_click_intent(source, data):
            print(f"\n[Agent] Intercepted click at {data['x']}, {data['y']}")
            
            # 1. Take Screenshot
            screenshot_bytes = page.screenshot()
            
            # 2. Ask Inference Bridge
            print("[Agent] Sending frame to SentinelVision...")
            start_t = time.time()
            verdict = bridge.predict(screenshot_bytes, target_coords=data)
            end_t = time.time()
            print(f"[Agent] Verdict: {verdict['action']} (took {end_t - start_t:.3f}s)")
            
            # 3. Act on Verdict
            if verdict['action'] == 'BLOCK':
                print("[Agent] Blocking click and drawing UI overlay.")
                draw_red_box(page, verdict['box'])
            elif verdict['action'] == 'PAUSE':
                print("[Agent] Pausing click for human review. Drawing orange UI overlay.")
                draw_red_box(page, verdict['box'], is_pause=True)
            else:
                print("[Agent] Allowing click.")
                # Dispatch the saved event programmatically with detail=999 to bypass our interceptor
                page.evaluate("""
                    if (window.interceptedEvent) {
                        let target = window.interceptedEvent.target;
                        let e = new MouseEvent('click', {
                            bubbles: true,
                            cancelable: true,
                            view: window,
                            detail: 999
                        });
                        target.dispatchEvent(e);
                        window.interceptedEvent = null;
                    }
                """)

        # Expose Python function to JS
        page.expose_binding("notifyPython", handle_click_intent)
        
        test_file = f"file:///{os.path.abspath('tests/test_agent.html').replace(os.sep, '/')}"
        print(f"[Agent] Navigating to {test_file}")
        page.goto(test_file)
        
        print("[Agent] Ready. Please interact with the browser.")
        print("[Agent] (Press Ctrl+C in terminal to exit)")
        
        # Keep alive for testing
        try:
            page.wait_for_timeout(300000) # Wait 5 minutes
        except KeyboardInterrupt:
            print("Exiting...")
        
        browser.close()

if __name__ == "__main__":
    run_agent()
