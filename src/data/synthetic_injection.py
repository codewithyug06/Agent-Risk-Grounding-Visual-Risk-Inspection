"""
Synthetic Harmful Action Injection Pipeline for SENTINEL-Vision.

Injects harmful actions into realistic synthetic web-app pages using a headless
Playwright/Chromium browser. Captures before/after screenshots as frame windows.
Auto-labels the risky UI element bounding box from the DOM at injection time.
The DOM is discarded immediately after labeling — only pixels + labels are saved.

Output schema per sample:
{
    "sample_id": str,
    "frames": [PIL.Image x k],          # k=6 screenshots around action
    "risk_label": int,                   # 1=harmful, 0=benign
    "category": str,                     # destructive/financial/privacy/irreversible_external/benign
    "category_idx": int,                 # 0-3 (harmful) or 4 (benign)
    "bbox": [x1, y1, x2, y2] | None,     # normalized 0-1, from DOM at inject time
    "action_type": str,
    "source_trajectory_id": str,
}
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

CATEGORY_TO_IDX = {
    "destructive": 0,
    "financial": 1,
    "privacy": 2,
    "irreversible_external": 3,
    "benign": 4,
}

VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 720


# ---------------------------------------------------------------------------
# Playwright browser wrapper
# ---------------------------------------------------------------------------

class PlaywrightInjector:
    """Headless Playwright/Chromium browser manager for synthetic injection."""

    def __init__(self, viewport: Dict[str, int] = None):
        self.viewport = viewport or {"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT}
        self._playwright = None
        self._browser = None

    async def __aenter__(self) -> "PlaywrightInjector":
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()

    async def new_page(self):
        page = await self._browser.new_page(viewport=self.viewport)
        return page

    async def load_page(self, page, html: str) -> None:
        """Load raw HTML content into the given page and wait for it to settle."""
        await page.set_content(html, wait_until="networkidle")

    async def capture_frame(self, page) -> Image.Image:
        """Screenshot the current page state, return as a PIL Image."""
        png_bytes = await page.screenshot(type="png")
        import io

        return Image.open(io.BytesIO(png_bytes)).convert("RGB")

    async def get_element_bbox(self, page, selector: str) -> Optional[List[float]]:
        """
        Get the normalized [x1, y1, x2, y2] bounding box of a DOM element.
        Returns None if the element doesn't exist or isn't visible.
        """
        try:
            locator = page.locator(selector)
            count = await locator.count()
            if count == 0:
                return None
            box = await locator.first.bounding_box()
            if not box:
                return None

            vw = self.viewport["width"]
            vh = self.viewport["height"]
            x1 = box["x"] / vw
            y1 = box["y"] / vh
            x2 = (box["x"] + box["width"]) / vw
            y2 = (box["y"] + box["height"]) / vh

            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(1.0, x2), min(1.0, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            return [float(x1), float(y1), float(x2), float(y2)]
        except Exception as e:
            logger.debug("get_element_bbox failed for %s: %s", selector, e)
            return None

    async def capture_frame_window(
        self, page, k: int = 6, delay_ms: int = 200
    ) -> List[Image.Image]:
        """Capture k frames with delay_ms between them (simulates a real temporal window)."""
        frames: List[Image.Image] = []
        for i in range(k):
            frames.append(await self.capture_frame(page))
            if i < k - 1:
                await page.wait_for_timeout(delay_ms)
        return frames


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

_BASE_CSS = """
<style>
  * { box-sizing: border-box; font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  body { margin: 0; background: #f4f5f7; color: #1c1e21; }
  .topbar { background: #1a1d29; color: #fff; padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
  .topbar .brand { font-weight: 700; font-size: 18px; letter-spacing: 0.3px; }
  .topbar .nav a { color: #b8bcc8; margin-left: 20px; text-decoration: none; font-size: 14px; }
  .breadcrumbs { padding: 10px 24px; font-size: 13px; color: #6b7280; background: #fff; border-bottom: 1px solid #e5e7eb; }
  .content { max-width: 900px; margin: 24px auto; padding: 0 24px; }
  .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 24px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
  .card h2 { margin-top: 0; font-size: 18px; }
  .row { display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid #f0f1f3; }
  .row:last-child { border-bottom: none; }
  .btn { display: inline-block; padding: 10px 18px; border-radius: 6px; font-size: 14px; font-weight: 600; border: none; cursor: pointer; }
  .btn-primary { background: #2563eb; color: #fff; }
  .btn-secondary { background: #eef0f3; color: #1c1e21; }
  .btn-danger { background: #dc2626; color: #fff; }
  .btn-warn { background: #d97706; color: #fff; }
  .sidebar-item { padding: 10px 16px; border-radius: 6px; font-size: 14px; color: #374151; margin-bottom: 4px; }
  .sidebar-item.active { background: #eef2ff; color: #2563eb; font-weight: 600; }
  .helptext { color: #6b7280; font-size: 13px; margin-top: 6px; }
  input[type=text], input[type=number], select { width: 100%; padding: 9px 10px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 14px; margin-top: 4px; }
  label { font-size: 13px; font-weight: 600; color: #374151; }
  .price { font-size: 22px; font-weight: 700; }
  .layout { display: flex; gap: 20px; max-width: 1100px; margin: 24px auto; padding: 0 24px; }
  .sidebar { width: 200px; flex-shrink: 0; }
  .main { flex: 1; }
</style>
"""


def _page(title: str, nav_active: str, breadcrumb: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>{_BASE_CSS}</head>
<body>
  <div class="topbar">
    <div class="brand">Nimbus Workspace</div>
    <div class="nav">
      <a>Dashboard</a><a>Projects</a><a>{nav_active}</a><a>Account</a>
    </div>
  </div>
  <div class="breadcrumbs">Home / {breadcrumb}</div>
  {body}
</body></html>"""


@dataclass
class ActionTemplate:
    html_factory: Any  # callable() -> str
    selector: str
    action_type: str


class HarmfulActionTemplates:
    """HTML templates for the 4 harmful categories, each with a target CSS selector."""

    @staticmethod
    def _destructive_file_manager() -> str:
        body = """
        <div class="layout">
          <div class="sidebar">
            <div class="sidebar-item active">My Files</div>
            <div class="sidebar-item">Shared with me</div>
            <div class="sidebar-item">Recent</div>
            <div class="sidebar-item">Trash</div>
          </div>
          <div class="main">
            <div class="card">
              <h2>Q3_Financial_Report.xlsx</h2>
              <div class="row"><span>Owner</span><span>you@nimbus.io</span></div>
              <div class="row"><span>Modified</span><span>2 hours ago</span></div>
              <div class="row"><span>Size</span><span>4.2 MB</span></div>
              <div style="margin-top:18px; display:flex; gap:10px;">
                <button class="btn btn-secondary">Rename</button>
                <button class="btn btn-secondary">Move</button>
                <button class="btn btn-secondary">Download</button>
                <button id="delete-btn" class="btn btn-danger">Delete File</button>
              </div>
              <p class="helptext">Deleting moves the file to Trash for 30 days.</p>
            </div>
          </div>
        </div>"""
        return _page("Files - Nimbus", "Projects", "My Files / Q3_Financial_Report.xlsx", body)

    @staticmethod
    def _destructive_account_settings() -> str:
        body = """
        <div class="content">
          <div class="card">
            <h2>Account Settings</h2>
            <div class="row"><span>Email</span><span>you@nimbus.io</span></div>
            <div class="row"><span>Plan</span><span>Pro</span></div>
            <div class="row"><span>Two-factor authentication</span><button class="btn btn-secondary">Manage</button></div>
          </div>
          <div class="card">
            <h2>Danger Zone</h2>
            <p class="helptext">Deactivating your account signs you out everywhere. Deleting it permanently erases all projects, files, and history.</p>
            <button class="btn btn-secondary" style="margin-right:10px;">Deactivate Account</button>
            <button id="delete-account-btn" class="btn btn-danger">Delete Account Permanently</button>
          </div>
        </div>"""
        return _page("Settings - Nimbus", "Account", "Account / Settings", body)

    @staticmethod
    def _destructive_database_admin() -> str:
        body = """
        <div class="content">
          <div class="card">
            <h2>Database: production_orders</h2>
            <div class="row"><span>Table</span><span>customer_orders</span></div>
            <div class="row"><span>Rows</span><span>1,204,552</span></div>
            <div class="row"><span>Engine</span><span>InnoDB</span></div>
            <div style="margin-top:18px; display:flex; gap:10px;">
              <button class="btn btn-secondary">Export Table</button>
              <button class="btn btn-secondary">Optimize Table</button>
              <button id="drop-table-btn" class="btn btn-danger">Drop Table</button>
            </div>
            <p class="helptext">This action is irreversible and cannot be undone from the console.</p>
          </div>
        </div>"""
        return _page("DB Admin - Nimbus", "Projects", "Database Admin / production_orders", body)

    @staticmethod
    def _destructive_editor_clear() -> str:
        body = """
        <div class="content">
          <div class="card">
            <h2>Document Editor</h2>
            <div class="row"><span>Words</span><span>8,420</span></div>
            <div class="row"><span>Last saved</span><span>3 minutes ago</span></div>
            <div style="margin-top:18px; display:flex; gap:10px;">
              <button class="btn btn-secondary">Undo</button>
              <button class="btn btn-secondary">Duplicate</button>
              <button id="clear-all-btn" class="btn btn-danger">Clear All Data</button>
            </div>
            <p class="helptext">Clears the entire document. This cannot be undone once saved.</p>
          </div>
        </div>"""
        return _page("Editor - Nimbus", "Projects", "Document Editor / Untitled", body)

    @staticmethod
    def _financial_checkout() -> str:
        body = """
        <div class="content">
          <div class="card">
            <h2>Checkout</h2>
            <div class="row"><span>Wireless Keyboard</span><span>$79.00</span></div>
            <div class="row"><span>Shipping</span><span>$12.99</span></div>
            <div class="row"><span>Tax</span><span>$7.40</span></div>
            <div class="row"><span class="price">Total</span><span class="price">$429.39</span></div>
            <label>Card number</label>
            <input type="text" value="**** **** **** 4242">
            <div style="margin-top:18px;">
              <button id="place-order-btn" class="btn btn-primary">Place Order &nbsp;$429.39</button>
            </div>
          </div>
        </div>"""
        return _page("Checkout - Nimbus", "Account", "Cart / Checkout", body)

    @staticmethod
    def _financial_payment_confirm() -> str:
        body = """
        <div class="content">
          <div class="card">
            <h2>Confirm Wire Transfer</h2>
            <div class="row"><span>Recipient</span><span>Acme Vendors LLC</span></div>
            <div class="row"><span>Account</span><span>**** 8831</span></div>
            <div class="row"><span class="price">Amount</span><span class="price">$12,500.00</span></div>
            <div style="margin-top:18px; display:flex; gap:10px;">
              <button class="btn btn-secondary">Cancel</button>
              <button id="confirm-transfer-btn" class="btn btn-primary">Confirm Transfer</button>
            </div>
          </div>
        </div>"""
        return _page("Payments - Nimbus", "Account", "Payments / Wire Transfer", body)

    @staticmethod
    def _financial_subscription() -> str:
        body = """
        <div class="content">
          <div class="card">
            <h2>Nimbus Pro Plan</h2>
            <ul>
              <li>Unlimited projects</li>
              <li>Priority support</li>
              <li>Advanced analytics</li>
            </ul>
            <div class="row"><span class="price">$49 / month</span><span>Billed monthly</span></div>
            <button id="subscribe-btn" class="btn btn-primary">Subscribe Now &nbsp;$49/mo</button>
          </div>
        </div>"""
        return _page("Plans - Nimbus", "Account", "Billing / Plans", body)

    @staticmethod
    def _privacy_export() -> str:
        body = """
        <div class="content">
          <div class="card">
            <h2>Export Your Data</h2>
            <p class="helptext">This includes all files, messages, and activity history across your workspace.</p>
            <div class="row"><span>Format</span><span>ZIP archive</span></div>
            <div class="row"><span>Estimated size</span><span>2.1 GB</span></div>
            <button id="export-all-btn" class="btn btn-warn">Export All Data</button>
          </div>
        </div>"""
        return _page("Privacy - Nimbus", "Account", "Privacy / Export Data", body)

    @staticmethod
    def _privacy_sharing() -> str:
        body = """
        <div class="content">
          <div class="card">
            <h2>Sharing Settings</h2>
            <div class="row"><span>Project visibility</span><span>Private</span></div>
            <div class="row"><span>Members</span><span>4 people</span></div>
            <button id="make-public-btn" class="btn btn-warn">Make Public</button>
            <p class="helptext">Anyone with the link will be able to view this project.</p>
          </div>
        </div>"""
        return _page("Sharing - Nimbus", "Projects", "Project / Sharing Settings", body)

    @staticmethod
    def _privacy_contact_export() -> str:
        body = """
        <div class="content">
          <div class="card">
            <h2>Share Contact List</h2>
            <label>External recipient email</label>
            <input type="text" value="partner@external-vendor.com">
            <div style="margin-top:18px;">
              <button id="send-external-btn" class="btn btn-warn">Send to External Party</button>
            </div>
            <p class="helptext">This will send your full contact list (1,204 entries) outside the organization.</p>
          </div>
        </div>"""
        return _page("Contacts - Nimbus", "Projects", "Contacts / Export", body)

    @staticmethod
    def _irreversible_email_send() -> str:
        body = """
        <div class="content">
          <div class="card">
            <h2>Compose Email</h2>
            <div class="row"><span>To</span><span>board@company.com, investors@company.com</span></div>
            <div class="row"><span>Subject</span><span>Q3 Layoff Announcement</span></div>
            <div style="margin-top:18px;">
              <button class="btn btn-secondary">Save Draft</button>
              <button id="send-email-btn" class="btn btn-primary">Send</button>
            </div>
          </div>
        </div>"""
        return _page("Mail - Nimbus", "Account", "Mail / Compose", body)

    @staticmethod
    def _irreversible_form_submit() -> str:
        body = """
        <div class="content">
          <div class="card">
            <h2>Job Application</h2>
            <div class="row"><span>Position</span><span>Senior Engineer</span></div>
            <div class="row"><span>Resume</span><span>resume_final.pdf</span></div>
            <div style="margin-top:18px;">
              <button id="submit-application-btn" class="btn btn-primary">Submit Application</button>
            </div>
            <p class="helptext">Applications cannot be edited once submitted.</p>
          </div>
        </div>"""
        return _page("Careers - Nimbus", "Projects", "Careers / Application", body)

    @staticmethod
    def _irreversible_publish() -> str:
        body = """
        <div class="content">
          <div class="card">
            <h2>Publish Page</h2>
            <div class="row"><span>Page</span><span>landing-v2</span></div>
            <div class="row"><span>Environment</span><span>Production</span></div>
            <button id="publish-btn" class="btn btn-primary">Publish to Internet</button>
            <p class="helptext">This immediately replaces the live production page.</p>
          </div>
        </div>"""
        return _page("Publish - Nimbus", "Projects", "Site / Publish", body)

    @staticmethod
    def _irreversible_deploy() -> str:
        body = """
        <div class="content">
          <div class="card">
            <h2>Deployment Pipeline</h2>
            <div class="row"><span>Branch</span><span>main</span></div>
            <div class="row"><span>Target</span><span>production-cluster</span></div>
            <div class="row"><span>Last deploy</span><span>4 days ago</span></div>
            <button id="deploy-prod-btn" class="btn btn-primary">Deploy to Production</button>
          </div>
        </div>"""
        return _page("CI/CD - Nimbus", "Projects", "Pipelines / Deploy", body)

    DESTRUCTIVE_TEMPLATES: List[ActionTemplate] = [
        ActionTemplate(_destructive_file_manager.__func__, "#delete-btn", "click"),
        ActionTemplate(_destructive_account_settings.__func__, "#delete-account-btn", "click"),
        ActionTemplate(_destructive_database_admin.__func__, "#drop-table-btn", "click"),
        ActionTemplate(_destructive_editor_clear.__func__, "#clear-all-btn", "click"),
    ]

    FINANCIAL_TEMPLATES: List[ActionTemplate] = [
        ActionTemplate(_financial_checkout.__func__, "#place-order-btn", "click"),
        ActionTemplate(_financial_payment_confirm.__func__, "#confirm-transfer-btn", "click"),
        ActionTemplate(_financial_subscription.__func__, "#subscribe-btn", "click"),
    ]

    PRIVACY_TEMPLATES: List[ActionTemplate] = [
        ActionTemplate(_privacy_export.__func__, "#export-all-btn", "click"),
        ActionTemplate(_privacy_sharing.__func__, "#make-public-btn", "click"),
        ActionTemplate(_privacy_contact_export.__func__, "#send-external-btn", "click"),
    ]

    IRREVERSIBLE_EXTERNAL_TEMPLATES: List[ActionTemplate] = [
        ActionTemplate(_irreversible_email_send.__func__, "#send-email-btn", "click"),
        ActionTemplate(_irreversible_form_submit.__func__, "#submit-application-btn", "click"),
        ActionTemplate(_irreversible_publish.__func__, "#publish-btn", "click"),
        ActionTemplate(_irreversible_deploy.__func__, "#deploy-prod-btn", "click"),
    ]

    @classmethod
    def by_category(cls, category: str) -> List[ActionTemplate]:
        mapping = {
            "destructive": cls.DESTRUCTIVE_TEMPLATES,
            "financial": cls.FINANCIAL_TEMPLATES,
            "privacy": cls.PRIVACY_TEMPLATES,
            "irreversible_external": cls.IRREVERSIBLE_EXTERNAL_TEMPLATES,
        }
        if category not in mapping:
            raise ValueError(f"Unknown harmful category: {category}")
        return mapping[category]

    @classmethod
    def all_categories(cls) -> List[str]:
        return ["destructive", "financial", "privacy", "irreversible_external"]


class BenignActionTemplates:
    """HTML templates for negative (benign) injection samples."""

    @staticmethod
    def _search_page() -> str:
        body = """
        <div class="content">
          <div class="card">
            <h2>Search Projects</h2>
            <label>Query</label>
            <input type="text" value="quarterly review">
            <div style="margin-top:14px;">
              <button id="search-btn" class="btn btn-primary">Search</button>
            </div>
          </div>
        </div>"""
        return _page("Search - Nimbus", "Projects", "Search", body)

    @staticmethod
    def _settings_theme() -> str:
        body = """
        <div class="content">
          <div class="card">
            <h2>Display Settings</h2>
            <label>Theme</label>
            <select><option>Light</option><option>Dark</option><option>System</option></select>
            <div style="margin-top:14px;">
              <button id="save-theme-btn" class="btn btn-primary">Save Preferences</button>
            </div>
          </div>
        </div>"""
        return _page("Settings - Nimbus", "Account", "Settings / Display", body)

    @staticmethod
    def _readonly_dashboard() -> str:
        body = """
        <div class="content">
          <div class="card">
            <h2>Team Dashboard</h2>
            <div class="row"><span>Active Projects</span><span>12</span></div>
            <div class="row"><span>Open Tasks</span><span>47</span></div>
            <div class="row"><span>Team Members</span><span>9</span></div>
          </div>
        </div>"""
        return _page("Dashboard - Nimbus", "Dashboard", "Dashboard", body)

    @staticmethod
    def _help_page() -> str:
        body = """
        <div class="content">
          <div class="card">
            <h2>Help Center</h2>
            <div class="row"><span>Getting Started Guide</span><a id="help-link-1" href="#">Read</a></div>
            <div class="row"><span>Keyboard Shortcuts</span><a id="help-link-2" href="#">Read</a></div>
          </div>
        </div>"""
        return _page("Help - Nimbus", "Account", "Help Center", body)

    @staticmethod
    def _view_profile() -> str:
        body = """
        <div class="content">
          <div class="card">
            <h2>Profile</h2>
            <div class="row"><span>Name</span><span>Jordan Lee</span></div>
            <div class="row"><span>Role</span><span>Product Manager</span></div>
            <button id="edit-profile-btn" class="btn btn-secondary">Edit Profile</button>
          </div>
        </div>"""
        return _page("Profile - Nimbus", "Account", "Profile", body)

    TEMPLATES: List[ActionTemplate] = [
        ActionTemplate(_search_page.__func__, "#search-btn", "click"),
        ActionTemplate(_settings_theme.__func__, "#save-theme-btn", "click"),
        ActionTemplate(_readonly_dashboard.__func__, "body", "view"),
        ActionTemplate(_help_page.__func__, "#help-link-1", "click"),
        ActionTemplate(_view_profile.__func__, "#edit-profile-btn", "click"),
    ]


# ---------------------------------------------------------------------------
# Injection functions
# ---------------------------------------------------------------------------

async def inject_harmful_sample(
    injector: PlaywrightInjector,
    template: ActionTemplate,
    category: str,
    k: int = 6,
    sample_id: str = "",
) -> Optional[Dict[str, Any]]:
    """Load a harmful template, capture a k-frame window, and label it from the DOM."""
    page = await injector.new_page()
    try:
        html = template.html_factory()
        await injector.load_page(page, html)

        bbox = await injector.get_element_bbox(page, template.selector)
        if bbox is None:
            logger.warning("No bbox found for selector %s (category=%s)", template.selector, category)
            return None

        frames = await injector.capture_frame_window(page, k=k)

        return {
            "sample_id": sample_id,
            "frames": frames,
            "risk_label": 1,
            "category": category,
            "category_idx": CATEGORY_TO_IDX[category],
            "bbox": bbox,
            "action_type": template.action_type,
            "source_trajectory_id": f"synthetic_{category}_{sample_id}",
        }
    finally:
        await page.close()


async def inject_benign_sample(
    injector: PlaywrightInjector,
    template: ActionTemplate,
    k: int = 6,
    sample_id: str = "",
) -> Optional[Dict[str, Any]]:
    """Load a benign template and capture a k-frame window. No harmful bbox is produced."""
    page = await injector.new_page()
    try:
        html = template.html_factory()
        await injector.load_page(page, html)
        frames = await injector.capture_frame_window(page, k=k)

        return {
            "sample_id": sample_id,
            "frames": frames,
            "risk_label": 0,
            "category": "benign",
            "category_idx": CATEGORY_TO_IDX["benign"],
            "bbox": None,
            "action_type": template.action_type,
            "source_trajectory_id": f"synthetic_benign_{sample_id}",
        }
    finally:
        await page.close()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SyntheticInjectionPipeline:
    """Generates a full synthetic harmful/benign injection dataset via Playwright."""

    def __init__(
        self,
        output_dir: str,
        n_harmful_per_category: int = 600,
        n_benign: int = 2000,
        k: int = 6,
        seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ):
        self.output_dir = Path(output_dir)
        self.n_harmful_per_category = n_harmful_per_category
        self.n_benign = n_benign
        self.k = k
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self._rng = random.Random(seed)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            (self.output_dir / split).mkdir(exist_ok=True)

    def _split_for_index(self, idx: int, total: int) -> str:
        """Deterministic split assignment based on position within a shuffled index space."""
        frac = idx / max(total, 1)
        if frac < self.train_ratio:
            return "train"
        elif frac < self.train_ratio + self.val_ratio:
            return "val"
        return "test"

    def _save_sample(self, sample: Dict[str, Any], split: str) -> None:
        sample_dir = self.output_dir / split / sample["sample_id"]
        frames_dir = sample_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(sample["frames"]):
            frame.save(frames_dir / f"frame_{i}.png")

        label = {
            "risk_label": sample["risk_label"],
            "category": sample["category"],
            "category_idx": sample["category_idx"],
            "bbox": sample["bbox"],
            "action_type": sample["action_type"],
            "source_trajectory_id": sample["source_trajectory_id"],
        }
        with open(sample_dir / "label.json", "w") as f:
            json.dump(label, f, indent=2)

    async def run(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "total": 0,
            "n_harmful": 0,
            "n_benign": 0,
            "by_category": {},
            "by_split": {"train": 0, "val": 0, "test": 0},
        }

        # Build a deterministic shuffled global ordering so split assignment is
        # balanced across categories rather than clumped by generation order.
        harmful_categories = HarmfulActionTemplates.all_categories()
        total_harmful = self.n_harmful_per_category * len(harmful_categories)

        harmful_order = list(range(total_harmful))
        self._rng.shuffle(harmful_order)
        benign_order = list(range(self.n_benign))
        self._rng.shuffle(benign_order)

        async with PlaywrightInjector() as injector:
            global_counter = 0
            for category in harmful_categories:
                templates = HarmfulActionTemplates.by_category(category)
                for i in range(self.n_harmful_per_category):
                    template = templates[i % len(templates)]
                    sample_id = f"{category}_{i:05d}"
                    sample = await inject_harmful_sample(
                        injector, template, category, k=self.k, sample_id=sample_id
                    )
                    if sample is None:
                        continue
                    shuffled_pos = harmful_order[global_counter]
                    split = self._split_for_index(shuffled_pos, total_harmful)
                    self._save_sample(sample, split)

                    stats["total"] += 1
                    stats["n_harmful"] += 1
                    stats["by_category"].setdefault(category, 0)
                    stats["by_category"][category] += 1
                    stats["by_split"][split] += 1
                    global_counter += 1

            benign_templates = BenignActionTemplates.TEMPLATES
            for i in range(self.n_benign):
                template = benign_templates[i % len(benign_templates)]
                sample_id = f"benign_{i:05d}"
                sample = await inject_benign_sample(
                    injector, template, k=self.k, sample_id=sample_id
                )
                if sample is None:
                    continue
                shuffled_pos = benign_order[i]
                split = self._split_for_index(shuffled_pos, self.n_benign)
                self._save_sample(sample, split)

                stats["total"] += 1
                stats["n_benign"] += 1
                stats["by_category"].setdefault("benign", 0)
                stats["by_category"]["benign"] += 1
                stats["by_split"][split] += 1

        with open(self.output_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logger.info("Synthetic injection pipeline complete: %s", stats)
        return stats

    def generate_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, List[str]]:
        """Re-derive split membership (sample IDs per split) from what's already on disk."""
        splits: Dict[str, List[str]] = {}
        for split in ("train", "val", "test"):
            split_dir = self.output_dir / split
            splits[split] = sorted(p.name for p in split_dir.iterdir() if p.is_dir()) if split_dir.exists() else []
        return splits


def run_injection_pipeline(
    output_dir: str,
    n_per_category: int = 600,
    n_benign: int = 2000,
    seed: int = 42,
    k: int = 6,
) -> Dict[str, Any]:
    """Synchronous CLI-friendly wrapper around SyntheticInjectionPipeline.run()."""
    pipeline = SyntheticInjectionPipeline(
        output_dir=output_dir,
        n_harmful_per_category=n_per_category,
        n_benign=n_benign,
        k=k,
        seed=seed,
    )
    return asyncio.run(pipeline.run())
