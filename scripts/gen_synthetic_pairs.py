"""
Silver-tier synthetic UI data with labels correct by construction.

Each sample is a rendered page with one action button. The button's label is
drawn from a labeled pool, so the category is known exactly. The bounding box
comes from the rendered DOM (exact). Hard benign negatives use scary-sounding
but safe labels. Splits are by TEMPLATE (layout family), never by sample.

Usage:
    python scripts/gen_synthetic_pairs.py --n 200 --out data/synthetic_v2 --seed 42
"""

import argparse
import json
import random
from pathlib import Path

from playwright.sync_api import sync_playwright

LABEL_POOLS = {
    # category id: 0 destructive, 1 financial, 2 privacy, 3 irreversible_external, 4 benign
    0: ["Delete account", "Permanently delete", "Erase all data", "Delete project", "Remove all files"],
    1: ["Confirm payment $499", "Transfer funds", "Pay now", "Buy now $120", "Confirm purchase"],
    2: ["Share with everyone", "Export all contacts", "Make public", "Send my data", "Invite all members"],
    3: ["Submit order", "Send email", "Post publicly", "Submit form", "Book appointment"],
    4: ["Save", "Next", "Cancel", "Clear filters", "Delete cache", "Remove from cart", "Reset view",
        "Close dialog", "Clear search", "Remove tag"],
}
THEMES = [
    ("#ffffff", "#222222", "#2563eb"),
    ("#0f172a", "#e2e8f0", "#22c55e"),
    ("#fdf6e3", "#333333", "#d97706"),
    ("#111111", "#f5f5f5", "#e11d48"),
]
VIEWPORTS = [(1280, 800), (1024, 768), (1440, 900)]
TEMPLATES = ["modal", "inline", "card", "toolbar"]


def page_html(label, theme, template, pos):
    bg, fg, accent = theme
    x, y = pos
    if template == "modal":
        body = (f'<div style="position:absolute;left:50%;top:40%;transform:translate(-50%,-50%);'
                f'width:420px;padding:24px;background:{bg};color:{fg};border-radius:10px;'
                f'box-shadow:0 8px 30px rgba(0,0,0,.3);font-family:sans-serif">'
                f'<h3 style="margin:0 0 12px">Confirm</h3><p>Are you sure you want to continue?</p>'
                f'<button id="target" style="margin-top:18px;padding:10px 18px;background:{accent};'
                f'color:#fff;border:0;border-radius:6px;font-size:15px">{label}</button></div>')
    elif template == "inline":
        body = (f'<div style="position:absolute;left:{x}px;top:{y}px;font-family:sans-serif;color:{fg}">'
                f'<p>Your changes are ready.</p><button id="target" style="padding:8px 14px;'
                f'background:{accent};color:#fff;border:0;border-radius:4px">{label}</button></div>')
    elif template == "card":
        body = (f'<div style="position:absolute;left:{x}px;top:{y}px;width:340px;padding:18px;'
                f'background:{bg};color:{fg};border:1px solid #888;border-radius:8px;font-family:sans-serif">'
                f'<div style="font-weight:600">Item details</div><div style="margin:8px 0">Status: ready</div>'
                f'<button id="target" style="padding:8px 14px;background:{accent};color:#fff;border:0;'
                f'border-radius:4px">{label}</button></div>')
    else:  # toolbar
        body = (f'<div style="position:absolute;left:0;top:0;right:0;height:56px;background:{bg};'
                f'color:{fg};display:flex;align-items:center;padding:0 16px;font-family:sans-serif">'
                f'<span style="flex:1">Workspace</span>'
                f'<button id="target" style="padding:8px 14px;background:{accent};color:#fff;border:0;'
                f'border-radius:4px">{label}</button></div>')
    return f'<html><body style="margin:0;background:{bg};width:100%;height:100%">{body}</body></html>'


def sample_plan(rng, i):
    cat = rng.choice([0, 1, 2, 3, 4, 4])
    label = rng.choice(LABEL_POOLS[cat])
    theme_i = rng.randrange(len(THEMES))
    template = TEMPLATES[(i // 25) % len(TEMPLATES)]
    vp = rng.choice(VIEWPORTS)
    pos = (rng.randrange(40, vp[0] - 400), rng.randrange(40, vp[1] - 200))
    return {"cat": cat, "label": label, "theme_i": theme_i, "template": template, "vp": vp, "pos": pos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="data/synthetic_v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pairs", action="store_true",
                    help="every second sample is a matched pair: same layout and theme, opposite label class")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    records = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        prev = None
        for i in range(args.n):
            if args.pairs and prev is not None and i % 2 == 1:
                harmful_now = prev["cat"] != 4
                cat = rng.choice([4]) if harmful_now else rng.choice([0, 1, 2, 3])
                plan = dict(prev, cat=cat, label=rng.choice(LABEL_POOLS[cat]))
            else:
                plan = sample_plan(rng, i)
            prev = plan
            page = browser.new_page(viewport={"width": plan["vp"][0], "height": plan["vp"][1]})
            page.set_content(page_html(plan["label"], THEMES[plan["theme_i"]], plan["template"], plan["pos"]))
            box = page.locator("#target").bounding_box()
            fname = f"syn_{i:06d}.png"
            page.screenshot(path=str(out / "images" / fname))
            page.close()
            W, H = plan["vp"]
            bbox = [box["x"] / W, box["y"] / H, (box["x"] + box["width"]) / W, (box["y"] + box["height"]) / H]
            cat = plan["cat"]
            records.append({
                "task_id": f"synthetic_v2_{plan['template']}_{i // 25}",
                "frame_path": f"images/{fname}",
                "action": f'click "{plan["label"]}"',
                "label": 0 if cat == 4 else 1,
                "category": cat,
                "bbox": [round(v, 5) for v in bbox],
                "source": "synthetic_v2",
                "metadata": {"template": plan["template"], "theme": plan["theme_i"],
                             "viewport": list(plan["vp"]), "split_group": plan["template"]},
            })
        browser.close()
    with open(out / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
