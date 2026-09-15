#!/usr/bin/env python3
"""
Run a CoralscapesV1/V2 Dino models on an image / folder of images and
generate an interactive HTML viewer or a static side-by-side png/jpg.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import math
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Encoding settings
JPEG_QUALITY = 90
PNG_COMPRESS_LEVEL = 6


def _pick_device(user_device: str | None) -> torch.device:
    if user_device:
        return torch.device(user_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_hub_module(repo_id: str) -> tuple[Any, Path]:
    root = Path(snapshot_download(repo_id=repo_id))
    hub_py = root / "coralscapes_hub_model.py"
    if not hub_py.is_file():
        raise FileNotFoundError(f"Missing {hub_py} in Hub snapshot for repo_id={repo_id!r}")
    spec = importlib.util.spec_from_file_location("coralscapes_hub_model", hub_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {hub_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, root


def _is_int_like(x: Any) -> bool:
    try:
        int(x)
        return True
    except Exception:
        return False


def _load_id_to_colour(
    colours_json: Path, classes_json: Path | None = None
) -> dict[int, tuple[int, int, int]]:
    colours = json.loads(colours_json.read_text())
    out: dict[int, tuple[int, int, int]] = {}

    if isinstance(colours, list):
        for idx, rgb in enumerate(colours):
            if isinstance(rgb, (list, tuple)) and len(rgb) == 3:
                out[int(idx)] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        return out

    if not isinstance(colours, dict):
        raise TypeError(f"Unsupported colours.json format: {type(colours)!r}")

    numeric_keys = all(_is_int_like(k) for k in colours.keys())

    if numeric_keys:
        for k, rgb in colours.items():
            if isinstance(rgb, (list, tuple)) and len(rgb) == 3:
                out[int(k)] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        return out

    if classes_json is None or not classes_json.is_file():
        raise FileNotFoundError(
            "colours.json uses class names, so classes.json is required to map names->ids."
        )
    classes = json.loads(classes_json.read_text())
    for class_name, cid in classes.items():
        if class_name not in colours:
            continue
        rgb = colours[class_name]
        if isinstance(rgb, (list, tuple)) and len(rgb) == 3:
            out[int(cid)] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return out


def _load_id_to_name(classes_json: Path) -> dict[int, str]:
    """Load class-id -> class-name mapping (list, {id: name}, or {name: id})."""
    data = json.loads(classes_json.read_text())
    out: dict[int, str] = {}

    if isinstance(data, list):
        for idx, name in enumerate(data):
            out[int(idx)] = str(name)
        return out

    if isinstance(data, dict):
        keys_numeric = all(_is_int_like(k) for k in data.keys())
        if keys_numeric:
            for k, v in data.items():
                out[int(k)] = str(v)
        else:
            for name, v in data.items():
                if _is_int_like(v):
                    out[int(v)] = str(name)
        return out

    raise TypeError(f"Unsupported classes.json format: {type(data)!r}")


def _build_colour_lut(id_to_colour: dict[int, tuple[int, int, int]]) -> np.ndarray:
    max_id = max(id_to_colour.keys()) if id_to_colour else 0
    lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for cid, rgb in id_to_colour.items():
        lut[int(cid)] = np.asarray(rgb, dtype=np.uint8)
    return lut


def _colourize(mask_hw: np.ndarray, lut: np.ndarray) -> np.ndarray:
    clipped = np.clip(mask_hw, 0, lut.shape[0] - 1).astype(np.int64, copy=False)
    return lut[clipped]


def _load_train_sizes(config_path: Path) -> list[tuple[int, int]]:
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    sizes_raw = cfg.get("augment", {}).get("train_sizes")
    if not sizes_raw:
        raise KeyError(f"augment.train_sizes not found in {config_path}")
    sizes: list[tuple[int, int]] = []
    for hw in sizes_raw:
        if len(hw) != 2:
            raise ValueError(f"Expected [H, W], got {hw!r} in {config_path}")
        sizes.append((int(hw[0]), int(hw[1])))
    return sizes


def _pick_closest_size(
    sizes: list[tuple[int, int]], native_h: int, native_w: int
) -> tuple[int, int]:
    if native_h <= 0 or native_w <= 0:
        raise ValueError(f"Invalid native size: ({native_h}, {native_w})")
    target = math.log(native_w / native_h)
    return min(sizes, key=lambda hw: abs(math.log(hw[1] / hw[0]) - target))


@lru_cache(maxsize=None)
def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _make_legend_image(
    class_ids: Iterable[int],
    id_to_colour: dict[int, tuple[int, int, int]],
    id_to_name: dict[int, str],
    swatch: int = 36,
    row_h: int = 48,
    pad: int = 18,
    font_size: int = 26,
) -> np.ndarray:
    ids = sorted(set(int(c) for c in class_ids))
    font = _load_font(font_size)
    labels = [id_to_name.get(cid, f"class {cid}") for cid in ids]

    dummy = Image.new("RGB", (10, 10))
    ddraw = ImageDraw.Draw(dummy)
    text_w = max((ddraw.textlength(lbl, font=font) for lbl in labels), default=60)

    width = int(pad * 3 + swatch + text_w)
    height = pad * 2 + row_h * max(len(ids), 1)
    legend = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(legend)

    y = pad
    for cid, label in zip(ids, labels):
        colour = tuple(int(c) for c in id_to_colour.get(cid, (128, 128, 128)))
        draw.rectangle([pad, y, pad + swatch, y + swatch], fill=colour, outline=(0, 0, 0))
        draw.text(
            (pad * 2 + swatch, y + (swatch - font_size) // 2),
            label,
            fill=(0, 0, 0),
            font=font,
        )
        y += row_h

    return np.asarray(legend, dtype=np.uint8)


def _hstack_with_legend(image_rgb: np.ndarray, legend_rgb: np.ndarray) -> np.ndarray:
    h = max(image_rgb.shape[0], legend_rgb.shape[0])

    def _pad(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == h:
            return img
        filler = np.full((h - img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
        return np.concatenate([img, filler], axis=0)

    divider = np.full((h, 2, 3), 200, dtype=np.uint8)
    return np.concatenate([_pad(image_rgb), divider, _pad(legend_rgb)], axis=1)


def _find_images(input_dir: Path, recursive: bool) -> list[Path]:
    it = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _b64_jpeg(arr: np.ndarray) -> str:
    """Full-res JPEG data URI. optimize+progressive shave a few % off size for free."""
    buf = BytesIO()
    Image.fromarray(arr).save(
        buf, format="JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True
    )
    return _b64_data_uri(buf.getvalue(), "image/jpeg")


def _b64_png_l(mask: np.ndarray) -> str:
    """Lossless grayscale PNG data URI encoding per-pixel class ids (0-255)."""
    buf = BytesIO()
    Image.fromarray(mask.astype(np.uint8), mode="L").save(buf, format="PNG", compress_level=9)
    return _b64_data_uri(buf.getvalue(), "image/png")


def _b64_data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


_HTML_VIEWER_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script>
// Runs before first paint to avoid a theme flash.
(function() {{
  var t = null;
  try {{ t = new URLSearchParams(window.location.search).get('theme'); }} catch (e) {{}}
  document.documentElement.setAttribute('data-theme', t === 'light' ? 'light' : 'dark');
}})();
</script>
<style>
  :root {{
    --bg:#111; --fg:#eee; --panel:#1a1a1a; --border:#333; --border2:#444;
    --btn-bg:#2a2a2a; --btn-fg:#ddd; --btn-hover:#333; --accent:#3a6df0;
    --muted:#888; --sw-border:rgba(255,255,255,0.4); --tooltip-bg:rgba(20,20,20,0.95);
  }}
  html[data-theme="light"] {{
    --bg:#f4f4f6; --fg:#171717; --panel:#ffffff; --border:#e0e0e3; --border2:#d0d0d5;
    --btn-bg:#eeeef0; --btn-fg:#222; --btn-hover:#e2e2e6; --accent:#3a6df0;
    --muted:#666; --sw-border:rgba(0,0,0,0.35); --tooltip-bg:rgba(255,255,255,0.97);
  }}
  html, body {{ margin:0; padding:0; background:var(--bg); color:var(--fg);
    font-family: -apple-system, Segoe UI, Arial, sans-serif; }}
  #bar {{ padding:10px 14px; background:var(--panel); border-bottom:1px solid var(--border);
    display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
  #bar h1 {{ font-size:14px; font-weight:600; margin:0; color:var(--muted); flex:1 1 auto;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  button.viewbtn, button.iconbtn {{ background:var(--btn-bg); color:var(--btn-fg);
    border:1px solid var(--border2); border-radius:5px; padding:6px 12px; font-size:13px;
    cursor:pointer; }}
  button.viewbtn:hover, button.iconbtn:hover {{ background:var(--btn-hover); }}
  button.viewbtn .active-word {{ color:var(--accent); font-weight:600; }}
  a.navbtn#galleryBtn {{ margin-right:8px; }}
  #nav {{ display:flex; align-items:center; gap:6px; }}
  a.navbtn {{ background:var(--btn-bg); color:var(--btn-fg); border:1px solid var(--border2);
    border-radius:5px; padding:6px 12px; font-size:13px; cursor:pointer; text-decoration:none; }}
  a.navbtn:hover {{ background:var(--btn-hover); }}
  a.navbtn.disabled {{ opacity:0.35; pointer-events:none; }}
  #navpos {{ font-size:12px; color:var(--muted); padding:0 4px; white-space:nowrap; }}
  #layout {{ display:flex; align-items:flex-start; gap:0; width:100%; box-sizing:border-box; }}
  #viewportCol {{ flex:1 1 auto; display:flex; flex-direction:column;
    align-items:center; min-width:0; }}
  /* Sized from known dimensions, not decoded image -- avoids layout shift. */
  #viewport {{ position:relative; overflow:hidden; border:1px solid var(--border);
    border-radius:6px;
    width: min(calc(100vw - 260px), calc((100vh - 90px) * {native_w} / {native_h}));
    aspect-ratio: {native_w} / {native_h}; }}
  #stage {{ position:relative; display:inline-block; transform-origin:0 0; width:100%; }}
  #stage img {{ display:block; width:100%; height:auto; }}
  #overlayCanvas {{ position:absolute; top:0; left:0; width:100%; height:100%;
    display:none; }}
  #stage canvas {{ position:absolute; top:0; left:0; width:100%; height:100%;
    pointer-events:none; }}
  #zoomHint {{ font-size:11px; color:var(--muted); margin:6px 2px 0; }}
  #tooltip {{ position:fixed; pointer-events:none; background:var(--tooltip-bg);
    border:1px solid var(--border2); border-radius:6px; padding:6px 10px; font-size:13px;
    color:var(--fg); display:none; z-index:10; white-space:nowrap; }}
  #tooltip .sw {{ display:inline-block; width:11px; height:11px; border-radius:2px;
    margin-right:6px; vertical-align:middle; border:1px solid var(--sw-border); }}
  #legend {{ width:250px; flex:0 0 250px; max-height:calc(100vh - 60px); overflow-y:auto;
    padding:10px; box-sizing:border-box; }}
  #legend h2 {{ font-size:12px; text-transform:uppercase; letter-spacing:.04em;
    color:var(--muted); margin:4px 6px 8px; }}
  .lrow {{ display:flex; align-items:center; gap:8px; padding:5px 6px; border-radius:5px;
    cursor:pointer; font-size:13px; border:1px solid transparent; }}
  .lrow:hover {{ background:var(--btn-hover); }}
  .lrow.selected {{ background:var(--btn-hover); border-color:var(--accent);
    box-shadow: inset 0 0 0 1px var(--accent); font-weight:600; }}
  .lrow .sw {{ width:14px; height:14px; border-radius:3px; flex:0 0 14px;
    border:1px solid var(--sw-border); }}
</style>
</head>
<body>
<div id="bar">
  {gallery_link}
  <h1>{title}</h1>
  {view_buttons}
  <div id="nav">
    <a class="navbtn{prev_disabled}" id="prevBtn" href="{prev_href_attr}"
       title="Go to the previous image (Left Arrow)">Prev (&larr;)</a>
    <span id="navpos">{position_label}</span>
    <a class="navbtn{next_disabled}" id="nextBtn" href="{next_href_attr}"
       title="Go to the next image (Right Arrow)">Next (&rarr;)</a>
  </div>
  <button class="iconbtn" id="themeBtn" onclick="toggleTheme()"
    title="Switch between light and dark mode">
    <span id="themeIcon">&#127769;</span> Theme
  </button>
</div>
<div id="layout">
  <div id="viewportCol">
    <div id="viewport">
      <div id="stage">
        <img id="mainImg" src="{initial_src}" width="{native_w}" height="{native_h}">
        <canvas id="overlayCanvas"></canvas>
        <canvas id="hl"></canvas>
      </div>
    </div>
    <div id="zoomHint">Ctrl+scroll to zoom &middot; double-click to reset</div>
  </div>
  <div id="legend">
    <h2>Classes in this image</h2>
    <div id="legendRows"></div>
  </div>
</div>
<div id="tooltip"><span class="sw" id="ttSw"></span><span id="ttText"></span></div>
<script>
const ORIGINAL_SRC = {original_src_js};
const MASK_SRC = {mask_src_js};
const CLASS_NAMES = {class_names_js};
const CLASS_COLOURS = {class_colours_js};
const PRESENT_IDS = {present_ids_js};

const mainImg = document.getElementById('mainImg');
const overlayCanvas = document.getElementById('overlayCanvas');
const hl = document.getElementById('hl');
const hlCtx = hl.getContext('2d');
const tooltip = document.getElementById('tooltip');
const ttSw = document.getElementById('ttSw');
const ttText = document.getElementById('ttText');
const legendRows = document.getElementById('legendRows');

// Ctrl+scroll zoom: CSS transform only, no re-render needed.
const viewport = document.getElementById('viewport');
const stageEl = document.getElementById('stage');
let zoomLevel = 1;
const ZOOM_MIN = 1, ZOOM_MAX = 6, ZOOM_STEP = 1.15;

function applyZoom() {{
  stageEl.style.transform = 'scale(' + zoomLevel + ')';
  // Only scrollable when zoomed in (avoids a spurious scrollbar at 1x).
  viewport.style.overflow = zoomLevel > 1.001 ? 'auto' : 'hidden';
}}

viewport.addEventListener('wheel', function(e) {{
  if (!e.ctrlKey) return;
  e.preventDefault();
  const rect = viewport.getBoundingClientRect();
  const px = e.clientX - rect.left + viewport.scrollLeft;
  const py = e.clientY - rect.top + viewport.scrollTop;
  const prevZoom = zoomLevel;
  zoomLevel = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, zoomLevel * (e.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP)));
  const scaleChange = zoomLevel / prevZoom;
  applyZoom();
  viewport.scrollLeft = px * scaleChange - (e.clientX - rect.left);
  viewport.scrollTop = py * scaleChange - (e.clientY - rect.top);
}}, {{ passive: false }});

viewport.addEventListener('dblclick', function() {{
  zoomLevel = 1;
  applyZoom();
  viewport.scrollLeft = 0;
  viewport.scrollTop = 0;
}});

let maskData = null, maskW = 0, maskH = 0;
let smallMaskData = null, smallW = 0, smallH = 0;
let origData = null;
let overlayReady = false;

const HIGHLIGHT_MAX_DIM = 700; // highlight tint doesn't need full res; cuts hover CPU cost a lot

const maskImg = new Image();
maskImg.onload = function() {{
  maskW = maskImg.naturalWidth;
  maskH = maskImg.naturalHeight;
  const off = document.createElement('canvas');
  off.width = maskW; off.height = maskH;
  const octx = off.getContext('2d');
  octx.drawImage(maskImg, 0, 0);
  maskData = octx.getImageData(0, 0, maskW, maskH).data;

  // Downsampled mask for highlight rendering only (tooltip uses full-res maskData).
  const scale = Math.min(1, HIGHLIGHT_MAX_DIM / Math.max(maskW, maskH));
  smallW = Math.max(1, Math.round(maskW * scale));
  smallH = Math.max(1, Math.round(maskH * scale));
  smallMaskData = new Uint8ClampedArray(smallW * smallH);
  for (let sy = 0; sy < smallH; sy++) {{
    const fy = Math.min(maskH - 1, Math.floor(sy / scale));
    for (let sx = 0; sx < smallW; sx++) {{
      const fx = Math.min(maskW - 1, Math.floor(sx / scale));
      smallMaskData[sy * smallW + sx] = maskData[(fy * maskW + fx) * 4];
    }}
  }}

  hl.width = smallW; hl.height = smallH;
  tryRenderOverlay();
}};
maskImg.src = MASK_SRC;

// Overlay = 0.5*original + 0.5*class_colour(mask), computed client-side, never stored.
function grabOrigData() {{
  const off = document.createElement('canvas');
  off.width = mainImg.naturalWidth;
  off.height = mainImg.naturalHeight;
  off.getContext('2d').drawImage(mainImg, 0, 0);
  origData = off.getContext('2d').getImageData(0, 0, off.width, off.height).data;
  tryRenderOverlay();
}}
if (ORIGINAL_SRC) {{
  if (mainImg.complete && mainImg.naturalWidth > 0) grabOrigData();
  else mainImg.addEventListener('load', grabOrigData);
}}

function tryRenderOverlay() {{
  if (!ORIGINAL_SRC || !maskData || !origData || overlayReady) return;
  const out = new Uint8ClampedArray(maskW * maskH * 4);
  const cache = {{}};
  for (let i = 0; i < maskW * maskH; i++) {{
    const cid = maskData[i * 4];
    let t = cache[cid];
    if (!t) {{ t = colourTripleFor(cid); cache[cid] = t; }}
    const o = i * 4;
    out[o]     = Math.round(0.5 * origData[o]     + 0.5 * t[0]);
    out[o + 1] = Math.round(0.5 * origData[o + 1] + 0.5 * t[1]);
    out[o + 2] = Math.round(0.5 * origData[o + 2] + 0.5 * t[2]);
    out[o + 3] = 255;
  }}
  overlayCanvas.width = maskW;
  overlayCanvas.height = maskH;
  overlayCanvas.getContext('2d').putImageData(new ImageData(out, maskW, maskH), 0, 0);
  overlayReady = true;
  if (currentView === 'overlay') overlayCanvas.style.display = 'block';
}}

function classIdAt(px, py) {{
  if (!maskData) return null;
  const x = Math.min(maskW - 1, Math.max(0, px));
  const y = Math.min(maskH - 1, Math.max(0, py));
  return maskData[(y * maskW + x) * 4];
}}

function nameFor(id) {{ return CLASS_NAMES[id] || ('class ' + id); }}
function colourFor(id) {{ return CLASS_COLOURS[id] || 'rgb(128,128,128)'; }}

mainImg.addEventListener('mousemove', function(e) {{
  if (!maskData) return;
  const rect = mainImg.getBoundingClientRect();
  const relX = e.clientX - rect.left, relY = e.clientY - rect.top;
  const nx = Math.floor(relX * maskW / rect.width);
  const ny = Math.floor(relY * maskH / rect.height);
  const id = classIdAt(nx, ny);
  if (id === null) return;
  ttSw.style.background = colourFor(id);
  ttText.textContent = nameFor(id);
  tooltip.style.display = 'block';
  tooltip.style.left = (e.clientX + 14) + 'px';
  tooltip.style.top = (e.clientY + 12) + 'px';
}});
mainImg.addEventListener('mouseleave', function() {{ tooltip.style.display = 'none'; }});

function colourTripleFor(id) {{
  const m = /rgb\\((\\d+),\\s*(\\d+),\\s*(\\d+)\\)/.exec(colourFor(id));
  return m ? [+m[1], +m[2], +m[3]] : [255, 0, 0];
}}
function renderHighlightIds(ids) {{
  if (!smallMaskData || ids.size === 0) {{ clearHighlight(); return; }}
  const triples = {{}};
  ids.forEach(function(id) {{ triples[id] = colourTripleFor(id); }});
  const out = new Uint8ClampedArray(smallW * smallH * 4);
  for (let i = 0; i < smallW * smallH; i++) {{
    const t = triples[smallMaskData[i]];
    if (t) {{
      out[i * 4] = t[0]; out[i * 4 + 1] = t[1]; out[i * 4 + 2] = t[2]; out[i * 4 + 3] = 170;
    }}
  }}
  hlCtx.putImageData(new ImageData(out, smallW, smallH), 0, 0);
}}
function clearHighlight() {{ hlCtx.clearRect(0, 0, smallW, smallH); }}

const selectedIds = new Set();
let hoverTimer = null;
const HOVER_DEBOUNCE_MS = 45; // avoid rendering while scanning across rows

for (const id of PRESENT_IDS) {{
  const row = document.createElement('div');
  row.className = 'lrow';
  row.title = 'Click to pin this class highlighted (multiple can be pinned); click again to unpin.';
  const sw = document.createElement('span');
  sw.className = 'sw';
  sw.style.background = colourFor(id);
  const label = document.createElement('span');
  label.textContent = nameFor(id);
  row.appendChild(sw);
  row.appendChild(label);
  row.addEventListener('mouseenter', function() {{
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(function() {{
      renderHighlightIds(new Set([...selectedIds, id]));
    }}, HOVER_DEBOUNCE_MS);
  }});
  row.addEventListener('mouseleave', function() {{
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(function() {{ renderHighlightIds(selectedIds); }}, HOVER_DEBOUNCE_MS);
  }});
  row.addEventListener('click', function() {{
    clearTimeout(hoverTimer); // clicks render immediately
    if (selectedIds.has(id)) {{
      selectedIds.delete(id);
      row.classList.remove('selected');
    }} else {{
      selectedIds.add(id);
      row.classList.add('selected');
    }}
    renderHighlightIds(selectedIds);
  }});
  row.id = 'lrow_' + id;
  legendRows.appendChild(row);
}}

function setView(which) {{
  currentView = which;
  const wOverlay = document.getElementById('vw_overlay');
  const wOriginal = document.getElementById('vw_original');
  if (wOverlay) wOverlay.classList.toggle('active-word', which === 'overlay');
  if (wOriginal) wOriginal.classList.toggle('active-word', which === 'original');
  overlayCanvas.style.display = (which === 'overlay' && overlayReady) ? 'block' : 'none';
}}
function toggleView() {{
  if (!ORIGINAL_SRC) return;
  setView(currentView === 'overlay' ? 'original' : 'overlay');
  syncNavLinksPrefs();
}}
let currentView = 'overlay';

function currentTheme() {{ return document.documentElement.getAttribute('data-theme') || 'dark'; }}

function withPrefs(href) {{
  if (!href || href === '#') return href;
  return href.split('?')[0] + '?theme=' + currentTheme() + '&view=' + currentView;
}}
function syncNavLinksPrefs() {{
  ['prevBtn', 'nextBtn', 'galleryBtn'].forEach(function(id) {{
    const el = document.getElementById(id);
    if (!el) return;
    const href = el.getAttribute('href');
    if (!href || href === '#') return;
    el.setAttribute('href', withPrefs(href));
  }});
}}

function toggleTheme() {{
  const next = currentTheme() === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', next);
  const icon = document.getElementById('themeIcon');
  if (icon) icon.textContent = next === 'light' ? '\\u2600' : '\\u{{1F313}}';
  syncNavLinksPrefs();
}}
(function() {{
  let v = null;
  try {{ v = new URLSearchParams(window.location.search).get('view'); }} catch (e) {{}}
  if (v === 'original' && ORIGINAL_SRC) setView('original');

  const icon = document.getElementById('themeIcon');
  if (icon) icon.textContent = currentTheme() === 'light' ? '\\u2600' : '\\u{{1F313}}';
  syncNavLinksPrefs();
}})();

const PREV_HREF = {prev_href_js};
const NEXT_HREF = {next_href_js};
document.addEventListener('keydown', function(e) {{
  if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
  if (e.key === 'ArrowLeft' && PREV_HREF) window.location.href = withPrefs(PREV_HREF);
  if (e.key === 'ArrowRight' && NEXT_HREF) window.location.href = withPrefs(NEXT_HREF);
  if (e.key === 'o' || e.key === 'O') toggleView();
}});
</script>
</body>
</html>
"""


def _build_html_viewer(
    *,
    title: str,
    frame_rgb: np.ndarray,
    view_img: np.ndarray,
    mask_only: bool,
    pred: np.ndarray,
    id_to_colour: dict[int, tuple[int, int, int]],
    id_to_name: dict[int, str],
    prev_href: str | None = None,
    next_href: str | None = None,
    position_label: str = "",
    gallery_href: str | None = None,
) -> tuple[str, str]:
    """Returns (html_string, thumbnail_b64).

    Original + mask are embedded inline. Overlay is derived client-side from original + mask.
    """
    mask_src = _b64_png_l(pred)
    original_src = None if mask_only else _b64_jpeg(frame_rgb)
    baked_src = _b64_jpeg(view_img) if mask_only else None

    initial_src = baked_src if mask_only else original_src
    if original_src is not None:
        view_buttons = (
            '<button class="viewbtn" id="toggleViewBtn" onclick="toggleView()" '
            'title="Toggle between the colour overlay and the original photo (press O)">'
            '<span id="vw_overlay" class="active-word">Overlay</span>/'
            '<span id="vw_original">Original</span> (O)'
            "</button>"
        )
    else:
        view_buttons = ""

    present_ids = sorted(int(c) for c in np.unique(pred).tolist())
    class_names_js = json.dumps(
        {str(i): id_to_name.get(i, f"class {i}") for i in present_ids}
    )
    class_colours_js = json.dumps(
        {str(i): f"rgb({r},{g},{b})" for i, (r, g, b) in
         ((i, id_to_colour.get(i, (128, 128, 128))) for i in present_ids)}
    )
    present_ids_js = json.dumps(present_ids)

    gallery_link = (
        f'<a class="navbtn" id="galleryBtn" href="{gallery_href}">&#9776; Gallery</a>'
        if gallery_href else ""
    )

    html = _HTML_VIEWER_TEMPLATE.format(
        title=title,
        view_buttons=view_buttons,
        gallery_link=gallery_link,
        initial_src=initial_src,
        native_w=pred.shape[1],
        native_h=pred.shape[0],
        original_src_js=json.dumps(original_src),
        mask_src_js=json.dumps(mask_src),
        class_names_js=class_names_js,
        class_colours_js=class_colours_js,
        present_ids_js=present_ids_js,
        prev_href_attr=prev_href or "#",
        next_href_attr=next_href or "#",
        prev_disabled="" if prev_href else " disabled",
        next_disabled="" if next_href else " disabled",
        prev_href_js=json.dumps(prev_href),
        next_href_js=json.dumps(next_href),
        position_label=position_label,
    )

    thumb = Image.fromarray(view_img)
    thumb.thumbnail((320, 320), Image.BILINEAR)
    thumb_buf = BytesIO()
    thumb.save(thumb_buf, format="JPEG", quality=80)
    thumb_b64 = _b64_data_uri(thumb_buf.getvalue(), "image/jpeg")

    return html, thumb_b64


_GALLERY_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script>
(function() {{
  var t = null;
  try {{ t = new URLSearchParams(window.location.search).get('theme'); }} catch (e) {{}}
  document.documentElement.setAttribute('data-theme', t === 'light' ? 'light' : 'dark');
}})();
</script>
<style>
  :root {{
    --bg:#111; --fg:#eee; --panel:#1a1a1a; --border:#2c2c2c; --accent:#3a6df0; --muted:#ccc;
  }}
  html[data-theme="light"] {{
    --bg:#f4f4f6; --fg:#171717; --panel:#ffffff; --border:#e0e0e3; --accent:#3a6df0; --muted:#444;
  }}
  html, body {{ margin:0; padding:20px; background:var(--bg); colour:var(--fg);
    font-family: -apple-system, Segoe UI, Arial, sans-serif; }}
  #head {{ display:flex; align-items:center; gap:10px; margin:0 0 16px; }}
  h1 {{ font-size:16px; colour:var(--muted); margin:0; flex:1 1 auto; }}
  button.iconbtn {{ background:var(--panel); colour:var(--fg); border:1px solid var(--border);
    border-radius:5px; padding:6px 12px; font-size:13px; cursor:pointer; }}
  .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap:14px; }}
  a.card {{ colour:var(--fg); text-decoration:none; background:var(--panel);
    border:1px solid var(--border); border-radius:8px; overflow:hidden; display:block; }}
  a.card:hover {{ border-colour:var(--accent); }}
  a.card img {{ width:100%; display:block; aspect-ratio:4/3; object-fit:cover; }}
  a.card .cap {{ padding:8px 10px; font-size:12px; overflow:hidden; text-overflow:ellipsis;
    white-space:nowrap; }}
</style>
</head>
<body>
<div id="head">
  <h1>{title}</h1>
  <button class="iconbtn" id="themeBtn" onclick="toggleTheme()"
    title="Switch between light and dark mode">
    <span id="themeIcon">&#127769;</span> Theme
  </button>
</div>
<div class="grid">
{cards}
</div>
<script>
function currentTheme() {{ return document.documentElement.getAttribute('data-theme') || 'dark'; }}

function preferredView() {{
  let v = null;
  try {{ v = new URLSearchParams(window.location.search).get('view'); }} catch (e) {{}}
  return v === 'original' ? 'original' : 'overlay';
}}
function withPrefs(href) {{
  if (!href || href === '#') return href;
  return href.split('?')[0] + '?theme=' + currentTheme() + '&view=' + preferredView();
}}
function syncCardLinksPrefs() {{
  document.querySelectorAll('a.card').forEach(function(el) {{
    el.setAttribute('href', withPrefs(el.getAttribute('href')));
  }});
}}

function toggleTheme() {{
  const next = currentTheme() === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', next);
  const icon = document.getElementById('themeIcon');
  if (icon) icon.textContent = next === 'light' ? '\\u2600' : '\\u{{1F313}}';
  syncCardLinksPrefs();
}}
(function() {{
  const icon = document.getElementById('themeIcon');
  if (icon) icon.textContent = currentTheme() === 'light' ? '\\u2600' : '\\u{{1F313}}';
  syncCardLinksPrefs();
}})();
</script>
</body>
</html>
"""


def _build_gallery_html(title: str, entries: list[tuple[str, str, str]]) -> str:
    """entries: list of (relative_html_href, thumb_b64, caption)."""
    cards = "\n".join(
        f'<a class="card" href="{href}"><img src="{thumb}"><div class="cap">{caption}</div></a>'
        for href, thumb, caption in entries
    )
    return _GALLERY_TEMPLATE.format(title=title, cards=cards)


def _process_one(
    image_path: Path,
    output_path: Path,
    *,
    model: Any,
    device: torch.device,
    lut: np.ndarray,
    id_to_colour: dict[int, tuple[int, int, int]],
    id_to_name: dict[int, str],
    train_sizes: list[tuple[int, int]],
    mask_only: bool,
    legend: bool,
    legend_all_classes: bool,
    prev_href: str | None = None,
    next_href: str | None = None,
    position_label: str = "",
    gallery_href: str | None = None,
) -> tuple[str, str] | None:
    """Returns (thumb_b64, title) when an HTML viewer was written, else None."""
    frame_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    H_v, W_v = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
    Hi, Wi = _pick_closest_size(train_sizes, H_v, W_v)

    img = Image.fromarray(frame_rgb).resize((Wi, Hi), Image.BILINEAR)
    px = model.processor(images=img, return_tensors="pt", do_resize=False)["pixel_values"].to(
        device
    )
    with torch.inference_mode():
        logits = model(px)

    logits = F.interpolate(logits.float(), size=(H_v, W_v), mode="bilinear", align_corners=False)
    pred = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)

    pred_colour = _colourize(pred, lut)

    ext = output_path.suffix.lower()

    if ext == ".html":
        view_img = pred_colour if mask_only else (
            (frame_rgb.astype(np.float32) * 0.5) + (pred_colour.astype(np.float32) * 0.5)
        ).clip(0, 255).astype(np.uint8)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html, thumb_b64 = _build_html_viewer(
            title=image_path.name,
            frame_rgb=frame_rgb,
            view_img=view_img,
            mask_only=mask_only,
            pred=pred,
            id_to_colour=id_to_colour,
            id_to_name=id_to_name,
            prev_href=prev_href,
            next_href=next_href,
            position_label=position_label,
            gallery_href=gallery_href,
        )
        output_path.write_text(html, encoding="utf-8")
        return thumb_b64, image_path.name

    if mask_only:
        out_img = pred_colour
    else:
        blend = (
            (frame_rgb.astype(np.float32) * 0.5) + (pred_colour.astype(np.float32) * 0.5)
        ).clip(0, 255).astype(np.uint8)
        out_img = np.concatenate([frame_rgb, blend], axis=1)

    if legend:
        class_ids = id_to_colour.keys() if legend_all_classes else np.unique(pred).tolist()
        legend_img = _make_legend_image(class_ids, id_to_colour, id_to_name)
        out_img = _hstack_with_legend(out_img, legend_img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if ext in (".jpg", ".jpeg"):
        save_kwargs = dict(quality=JPEG_QUALITY)
    elif ext == ".png":
        save_kwargs = dict(compress_level=PNG_COMPRESS_LEVEL)
    else:
        save_kwargs = {}
    Image.fromarray(out_img).save(output_path, **save_kwargs)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--repo-id",
        type=str,
        default="EPFL-ECEO/coralscapesv2-dinov3-vitb-lora-dpt",
        help="Hugging face model repository"
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input image file path or directory",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output name (single input) or output directory name (directory input)"
    )
    p.add_argument(
        "--output-ext",
        type=str,
        default=".html",
        help="Output type (.html | .jpg | .png)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vit_b.yaml"),
        help="Model config, set based on backbone size",
    )
    p.add_argument(
        "--colours",
        type=Path,
        default=Path("dataset_metadata/colours_39.json"),
        help="Class colour json, set based on 39/95 class model",
    )
    p.add_argument(
        "--classes",
        type=Path,
        default=Path("dataset_metadata/classes_39.json"),
        help="Class name json, set based on 39/95 class model",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference, (cuda | cuda:0 | cpu)",
    )
    p.add_argument(
        "--mask-only",
        action="store_true",
        help="Create only the mask, no original comparison.",
    )
    p.add_argument(
        "--no-legend",
        action="store_true",
        help="Don't include a colour-key legend (png/jpg outputs only).",
    )
    p.add_argument(
        "--legend-all-classes",
        action="store_true",
        help="List every known class in the legend (png/jpg outputs only).",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="When --input is a directory, search it recursively for images.",
    )
    args = p.parse_args()

    # --output is just a name; the extension always comes from --output-ext.
    # A missing --output defaults to "<input>_segmented" (no extension yet
    # for directory input -- it becomes a directory, not a file).
    if args.output is None:
        stem = args.input.name if args.input.is_dir() else args.input.stem
        args.output = Path(f"{stem}_segmented")
    if not args.input.is_dir():
        args.output = args.output.with_suffix(args.output_ext)

    device = _pick_device(args.device)
    print(f"device: {device}")

    id_to_colour = _load_id_to_colour(
        args.colours,
        classes_json=args.classes if args.classes.exists() else None,
    )
    id_to_name = (
        _load_id_to_name(args.classes) if args.classes.exists() else {}
    )
    lut = _build_colour_lut(id_to_colour)

    train_sizes = _load_train_sizes(args.config)

    mod, model_root = _load_hub_module(args.repo_id)
    model = mod.Dinov3DPTSegmenter.from_pretrained(
        model_root,
        map_location=device,
        local_files_only=True,
    ).eval()

    common_kwargs = dict(
        model=model,
        device=device,
        lut=lut,
        id_to_colour=id_to_colour,
        id_to_name=id_to_name,
        train_sizes=train_sizes,
        mask_only=args.mask_only,
        legend=not args.no_legend,
        legend_all_classes=args.legend_all_classes,
    )

    if args.input.is_dir():
        image_paths = _find_images(args.input, recursive=args.recursive)
        if not image_paths:
            raise FileNotFoundError(f"No images found in {args.input}")
        print(f"found {len(image_paths)} images in {args.input}")

        # Compute paths up front so viewers can link to sibling files.
        output_paths = [
            (args.output / image_path.relative_to(args.input)).with_suffix(args.output_ext)
            for image_path in image_paths
        ]

        def _rel_href(from_dir: Path, to_path: Path) -> str:
            return os.path.relpath(to_path, from_dir).replace(os.sep, "/")

        gallery_entries: list[tuple[str, str, str]] = []
        n = len(image_paths)
        index_path = args.output / "index.html"
        for i, image_path in enumerate(tqdm(image_paths, desc="images")):
            output_path = output_paths[i]
            extra = {}
            if args.output_ext.lower() == ".html":
                extra["prev_href"] = (
                    _rel_href(output_path.parent, output_paths[i - 1]) if i > 0 else None
                )
                extra["next_href"] = (
                    _rel_href(output_path.parent, output_paths[i + 1]) if i < n - 1 else None
                )
                extra["position_label"] = f"{i + 1} / {n}"
                extra["gallery_href"] = _rel_href(output_path.parent, index_path)
            result = _process_one(image_path, output_path, **common_kwargs, **extra)
            if result is not None:
                thumb_b64, title = result
                href = output_path.relative_to(args.output).as_posix()
                gallery_entries.append((href, thumb_b64, title))
        print(f"Wrote {len(image_paths)} images -> {args.output}")
        if gallery_entries:
            gallery_html = _build_gallery_html(
                title=f"{args.output.name} ({len(gallery_entries)} images)",
                entries=gallery_entries,
            )
            index_path.write_text(gallery_html, encoding="utf-8")
            print(f"Wrote gallery -> {index_path}")
    else:
        if not args.input.is_file():
            raise FileNotFoundError(f"Input not found: {args.input}")
        H_v_W_v = np.asarray(Image.open(args.input)).shape[:2]
        print(f"image: {H_v_W_v[1]}x{H_v_W_v[0]}")
        _process_one(args.input, args.output, **common_kwargs)
        print(f"Wrote -> {args.output}")


if __name__ == "__main__":
    main()
