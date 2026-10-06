import os

os.environ.setdefault("MPLBACKEND", "Agg")  # must be set before ANY matplotlib import (macOS GUI backend crashes off the main thread)

import sys
from pathlib import Path

# Streamlit puts only the script's own folder on sys.path. If this file lives inside engine/ (or any subfolder),
# `from engine... import` / `from data... import` fail with ModuleNotFoundError. Add the app folder AND its parent
# (the project root that contains the engine/ and data/ packages) so imports work in either layout.
APP_DIR = Path(__file__).resolve().parent
for _p in (APP_DIR, APP_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import datetime
import difflib
import hashlib
import html
import io
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")  # headless backend: no GUI overhead
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.collections import LineCollection
from matplotlib.patches import Arc, Circle, Rectangle

# Optional paid StatsBomb access. statsbombpy reads SB_USERNAME / SB_PASSWORD from the environment,
# so this also covers calls made inside data/fetcher.py. Set them in .streamlit/secrets.toml or your shell.
try:
    for _k in ("SB_USERNAME", "SB_PASSWORD"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets file: fall back to the free open data
HAS_CREDS = bool(os.environ.get("SB_USERNAME") and os.environ.get("SB_PASSWORD"))

from statsbombpy import sb  # noqa: E402  (must come after the credentials are in the environment)

from engine.adaptive_pagerank import sports_adaptive_pagerank, system_dependency
from data.fetcher import get_match_passing_matrix

st.set_page_config(page_title="PageRank Match Intelligence", layout="wide", page_icon="⚽")

# --- CONFIG ---
# If passing_matrix[i, j] = passes FROM i TO j, outbound volume is the row sum (axis=1).
# If it is passes FROM j TO i, use axis=0. Verify against get_match_passing_matrix.
OUTBOUND_AXIS = 0
BOTTLENECK_THRESHOLD = 1.5
# Parallel catalog fetching is faster but uses threads; keep False until the app is stable on your machine.
PARALLEL_FETCH = False

# st.fragment reruns only the decorated block on widget interaction (Streamlit >= 1.37)
fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None) or (lambda f: f)

# --- THEMES (dark / light). Everything visual reads from these tokens. ---
THEMES = {
    "dark": {
        "bg": "#0b1220", "surface": "#121a2b", "surface-2": "#18233a", "border": "#24304a",
        "text": "#e6edf8", "muted": "#8fa0bd", "field": "#16213a", "field-border": "#33425f", "accent": "#38bdf8", "on-accent": "#04121f", "gold": "#facc15",
        "info-bg": "rgba(56,189,248,.10)", "warn-bg": "rgba(251,191,36,.12)",
        "ok-bg": "rgba(52,211,153,.12)", "err-bg": "rgba(248,113,113,.12)",
        "shadow": "0 1px 2px rgba(0,0,0,.35)",
    },
    "light": {
        "bg": "#f5eed9", "surface": "#fcf7e8", "surface-2": "#efe5c9", "border": "#e0d4b4",
        "text": "#2a2417", "muted": "#7a6d52", "field": "#fffdf5", "field-border": "#c9bb93", "accent": "#0369a1", "on-accent": "#ffffff", "gold": "#d97706",
        "info-bg": "rgba(2,132,199,.08)", "warn-bg": "rgba(245,158,11,.14)",
        "ok-bg": "rgba(5,150,105,.10)", "err-bg": "rgba(220,38,38,.08)",
        "shadow": "0 1px 2px rgba(90,70,20,.10)",
    },
}

BASE_CSS = """
/* Force our surfaces + text colours so Streamlit's own dark/light theme can never leak through
   (this is what made some text invisible in light mode: dark text on leftover dark widget backgrounds). */
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background-color:var(--bg) !important}
[data-testid="stSidebar"],[data-testid="stSidebar"]>div,[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"],[data-testid="stSidebarHeader"]{background-color:var(--surface) !important}
[data-testid="stMain"] *,[data-testid="stSidebar"] *{color:var(--text)}
[data-testid="stHeader"] *,[data-testid="stToolbar"] *{color:var(--muted)}
* {scrollbar-color:var(--border) transparent}
"""

STATIC_CSS = """
html,body,.stApp{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.stApp{background:var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:transparent}
footer{visibility:hidden}
.block-container{padding:1.2rem 1.4rem 3rem;max-width:1320px}
.stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp li,.stApp label,
[data-testid="stWidgetLabel"] p,[data-testid="stExpander"] summary p,[data-testid="stExpander"] summary span{color:var(--text)}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{color:var(--muted)}
[data-testid="stSpinner"] *{color:var(--muted)}

/* inputs */
[data-baseweb="select"]>div{background:var(--surface);border:1px solid var(--border);border-radius:10px}
[data-baseweb="select"] *{color:var(--text)}
[data-baseweb="select"] svg{fill:var(--muted)}
[data-baseweb="popover"] [role="listbox"],[data-baseweb="popover"] ul{background:var(--surface)}
[data-baseweb="popover"] [role="option"]{background:var(--surface);color:var(--text)}
[data-baseweb="popover"] [role="option"]:hover{background:var(--surface-2)}
[data-testid="stExpander"]{background:var(--surface);border:1px solid var(--border);border-radius:14px}
[data-testid="stExpander"] details{border:none}
.stImage img{border-radius:14px}

/* header */
.app-title{font-size:clamp(22px,3.4vw,32px);font-weight:800;margin:0;letter-spacing:-.02em}
.app-sub{color:var(--muted);font-size:14px;margin:2px 0 14px}
.section{font-size:18px;font-weight:700;margin:26px 0 10px}

/* cards */
.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:16px 18px;box-shadow:var(--shadow)}
.sb{display:grid;grid-template-columns:1fr auto 1fr;gap:16px;align-items:start}
.sb .home{text-align:right}.sb .away{text-align:left}
.sb .team{font-size:clamp(17px,2.6vw,26px);font-weight:700}
.sb .score{color:var(--accent);font-size:clamp(28px,5vw,40px);font-weight:800;line-height:1;text-align:center}
.sb .score.na{font-size:16px;color:var(--muted);font-weight:600}
.sb .scorer{color:var(--muted);font-size:13px;margin-top:2px}
.sb .note{color:var(--muted);font-size:12px;margin-top:6px;text-align:center}

.panel-title{display:flex;align-items:center;flex-wrap:wrap;gap:10px;font-size:20px;font-weight:700;margin:4px 0 8px}
.badge{font-size:12px;font-weight:600;padding:2px 9px;border-radius:999px;background:var(--surface-2);border:1px solid var(--border);color:var(--muted)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px;margin:6px 0 12px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:10px 12px}
.tile .k{font-size:12px;color:var(--muted)}
.tile .v{font-size:22px;font-weight:700;color:var(--text)}

.callout{border-radius:12px;padding:10px 14px;font-size:14px;border:1px solid var(--border);margin:8px 0;color:var(--text)}
.callout.info{background:var(--info-bg)}.callout.warn{background:var(--warn-bg)}
.callout.ok{background:var(--ok-bg)}.callout.err{background:var(--err-bg)}

.mvp{background:var(--surface);padding:14px 16px;border-radius:14px;border:1px solid var(--border);
     display:flex;align-items:center;gap:14px;transition:transform .15s ease,border-color .15s ease}
.mvp:hover{transform:translateY(-2px);border-color:var(--accent)}
.avatar{flex:0 0 56px;width:56px;height:56px;border-radius:50%;display:grid;place-items:center;
        background:var(--surface-2);border:2px solid var(--accent);color:var(--accent);font-weight:700;font-size:18px}
.mvp .lbl{color:var(--muted);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em}
.mvp .nm{font-size:19px;font-weight:700;color:var(--text);margin:1px 0}
.mvp .inf{color:var(--accent);font-size:13px;font-weight:600}

/* table */
.tbl-wrap{max-height:380px;overflow:auto;border:1px solid var(--border);border-radius:12px;background:var(--surface)}
table.tbl{width:100%;border-collapse:collapse;font-size:13px}
.tbl th{position:sticky;top:0;background:var(--surface-2);color:var(--muted);text-align:right;font-weight:600;padding:8px 10px;white-space:nowrap}
.tbl td{padding:7px 10px;border-top:1px solid var(--border);text-align:right;color:var(--text);white-space:nowrap}
.tbl th:nth-child(2),.tbl td:nth-child(2){text-align:left}
.tbl th:first-child,.tbl td:first-child{text-align:center;color:var(--muted);width:32px}
.tbl tr:first-child td{font-weight:700}
.bar{display:inline-block;position:relative;width:64px;height:6px;border-radius:99px;background:var(--surface-2);margin-left:8px;vertical-align:middle;overflow:hidden}
.bar>i{position:absolute;left:0;top:0;bottom:0;background:var(--accent);border-radius:99px}

@media (max-width:640px){
  .block-container{padding:.8rem .8rem 2rem}
  .sb{grid-template-columns:1fr;text-align:center}
  .sb .home,.sb .away{text-align:center}
  .mvp{padding:12px;gap:12px}
}
"""


EXTRA_CSS = """
html{scroll-behavior:smooth}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.block-container{padding-top:1rem}

/* rows marked with .keep-row never stack: team A / team B stay side by side on every screen size */
.keep-row{display:none}
div[data-testid="stElementContainer"]:has(.keep-row),.element-container:has(.keep-row){display:none}
div[data-testid="stHorizontalBlock"]:has(.keep-row){flex-wrap:nowrap !important;gap:.75rem !important}
div[data-testid="stHorizontalBlock"]:has(.keep-row)>div[data-testid="stColumn"],
div[data-testid="stHorizontalBlock"]:has(.keep-row)>div[data-testid="column"]{min-width:0 !important;flex:1 1 0 !important;width:auto !important}
.stImage img,[data-testid="stImage"] img{width:100%;height:auto}

/* polish */
.section::before{content:"";display:inline-block;width:4px;height:1em;background:var(--accent);border-radius:2px;margin-right:8px;vertical-align:-.12em}
.tbl,.tile .v,.sb .score{font-variant-numeric:tabular-nums}
.tbl tbody tr:hover td{background:var(--surface-2)}
.tbl-wrap::-webkit-scrollbar{height:8px;width:8px}
.tbl-wrap::-webkit-scrollbar-thumb{background:var(--border);border-radius:8px}
.mini-title{font-size:14px;font-weight:700;margin:2px 0 6px}
.note-s{font-size:12px;color:var(--muted);margin-top:6px}
.tiles{grid-template-columns:repeat(auto-fit,minmax(84px,1fr))}

/* buttons (theme switch, chat results) */
.stButton button{width:100%;border-radius:10px;border:1px solid var(--border);background:var(--surface-2) !important;color:var(--text) !important;
  text-align:left;justify-content:flex-start;padding:.45rem .75rem;font-size:13px;line-height:1.3;
  transition:border-color .15s ease,transform .1s ease}
.stButton button p{color:var(--text);white-space:normal}
.stButton button:hover{border-color:var(--accent);color:var(--text)}
.stButton button:active{transform:scale(.98)}
.st-key-theme_btn button{justify-content:center;text-align:center;font-weight:600;padding:.55rem .6rem}

/* sidebar chat */
[data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--border)}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p{color:var(--text)}
.side-title{font-size:17px;font-weight:700;margin:0}
.side-sub{color:var(--muted);font-size:12.5px;margin:2px 0 10px}
.chat{display:flex;flex-direction:column;gap:8px;margin:6px 0 10px}
.bubble{max-width:94%;padding:8px 11px;border-radius:14px;font-size:13.5px;line-height:1.38;word-wrap:break-word}
.bubble.a{align-self:flex-start;background:var(--surface-2);border:1px solid var(--border);color:var(--text)}
.bubble.u{align-self:flex-end;background:var(--accent);color:var(--on-accent)}
.bubble.u,.bubble.u *{color:var(--on-accent) !important}
[data-testid="stChatInput"]{background:var(--surface-2);border:1px solid var(--border);border-radius:14px}
[data-testid="stChatInput"] textarea{background:transparent;color:var(--text)}
[data-testid="stChatInput"] textarea::placeholder{color:var(--muted)}

/* inputs: one themed surface, never black (every inner wrapper is transparent) */
[data-baseweb="input"],[data-baseweb="base-input"],[data-baseweb="textarea"],
[data-testid="stNumberInputContainer"],[data-baseweb="select"]>div{background-color:var(--surface-2) !important;border-radius:10px !important}
[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="select"]>div{border:1px solid var(--border) !important}
[data-baseweb="input"]:focus-within,[data-baseweb="textarea"]:focus-within,[data-baseweb="select"]>div:focus-within{
  border-color:var(--accent) !important;box-shadow:0 0 0 1px var(--accent) !important}
[data-baseweb="input"] *,[data-baseweb="base-input"] *,[data-baseweb="textarea"] *,[data-baseweb="select"] input{
  background-color:transparent !important;color:var(--text) !important}
input,textarea{color:var(--text) !important;caret-color:var(--accent)}
input::placeholder,textarea::placeholder{color:var(--muted) !important;opacity:1}
input:-webkit-autofill{-webkit-box-shadow:0 0 0 1000px var(--surface-2) inset !important;-webkit-text-fill-color:var(--text) !important}
[data-testid="stNumberInput"] button{background:transparent !important;color:var(--text) !important;border:none !important}
[data-testid="stChatInput"]{background:var(--surface-2) !important;border:1px solid var(--border) !important;border-radius:14px !important}
[data-testid="stChatInput"] *{background-color:transparent !important}
[data-testid="stChatInput"]:focus-within{border-color:var(--accent) !important}
[data-testid="stChatInput"] textarea{color:var(--text) !important}
[data-testid="stChatInput"] button{color:var(--accent) !important}
[data-testid="stFileUploaderDropzone"]{background:var(--surface-2) !important;border:1px dashed var(--border) !important;border-radius:12px !important}
[data-testid="stFileUploaderDropzone"] *{color:var(--text) !important}
[data-testid="stFileUploaderDropzone"] button{background:var(--surface) !important;border:1px solid var(--border) !important;border-radius:8px !important}
[data-testid="stFileUploaderFile"] *{color:var(--text) !important}
[data-baseweb="calendar"],[data-baseweb="calendar"] *{background-color:var(--surface) !important;color:var(--text) !important}
[data-baseweb="calendar"] [aria-selected="true"]{background-color:var(--accent) !important;color:var(--on-accent) !important}
button[data-baseweb="tab"] p{color:var(--muted)}
button[data-baseweb="tab"][aria-selected="true"] p{color:var(--text)}
[data-baseweb="tab-highlight"]{background-color:var(--accent) !important}
[data-baseweb="tab-border"]{background-color:var(--border) !important}
.stDownloadButton button{width:100%;border-radius:10px;border:1px solid var(--border);background:var(--surface-2);color:var(--text);font-size:13px}
.stDownloadButton button p{color:var(--text)}
button[kind="primary"]{background:var(--accent) !important;border-color:var(--accent) !important}
button[kind="primary"] p{color:var(--on-accent) !important;font-weight:600}

/* sidebar: wider, tidier, and an obvious "open" pill when it is collapsed */
.block-container{padding-top:3.6rem !important}
[data-testid="stSidebar"][aria-expanded="true"]{width:min(92vw,430px) !important;min-width:min(92vw,430px) !important;max-width:92vw !important}
[data-testid="stSidebarContent"]{padding:1rem 1.1rem 2rem}
.side-head{padding-bottom:10px;border-bottom:1px solid var(--border);margin-bottom:12px}
.side-title{font-size:19px;font-weight:800;margin:0}
.side-sec{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:700;margin:16px 0 6px}
[data-testid="stSidebarCollapseButton"] button{color:var(--text) !important}
[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{
  position:fixed;top:12px;left:12px;z-index:1000;display:flex;align-items:center;gap:6px;
  background:var(--accent);border-radius:999px;padding:4px 14px 4px 6px;
  box-shadow:0 6px 18px rgba(0,0,0,.28);transition:transform .15s ease}
[data-testid="stSidebarCollapsedControl"]:hover,[data-testid="collapsedControl"]:hover{transform:translateY(-1px) scale(1.03)}
[data-testid="stSidebarCollapsedControl"] *,[data-testid="collapsedControl"] *{color:var(--on-accent) !important;fill:var(--on-accent) !important}
[data-testid="stSidebarCollapsedControl"]::after,[data-testid="collapsedControl"]::after{
  content:"Match finder";font-weight:700;font-size:13px;color:var(--on-accent)}

/* tables adapt to the width of their own column (laptop with/without sidebar, phone) */
.tbl-wrap{container-type:inline-size}
@container (max-width:560px){.bar{display:none}}
@container (max-width:470px){
  .xs{display:none}
  .tbl{font-size:12px}
  .tbl td,.tbl th{padding:6px 7px}
}

/* native chrome that can keep Streamlit's own theme: force ours */
[data-baseweb="popover"],[data-baseweb="popover"]>div,[data-baseweb="menu"],
[data-testid="stMainMenuPopover"],[data-testid="stMainMenuPopover"] *{background-color:var(--surface) !important;color:var(--text) !important}
[data-testid="stTooltipContent"],[data-testid="stTooltipContent"] *{background-color:var(--surface-2) !important;color:var(--text) !important}
[data-testid="stExpander"] summary,[data-testid="stExpander"] details{background-color:transparent !important}
[data-testid="stExpander"] summary:hover{background-color:var(--surface-2) !important}
[data-testid="stExpander"] summary svg{color:var(--muted) !important;fill:var(--muted) !important}
code{background-color:var(--surface-2) !important;color:var(--text) !important;border-radius:6px;padding:1px 6px}
.stButton button:disabled{opacity:.5}
/* image toolbar: one slim round button with a clean fullscreen icon (drawn with a mask, so no stray fills) */
[data-testid="stElementToolbar"]{background:transparent !important;border:none !important;box-shadow:none !important}
[data-testid="stElementToolbar"] button,[data-testid="stElementToolbarButton"]{
  width:34px;height:34px;padding:0 !important;border-radius:10px !important;
  background-color:var(--surface) !important;border:1px solid var(--border) !important;box-shadow:var(--shadow);
  opacity:.92;transition:border-color .15s ease,opacity .15s ease}
[data-testid="stElementToolbar"] button:hover{border-color:var(--accent) !important;opacity:1}
[data-testid="stElementToolbar"] button svg{display:none !important}
[data-testid="stElementToolbar"] button::before{
  content:"";display:block;width:18px;height:18px;margin:auto;background-color:var(--text);
  -webkit-mask:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M8 3H3v5M16 3h5v5M21 16v5h-5M3 16v5h5'/%3E%3C/svg%3E") center/contain no-repeat;
  mask:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M8 3H3v5M16 3h5v5M21 16v5h-5M3 16v5h5'/%3E%3C/svg%3E") center/contain no-repeat}

/* one clean border per input: only the outer box has a border; the inner boxes are borderless */
[data-baseweb="base-input"],[data-baseweb="input"] input,[data-baseweb="textarea"] textarea{
  border:none !important;box-shadow:none !important;outline:none !important}
[data-testid="stChatInput"] *{border:none !important;box-shadow:none !important;outline:none !important}

/* match-finder form: full width, one clean input, square accent send button */
[data-testid="stForm"]{border:none !important;padding:0 !important;background:transparent !important;width:100% !important}
[data-testid="stForm"] div[data-testid="stHorizontalBlock"]{display:flex !important;flex-wrap:nowrap !important;
  align-items:stretch !important;gap:.5rem !important;width:100% !important}
[data-testid="stForm"] div[data-testid="stHorizontalBlock"]:has(.keep-row)>div[data-testid="stColumn"]:first-child,
[data-testid="stForm"] div[data-testid="stHorizontalBlock"]:has(.keep-row)>div[data-testid="column"]:first-child{
  flex:1 1 auto !important;min-width:0 !important;width:auto !important}
[data-testid="stForm"] div[data-testid="stHorizontalBlock"]:has(.keep-row)>div[data-testid="stColumn"]:last-child,
[data-testid="stForm"] div[data-testid="stHorizontalBlock"]:has(.keep-row)>div[data-testid="column"]:last-child{
  flex:0 0 2.8rem !important;min-width:2.8rem !important;width:2.8rem !important}
[data-testid="stForm"] [data-testid="stTextInput"],[data-testid="stForm"] [data-baseweb="input"]{width:100% !important}
[data-testid="stForm"] [data-testid="stTextInput"] input{padding:0 .9rem;height:2.7rem;font-size:14.5px}
[data-testid="stFormSubmitButton"],[data-testid="stFormSubmitButton"] button{width:100% !important}
[data-testid="stFormSubmitButton"] button{background:var(--accent) !important;border:1px solid var(--accent) !important;
  border-radius:10px !important;height:2.7rem;min-height:2.7rem;justify-content:center;padding:0 !important}
[data-testid="stFormSubmitButton"] button p{color:var(--on-accent) !important;font-size:16px;font-weight:700}

/* inputs v2: ONE clearly visible field per widget. Every inner layer is transparent, so Streamlit's own
   (dark) input backgrounds can't show through, whatever its internal structure is. */
[data-testid="stTextInput"] div,[data-testid="stNumberInput"] div,[data-testid="stSelectbox"] div,[data-testid="stDateInput"] div,
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stDateInput"] input,[data-testid="stSelectbox"] input{
  background-color:transparent !important;border:none !important;box-shadow:none !important}
[data-testid="stTextInput"] [data-testid="stTextInputRootElement"],
[data-testid="stNumberInput"] [data-testid="stNumberInputContainer"],
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stDateInput"] [data-baseweb="input"]{
  background-color:var(--field) !important;border:1px solid var(--field-border) !important;border-radius:10px !important;
  min-height:42px;transition:border-color .15s ease,box-shadow .15s ease}
[data-testid="stTextInput"] [data-testid="stTextInputRootElement"]:hover,
[data-testid="stNumberInput"] [data-testid="stNumberInputContainer"]:hover,
[data-testid="stSelectbox"] [data-baseweb="select"]:hover,
[data-testid="stDateInput"] [data-baseweb="input"]:hover{border-color:var(--muted) !important}
[data-testid="stTextInput"] [data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stNumberInput"] [data-testid="stNumberInputContainer"]:focus-within,
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
[data-testid="stDateInput"] [data-baseweb="input"]:focus-within{
  border-color:var(--accent) !important;box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 22%,transparent) !important}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stDateInput"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] *{color:var(--text) !important}
[data-testid="stSelectbox"] [data-baseweb="select"] svg{fill:var(--muted) !important;color:var(--muted) !important}

/* dependency card */
.dep{background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--accent);border-radius:12px;padding:12px 14px;margin-top:10px}
.dep .k{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:700}
.dep .nm{font-size:19px;font-weight:800;margin:2px 0 4px}
.dep .sub{font-size:13px;line-height:1.4}
.dep .sub2{font-size:12px;color:var(--muted);margin-top:4px}

/* phone-only team switcher (desktop never sees it) */
.st-key-m_team{display:none}
@media (max-width:640px){
  .st-key-m_team{display:block;margin:4px 0 10px}
  .st-key-m_team [role="radiogroup"]{display:flex;gap:8px}
  .st-key-m_team label{flex:1;justify-content:center;background:var(--surface-2);border:1px solid var(--border);
    border-radius:10px;padding:8px 10px;margin:0}
  .st-key-m_team label>div:first-child{display:none}
  .st-key-m_team label:has(input:checked){background:var(--accent);border-color:var(--accent)}
  .st-key-m_team label:has(input:checked) *{color:var(--on-accent) !important}
}
@media (hover:none) and (pointer:coarse){
  .stButton button,.stDownloadButton button{min-height:44px}
}
@media (max-width:640px){
  .tile .v{font-size:17px}.tile{padding:8px 9px}
  .panel-title{font-size:16px}
  .avatar{flex:0 0 40px;width:40px;height:40px;font-size:14px}
  .mvp .nm{font-size:15px}.mvp .inf{font-size:11.5px}.mvp{padding:10px;gap:10px}
}
"""


def build_css(theme: dict) -> str:
    variables = ";".join(f"--{k}:{v}" for k, v in theme.items())
    scheme = "dark" if theme is THEMES["dark"] else "light"
    variables = f"color-scheme:{scheme};" + variables
    return f"<style>:root{{{variables}}}{BASE_CSS}{STATIC_CSS}{EXTRA_CSS}</style>"


# Theme is read from session state BEFORE the toggle is drawn, so the first paint after a click is already correct.
DARK = st.session_state.get("dark", True)
T = THEMES["dark" if DARK else "light"]
st.markdown(build_css(T), unsafe_allow_html=True)


# --- DATA LAYER (everything heavy is cached) ---
def _fetch_comp_matches(comp: dict) -> list:
    try:
        matches = sb.matches(
            competition_id=int(comp["competition_id"]),
            season_id=int(comp["season_id"]),
        )
    except Exception:
        return []
    out = []
    for m in matches.to_dict("records"):
        date = str(m["match_date"])
        out.append({
            "id": int(m["match_id"]),
            "year": date[:4],
            "date": date,
            "home": m["home_team"],
            "away": m["away_team"],
            "label": f"{m['home_team']} vs {m['away_team']} ({date}) - {comp['competition_name']} [#{int(m['match_id'])}]",
        })
    return out


@st.cache_data(ttl=6 * 3600, persist="disk", show_spinner="⏳ Syncing StatsBomb catalog (first run only)...")
def load_catalog(has_creds: bool):  # has_creds only keys the cache (free vs paid catalogs)
    """Parallel catalog fetch, persisted to disk so restarts are instant.
    Raises when empty so a failed/offline fetch is never cached."""
    comps = sb.competitions().to_dict("records")
    if PARALLEL_FETCH:
        with ThreadPoolExecutor(max_workers=4) as pool:
            chunks = list(pool.map(_fetch_comp_matches, comps))
    else:
        chunks = [_fetch_comp_matches(c) for c in comps]
    matches = [m for chunk in chunks for m in chunk]
    if not matches:
        raise RuntimeError("Empty catalog")
    matches.sort(key=lambda m: m["date"], reverse=True)
    return {
        "matches": matches,
        "teams": sorted({m["home"] for m in matches} | {m["away"] for m in matches}),
        "years": sorted({m["year"] for m in matches}, reverse=True),
    }


def _scoreboard_from_events(events, home, away):
    scorers = {home: [], away: []}
    normal = {home: 0, away: 0}
    shootout = {home: 0, away: 0}
    opponent = {home: away, away: home}

    if "type" in events.columns and "shot_outcome" in events.columns:
        shots = events[events["type"] == "Shot"]
        goals = shots[shots["shot_outcome"] == "Goal"]
        for row in goals.to_dict("records"):
            team = row.get("team", "")
            if team not in scorers:
                continue
            player = row.get("player", "Unknown Player")
            minute = row.get("minute")
            period = int(row.get("period", 1))
            is_so = period == 5
            is_pen = is_so or row.get("shot_type", "") == "Penalty"
            tm = "PSO" if is_so else (f"{int(minute)}'" if pd.notna(minute) else "")
            scorers[team].append(f"{player} ({tm})" + (" (P)" if is_pen else ""))
            (shootout if is_so else normal)[team] += 1

    if "type" in events.columns:
        for row in events[events["type"] == "Own Goal Against"].to_dict("records"):
            conceding = row.get("team", "")
            if conceding not in opponent:
                continue
            benef = opponent[conceding]
            minute = row.get("minute")
            tm = f"{int(minute)}'" if pd.notna(minute) else ""
            scorers[benef].append(f"{row.get('player', 'Unknown Player')} ({tm}) (OG)")
            normal[benef] += 1

    has_pso = shootout[home] > 0 or shootout[away] > 0
    score = (f"{normal[home]} ({shootout[home]}) - ({shootout[away]}) {normal[away]}"
             if has_pso else f"{normal[home]} - {normal[away]}")
    return score, scorers[home], scorers[away], has_pso


def _alpha_for(alphas, player):
    if isinstance(alphas, dict):
        v = alphas.get(player)
        return float(v) if v is not None else None
    return float(alphas)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def get_display_names(match_id: int, squad: str) -> dict:
    """{full StatsBomb name: common name}. Full names are often 4+ words (e.g. 'Francisco Román Alarcón Suárez' = Isco)."""
    if match_id >= CUSTOM_ID_BASE:
        return {}
    try:
        df = sb.lineups(match_id=match_id)[squad]
    except Exception:
        return {}
    out = {}
    for r in df.to_dict("records"):
        full, nick = r.get("player_name"), r.get("player_nickname")
        if full:
            out[full] = nick.strip() if isinstance(nick, str) and nick.strip() else full
    return out


def _total_efficiency(W: np.ndarray) -> float:
    """Sum of 1/shortest-path-distance over all ordered pairs, with edge length = 1 / (passes between the pair).
    Many passes = short distance. Reversing every edge gives the same total, so the pass-matrix orientation doesn't matter."""
    n = len(W)
    if n < 2:
        return 0.0
    D = np.full((n, n), np.inf)
    pos = W > 0
    D[pos] = 1.0 / W[pos]
    np.fill_diagonal(D, 0.0)
    for k in range(n):  # Floyd-Warshall (teams have ~11-16 players, so this is instant)
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    off = ~np.eye(n, dtype=bool)
    return float((1.0 / D[off]).sum())  # unreachable pairs (inf) contribute 0


@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def analyze_team(match_id: int, squad: str):
    """All per-team math in one cached call; returns plain data or None."""
    if match_id >= CUSTOM_ID_BASE:  # imported match
        matrix, players = _custom_matrix(match_id, squad)
    else:
        matrix, players = get_match_passing_matrix(match_id=match_id, team_name=squad)
    if len(players) == 0:
        return None
    players = list(players)
    scores, alphas, density = sports_adaptive_pagerank(matrix, players)
    roster = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    out_vol = np.sum(matrix, axis=OUTBOUND_AXIS)
    total_out = float(out_vol.sum()) or 1.0
    bottleneck = {p: scores[p] / max(out_vol[i] / total_out, 0.01) for i, p in enumerate(players)}

    names = get_display_names(match_id, squad)
    disp = {p: names.get(p, p) for p in players}
    p_index = {p: i for i, p in enumerate(players)}
    passes_made = np.sum(matrix, axis=OUTBOUND_AXIS)
    passes_recv = np.sum(matrix, axis=1 - OUTBOUND_AXIS)

    top_name, top_score = roster[0]
    denom = max(len(roster) - 1, 1)
    total_passes = float(np.sum(matrix))
    # Volume-weighted score = PageRank influence share x the player's own outbound (completed) passes.
    # A flat team-wide multiplier would never change the ranking inside a team; this does.
    weighted = {p: scores[p] * float(passes_made[p_index[p]]) for p in players}
    rows_sorted = sorted(roster, key=lambda x: weighted[x[0]], reverse=True)
    mvp_name = max(players, key=lambda p: weighted[p])
    most_dependent_player, max_network_loss, dependency_scores = system_dependency(matrix, players)
    dependency = {
        "most_dependent_player": most_dependent_player,
        "max_network_loss": max_network_loss,
        "dependency_scores": dependency_scores,
    }
    return {
        "matrix": matrix,
        "players": players,
        "scores": scores,
        "roster": roster,
        "top": (top_name, top_score),
        "total_passes": total_passes,
        "weighted": weighted,
        "mvp": (mvp_name, scores[mvp_name], weighted[mvp_name]),
        "dependency": dependency,
        "display": disp,
        "density": density,
        "centralization": sum(top_score - s for _, s in roster) / denom,
        "primary_bottleneck": max(bottleneck, key=bottleneck.get),
        "bottleneck": bottleneck,
        "rows": [
            {
                "Player Identity": disp[p],
                "Weighted Score": weighted[p],
                "Influence %": s * 100,
                "Sigmoid Alpha": _alpha_for(alphas, p),
                "Bottleneck Index": float(bottleneck[p]),
                "Passes Made": int(passes_made[p_index[p]]),
                "Passes Received": int(passes_recv[p_index[p]]),
            }
            for p, s in rows_sorted
        ],
    }


# ---- Pitch-based passing network (players placed by the match's starting formation) ----
EDGE_MIN_PASSES = 3  # hide pairs with fewer combined passes than this

# StatsBomb coordinates: 120 x 80, own goal at x=0, attacking toward x=120, y runs top (0) to bottom (80),
# so a team's RIGHT side is the larger-y side when attacking left-to-right.
POSITION_XY = {
    "Goalkeeper": (8, 40),
    "Right Back": (30, 68), "Right Center Back": (27, 52), "Center Back": (27, 40),
    "Left Center Back": (27, 28), "Left Back": (30, 12),
    "Right Wing Back": (46, 70), "Left Wing Back": (46, 10),
    "Right Defensive Midfield": (45, 52), "Center Defensive Midfield": (45, 40), "Left Defensive Midfield": (45, 28),
    "Right Midfield": (62, 68), "Right Center Midfield": (62, 52), "Center Midfield": (62, 40),
    "Left Center Midfield": (62, 28), "Left Midfield": (62, 12),
    "Right Attacking Midfield": (80, 54), "Center Attacking Midfield": (80, 40), "Left Attacking Midfield": (80, 26),
    "Right Wing": (90, 68), "Left Wing": (90, 12),
    "Right Center Forward": (102, 50), "Center Forward": (102, 40), "Left Center Forward": (102, 30),
    "Secondary Striker": (92, 40),
}


def _tactics_from_events(events, teams):
    """Starting formation + {player: position} per team. Substitutes inherit the slot of the player they replace."""
    out = {t: {"formation": None, "positions": {}} for t in teams}
    if "type" not in events.columns or "tactics" not in events.columns:
        return out

    for row in events[events["type"] == "Starting XI"].to_dict("records"):
        team, tac = row.get("team"), row.get("tactics")
        if team not in out or not isinstance(tac, dict):
            continue
        f = tac.get("formation")
        if f:
            s = str(f)
            if "-" not in s:
                try:
                    s = str(int(float(s)))
                except ValueError:
                    pass
                s = "-".join(s)
            out[team]["formation"] = s
        for item in tac.get("lineup", []):
            name = (item.get("player") or {}).get("name")
            pos = (item.get("position") or {}).get("name")
            if name and pos:
                out[team]["positions"][name] = pos

    if "substitution_replacement" in events.columns:
        subs = events[events["type"] == "Substitution"]
        sort_cols = [c for c in ("period", "minute", "second") if c in subs.columns]
        if sort_cols:
            subs = subs.sort_values(sort_cols)
        for row in subs.to_dict("records"):
            team, off, on = row.get("team"), row.get("player"), row.get("substitution_replacement")
            if isinstance(on, dict):
                on = on.get("name")
            pos_map = out.get(team, {}).get("positions", {})
            if off in pos_map and on and on not in pos_map:
                pos_map[on] = pos_map[off]
    return out


@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def load_match_info(match_id: int, home: str, away: str):
    """One events download feeds both the scoreboard and the formation; only small results are cached."""
    if match_id >= CUSTOM_ID_BASE:  # imported match: score and formation were read at import time
        reg = custom_registry().get(match_id)
        if reg is None:
            raise RuntimeError("Imported match data not found (re-add it).")
        return {"scoreboard": reg["scoreboard"], "tactics": reg["tactics"]}
    events = sb.events(match_id=match_id)
    try:
        scoreboard = _scoreboard_from_events(events, home, away)
    except Exception:
        scoreboard = None
    try:
        tactics = _tactics_from_events(events, (home, away))
    except Exception:
        tactics = {t: {"formation": None, "positions": {}} for t in (home, away)}
    return {"scoreboard": scoreboard, "tactics": tactics}


def get_tactics(match_id: int, squad: str, home: str, away: str) -> dict:
    try:
        return load_match_info(match_id, home, away)["tactics"].get(squad) or {"formation": None, "positions": {}}
    except Exception:
        return {"formation": None, "positions": {}}


def _norm(s) -> str:
    return " ".join(str(s).lower().split())


def _place_players(players, pos_map):
    """Map each player to a pitch (x, y) via the formation slot they occupy."""
    by_norm = {_norm(n): p for n, p in pos_map.items()}
    by_last = {}
    for n, p in pos_map.items():
        if _norm(n):
            by_last.setdefault(_norm(n).split()[-1], []).append(p)

    coords, used = {}, {}
    for name in players:
        key = _norm(name)
        pos = by_norm.get(key)
        if pos is None and key:
            cands = by_last.get(key.split()[-1], [])
            pos = cands[0] if len(cands) == 1 else None
        if pos is None:
            continue
        x, y = POSITION_XY.get(pos, (60, 40))
        k = used.get(pos, 0)
        used[pos] = k + 1
        if k:  # a substitute sharing a slot: nudge so both stay visible
            y += (7 if k % 2 else -7) * ((k + 1) // 2)
            x += 3
        coords[name] = (x, min(max(y, 4), 76))
    return coords


MAX_EDGES = 18  # strongest links only: keeps the network readable
FIG_W, FIG_H = 4.4, 6.5   # network figure size in inches (was 5.0 x 7.4)
_SCALE = FIG_W / 5.0      # keeps nodes, text and lines in proportion to the smaller figure


def _draw_pitch_v(ax):
    """Vertical pitch (80 wide x 120 long), own goal at the bottom, attacking up."""
    for k in range(12):  # mowing stripes
        ax.add_patch(Rectangle((0, k * 10), 80, 10, facecolor="#2f8f46" if k % 2 == 0 else "#2a8040",
                               edgecolor="none", zorder=0))
    kw = dict(edgecolor="white", facecolor="none", linewidth=1.5, zorder=1)
    ax.add_patch(Rectangle((0, 0), 80, 120, **kw))                                  # outline
    ax.add_patch(Rectangle((18, 0), 44, 18, **kw))                                  # penalty areas
    ax.add_patch(Rectangle((18, 102), 44, 18, **kw))
    ax.add_patch(Rectangle((30, 0), 20, 6, **kw))                                   # six-yard boxes
    ax.add_patch(Rectangle((30, 114), 20, 6, **kw))
    ax.add_patch(Rectangle((36, -2), 8, 2, **kw))                                   # goals
    ax.add_patch(Rectangle((36, 120), 8, 2, **kw))
    ax.add_patch(Circle((40, 60), 10, **kw))                                        # centre circle
    ax.add_patch(Arc((40, 12), 20, 20, theta1=37, theta2=143, edgecolor="white", linewidth=1.5, zorder=1))
    ax.add_patch(Arc((40, 108), 20, 20, theta1=217, theta2=323, edgecolor="white", linewidth=1.5, zorder=1))
    ax.plot([0, 80], [60, 60], color="white", linewidth=1.5, zorder=1)             # halfway line
    ax.scatter([40, 40, 40], [12, 60, 108], s=10, color="white", zorder=1)          # spots


@st.cache_data(ttl=3600, show_spinner=False, max_entries=32)
def network_png(match_id: int, squad: str, home: str, away: str) -> bytes:
    """Passing network on a vertical green pitch, players at their formation slots; transparent PNG, cached."""
    a = analyze_team(match_id, squad)
    matrix, players, scores = a["matrix"], a["players"], a["scores"]
    n = len(players)
    mvp = a["mvp"][0]

    # StatsBomb (x = length, y = width, right side = larger y) -> vertical plot (lateral = y, depth = x)
    placed = _place_players(players, get_tactics(match_id, squad, home, away)["positions"])
    pc = {p: (y, x) for p, (x, y) in placed.items()}
    if not pc:  # no formation data (e.g. CSV import): spread players on an ellipse instead
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
        pc = {p: (40 + 28 * float(np.cos(t)), 60 + 44 * float(np.sin(t))) for p, t in zip(players, ang)}
    bench = [p for p in players if p not in pc]  # unplaced players sit in a strip below the pitch
    for k, p in enumerate(bench):
        pc[p] = (8 + k * 64 / (len(bench) - 1), -9) if len(bench) > 1 else (40, -9)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _draw_pitch_v(ax)

    # Edges: only the strongest pairs, thin and translucent
    if n > 1:
        sym = matrix + matrix.T
        iu, ju = np.triu_indices(n, 1)
        w = sym[iu, ju].astype(float)
        ok = np.array([players[i] not in bench and players[j] not in bench for i, j in zip(iu, ju)])
        if ok.any():
            cutoff = max(EDGE_MIN_PASSES, 0.2 * float(w[ok].max()))
            cand = np.where(ok & (w >= cutoff))[0]
            cand = cand[np.argsort(-w[cand])][:MAX_EDGES]
            if len(cand):
                rel = w[cand] / float(w[cand].max())
                segs = [[pc[players[iu[c]]], pc[players[ju[c]]]] for c in cand]
                colors = [(1, 1, 1, float(0.30 + 0.60 * r)) for r in rel]
                ax.add_collection(LineCollection(segs, linewidths=(0.8 + 3.6 * rel) * _SCALE, colors=colors, zorder=3))

    xs = [pc[p][0] for p in players]
    ys = [pc[p][1] for p in players]
    sizes = [max(220, min(900, scores.get(p, 0.05) * 3500)) * _SCALE ** 2 for p in players]
    rings = ["#facc15" if p == mvp else "white" for p in players]
    ax.scatter(xs, ys, s=sizes, c="#0f172a", edgecolors=rings, linewidths=2, zorder=4)

    for p, x, y, s in zip(players, xs, ys, sizes):
        shown = a["display"].get(p, p)
        label = shown.split()[-1] if shown.split() else shown
        ax.text(x, y - np.sqrt(s / np.pi) * (86 / (FIG_W * 72)) - 1.2, label, color="white", fontsize=7, weight="bold",
                ha="center", va="top", zorder=5,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="#0f172a")])

    if bench:
        ax.text(40, -3.5, "Position unavailable", color="#94a3b8", fontsize=7, ha="center", va="top")  # readable on dark and light pages

    ax.set_xlim(-3, 83)
    ax.set_ylim(-17 if bench else -6, 126)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0)  # transparent: blends with the dark or light page
    fig.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, transparent=True)
    plt.close(fig)
    return buf.getvalue()


# --- HTML HELPERS (one block each; colours come from CSS variables so they follow the theme) ---
def esc(x) -> str:
    return html.escape(str(x))


def callout(kind: str, text_html: str):
    st.markdown(f'<div class="callout {kind}">{text_html}</div>', unsafe_allow_html=True)


def scoreboard_html(teams, score, home_sc, away_sc, has_pso):
    def lst(items):
        return "".join(f'<div class="scorer">⚽ {esc(s)}</div>' for s in items)

    cls = "score" if score else "score na"
    txt = esc(score) if score else "Score unavailable"
    pso = '<div class="note">Shootout scores in brackets</div>' if has_pso else ""
    return (
        '<div class="card"><div class="sb">'
        f'<div class="home"><div class="team">{esc(teams[0])}</div>{lst(home_sc)}</div>'
        f'<div><div class="{cls}">{txt}</div>{pso}</div>'
        f'<div class="away"><div class="team">{esc(teams[1])}</div>{lst(away_sc)}</div>'
        "</div></div>"
    )


def tiles_html(items):
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>' for k, v in items
    ) + "</div>"


def table_html(rows):
    wmax = max((r["Weighted Score"] for r in rows), default=0) or 1.0
    alphas = {round(r["Sigmoid Alpha"], 3) for r in rows if r["Sigmoid Alpha"] is not None}
    show_alpha = len(alphas) > 1  # a constant alpha is shown once in a note instead of as a repeated column
    cols = [("#", ""), ("Player", ""), ("Weighted", ""), ("Influence", ""), ("Made", ""),
            ("Received", "xs"), ("Bottleneck", "xs")] + ([("α", "xs")] if show_alpha else [])  # "xs" hides on narrow screens
    head = "".join(f'<th class="{c}">{h}</th>' for h, c in cols)
    body = []
    for i, r in enumerate(rows, 1):
        pct = r["Weighted Score"] / wmax * 100
        vals = [
            (str(i), ""),
            (esc(r["Player Identity"]), ""),
            (f'{r["Weighted Score"]:.1f}<span class="bar"><i style="width:{pct:.0f}%"></i></span>', ""),
            (f'{r["Influence %"]:.2f}%', ""),
            (str(r["Passes Made"]), ""),
            (str(r["Passes Received"]), "xs"),
            (f'{r["Bottleneck Index"]:.2f}', "xs"),
        ]
        if show_alpha:
            vals.append(("" if r["Sigmoid Alpha"] is None else f'{r["Sigmoid Alpha"]:.3f}', "xs"))
        body.append("<tr>" + "".join(f'<td class="{c}">{v}</td>' for v, c in vals) + "</tr>")
    table = (
        f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )
    note = f"Sigmoid α = {next(iter(alphas)):.3f}" if len(alphas) == 1 else ""
    return table, note


def mvp_card_html(squad, name, score, weighted):
    initials = "".join(w[0] for w in name.split()[:2]).upper() or "?"
    return (
        '<div class="mvp">'
        f'<div class="avatar">{esc(initials)}</div>'
        "<div>"
        f'<div class="lbl">{esc(squad)} MVP</div>'
        f'<div class="nm">{esc(name)}</div>'
        f'<div class="inf">Weighted {weighted:.1f} · Influence {score * 100:.2f}%</div>'
        "</div></div>"
    )


def dependency_card_html(a: dict) -> str:
    """Render the dependency results returned by engine.adaptive_pagerank."""
    dependency = a.get("dependency") or {}
    player = dependency.get("most_dependent_player")
    loss = dependency.get("max_network_loss")
    scores = dependency.get("dependency_scores") or {}
    if not player or loss is None:
        return ""

    display_name = a["display"].get(player, player)
    if isinstance(scores, dict):
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    else:
        ranked = sorted(
            ((item.get("player"), item.get("loss", 0.0)) for item in scores),
            key=lambda item: item[1], reverse=True,
        )
    next_players = " · ".join(
        f'{esc(a["display"].get(name, name))} {score * 100:.1f}%'
        for name, score in ranked
        if name != player
    )
    next_players = " · ".join(next_players.split(" · ")[:2])
    return (
        '<div class="dep">'
        '<div class="k">🔗 Most depended-on player</div>'
        f'<div class="nm">{esc(display_name)}</div>'
        f'<div class="sub">Removing this player reduces the team network connectivity by '
        f'<b>{loss * 100:.1f}%</b>.</div>'
        + (f'<div class="sub2">Next: {next_players}</div>' if next_players else "")
        + "</div>"
    )


def keep_row():
    """Marker that stops this row of columns from stacking (see .keep-row in the CSS)."""
    st.markdown('<span class="keep-row"></span>', unsafe_allow_html=True)


# --- CUSTOM MATCHES: bring your own data (e.g. 2026 matches that aren't in the free StatsBomb set) ---
# Three ways in: (1) drop files into ./custom_matches (auto-loaded), (2) upload a file, (3) paste a link.
# Supported: StatsBomb-format events JSON (full match: score, formation and substitutes are read automatically)
# or a simple passes CSV with columns team, passer, recipient (optional: count, outcome).
CUSTOM_ID_BASE = 900_000_000  # custom match ids live above any real StatsBomb id
CUSTOM_DIR = os.environ.get("CUSTOM_MATCH_DIR", str(APP_DIR / "custom_matches"))
CSV_TEMPLATE = (
    "team,passer,recipient,count\n"
    "Home FC,Player A,Player B,12\n"
    "Home FC,Player B,Player C,9\n"
    "Away FC,Player X,Player Y,7\n"
    "Away FC,Player Y,Player Z,6\n"
)


@st.cache_resource
def custom_registry() -> dict:
    """Server-side store of imported matches, keyed by content-hash id (shared by the cached functions)."""
    return {}


def _custom_matrix(match_id: int, squad: str):
    reg = custom_registry().get(match_id)
    if not reg:
        return np.zeros((0, 0)), []
    counts = reg["passes"].get(squad, {})
    players = sorted({p for pair in counts for p in pair})
    idx = {p: i for i, p in enumerate(players)}
    matrix = np.zeros((len(players), len(players)))
    for (src, dst), c in counts.items():
        if OUTBOUND_AXIS == 0:
            matrix[idx[dst], idx[src]] += c  # same orientation convention as the rest of the app
        else:
            matrix[idx[src], idx[dst]] += c
    return matrix, players


def _parse_events_json(data: bytes) -> dict:
    events = json.loads(data)
    if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
        raise ValueError("Expected a JSON list of events (StatsBomb open-data format).")
    counts, rows, order = {}, [], []
    for e in events:
        etype = (e.get("type") or {}).get("name")
        team = (e.get("team") or {}).get("name")
        player = (e.get("player") or {}).get("name")
        if team and team not in order:
            order.append(team)
        if etype == "Pass":
            p = e.get("pass") or {}
            rec = (p.get("recipient") or {}).get("name")
            if team and player and rec and not p.get("outcome"):  # no outcome = completed
                pair = counts.setdefault(team, {})
                pair[(player, rec)] = pair.get((player, rec), 0) + 1
        if etype in ("Starting XI", "Shot", "Own Goal Against", "Substitution"):
            row = {"type": etype, "team": team, "player": player, "minute": e.get("minute"),
                   "second": e.get("second"), "period": e.get("period")}
            if etype == "Shot":
                s = e.get("shot") or {}
                row["shot_outcome"] = (s.get("outcome") or {}).get("name")
                row["shot_type"] = (s.get("type") or {}).get("name")
            elif etype == "Starting XI":
                row["tactics"] = e.get("tactics")
            elif etype == "Substitution":
                row["substitution_replacement"] = (((e.get("substitution") or {}).get("replacement")) or {}).get("name")
            rows.append(row)
    teams = [t for t in order if t in counts][:2]
    if len(teams) < 2:
        raise ValueError("Couldn't find completed passes for two teams in that file.")
    return {"kind": "json", "teams": teams, "counts": counts, "events": pd.DataFrame(rows)}


def _parse_passes_csv(data: bytes) -> dict:
    df = pd.read_csv(io.BytesIO(data))
    df.columns = [str(c).strip().lower() for c in df.columns]

    def pick(*names):
        return next((n for n in names if n in df.columns), None)

    c_team = pick("team", "squad", "club")
    c_src = pick("passer", "player", "from", "from_player", "pass_from", "source")
    c_dst = pick("recipient", "receiver", "to", "to_player", "pass_to", "pass_recipient", "target")
    c_n = pick("count", "passes", "n", "weight", "number")
    c_out = pick("outcome", "result")
    if not (c_team and c_src and c_dst):
        raise ValueError("CSV needs the columns: team, passer, recipient (optional: count, outcome).")

    df = df.dropna(subset=[c_team, c_src, c_dst]).copy()
    for c in (c_team, c_src, c_dst):
        df[c] = df[c].astype(str).str.strip()
    if c_out:
        good = {"", "nan", "complete", "completed", "success", "successful", "true", "1", "yes", "y"}
        df = df[df[c_out].astype(str).str.strip().str.lower().isin(good)]
    df = df[df[c_src] != df[c_dst]]
    df["_n"] = pd.to_numeric(df[c_n], errors="coerce").fillna(1) if c_n else 1

    teams = list(dict.fromkeys(df[c_team]))
    if len(teams) != 2:
        raise ValueError(f"CSV must contain exactly two teams (found {len(teams)}).")
    counts = {}
    for (t, a, b), n in df.groupby([c_team, c_src, c_dst])["_n"].sum().items():
        counts.setdefault(t, {})[(a, b)] = float(n)
    return {"kind": "csv", "teams": teams, "counts": counts, "events": None}


@st.cache_data(show_spinner=False, max_entries=8)
def parse_import(name: str, data: bytes) -> dict:
    if name.lower().endswith(".json") or data.lstrip()[:1] in (b"[", b"{"):
        return _parse_events_json(data)
    return _parse_passes_csv(data)


@st.cache_data(ttl=600, show_spinner="Downloading...", max_entries=8)
def fetch_url_bytes(url: str) -> bytes:
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("Please paste a full http(s) link.")
    if "github.com" in url and "/blob/" in url:  # accept normal GitHub page links
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    import requests

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    if len(resp.content) > 60_000_000:
        raise ValueError("That file is too large (60 MB limit).")
    return resp.content


def _build_custom(parsed: dict, home: str, away: str, date_str: str, comp: str, gh=None, ga=None):
    """Return (catalog-style match dict, registry payload); no side effects."""
    teams = (home, away)
    ev = parsed.get("events")
    if ev is not None and len(ev):
        try:
            scoreboard = _scoreboard_from_events(ev, home, away)
        except Exception:
            scoreboard = None
        tactics = _tactics_from_events(ev, teams)
    else:
        scoreboard = (f"{int(gh)} - {int(ga)}", [], [], False) if gh is not None and ga is not None else None
        tactics = {t: {"formation": None, "positions": {}} for t in teams}
    counts = {t: parsed["counts"].get(t, {}) for t in teams}
    sig = repr((home, away, date_str, comp, scoreboard, sorted((t, sorted(c.items())) for t, c in counts.items())))
    cid = CUSTOM_ID_BASE + int(hashlib.sha1(sig.encode()).hexdigest()[:8], 16) % 90_000_000
    match = {"id": cid, "year": date_str[:4], "date": date_str, "home": home, "away": away, "custom": True,
             "label": f"{home} vs {away} ({date_str}) - {comp} [#{cid}]"}
    return match, {"passes": counts, "scoreboard": scoreboard, "tactics": tactics}


def add_custom_match(parsed, home, away, date_str, comp, gh=None, ga=None) -> dict:
    match, payload = _build_custom(parsed, home, away, date_str, comp, gh, ga)
    custom_registry()[match["id"]] = payload
    kept = [m for m in st.session_state.get("custom_matches", []) if m["id"] != match["id"]]
    st.session_state["custom_matches"] = kept + [match]
    return match


@st.cache_data(show_spinner=False, max_entries=64)
def _load_folder_file(path: str, mtime: float):
    with open(path, "rb") as fh:
        parsed = parse_import(os.path.basename(path), fh.read())
    home, away = parsed["teams"][:2]
    dated = re.search(r"(20\d{2}-\d{2}-\d{2})", os.path.basename(path))  # e.g. 2026-06-15_spain_uruguay.json
    date_str = dated.group(1) if dated else datetime.date.fromtimestamp(mtime).isoformat()
    return _build_custom(parsed, home, away, date_str, "Custom folder")


def scan_custom_dir() -> list:
    """Auto-load every .json / .csv in the custom folder; edits are picked up on the next rerun."""
    out = []
    if not os.path.isdir(CUSTOM_DIR):
        return out
    for entry in sorted(os.scandir(CUSTOM_DIR), key=lambda e: e.name):
        if entry.is_file() and entry.name.lower().endswith((".json", ".csv")):
            try:
                match, payload = _load_folder_file(entry.path, entry.stat().st_mtime)
            except Exception:
                continue  # unreadable file: skip it
            custom_registry()[match["id"]] = payload
            out.append(match)
    return out


def merged_catalog(base: dict) -> dict:
    """Catalog = StatsBomb matches + folder matches + matches added this session (newest first)."""
    extra = scan_custom_dir() + st.session_state.get("custom_matches", [])
    if not extra:
        return base
    unique = {m["id"]: m for m in extra}
    matches = sorted(list(unique.values()) + base["matches"], key=lambda m: m["date"], reverse=True)
    return {
        "matches": matches,
        "teams": sorted({m["home"] for m in matches} | {m["away"] for m in matches}),
        "years": sorted({m["year"] for m in matches}, reverse=True),
    }


def team_row():
    """Marker for team A / team B rows: side by side on laptops, one team at a time on phones."""
    st.markdown('<span class="keep-row team-row"></span>', unsafe_allow_html=True)


# --- MATCH FINDER: sidebar chat that turns plain text into a match search ---
STOP_WORDS = {
    "vs", "v", "versus", "against", "match", "matches", "game", "games", "the", "in", "of", "between", "and", "at",
    "a", "an", "show", "me", "find", "get", "play", "played", "fixture", "fixtures", "please", "analyse", "analyze",
    "load", "open", "for", "with", "from", "season", "i", "want", "to", "see",
}
ALIASES = {
    "usa": "united states", "barca": "barcelona", "man utd": "manchester united", "man united": "manchester united",
    "man city": "manchester city", "atletico": "atletico madrid", "psg": "paris saint germain", "spurs": "tottenham",
    "holland": "netherlands", "korea": "south korea", "ivory coast": "cote divoire", "czech republic": "czech",
}


def _norm_txt(s) -> str:
    s = str(s).replace("’", "'")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().replace("'", "")
    return re.sub(r"[^a-z0-9# ]+", " ", s)


def _comp(m: dict) -> str:
    hit = re.search(r"\) - (.*) \[#\d+\]$", m["label"])
    return hit.group(1) if hit else ""


def parse_match_query(q: str, teams: list) -> dict:
    ql = " " + _norm_txt(q) + " "
    info = {"teams": [], "year": None, "id": None, "kw": []}

    hit = re.search(r"#\s*(\d{4,9})\b", ql) or re.search(r"\b(\d{5,9})\b", ql)
    if hit:
        info["id"] = int(hit.group(1))
        ql = ql.replace(hit.group(0), " ", 1)
    hit = re.search(r"\b((?:19|20)\d{2})\b", ql)
    if hit:
        info["year"] = hit.group(1)
        ql = ql.replace(hit.group(1), " ", 1)

    norm_teams = {t: _norm_txt(t).strip() for t in teams}
    for key, target in ALIASES.items():
        if f" {key} " in ql:
            cands = [t for t, n in norm_teams.items() if target in n]
            if cands:
                best = min(cands, key=lambda t: len(norm_teams[t]))
                ql = ql.replace(f" {key} ", f" {norm_teams[best]} ", 1)

    found = []  # (position in query, team); longest names first so "Germany Women's" beats "Germany"
    for t in sorted(teams, key=lambda t: -len(norm_teams[t])):
        n = norm_teams[t]
        if n and f" {n} " in ql:
            found.append((ql.index(f" {n} "), t))
            ql = ql.replace(f" {n} ", " ", 1)

    left = [w for w in ql.split() if w not in STOP_WORDS and not w.isdigit()]
    if len(found) < 2 and left:  # typo-tolerant pass over what is left
        pool = {n: t for t, n in norm_teams.items() if n}
        for size in (3, 2, 1):
            i = 0
            while i + size <= len(left) and len(found) < 2:
                phrase = " ".join(left[i:i + size])
                close = difflib.get_close_matches(phrase, list(pool), n=1, cutoff=0.84) if len(phrase) >= 4 else []
                if close and pool[close[0]] not in [t for _, t in found]:
                    found.append((10_000 + i, pool[close[0]]))
                    left = left[:i] + left[i + size:]
                    continue
                i += 1

    info["teams"] = [t for _, t in sorted(found)][:2]
    info["kw"] = [w for w in left if len(w) >= 3]
    return info


def search_matches(info: dict, matches: list) -> list:
    if info["id"] is not None:
        return [m for m in matches if m["id"] == info["id"]]
    res, ts = matches, info["teams"]
    if len(ts) >= 2:
        pair = set(ts)
        res = [m for m in res if {m["home"], m["away"]} == pair]
    elif len(ts) == 1:
        res = [m for m in res if ts[0] in (m["home"], m["away"])]
    if info["year"]:
        res = [m for m in res if m["year"] == info["year"]]
    if info["kw"]:
        narrowed = [m for m in res if all(k in _norm_txt(m["label"]) for k in info["kw"])]
        if narrowed or not (ts or info["year"]):
            res = narrowed
    if not (ts or info["year"] or info["kw"]):
        return []
    return res


def chat_reply(q: str, catalog: dict) -> dict:
    info = parse_match_query(q, catalog["teams"])
    res = search_matches(info, catalog["matches"])
    what = " vs ".join(info["teams"]) or "your search"
    if info["year"]:
        what += f" ({info['year']})"

    if info["id"] is not None and not res:
        text = f"Match <b>#{info['id']}</b> isn't in the current catalog."
    elif not res:
        text = (f"I couldn't find matches for <b>{esc(what)}</b>. Check the spelling or year — the free data only "
                "covers selected competitions. Try something like <b>Spain vs Russia 2018</b>.")
    elif len(res) == 1:
        text = f"Loaded <b>{esc(res[0]['home'])} vs {esc(res[0]['away'])}</b> ({esc(res[0]['date'])})."
    else:
        extra = " Showing the 6 most recent — add a year to narrow it." if len(res) > 6 else ""
        text = f"Found <b>{len(res)}</b> matches for <b>{esc(what)}</b>. Pick one:{extra}"
    return {"role": "assistant", "text": text, "ids": [m["id"] for m in res[:6]], "autopick": len(res) == 1}


def pick_match(m: dict):
    """Point the main filters at this match (also used as a button callback, so it runs before widgets exist)."""
    st.session_state["f_team1"] = m["home"]
    st.session_state["f_team2"] = m["away"]
    st.session_state["f_year"] = m["year"]
    st.session_state["f_match"] = m["label"]


# --- NETWORKS ROW: a fragment so the toggle only reruns this block ---
@fragment
def networks_row(match_id: int, teams: tuple, ok: tuple):
    if not st.toggle("Show passing networks", value=True, key=f"net_{match_id}"):
        return
    cols = st.columns(2, gap="small")
    for col, squad in zip(cols, teams):
        with col:
            team_row()
            if squad in ok:
                st.image(network_png(match_id, squad, teams[0], teams[1]))
            else:
                callout("warn", f"No passing data for {esc(squad)}.")
    st.caption(
        "Players at their starting-XI positions (or in a circle if no formation data), attacking up · node size = influence · "
        "line width = passes between a pair (strongest links only) · gold ring = MVP"
    )


# --- PAGE ---
def _flip_theme():
    st.session_state["dark"] = not st.session_state.get("dark", True)


head_l, head_r = st.columns([5, 1.5], gap="small")
with head_l:
    keep_row()
    st.markdown(
        '<h1 class="app-title">⚽ PageRank Match Intelligence</h1>'
        '<div class="app-sub">Passing-network analysis built on StatsBomb event data</div>',
        unsafe_allow_html=True,
    )
with head_r:
    # A plain button with a callback: one click always flips the theme
    st.button("☀️ Light mode" if DARK else "🌙 Dark mode", key="theme_btn", on_click=_flip_theme)

try:
    base_catalog = load_catalog(HAS_CREDS)
except Exception:
    callout("err", "Unable to reach the StatsBomb registry. Check your internet connection and credentials.")
    st.stop()

catalog = merged_catalog(base_catalog)
matches_all = catalog["matches"]
by_id = {m["id"]: m for m in matches_all}

# ---- Sidebar: match finder chat (runs BEFORE the filter widgets so it can set their values) ----
if "chat" not in st.session_state:
    st.session_state["chat"] = [{
        "role": "assistant", "ids": [],
        "text": "Hi! Tell me which match to analyze — teams, a year, or a match ID. "
                "Try <b>Spain vs Russia 2018</b>, <b>Barcelona 2019</b> or <b>#8657</b>. "
                "Got a 2026 match? Use <b>➕ Add your own match</b> below.",
    }]

def render_bubble(msg: dict):
    if msg["role"] == "user":
        st.markdown(f'<div class="chat"><div class="bubble u">{esc(msg["text"])}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat"><div class="bubble a">{msg["text"]}</div></div>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown(
        '<div class="side-head"><div class="side-title">💬 Match finder</div>'
        '<div class="side-sub">Ask for any match in plain text, or add your own data below.</div></div>',
        unsafe_allow_html=True,
    )
    history = st.container()  # filled at the end so messages from the import form show up immediately
    with st.form("finder_form", clear_on_submit=True):
        fc1, fc2 = st.columns([5, 1], gap="small")
        with fc1:
            keep_row()  # keep the box and the send button on one line, even on phones
            query = st.text_input("Ask for a match", placeholder="e.g. Spain vs Russia 2018",
                                  label_visibility="collapsed", key="finder_q")
        sent = fc2.form_submit_button("➤")
    prompt = query.strip() if (sent and query and query.strip()) else None
    if prompt:
        chat = st.session_state["chat"]
        chat.append({"role": "user", "text": prompt})
        reply = chat_reply(prompt, catalog)
        chat.append(reply)
        if reply["autopick"] and reply["ids"]:
            pick_match(by_id[reply["ids"][0]])
        del chat[:-14]  # keep the history short

    # ---- Add your own match (2026 or any) ----
    st.markdown('<div class="side-sec">Your own data</div>', unsafe_allow_html=True)
    with st.expander("➕ Add your own match (2026 or any)"):
        st.caption(
            "If a match isn't in the list (for example a 2026 game), bring your own data: a StatsBomb-format "
            "events JSON (score, formation and subs are read automatically) or a passes CSV."
        )
        tab_up, tab_link = st.tabs(["Upload file", "Paste link"])
        with tab_up:
            up = st.file_uploader("Events JSON or passes CSV", type=["json", "csv"], key="imp_file",
                                  label_visibility="collapsed")
        with tab_link:
            url = st.text_input("Link to an events JSON", placeholder="https://…/events/12345.json", key="imp_url",
                                label_visibility="collapsed")
        st.download_button("⬇ Download CSV template", CSV_TEMPLATE, file_name="passes_template.csv",
                           mime="text/csv", key="imp_tpl")

        parsed = None
        try:
            if up is not None:
                parsed = parse_import(up.name, up.getvalue())
            elif url.strip():
                parsed = parse_import(url.strip().rsplit("/", 1)[-1] or "events.json", fetch_url_bytes(url.strip()))
        except Exception as exc:
            callout("err", esc(str(exc)))

        if parsed:
            st.markdown('<div class="side-sec" style="margin-top:12px">Match details</div>', unsafe_allow_html=True)
            h1, h2 = st.columns(2)
            imp_home = h1.selectbox("Home team", parsed["teams"], key="imp_home")
            imp_away = h2.selectbox("Away team", [t for t in parsed["teams"] if t != imp_home], key="imp_away")
            d1, d2 = st.columns(2)
            imp_date = d1.date_input("Match date", value=datetime.date.today(), key="imp_date")
            imp_comp = d2.text_input("Competition", value="Custom match", key="imp_comp")
            gh = ga = None
            if parsed["kind"] == "csv":  # a CSV has no goal events, so ask for the score (optional)
                g1, g2 = st.columns(2)
                gh = g1.number_input(f"{imp_home} goals", min_value=0, step=1, value=None, key="imp_gh")
                ga = g2.number_input(f"{imp_away} goals", min_value=0, step=1, value=None, key="imp_ga")
            if st.button("Add match", key="imp_add", type="primary") and imp_away:
                added = add_custom_match(parsed, imp_home, imp_away, imp_date.isoformat(),
                                         imp_comp.strip() or "Custom match", gh, ga)
                pick_match(added)
                catalog = merged_catalog(base_catalog)  # refresh before the filters below are built
                matches_all = catalog["matches"]
                by_id = {m["id"]: m for m in matches_all}
                st.session_state["chat"].append({
                    "role": "assistant", "ids": [], "autopick": True,
                    "text": f"Added <b>{esc(added['home'])} vs {esc(added['away'])}</b> ({esc(added['date'])}).",
                })
                callout("ok", "Match added and selected.")

        n_folder = len(scan_custom_dir())
        found = f" ({n_folder} found)" if n_folder else ""
        st.markdown(
            f'<div class="note-s">Tip: drop .json / .csv files into <code>{esc(os.path.basename(CUSTOM_DIR.rstrip("/")))}/</code> and they load '
            f'automatically{found}. Name files like <code>2026-06-15_spain_uruguay.json</code> to set the date.</div>',
            unsafe_allow_html=True,
        )

    # ---- Utilities ----
    b1, b2 = st.columns(2)
    clear_clicked = b1.button("🗑 Clear chat", key="clear_chat", disabled=len(st.session_state["chat"]) <= 1)
    refresh_clicked = b2.button("↻ Refresh list", key="refresh_catalog", help="Re-check StatsBomb for newly released matches")
    if clear_clicked:
        del st.session_state["chat"]
        st.rerun()
    if refresh_clicked:
        load_catalog.clear()
        st.rerun()

    # ---- Chat history (last few messages; older ones are tucked away) ----
    chat = st.session_state["chat"]
    older, recent = chat[:-4], chat[-4:]
    with history:
        if older:
            with st.expander(f"Earlier messages ({len(older)})"):
                for msg in older:
                    render_bubble(msg)
        for j, msg in enumerate(recent):
            render_bubble(msg)
            is_last = (len(older) + j) == len(chat) - 1
            if is_last and msg["role"] == "assistant" and not msg.get("autopick"):
                for mid in msg["ids"]:
                    m = by_id.get(mid)
                    if m:
                        st.button(f"{m['home']} vs {m['away']} · {m['date']} · {_comp(m)}",
                                  key=f"pick_{len(older) + j}_{mid}", on_click=pick_match, args=(m,))


# ---- Main: manual filters ----
with st.expander("🔍 Find a match", expanded=True):
    f1, f2, f3 = st.columns(3)
    team_1 = f1.selectbox("Primary team", catalog["teams"], index=None, placeholder="Search team... (e.g. Spain)", key="f_team1")
    team_2 = f2.selectbox(
        "Secondary team (optional)",
        [t for t in catalog["teams"] if t != team_1],
        index=None,
        placeholder="Search opponent...",
        key="f_team2",
    )
    year = f3.selectbox("Season year (optional)", catalog["years"], index=None, placeholder="Any year", key="f_year")

    src = "StatsBomb paid API" if HAS_CREDS else "StatsBomb open data (free)"
    st.caption(f"Source: {src} · {len(matches_all):,} matches · seasons {catalog['years'][-1]}–{catalog['years'][0]}")
    if "2026" not in catalog["years"]:
        callout(
            "info",
            "No 2026 matches are in the current data source. StatsBomb's free open data covers selected competitions only. "
            "Add your own match from the sidebar (➕), drop files into <code>custom_matches/</code>, or add StatsBomb credentials "
            "(<code>SB_USERNAME</code> / <code>SB_PASSWORD</code>) for the licensed catalog.",
        )

    filtered = matches_all
    if team_1 and team_2:
        pair = {team_1, team_2}
        filtered = [m for m in filtered if {m["home"], m["away"]} == pair]
    elif team_1 or team_2:
        t = team_1 or team_2
        filtered = [m for m in filtered if t in (m["home"], m["away"])]
    if year:
        filtered = [m for m in filtered if m["year"] == year]

    filters_active = bool(team_1 or team_2 or year)
    pool = filtered if filters_active else matches_all[:15]  # catalog is newest-first
    match_labels = {m["label"]: m for m in pool}

    selected_label = None
    if not match_labels:
        callout("warn", "No matches found for those filters. Try adjusting them.")
    else:
        prompt_txt = (f"Select match ({len(pool)} found)" if filters_active
                      else "Recent matches (or use the filters above / the match finder)")
        selected_label = st.selectbox(prompt_txt, list(match_labels.keys()), key="f_match")

if not selected_label:
    callout("info", "Pick a match above, or ask the match finder in the sidebar.")
else:
    target = match_labels[selected_label]
    match_id = target["id"]
    teams = [target["home"], target["away"]]

    try:
        info = load_match_info(match_id, teams[0], teams[1])
    except Exception:
        info = None
    score, home_sc, away_sc, has_pso = (info["scoreboard"] if info and info["scoreboard"] else (None, [], [], False))
    st.markdown(scoreboard_html(teams, score, home_sc, away_sc, has_pso), unsafe_allow_html=True)

    # Phones: show one team at a time (full-width pitch + table). The switch is hidden on laptops.
    st.radio("Team", [0, 1], format_func=lambda i: teams[i], horizontal=True, key="m_team", label_visibility="collapsed")
    hide_nth = 2 if st.session_state.get("m_team", 0) == 0 else 1
    st.markdown(
        "<style>@media (max-width:640px){"
        "div[data-testid='stHorizontalBlock']:has(.team-row)>div[data-testid='stColumn']:nth-child(%d),"
        "div[data-testid='stHorizontalBlock']:has(.team-row)>div[data-testid='column']:nth-child(%d)"
        "{display:none !important}}</style>" % (hide_nth, hide_nth),
        unsafe_allow_html=True,
    )

    # Compute both squads once up front (cached; everything below reuses the results)
    with st.spinner("Analyzing passing networks..."):
        analyses = {squad: analyze_team(match_id, squad) for squad in teams}
    ok = tuple(s for s in teams if analyses[s] is not None)

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    # Row 1: titles + stat tiles
    for col, squad in zip(st.columns(2, gap="small"), teams):
        with col:
            team_row()
            tac = get_tactics(match_id, squad, teams[0], teams[1])
            badge = f'<span class="badge">{esc(tac["formation"])}</span>' if tac["formation"] else ""
            st.markdown(f'<div class="panel-title">🛡️ {esc(squad)} {badge}</div>', unsafe_allow_html=True)
            a = analyses[squad]
            if a is not None:
                st.markdown(
                    tiles_html([
                        ("Team passes", f"{int(a['total_passes'])}"),
                        ("Density", f"{a['density']:.1%}"),
                        ("Centralization", f"{a['centralization']:.1%}"),
                    ]),
                    unsafe_allow_html=True,
                )

    # Row 2: passing networks, side by side
    networks_row(match_id, tuple(teams), ok)

    # Row 3: metrics tables, side by side
    st.markdown('<div class="section">Player metrics</div>', unsafe_allow_html=True)
    for col, squad in zip(st.columns(2, gap="small"), teams):
        with col:
            team_row()
            a = analyses[squad]
            if a is None:
                callout("warn", f"No completed passing data for {esc(squad)}.")
                continue
            table, note = table_html(a["rows"])
            st.markdown(f'<div class="mini-title">{esc(squad)}</div>{table}', unsafe_allow_html=True)
            dep_html = dependency_card_html(a)
            if dep_html:
                st.markdown(dep_html, unsafe_allow_html=True)
            if note:
                st.markdown(f'<div class="note-s">{note}</div>', unsafe_allow_html=True)
    st.caption(
        "Weighted = influence share × the player's own completed outbound passes (rows are sorted by it). "
        "Dependency = drop in global passing-network efficiency when that player is removed (computed by the adaptive PageRank module)."
    )

    # Row 4: bottleneck notes, below the networks and tables
    flagged = {}
    for squad in ok:
        a = analyses[squad]
        pb = a["primary_bottleneck"]
        if a["bottleneck"][pb] > BOTTLENECK_THRESHOLD:
            flagged[squad] = a["display"].get(pb, pb)
    if flagged:
        for col, squad in zip(st.columns(2, gap="small"), teams):
            with col:
                team_row()
                if squad in flagged:
                    callout("warn", f"⚠️ <b>{esc(squad)} bottleneck:</b> {esc(flagged[squad])} — high influence, low outbound distribution.")

    # Spotlight + summary
    available = {s: analyses[s] for s in ok}
    if available:
        st.markdown('<div class="section">⭐ Elite performers</div>', unsafe_allow_html=True)
        for col, squad in zip(st.columns(2, gap="small"), teams):
            with col:
                if squad in available:
                    name, sc, wsc = available[squad]["mvp"]
                    name = available[squad]["display"].get(name, name)
                    st.markdown(mvp_card_html(squad, name, sc, wsc), unsafe_allow_html=True)

        best_squad, best_name = max(
            ((s, p) for s, a_ in available.items() for p in a_["weighted"]),
            key=lambda sp: available[sp[0]]["weighted"][sp[1]],
        )
        best_score = available[best_squad]["scores"][best_name]
        best_weighted = available[best_squad]["weighted"][best_name]
        best_name = available[best_squad]["display"].get(best_name, best_name)

        st.markdown('<div class="section">📊 Match summary</div>', unsafe_allow_html=True)
        callout(
            "ok",
            f"<b>{esc(best_name)} ({esc(best_squad)})</b> was the top distribution hub on the pitch "
            f"(influence weighted by their own outbound passes: <b>{best_weighted:.1f}</b>), holding "
            f"<b>{best_score:.2%}</b> of their team's passing-network influence.",
        )
    else:
        callout("info", "No passing data available to compute a match summary.")