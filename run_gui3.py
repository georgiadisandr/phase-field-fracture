#!/usr/bin/env python3
"""
Set up, run and watch a phase-field solve.

THIRD ITERATION. run_gui.py is the untouched original and run_gui2.py is the
previous working version.  This file is the visual/interaction refinement, so
all three can be compared and an earlier version can be run if something
regresses.

    pip install matplotlib
    python run_gui3.py
    python run_gui3.py config_ambati_3pb.toml

Left: the force-displacement curve building itself live, and the solver log.
Right: every run setting as a field. Import loads an existing TOML into the
form; Save writes the form back out.

HOW IT TALKS TO THE SOLVER
    The form is serialised to a TOML and phasefield_1.exe is launched on it as
    a subprocess -- exactly what you would do by hand. Nothing in the C++
    changes, the solver has no idea it is being driven, and if the GUI dies the
    run continues.

    The generated file is written to _gui_run.toml in the project folder and
    copied into the output directory beside the results, so "what actually
    ran" is always recoverable. A GUI that runs from in-memory state you cannot
    inspect afterwards is not reproducible, and these runs take hours.

    The live curve works because the solver flushes its force-displacement CSV
    every step. No instrumentation needed.
"""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter import font as tkfont

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from matplotlib.collections import PolyCollection, LineCollection
import matplotlib.tri as mtri
import matplotlib.patheffects as mpe
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
GUI_STATE = os.path.join(PROJECT_DIR, ".phasefield_gui3_state.json")
MAX_RECENT_CONFIGS = 8


# ---------------------------------------------------------------------------
# Display scaling
#
# WHY THIS IS NOT OPTIONAL ON WINDOWS. A process that never declares itself
# DPI-aware gets lied to: Windows reports 96 dpi whatever the panel really is,
# lets the app lay itself out for that, then BITMAP-SCALES the finished window
# up to the real resolution. At 150% every glyph and every one-pixel plot line
# is resampled from a smaller image -- which is the soft, slightly smeared look
# this window had, and nothing inside Tk or matplotlib can undo it, because by
# the time they draw the resampling is downstream of them.
#
# Declaring awareness stops that and hands us the true pixel grid, along with
# the job of sizing everything ourselves. Two things follow:
#
#   tk scaling   Tk's PIXELS PER POINT. Set it and every font given a POSITIVE
#                (= point) size grows with the display -- which is all of them
#                here. It is ALSO how matplotlib's Tk backend detects a HiDPI
#                screen: _update_device_pixel_ratio reads this exact value and
#                rescales the figure dpi on its own. So the Figure(dpi=100)
#                calls below must STAY at 100. Multiplying them by the scale as
#                well counts it twice -- 2.25x text on a 150% display.
#
#   px()         Everything measured in raw pixels -- window size, pane widths,
#                sash thickness, wrap widths -- is a number chosen against a
#                96 dpi screen, so it has to be multiplied. A width in
#                CHARACTERS does not: it follows the font, which already
#                scaled.
#
# SYSTEM-aware rather than PER-MONITOR: under per-monitor awareness Windows
# stops rescaling when the window is dragged to a display at a different scale
# factor, and Tk 8.6 cannot re-lay-out for the new dpi -- so the window would
# sit there crisp at the WRONG PHYSICAL SIZE. System-aware keeps it sharp on
# the primary display and correctly sized on every other one, which is the more
# forgiving failure. --scale overrides the whole calculation.
# ---------------------------------------------------------------------------
SCALE = 1.0        # device pixels per 96-dpi pixel: 1.0 at 100%, 1.5 at 150%
SCALE_INFO = ""    # what was detected; reported into the log at startup


def enable_dpi_awareness() -> None:
    """Declare the process DPI-aware. MUST run before the first tk.Tk()."""
    if os.name != "nt":
        return
    try:
        import ctypes
    except ImportError:
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)     # Windows 8.1+
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()          # Vista .. 8.0
    except (AttributeError, OSError):
        pass


def init_scaling(root: tk.Tk, forced: float = 0.0) -> float:
    """Detect the display scale and make Tk agree with it."""
    global SCALE, SCALE_INFO
    try:
        aqua = root.tk.call("tk", "windowingsystem") == "aqua"
    except tk.TclError:
        aqua = False

    if forced > 0.0:
        SCALE, how = forced, "--scale"
    elif aqua:
        # macOS renders the whole app at the backing scale factor itself;
        # multiplying again here would double-count it.
        SCALE, how = 1.0, "macOS, handled by the OS"
    else:
        dpi = 0.0
        if os.name == "nt":
            try:
                import ctypes
                dpi = float(ctypes.windll.user32.GetDpiForSystem())   # Win10+
            except (AttributeError, OSError, ImportError, ValueError):
                dpi = 0.0
        if dpi <= 0.0:
            # Tk derives this from the screen size in pixels and in mm, which
            # on a DPI-aware process is the real thing.
            try:
                dpi = float(root.winfo_fpixels("1i"))
            except tk.TclError:
                dpi = 96.0
        SCALE, how = dpi / 96.0, f"{dpi:.0f} dpi"

    # Never below 1: every size here was chosen at 96 dpi, so shrinking them
    # only clips text. The upper cap is against X servers that report an
    # invented physical screen size -- 0.8 and 1.7 both turn up for perfectly
    # ordinary monitors.
    SCALE = min(max(SCALE, 1.0), 4.0)
    if not aqua:
        root.tk.call("tk", "scaling", SCALE * 96.0 / 72.0)
    try:
        got = float(root.tk.call("tk", "scaling"))
    except tk.TclError:
        got = 0.0
    SCALE_INFO = (f"[gui] display scale {SCALE:.2f}x ({how}), "
                  f"tk scaling {got:.3f}\n")
    return SCALE


def px(n: float) -> int:
    """A length in device pixels, from a measurement chosen at 96 dpi."""
    return int(round(n * SCALE))


def child_env():
    """Environment for a subprocess, with Python's own buffering disabled.

    POLLING FASTER DOES NOTHING IF THE CHILD IS NOT TALKING. A Python process
    whose stdout is a pipe rather than a terminal switches to BLOCK buffering,
    so mesh/from_spec.py holds several KB of progress and releases it in one
    lump -- usually at exit. That looks exactly like a slow GUI and is not.
    PYTHONUNBUFFERED=1 puts it back to line buffering.

    Note this cannot help the C++ solver: its buffering is decided inside its
    own CRT. If solver output still arrives in bursts, that is where to look,
    not here.
    """
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def process_group_kwargs():
    """Start tools in their own process group so Stop can include children."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def signal_process_tree(proc, hard=False):
    """Signal a child and anything it launched, without blocking the UI."""
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        if not hard:
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
                return
            except (OSError, ValueError):
                pass
        # /T includes descendants; /F is reserved for the fallback/quit path.
        cmd = ["taskkill", "/PID", str(proc.pid), "/T"]
        if hard:
            cmd.append("/F")
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill() if hard else proc.terminate()
            except OSError:
                pass
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL if hard else signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            proc.kill() if hard else proc.terminate()
        except OSError:
            pass
# Per PROCESS, so two GUI windows can run at once. Both used to write the same
# _gui_run.toml and hand that one filename to the solver, so launching a second
# run while the first was starting could silently give it the other window's
# settings. Cleaned up on exit.
GENERATED = f"_gui_run_{os.getpid()}.toml"

# A run is treated as "live" if its CSV was flushed within this many seconds.
# main.cpp flushes the force-displacement CSV every accepted step, so a running
# solve always has a fresh mtime; anything older is a finished run.
LIVE_RUN_SECONDS = 20
EXE_CANDIDATES = ["build/phasefield_1.exe", "build/Release/phasefield_1.exe",
                  "build/Debug/phasefield_1.exe", "build/phasefield_1"]
# How often the log pane drains the subprocess pipe. This is the number that
# decides how "live" a run feels, so it is deliberately short.
#
# 25 ms = 40 Hz. Going much below this buys nothing: Tk's after() resolves to
# roughly 10-15 ms on Windows, so the timer itself becomes the floor, and 40 Hz
# is already past the point where text stops looking stepped. If the log still
# arrives in bursts at this rate the cause is the SOLVER's own stdout
# buffering, not the GUI -- see child_env().
LOG_POLL_MS = 25
# The force-displacement curve is redrawn on its own, much slower cadence:
# it re-reads the whole CSV and asks matplotlib to redraw, which costs far more
# than appending text. Redrawing it at LOG_POLL_MS would spend the entire UI
# budget on a plot that gains one point every few seconds.
PLOT_MS = 500
POLL_MS = LOG_POLL_MS        # the main tick interval
# Time budget per tick for draining the queue. A burst of thousands of lines
# (gmsh is chatty) must not freeze the window: whatever is left waits one tick.
# Kept well under LOG_POLL_MS so the drain can never monopolise the event loop.
DRAIN_MS = 10
LOG_LINES = 400
# Keep a larger plain-text history for Save even though the on-screen widget is
# trimmed aggressively for responsiveness.
LOG_HISTORY_CHARS = 5_000_000
# Most converged-step markers drawn at once. Above this they are thinned by a
# stride (and the legend says so); non-converged markers are always drawn in
# full, however many there are.
MAX_MARKERS = 400

# Above this many elements IN VIEW the Mesh tab shows an element-size map
# instead of the wireframe. Chosen from the plot width: at ~700 px, more than
# roughly this many elements are sub-pixel, so the wireframe conveys nothing
# and only costs time.
MESH_DETAIL_LIMIT = 60_000

# Mesh tab view modes.
#   auto       size map when the wireframe would be sub-pixel, elements when
#              it would not. The sensible default.
#   elements   ALWAYS the real wireframe, however long it takes. At full
#              zoom-out on a fine mesh this fills in solid -- which is not a
#              bug, it is what a mesh finer than the screen looks like.
#   size map   ALWAYS the size map, even on a small mesh, which is the quickest
#              way to check a refinement is where you asked for it.
MESH_VIEWS = ["auto", "elements", "size map"]

# Colour maps offered for the phi contour. All SEQUENTIAL: phi runs 0 -> 1 from
# intact to fully broken, so the colours must too. A diverging map (coolwarm and
# friends) puts its neutral colour in the MIDDLE of the range, which for damage
# is a meaningless place to draw the eye.
#
# inferno/magma/viridis are perceptually uniform -- equal steps in phi look like
# equal steps in colour. jet and turbo are not: jet in particular invents banding
# where the field is smooth, which on a smeared phase field can read as detail
# that is not there. They are offered because most of the phase-field literature
# plots blue-to-red, so they make comparison with published figures easier.
CMAPS = ["inferno", "magma", "viridis", "plasma", "turbo", "jet",
         "hot", "afmhot", "cividis", "gray"]

# Iso-line colour per map. The line marks the crack, so it must not disappear
# into the colours around phi = 0.9 -- which is the BRIGHT end of every
# sequential map, so a light line is the wrong default.
ISO_COLOR = {
    "inferno": "#00e5ff", "magma": "#00e5ff", "plasma": "#00e5ff",
    "hot": "#00e5ff", "afmhot": "#0066ff",
    "viridis": "#ff1e6e", "cividis": "#ff1e6e",   # yellow at the top end
    "turbo": "#ffffff", "jet": "#ffffff",         # red at the top end
    "gray": "#ff1e6e",
}

# ---------------------------------------------------------------------------
# Dark theme
# ---------------------------------------------------------------------------
DARK = {
    "bg": "#1e1f22", "panel": "#26282c", "field": "#2f3237", "hover": "#3a3f46",
    # Text toned down across the board. Near-white on near-black is a lot of
    # contrast to sit in front of for an afternoon; all of these still clear
    # 4.5:1 against the background.
    "fg": "#c5c8cd", "muted": "#868b92", "accent": "#6f9ed4",
    "warn": "#bd8f52", "err": "#cf6b6b", "grid": "#3a3d42", "line": "#5ac8fa",
    "warnbg": "#2a2620",
    # Softer lines. "grid" was doing double duty as the section outline AND
    # the field border, which made a row of seven entries read as a cage.
    "accent_hi": "#8ab4e8", "onaccent": "#0d1117", "errbg": "#2e2022",
    "edge": "#33363b", "outline": "#2c2f34", "cell": "#2b2e33",
    # Mesh view: near-white wireframe on a face darker than the panel, so the
    # elements read as a mesh rather than as texture.
    "mesh_edge": "#dfe6ee", "mesh_face": "#15171a",
    "success": "#5fc98b",
}
LIGHT = {
    "bg": "#f0f0f0", "panel": "#e4e4e4", "field": "#ffffff", "hover": "#d8d8d8",
    "fg": "#101010", "muted": "#555555", "accent": "#0a5fb4",
    "warn": "#b36b00", "err": "#b00020", "grid": "#cccccc", "line": "#1f77b4",
    "warnbg": "#fffaf0",
    "accent_hi": "#0b6ac9", "onaccent": "#ffffff", "errbg": "#fdeaea",
    "edge": "#c4c4c4", "outline": "#d2d2d2", "cell": "#fbfbfb",
    "mesh_edge": "#222222", "mesh_face": "#ffffff",
    "success": "#167447",
}
T = DARK          # active palette; swapped by --light before the window is built


def apply_theme(root: tk.Tk) -> None:
    """Recolour ttk and the plain-tk widgets.

    The theme MUST be switched to 'clam' first. Windows defaults to 'vista',
    whose Entry, Combobox and Button elements are drawn by the OS and silently
    ignore background/foreground settings -- you get a dark window with white
    input boxes and no error to explain why. 'clam' is drawn by Tk itself, so
    every colour below actually applies.
    """
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass                                   # exotic build: keep what we have

    root.configure(background=T["bg"])
    style.configure(".", background=T["bg"], foreground=T["fg"],
                    fieldbackground=T["field"], bordercolor=T["grid"],
                    lightcolor=T["panel"], darkcolor=T["panel"],
                    troughcolor=T["bg"], insertcolor=T["fg"],
                    focuscolor=T["accent"])
    style.configure("TFrame", background=T["bg"])
    style.configure("TLabel", background=T["bg"], foreground=T["fg"])
    style.configure("Title.TLabel", background=T["bg"], foreground=T["fg"],
                    font=("TkDefaultFont", 17, "bold"))
    style.configure("Subtitle.TLabel", background=T["bg"],
                    foreground=T["muted"], font=("TkDefaultFont", 9))
    style.configure("SectionHint.TLabel", background=T["bg"],
                    foreground=T["muted"], font=("TkDefaultFont", 8))
    style.configure("Good.TLabel", background=T["bg"],
                    foreground=T["success"], font=("TkDefaultFont", 9, "bold"))
    style.configure("Warning.TLabel", background=T["bg"],
                    foreground=T["warn"], font=("TkDefaultFont", 9, "bold"))
    style.configure("Status.TLabel", background=T["panel"], foreground=T["fg"],
                    padding=[px(10), px(7)], font=("TkDefaultFont", 10, "bold"))
    style.configure("Activity.TLabel", background=T["bg"], foreground=T["accent"],
                    font=("TkDefaultFont", 10, "bold"))
    style.configure("RunDetail.TLabel", background=T["bg"], foreground=T["fg"],
                    font=("TkFixedFont", 9))
    style.configure("TLabelframe", background=T["bg"],
                    bordercolor=T["outline"], lightcolor=T["outline"],
                    darkcolor=T["outline"])
    style.configure("TLabelframe.Label", background=T["bg"],
                    foreground=T["accent"])
    # bordercolor alone does nothing about a bright button edge: clam draws a
    # 3D bevel from lightcolor (near-white by default) and darkcolor, so all
    # three must be flattened. focuscolor is the dotted ring, otherwise drawn
    # in the foreground colour.
    style.configure("TButton", background=T["panel"], foreground=T["muted"],
                    bordercolor=T["edge"], lightcolor=T["edge"],
                    darkcolor=T["edge"], focuscolor=T["bg"],
                    relief="flat", borderwidth=1, padding=px(4))
    style.map("TButton",
              background=[("active", T["hover"]), ("disabled", T["bg"])],
              foreground=[("active", T["fg"]), ("disabled", T["muted"])],
              bordercolor=[("active", T["grid"])],
              lightcolor=[("active", T["grid"])],
              darkcolor=[("active", T["grid"])])
    style.configure("Primary.TButton", background=T["accent"],
                    foreground=T["onaccent"], bordercolor=T["accent"],
                    lightcolor=T["accent"], darkcolor=T["accent"],
                    focuscolor=T["accent"], relief="flat", borderwidth=1,
                    padding=(px(12), px(4)),
                    font=("TkDefaultFont", 9, "bold"))
    style.map("Primary.TButton",
              background=[("active", T["accent_hi"]),
                          ("disabled", T["panel"])],
              foreground=[("disabled", T["muted"])],
              bordercolor=[("active", T["accent_hi"]), ("disabled", T["edge"])],
              lightcolor=[("active", T["accent_hi"]), ("disabled", T["edge"])],
              darkcolor=[("active", T["accent_hi"]), ("disabled", T["edge"])])
    style.configure("Danger.TButton", background=T["panel"],
                    foreground=T["err"], bordercolor=T["edge"],
                    lightcolor=T["edge"], darkcolor=T["edge"],
                    focuscolor=T["bg"], relief="flat", borderwidth=1,
                    padding=px(4))
    style.map("Danger.TButton",
              background=[("active", T["errbg"]), ("disabled", T["bg"])],
              foreground=[("active", T["err"]), ("disabled", T["muted"])],
              bordercolor=[("active", T["err"])],
              lightcolor=[("active", T["err"])],
              darkcolor=[("active", T["err"])])
    style.configure("Tool.TButton", padding=px(4), font=("TkDefaultFont", 9))
    style.configure("Section.TButton", background=T["panel"], foreground=T["fg"],
                    bordercolor=T["outline"], lightcolor=T["outline"],
                    darkcolor=T["outline"], anchor="w", relief="flat",
                    padding=[px(9), px(6)], font=("TkDefaultFont", 9, "bold"))
    style.map("Section.TButton",
              background=[("active", T["hover"])],
              foreground=[("active", T["fg"])])
    style.configure("Summary.TFrame", background=T["panel"])
    style.configure("SummaryTitle.TLabel", background=T["panel"],
                    foreground=T["accent"], font=("TkDefaultFont", 9, "bold"))
    style.configure("SummaryText.TLabel", background=T["panel"],
                    foreground=T["fg"], font=("TkDefaultFont", 9))
    style.configure("Invalid.TEntry", fieldbackground=T["field"],
                    foreground=T["fg"], insertcolor=T["fg"],
                    bordercolor=T["err"], lightcolor=T["err"],
                    darkcolor=T["err"], borderwidth=2)
    style.configure("Invalid.TCombobox", fieldbackground=T["field"],
                    foreground=T["fg"], background=T["panel"],
                    arrowcolor=T["fg"], bordercolor=T["err"],
                    lightcolor=T["err"], darkcolor=T["err"], borderwidth=2)

    # Table cells: flush, flat, dim border. Same reasoning as the buttons --
    # the bevel is what made a row of entries look like a cage.
    style.configure("Cell.TEntry", fieldbackground=T["cell"],
                    foreground=T["fg"], bordercolor=T["edge"],
                    lightcolor=T["edge"], darkcolor=T["edge"],
                    insertcolor=T["fg"], borderwidth=1, relief="flat",
                    padding=px(3))
    style.map("Cell.TEntry",
              bordercolor=[("focus", T["accent"])],
              lightcolor=[("focus", T["accent"])],
              darkcolor=[("focus", T["accent"])])
    style.configure("Cell.TCombobox", fieldbackground=T["cell"],
                    foreground=T["fg"], background=T["panel"],
                    arrowcolor=T["muted"], bordercolor=T["edge"],
                    lightcolor=T["edge"], darkcolor=T["edge"],
                    borderwidth=1, relief="flat", padding=px(2))
    style.map("Cell.TCombobox",
              fieldbackground=[("readonly", T["cell"])],
              foreground=[("readonly", T["fg"])])
    style.configure("TEntry", fieldbackground=T["field"], foreground=T["fg"],
                    insertcolor=T["fg"], bordercolor=T["grid"])
    style.configure("TCombobox", fieldbackground=T["field"], foreground=T["fg"],
                    background=T["panel"], arrowcolor=T["fg"],
                    bordercolor=T["grid"])
    style.map("TCombobox",
              fieldbackground=[("readonly", T["field"])],
              foreground=[("readonly", T["fg"])],
              background=[("active", T["hover"])])
    style.configure("TCheckbutton", background=T["bg"], foreground=T["fg"],
                    indicatorcolor=T["field"])
    style.map("TCheckbutton",
              background=[("active", T["bg"])],
              indicatorcolor=[("selected", T["accent"])])
    # Tabs are navigation rather than actions.  The selected tab joins the
    # content area and uses an accent label, matching the calmer gui2 design.
    style.configure("TNotebook", background=T["bg"], bordercolor=T["grid"],
                    tabmargins=[px(2), px(6), px(2), 0])
    style.configure("TNotebook.Tab", background=T["panel"],
                    foreground=T["muted"], bordercolor=T["grid"],
                    padding=[px(14), px(6)],
                    font=("TkDefaultFont", 9))
    style.map("TNotebook.Tab",
              background=[("selected", T["bg"]), ("active", T["hover"])],
              foreground=[("selected", T["accent"]), ("active", T["fg"])],
              expand=[("selected", [1, 1, 1, 0])])

    # The sash is invisible at its default 2 px on a dark theme; make it wide
    # enough to grab and light enough to see.
    style.configure("TPanedwindow", background=T["grid"])
    style.configure("Sash", sashthickness=px(7), gripcount=12,
                    background=T["grid"], lightcolor=T["hover"],
                    bordercolor=T["bg"], handlepad=px(60))

    style.configure("TScrollbar", background=T["panel"], troughcolor=T["bg"],
                    bordercolor=T["bg"], arrowcolor=T["fg"])
    style.map("TScrollbar", background=[("active", T["hover"])])

    # The Combobox dropdown is a plain tk Listbox living in its own toplevel,
    # so ttk styling never reaches it. Without these it stays white.
    root.option_add("*TCombobox*Listbox.background", T["field"])
    root.option_add("*TCombobox*Listbox.foreground", T["fg"])
    root.option_add("*TCombobox*Listbox.selectBackground", T["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", "#000000")


def add_toolbar(canvas, parent):
    """matplotlib's pan/zoom/home toolbar, recoloured to match.

    Its buttons are plain tk widgets, so ttk styling never reaches them --
    without the loop below you get a bright grey strip under a dark plot.
    """
    try:
        tb = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
    except TypeError:
        tb = NavigationToolbar2Tk(canvas, parent)   # matplotlib < 3.5
    tb.update()
    try:
        tb.config(background=T["bg"])
        for child in tb.winfo_children():
            try:
                child.config(background=T["bg"])
            except tk.TclError:
                pass
        if hasattr(tb, "_message_label"):
            tb._message_label.config(background=T["bg"], foreground=T["fg"])
    except tk.TclError:
        pass
    return tb


def style_axes(fig, ax) -> None:
    """Recolour a matplotlib axes. Must be re-applied after every ax.clear(),
    which resets tick colours and spines back to the default black."""
    fig.patch.set_facecolor(T["bg"])
    ax.set_facecolor(T["panel"])
    ax.tick_params(colors=T["fg"], which="both")
    for s in ax.spines.values():
        s.set_color(T["grid"])
    ax.xaxis.label.set_color(T["fg"])
    ax.yaxis.label.set_color(T["fg"])
    ax.grid(True, alpha=0.25, color=T["grid"])


SPLITS = ["none", "lancioni", "amor", "spectral"]
SCHEMES = ["staggered", "monolithic"]
STAGGER_STOPS = ["energy", "residual"]
STEP_MODES = ["two_stage", "three_stage", "uniform"]
NTYPES = {"plane stress": 1, "plane strain": 2}

# ---------------------------------------------------------------------------
# The repeatable STEP-geometry arguments, as tables.
#
# Each entry is (title, flag, hint, template, columns). The template drives
# both the writer and the reader (see compile_template), so the table view and
# the text view cannot disagree about the format.
# ---------------------------------------------------------------------------
NUMW = 8      # 0.00333 is 7 characters; 6 clips it
_P0 = ("First point of the SEGMENT. A refine line is two POINTS "
       "(x0,y0)->(x1,y1) -- unlike a refine box, which is two RANGES.")
_P1 = ("Second point of the SEGMENT. The band follows the straight line "
       "between the two points, at any angle.")
_HL = ("Element size inside the fully-refined band, in mm. This is the number "
       "that has to resolve l0: l0/h >= 2 is the minimum, >= 4 is where "
       "results stop moving.")
_NEAR = ("How far the fine size h is HELD either side of the segment, in mm. "
         "A RADIUS, not a fraction of the specimen. Wide enough that the "
         "crack cannot wander out of it.")
_FAR = ("Distance at which the size has grown back to the bulk h. Must be "
        "greater than 'near'. Omitted, it defaults to 5x near -- usually "
        "wider than intended.")

GEOM_TABLES = [
    ("refine lines", "--refine-line",
     "a graded band around a segment at any angle -- two POINTS",
     "{x0},{y0},{x1},{y1}:{h}:{near}:{far}",
     [("x0", "x0", NUMW, "num", _P0), ("y0", "y0", NUMW, "num", _P0),
      ("x1", "x1", NUMW, "num", _P1), ("y1", "y1", NUMW, "num", _P1),
      ("h", "h", NUMW, "num", _HL), ("near", "near", NUMW, "num", _NEAR),
      ("far", "far", NUMW, "num", _FAR)]),

    ("refine boxes", "--refine-box",
     "refine a whole REGION -- two RANGES, note the order differs from a line",
     "{x0},{x1},{y0},{y1}:{h}:{t}",
     [("x0", "x0", NUMW, "num",
       "x RANGE of the box: from x0 to x1. NOTE the ordering differs from a "
       "refine line, which is two points."),
      ("x1", "x1", NUMW, "num", "Upper x of the box."),
      ("y0", "y0", NUMW, "num", "y RANGE of the box: from y0 to y1."),
      ("y1", "y1", NUMW, "num", "Upper y of the box."),
      ("h", "h", NUMW, "num",
       "Element size inside the box. Use a box when the crack path is NOT "
       "known in advance -- a distance field can only refine around geometry "
       "that already exists."),
      ("t", "thick", NUMW, "num",
       "Width of the graded collar OUTSIDE the box. Without it the size jumps "
       "from h straight to the bulk across one element and distorts every "
       "element on the boundary. Defaults to 4h.")]),

    ("supports / pads", "--pad",
     "cuts a named piece out of a boundary edge",
     "{name}={x0},{y0},{x1},{y1}",
     [("name", "name", 12, "text",
       "Physical group name -- the handle the solver uses in [[bcs]], so it "
       "must match the config exactly."),
      ("x0", "x0", NUMW, "num",
       "Start of the pad, ON the existing boundary. A pad SPLITS a long CAD "
       "edge so a support can be named in the middle of it, which --edge "
       "cannot do."),
      ("y0", "y0", NUMW, "num", "Start of the pad (y)."),
      ("x1", "x1", NUMW, "num",
       "End of the pad. Keep it a few elements long: a one-element pad "
       "carries the whole reaction on two nodes, which is the point restraint "
       "the pad exists to avoid."),
      ("y1", "y1", NUMW, "num", "End of the pad (y).")]),

    # First point is always the MOUTH, second is always the TIP. The mesher
    # still accepts an explicit open_at, but offering the choice here only
    # created a way to get it backwards -- and getting it backwards fails
    # quietly: the mesh writes, the solve runs, the answer is wrong. For an
    # embedded flaw (both ends welded) use from_step.py directly with ":none".
    ("cracks", "--crack",
     "first point = mouth (opens at the surface), second point = crack tip",
     "{x0},{y0},{x1},{y1}:{name}",
     [("x0", "x0", NUMW, "num",
       "Crack MOUTH -- the end that meets a free surface, where the nodes are "
       "split open. On a boundary edge the fragment splits the edge there; a "
       "mouth mid-edge with no split meshes NO surface and reports success."),
      ("y0", "y0", NUMW, "num", "Crack mouth (y)."),
      ("x1", "x1", NUMW, "num",
       "Crack TIP -- always the second point. Stays welded (one shared node), "
       "which is what makes it a tip inside the material rather than a cut "
       "that severs the part. This is where the refinement rosette goes."),
      ("y1", "y1", NUMW, "num", "Crack tip (y)."),
      ("name", "name", 10, "text",
       "Physical group name. Also creates <Name>Mouth and <Name>Tip as 0D "
       "groups, usable by [[initial_phi]] and [[point_load]].")],
     ("{x0},{y0},{x1},{y1}:{name}:{open}",)),

    ("named points", "--point",
     "a 0D group, for a point load or an initial phi",
     "{name}={x},{y}",
     [("name", "name", 14, "text",
       "Physical group name. [[point_load]] and [[initial_phi]] resolve "
       "against dimension 0 ONLY; a Dirichlet BC tries 1D first, then 0D."),
      ("x", "x", NUMW, "num",
       "On a boundary edge this splits the edge to create the vertex; inside "
       "the part the node is embedded."),
      ("y", "y", NUMW, "num", "Point location (y).")]),

    ("extra edges", "--edge",
     "names every WHOLE edge inside this window",
     "{name}={x0},{x1},{y0},{y1}",
     [("name", "name", 12, "text",
       "Physical group name for every edge found in the window."),
      ("x0", "x0", NUMW, "num",
       "A WINDOW (x0..x1, y0..y1) that must fully CONTAIN the edge. This "
       "names whole edges only -- to name part of a long edge, use a pad."),
      ("x1", "x1", NUMW, "num", "Upper x of the window."),
      ("y0", "y0", NUMW, "num", "Lower y of the window."),
      ("y1", "y1", NUMW, "num", "Upper y of the window.")]),
]

# Plain fields, shown inside the section they belong to rather than in one
# global sizing block: crack size/radius belong with the cracks, pad size with
# the pads.
GEOM_EXTRAS = {
    "cracks": (
        ("crack size", "h_crack", "",
         "Element size along the crack and around its tip, in mm. Must "
         "resolve l0: aim for l0/h >= 4."),
        ("crack radius", "r_crack", "",
         "near[:far] in mm -- the fine size is held out to 'near' and is back "
         "to the bulk by 'far'. A bare value means near:5*near."),
    ),
    "supports / pads": (
        ("pad size", "h_pad", "",
         "Element size on the pads, in mm. Default bulk/4, about 5 nodes per "
         "pad. An unrefined pad behaves like the point restraint it replaces."),
    ),
    "extra edges": (
        ("edge size", "h_fine", "",
         "Element size on the named edges, in mm. Applies to EVERY named "
         "edge at once -- with 'name outer edges' on, that is all four sides."),
        ("edge radius", "r_fine", "",
         "near[:far] in mm. On a small part a radius above about a quarter of "
         "the width covers the whole domain."),
    ),
}
# The element types GmshReader accepts. Mirrors mesh/common.ELEMENT_TYPES;
# anything else the mesher can produce (quad9, tri6) is rejected by the solver.
ELEMENTS = ["quad4", "tri3", "quad8"]


# ===========================================================================
#  Pure logic -- no Tk, so it can be tested without a display
# ===========================================================================
def find_exe() -> str:
    for rel in EXE_CANDIDATES:
        p = os.path.join(PROJECT_DIR, rel)
        if os.path.isfile(p):
            return p
    return ""


def _toml_load(path: str) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib          # pip install tomli on Python < 3.11
    with open(path, "rb") as f:
        return tomllib.load(f)


DEFAULTS = {
    "mesh_source": "file", "mesh_path": "", "base_name": "run",
    "ntype": 2, "ngaus": 2, "energy_split": "spectral", "hybrid": True,
    "mat_name": "material", "E": 20800.0, "nu": 0.3, "Gc": 0.54,
    "l0": 0.03, "k": 1e-9, "domain": "Domain",
    "bcs": [],
    "scheme": "staggered", "stagger_stop": "energy", "max_staggered": 20,
    "gamma_tol": 10.0, "tol_rel": 1e-6, "tol_abs": 1e-8, "max_iter": 100,
    "step_mode": "two_stage", "N_steps": 20, "du": 1e-4, "du_coarse": 1e-3,
    "u_switch": 0.02, "du_fine": 1e-4,
    "u_switch2": 0.0, "du_final": 0.0, "max_subdivs": 4,
    "output_dir": ".", "vtk_every": 10, "profile_every": 0, "write_log": True,
}

# Presets intentionally touch solver/run controls only.  Specimen geometry,
# material values and boundary conditions remain the user's model.
BUILTIN_PRESETS = {
    "Robust staggered": {
        "scheme": "staggered", "stagger_stop": "energy",
        "max_staggered": 30, "max_subdivs": 4,
        "tol_rel": 1e-6, "tol_abs": 1e-8,
    },
    "Fast preview": {
        "scheme": "staggered", "stagger_stop": "energy",
        "max_staggered": 10, "max_subdivs": 1,
        "vtk_every": 25, "profile_every": 0,
    },
    "Monolithic": {
        "scheme": "monolithic", "max_subdivs": 4,
        "tol_rel": 1e-6, "tol_abs": 1e-8, "max_iter": 100,
    },
}


def config_to_form(cfg: dict) -> dict:
    """Flatten a parsed TOML into the flat dict the form holds."""
    f = dict(DEFAULTS)
    mesh, fem, sol, run = (cfg.get("mesh", {}), cfg.get("fem", {}),
                           cfg.get("solver", {}), cfg.get("run", {}))
    f["mesh_source"] = mesh.get("source", "builtin")
    f["mesh_path"] = mesh.get("path", "")
    f["base_name"] = mesh.get("base_name", "run")

    f["ntype"] = int(fem.get("ntype", 2))
    f["ngaus"] = int(fem.get("ngaus", 2))
    f["energy_split"] = fem.get("energy_split", "spectral")
    f["hybrid"] = bool(fem.get("hybrid", True))

    mats = cfg.get("materials") or []
    if mats:
        m = mats[0]
        f["mat_name"] = m.get("name", "material")
        p = list(m.get("props", []))
        # props is positional: { E, nu, Gc, l0, k }
        for i, key in enumerate(("E", "nu", "Gc", "l0", "k")):
            if i < len(p):
                f[key] = float(p[i])
    mfg = cfg.get("material_for_group") or []
    if mfg:
        f["domain"] = mfg[0].get("group", "Domain")

    rows = [{"kind": "displacement", "group": b.get("physical_name", ""),
             "fix_x": int((b.get("flags") or [0, 0])[0]),
             "fix_y": int((b.get("flags") or [0, 0])[1]),
             "vx": float((b.get("values") or [0.0, 0.0])[0]),
             "vy": float((b.get("values") or [0.0, 0.0])[1])}
            for b in (cfg.get("bcs") or [])]
    rows += [{"kind": "force", "group": b.get("physical_name", ""),
              "fix_x": 0, "fix_y": 0,
              "vx": float((b.get("force") or [0.0, 0.0])[0]),
              "vy": float((b.get("force") or [0.0, 0.0])[1])}
             for b in (cfg.get("point_load") or [])]
    rows += [{"kind": "traction", "group": b.get("physical_name", ""),
              "fix_x": 0, "fix_y": 0,
              "vx": float((b.get("traction") or [0.0, 0.0])[0]),
              "vy": float((b.get("traction") or [0.0, 0.0])[1])}
             for b in (cfg.get("neumann") or [])]
    f["bcs"] = rows

    f["scheme"] = sol.get("scheme", "staggered")
    f["stagger_stop"] = sol.get("stagger_stop", "energy")
    f["max_staggered"] = int(sol.get("max_staggered", 20))
    f["gamma_tol"] = float(sol.get("stagger_gamma_tol_deg", 10.0))
    f["tol_rel"] = float(sol.get("tol_rel", 1e-6))
    f["tol_abs"] = float(sol.get("tol_abs", 1e-8))
    f["max_iter"] = int(sol.get("max_iter", 100))

    f["step_mode"] = run.get("step_mode", "two_stage")
    f["N_steps"] = int(run.get("N_steps", 20))
    f["du"] = float(run.get("du", 0.0))
    f["du_coarse"] = float(run.get("du_coarse", 1e-3))
    f["u_switch"] = float(run.get("u_switch", 0.02))
    f["du_fine"] = float(run.get("du_fine", 1e-4))
    f["u_switch2"] = float(run.get("u_switch2", 0.0))
    f["du_final"] = float(run.get("du_final", 0.0))
    f["max_subdivs"] = int(run.get("max_subdivs", 4))
    f["output_dir"] = run.get("output_dir", ".")
    f["vtk_every"] = int(run.get("vtk_every", 10))
    f["profile_every"] = int(run.get("profile_every", 0))
    f["write_log"] = bool(run.get("write_log", True))
    return f


def _str(x) -> str:
    r"""A TOML basic string, correctly escaped.

    THE BUG THIS FIXES: a Windows path pasted into output_dir or mesh path was
    written verbatim, so

        output_dir    = "C:\Users\andre\runs"

    made the solver die at startup with

        TOML parse error: Error while parsing unicode scalar sequence:
        expected hex digit, saw 's'

    -- because inside a TOML basic string \U begins a unicode escape and
    "sers" is not hex. \n, \t and friends are just as capable of silently
    corrupting a path. Every quoted value goes through here now.
    """
    s = str(x)
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _num(x) -> str:
    """TOML number, keeping exponent form readable for the tiny values here."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    s = repr(float(x))
    return s


def form_to_toml(f: dict) -> str:
    """Serialise the form to a run config. Written by hand rather than with a
    TOML library so there is no extra dependency and the output stays readable
    -- you are meant to be able to open this file and see what ran."""
    L = ["# Generated by run_gui3.py -- edit the GUI, not this file.",
         f"# {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
         "[mesh]",
         f'source    = ' + _str(f["mesh_source"])]
    if f["mesh_source"] == "file":
        L.append(f'path      = ' + _str(f["mesh_path"]))
    L += [f'base_name = ' + _str(f["base_name"]), "",
          "[fem]",
          f'ntype        = {int(f["ntype"])}',
          "ndofn        = 2",
          f'ngaus        = {int(f["ngaus"])}',
          "nstre        = 3",
          f'energy_split = ' + _str(f["energy_split"]),
          f'hybrid       = {_num(bool(f["hybrid"]))}', "",
          "[[materials]]",
          f'name  = ' + _str(f["mat_name"]),
          "props = [ " + ", ".join(_num(float(f[k]))
                                   for k in ("E", "nu", "Gc", "l0", "k"))
          + " ]   # { E, nu, Gc, l0, k }", "",
          "[[material_for_group]]",
          f'group    = ' + _str(f["domain"]),
          f'material = ' + _str(f["mat_name"]), ""]

    # Each kind goes to the section the solver reads it from.
    for b in f["bcs"]:
        if not b.get("group") or b.get("kind", "displacement") != "displacement":
            continue
        L += ["[[bcs]]",
              f'physical_name = ' + _str(b["group"]),
              f'flags         = [{int(b["fix_x"])}, {int(b["fix_y"])}]',
              f'values        = [{_num(float(b["vx"]))}, {_num(float(b["vy"]))}]',
              ""]
    for b in f["bcs"]:
        if not b.get("group") or b.get("kind") != "force":
            continue
        L += ["[[point_load]]   # N, scaled by load_factor each step",
              f'physical_name = ' + _str(b["group"]),
              f'force         = [{_num(float(b["vx"]))}, {_num(float(b["vy"]))}]',
              ""]
    for b in f["bcs"]:
        if not b.get("group") or b.get("kind") != "traction":
            continue
        L += ["[[neumann]]      # N/mm of edge, scaled by load_factor each step",
              f'physical_name = ' + _str(b["group"]),
              f'traction      = [{_num(float(b["vx"]))}, {_num(float(b["vy"]))}]',
              ""]

    L += ["[solver]",
          f'tol_rel               = {_num(float(f["tol_rel"]))}',
          f'tol_abs               = {_num(float(f["tol_abs"]))}',
          f'max_iter              = {int(f["max_iter"])}',
          "verbose               = true",
          f'scheme                = ' + _str(f["scheme"]),
          f'max_staggered         = {int(f["max_staggered"])}',
          f'stagger_stop          = ' + _str(f["stagger_stop"]),
          f'stagger_gamma_tol_deg = {_num(float(f["gamma_tol"]))}', "",
          "[run]",
          f'step_mode     = ' + _str(f["step_mode"]),
          f'du            = {_num(float(f["du"]))}'
          + '   # uniform mode: displacement increment (0 = use N_steps)',
          f'N_steps       = {int(f["N_steps"])}',
          f'du_coarse     = {_num(float(f["du_coarse"]))}',
          f'u_switch      = {_num(float(f["u_switch"]))}',
          f'du_fine       = {_num(float(f["du_fine"]))}',
          f'u_switch2     = {_num(float(f["u_switch2"]))}'
          + '   # three_stage only',
          f'du_final      = {_num(float(f["du_final"]))}',
          f'max_subdivs   = {int(f["max_subdivs"])}',
          "show_gui      = false",
          "preview_mesh  = false",
          f'output_dir    = ' + _str(f["output_dir"]),
          f'vtk_every     = {int(f["vtk_every"])}',
          f'write_log     = {_num(bool(f["write_log"]))}',
          f'profile_every = {int(f["profile_every"])}', ""]
    return "\n".join(L)


def validate_form(f: dict) -> list[str]:
    """Problems worth stopping for. Returns a list of messages, empty if fine."""
    p = []
    if not f["base_name"].strip():
        p.append("base_name is empty -- it names the output folder.")
    if f["mesh_source"] == "file":
        if not f["mesh_path"].strip():
            p.append("Mesh source is 'file' but no mesh path is set.")
        elif not os.path.isfile(os.path.join(PROJECT_DIR, f["mesh_path"])):
            p.append(f"Mesh file not found: {f['mesh_path']}")
    rows = [b for b in f["bcs"] if b.get("group")]
    if not any(b.get("kind", "displacement") == "displacement" for b in rows):
        p.append("No displacement BC -- nothing would restrain the specimen "
                 "(a force or traction alone leaves it free to fly away).")
    if not any(float(b["vx"]) or float(b["vy"]) for b in rows):
        p.append("Every prescribed value is zero -- nothing is loaded.")
    for key in ("E", "Gc", "l0"):
        if float(f[key]) <= 0:
            p.append(f"{key} must be positive.")
    if not 0.0 < float(f["nu"]) < 0.5:
        p.append("nu should be in (0, 0.5).")
    if f["step_mode"] == "three_stage":
        if float(f["du_final"]) <= 0:
            p.append("three_stage needs du_final > 0.")
        if float(f["u_switch2"]) <= float(f["u_switch"]):
            p.append("u_switch2 must be greater than u_switch, or the second "
                     "stage has zero width and is skipped silently.")
    if f["step_mode"] in ("two_stage", "three_stage") \
            and float(f["du_fine"]) > float(f["du_coarse"]):
        p.append("du_fine should be <= du_coarse (the fine stage is the refined one).")
    return p


def warn_form(f: dict) -> list[str]:
    """Things that are legal but usually a mistake. Shown, not blocking."""
    w = []
    if int(f["max_subdivs"]) == 0:
        w.append(
            "max_subdivs = 0 disables adaptive halving. A non-converged step is "
            "then FORCE-ACCEPTED and its history field is committed permanently, "
            "steering the crack for the rest of the run. Consider 4.")
    if f["scheme"] == "staggered" and int(f["max_staggered"]) < 10:
        w.append(
            f"max_staggered = {int(f['max_staggered'])} is few sweeps; the cycle "
            "will often exit unconverged during crack growth.")
    if f["step_mode"] == "two_stage":
        disp = [b for b in f["bcs"] if b.get("group")
                and b.get("kind", "displacement") == "displacement"]
        umax = max((max(abs(float(b["vx"])), abs(float(b["vy"]))) for b in disp),
                   default=0.0)
        if umax and float(f["u_switch"]) >= umax:
            w.append("u_switch is at or beyond the full applied displacement, so "
                     "the fine stage never starts.")
        if umax and float(f["du_fine"]) > 0:
            n = (umax - float(f["u_switch"])) / float(f["du_fine"])
            if n > 3000:
                w.append(f"That schedule is about {n:,.0f} fine steps. "
                         "Raise du_fine unless you mean to run for days.")
    return w


# ---------------------------------------------------------------------------
# Load schedule: how many steps this config will actually take
#
# Mirrors main.cpp exactly rather than approximating it, because the two must
# agree or the progress bar lies. The rules there are:
#
#   u_ref  = the largest |prescribed displacement| over every constrained DOF
#            of every fixed node. Zero-valued constraints are supports, not
#            loading, so they never win.       (main.cpp "Loaded direction")
#   uniform    -> N_steps equal increments of load factor
#   two_stage  -> advance u by du_coarse until u reaches u_switch, then du_fine,
#                 landing EXACTLY on u_switch so the stage boundary is clean.
#
# Subdivisions (max_subdivs > 0) can only ADD steps on top of this, so treat the
# result as the planned minimum.
# ---------------------------------------------------------------------------
def u_ref_of(f: dict) -> float:
    """Magnitude of the full-load prescribed displacement, as the solver sees it."""
    best = 0.0
    for b in f.get("bcs", []):
        if not b.get("group") or b.get("kind", "displacement") != "displacement":
            continue
        # A component only counts if it is actually constrained AND nonzero.
        if int(b.get("fix_x", 0)):
            best = max(best, abs(float(b.get("vx", 0.0))))
        if int(b.get("fix_y", 0)):
            best = max(best, abs(float(b.get("vy", 0.0))))
    return best


def load_stages(f: dict):
    """The displacement stages as ``(name, start, end, du, steps)`` tuples.

    This is kept separate from :func:`planned_steps` because the live ETA needs
    the individual stage sizes and rates.  The end points and rounding mirror
    main.cpp's land-exactly-on-the-boundary behaviour.
    """
    try:
        mode = f.get("step_mode")
        u_ref = u_ref_of(f)
        if mode not in ("two_stage", "three_stage") or u_ref <= 0.0:
            return None
        du1 = float(f["du_coarse"])
        du2 = float(f["du_fine"])
        u1 = float(f["u_switch"])
        if min(du1, du2, u1) <= 0.0:
            return None
        if mode == "three_stage":
            u2 = float(f["u_switch2"])
            du3 = float(f["du_final"])
            if du3 <= 0.0 or u2 <= u1:
                return None
            raw = (("coarse", 0.0, min(u1, u_ref), du1),
                   ("fine", min(u1, u_ref), min(u2, u_ref), du2),
                   ("final", min(u2, u_ref), u_ref, du3))
        else:
            raw = (("coarse", 0.0, min(u1, u_ref), du1),
                   ("fine", min(u1, u_ref), u_ref, du2))
    except (KeyError, TypeError, ValueError):
        return None

    stages = []
    for name, start, end, du in raw:
        width = max(0.0, end - start)
        steps = max(0, math.ceil(width / du - 1e-12))
        stages.append((name, start, end, du, steps))
    return stages, u_ref


def remaining_stage_steps(stages, u_now: float):
    """Nominal steps still required in each stage from an actual displacement."""
    u_now = max(0.0, abs(float(u_now)))
    left = []
    for _name, start, end, du, _steps in stages:
        width = max(0.0, end - max(start, u_now))
        # CSV values have finite text precision.  The slightly wider epsilon
        # prevents a value printed at a boundary from inventing one extra step.
        left.append(max(0, math.ceil(width / du - 1e-7)))
    return left


def stage_for_displacement(stages, u_now: float) -> int:
    """Stage that produced an accepted endpoint displacement."""
    u_now = abs(float(u_now))
    for i, (_name, _start, end, du, _steps) in enumerate(stages):
        if u_now <= end + max(1e-12, du * 1e-4):
            return i
    return max(0, len(stages) - 1)


def planned_steps(f: dict):
    """(n_total, n_first, n_rest, u_ref) or None if it cannot be determined."""
    staged = load_stages(f)
    if staged:
        stages, u_ref = staged
        counts = [s[4] for s in stages]
        return sum(counts), counts[0], sum(counts[1:]), u_ref
    try:
        if f.get("step_mode") in ("two_stage", "three_stage"):
            return None
        u_ref = u_ref_of(f)
        du = float(f.get("du", 0.0) or 0.0)
        # Mirrors main.cpp: run.du wins when set, N_steps is the fallback.
        if du > 0.0 and u_ref > 0.0:
            n = math.ceil(u_ref / du - 1e-12)
            return n, n, 0, u_ref
        n = int(f.get("N_steps", 0))
        return (n, n, 0, u_ref) if n > 0 else None
    except (TypeError, ValueError):
        return None


def peak_index(F, min_drop: float = 0.10):
    """Index of peak |F|, but ONLY once the curve has clearly turned over.

    Returns None while the response is still rising. That distinction is the
    whole point: max(|F|) on a rising curve is just the last point, and
    reporting it as "the peak" would fire the coarse-stepping warning below on
    every run, from the very first step, until it meant nothing.

    A peak counts as real when everything after it has fallen at least
    `min_drop` below it.
    """
    if len(F) < 6:
        return None
    a = [abs(x) for x in F]
    i = max(range(len(a)), key=a.__getitem__)
    if a[i] <= 0.0:
        # An all-zero curve. main.cpp warns about this when no node carries a
        # nonzero prescribed displacement; without this guard the "peak" would
        # be index 0 and we would annotate a peak load of zero.
        return None
    if i >= len(a) - 3:
        return None                       # too close to the end to call
    # MIN of the tail, not max. The point immediately after the peak is always
    # close to it on a finely stepped curve, so testing max() here asks "did it
    # collapse in one step" -- which never happens, and the peak would never be
    # reported at all. What we want is "has it fallen `min_drop` below the peak
    # at some point since", which is the tail minimum.
    if min(a[i + 1:]) > (1.0 - min_drop) * a[i]:
        return None                       # hasn't really come down yet
    return i


def fmt_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds or seconds == float("inf"):
        return "?"
    s = int(seconds)
    if s < 90:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def check_group_dims(rows, mesh_groups: dict) -> list[str]:
    """Catch a load pointed at a group of the wrong dimension.

    The solver resolves point loads against 0D groups only and tractions
    against 1D only, so naming an edge in a force row throws at startup. The
    mesh already tells us each group's dimension, so say it here instead.
    """
    out = []
    if not mesh_groups:
        return out
    for b in rows:
        g = b.get("group")
        if not g or g not in mesh_groups:
            continue
        dim = mesh_groups[g]
        kind = b.get("kind", "displacement")
        if kind == "force" and dim != 0:
            out.append(f"'{g}' is a {'surface' if dim == 2 else 'edge'} group, "
                       "but a force needs a POINT (0D) group.")
        elif kind == "traction" and dim != 1:
            out.append(f"'{g}' is a {'surface' if dim == 2 else 'point'} group, "
                       "but a traction needs an EDGE (1D) group.")
        elif kind == "displacement" and dim == 2:
            out.append(f"'{g}' is a surface group; a displacement BC resolves "
                       "against edges (1D) or points (0D) only.")
    return out


def explain_exit_code(code: int) -> str:
    """Turn a bare Windows crash code into something actionable.

    These arrive as huge unsigned numbers with no output whatsoever, because
    the process died before it could print anything -- so the log shows only
    the number, and the number is the only evidence there is.
    """
    NT = {
        0xC0000135: (
            "STATUS_DLL_NOT_FOUND -- the exe never started; Windows could not "
            "load a DLL it\n"
            "  needs. It fails identically with --mesh-only, which is the "
            "fingerprint: nothing\n"
            "  in the config or the mesh can be at fault this early.\n"
            "  A MinGW/MSYS2 build needs libgcc_s_seh-1.dll and libstdc++-6.dll "
            "(from\n"
            "  C:\\msys64\\ucrt64\\bin) plus gmsh-4.15.dll beside the exe. "
            "Rebuild -- CMakeLists\n"
            "  now links the first two statically and copies them as a "
            "fallback:\n"
            "      cmake -S . -B build -G \"MinGW Makefiles\" "
            "-DCMAKE_BUILD_TYPE=Release\n"
            "      cmake --build build"),
        0xC0000005: (
            "ACCESS_VIOLATION -- the solver crashed. If it died immediately, "
            "suspect a\n"
            "  mesh/config mismatch; if mid-run, capture the last log lines."),
        0xC000007B: (
            "INVALID_IMAGE_FORMAT -- a 32-bit DLL next to a 64-bit exe (or the "
            "reverse).\n"
            "  Check that gmsh-4.15.dll is the Windows64 SDK build."),
        0xC00000FD: (
            "STACK_OVERFLOW -- usually gmsh's Blossom recombination on a very "
            "large mesh.\n"
            "  Regenerate with a coarser h, or --recombine-algo simple."),
    }
    hit = NT.get(code & 0xFFFFFFFF)
    if not hit:
        return ""
    return f"[gui] exit {code} = 0x{code & 0xFFFFFFFF:08X}: {hit}\n"


def csv_path_for(f: dict) -> str:
    if not f.get("base_name"):
        return ""
    return os.path.join(f.get("output_dir") or ".", f["base_name"],
                        f["base_name"] + "_force_disp.csv")


# ---------------------------------------------------------------------------
# Crack view: reading the solver's per-step VTK snapshots
#
# Output.cpp writes legacy ASCII VTK, so this parser only has to handle the
# exact layout that writeVTK() produces -- not VTK in general:
#
#     POINTS      <npoin> float      x y 0
#     CELLS       <nelem> <total>    <nn> n0 n1 ...
#     CELL_TYPES  <nelem>            5 | 9 | 23
#     POINT_DATA  <npoin>
#     SCALARS phi float 1
#     LOOKUP_TABLE default           <npoin values>
#     VECTORS displacement float
#     CELL_DATA ...                  per-element stresses
#
# THE GEOMETRY IS IDENTICAL IN EVERY SNAPSHOT -- same nodes, same connectivity,
# every step. So it is parsed ONCE and only the phi block is re-read per frame,
# which is what keeps a live view affordable on a large mesh.
# ---------------------------------------------------------------------------
def vtk_dir_for(f: dict) -> str:
    if not f.get("base_name"):
        return ""
    return os.path.join(f.get("output_dir") or ".", f["base_name"], "vtk")


_STEP_RE = re.compile(r"_step_(\d+)\.vtk$", re.IGNORECASE)

# main.cpp prints, per ACCEPTED step:
#   "  [time] step 137 = 0.84 s   (cumulative solve = 115.20 s)"
# That is the only place a real per-step cost is available, and it is already
# streaming into the log, so the ETA costs one regex per line and nothing else.
_TIME_RE = re.compile(r"\[time\]\s+step\s+(\d+)\s*=\s*([0-9.]+)\s*s")

# How many recent steps the rate is averaged over. Short on purpose: steps in
# the coarse stage are far cheaper than steps during crack growth, so a
# cumulative mean would keep predicting a run that finishes much too early.
RATE_WINDOW = 25


def newest_vtk(vdir: str):
    """(path, step) of the highest-numbered snapshot, or (None, -1).

    Sorted by the STEP NUMBER parsed from the name, not by mtime or asciibetics:
    setw(3) stops zero-padding at step 999, so "_step_1000.vtk" sorts before
    "_step_999.vtk" as text.
    """
    try:
        entries = os.listdir(vdir)
    except OSError:
        return None, -1
    best, best_step = None, -1
    for name in entries:
        m = _STEP_RE.search(name)
        if m and int(m.group(1)) > best_step:
            best_step = int(m.group(1))
            best = os.path.join(vdir, name)
    return best, best_step


def vtk_snapshots(vdir: str):
    """All VTK snapshots as sorted ``(step, path)`` pairs."""
    try:
        entries = os.listdir(vdir)
    except OSError:
        return []
    found = []
    for name in entries:
        m = _STEP_RE.search(name)
        if m:
            found.append((int(m.group(1)), os.path.join(vdir, name)))
    found.sort(key=lambda item: item[0])
    return found


def _header(lines, keyword):
    """Index of the line beginning with `keyword`, or -1."""
    for i, ln in enumerate(lines):
        if ln.startswith(keyword):
            return i
    return -1


def _floats(lines, a, b):
    return np.array(" ".join(lines[a:b]).split(), dtype=np.float64)


def read_vtk_geometry(path: str):
    """-> (Triangulation, npoin). Quads are split into two triangles.

    tricontourf needs a triangulation; the solver's elements are quads. Corner
    nodes only for Q8 -- VTK's Q8 ordering puts the four corners first, so
    idx[:4] is correct and the midside nodes are simply not used for contouring.
    """
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()

    i = _header(lines, "POINTS")
    if i < 0:
        raise ValueError("no POINTS block")
    npoin = int(lines[i].split()[1])
    pts = _floats(lines, i + 1, i + 1 + npoin).reshape(npoin, 3)

    j = _header(lines, "CELLS")
    if j < 0:
        raise ValueError("no CELLS block")
    nelem = int(lines[j].split()[1])
    tris = []
    for ln in lines[j + 1:j + 1 + nelem]:
        v = ln.split()
        nn = int(v[0])
        idx = [int(t) for t in v[1:1 + nn]]
        if nn == 3:
            tris.append(idx)
        elif nn in (4, 8):
            q = idx[:4]
            tris.append([q[0], q[1], q[2]])
            tris.append([q[0], q[2], q[3]])
    if not tris:
        raise ValueError("no supported cells")
    return mtri.Triangulation(pts[:, 0], pts[:, 1], np.array(tris)), npoin


def read_vtk_phi(path: str, npoin: int):
    """Just the phi field. Deliberately skips POINTS/CELLS parsing."""
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()
    i = _header(lines, "SCALARS phi")
    if i < 0:
        raise ValueError("no 'SCALARS phi' block")
    # LOOKUP_TABLE always follows SCALARS in what writeVTK emits.
    start = i + 2
    phi = _floats(lines, start, start + npoin)
    if phi.size != npoin:
        raise ValueError(f"phi has {phi.size} values, expected {npoin}")
    return phi


_CURVE_CACHE = {}


def reset_curve_cache(path: str) -> None:
    """Forget the incremental reader state before a new solve reuses a path."""
    if path:
        _CURVE_CACHE.pop(os.path.abspath(path), None)


def read_curve(path: str):
    """Read the force-displacement CSV -> (u, F, phi, converged, labels).

    Column names carry the loaded direction (applied_uy vs applied_ux), so they
    are found by prefix. The file is read while the solver writes it, so the
    last line is routinely half-written: the WHOLE row is parsed before
    anything is appended, otherwise the lists end up different lengths and
    matplotlib raises on mismatched x and y.
    """
    if not path or not os.path.isfile(path):
        return [], [], [], [], ("u", "F")
    key = os.path.abspath(path)
    try:
        size = os.path.getsize(path)
    except OSError:
        return [], [], [], [], ("u", "F")

    state = _CURVE_CACHE.get(key)
    # Truncation means a fresh solve has replaced the old file.  Normally the
    # launch path resets this explicitly; this also makes the reader robust if
    # the file is replaced externally.
    if state is None or size < state["offset"]:
        state = {
            "offset": 0, "tail": b"", "header": None, "cols": None,
            "u": [], "F": [], "phi": [], "conv": [], "labels": ("u", "F"),
        }
        _CURVE_CACHE[key] = state

    if size == state["offset"]:
        return (state["u"], state["F"], state["phi"], state["conv"],
                state["labels"])

    try:
        with open(path, "rb") as fh:
            fh.seek(state["offset"])
            chunk = fh.read()
            state["offset"] = fh.tell()
    except OSError:
        return (state["u"], state["F"], state["phi"], state["conv"],
                state["labels"])

    data = state["tail"] + chunk
    lines = data.splitlines(keepends=True)
    if lines and not lines[-1].endswith((b"\n", b"\r")):
        state["tail"] = lines.pop()
    else:
        state["tail"] = b""

    for raw in lines:
        try:
            r = next(csv.reader([raw.decode("utf-8", errors="replace")]))
        except (csv.Error, StopIteration):
            continue
        if state["header"] is None:
            header = [h.strip() for h in r]
            def col(prefix):
                return next((i for i, h in enumerate(header)
                             if h.startswith(prefix)), None)
            iu, iF, iphi, iconv = (col("applied_u"), col("reaction_F"),
                                   col("max_phi"), col("converged"))
            state["header"] = header
            state["cols"] = (iu, iF, iphi, iconv)
            if iu is not None and iF is not None:
                state["labels"] = (header[iu], header[iF])
            continue

        iu, iF, iphi, iconv = state["cols"]
        if iu is None or iF is None:
            continue
        try:
            vu, vF = float(r[iu]), float(r[iF])
            vphi = float(r[iphi]) if iphi is not None else 0.0
            vconv = int(float(r[iconv])) if iconv is not None else 1
        except (ValueError, IndexError):
            continue
        state["u"].append(vu)
        state["F"].append(vF)
        state["phi"].append(vphi)
        state["conv"].append(vconv)
    return (state["u"], state["F"], state["phi"], state["conv"],
            state["labels"])


_MESH_CACHE = {}          # path -> (mtime, data)


def _gmsh_initialize(gmsh):
    """Initialize Gmsh safely from either the Tk or a worker thread."""
    try:
        # Gmsh's default installs a SIGINT handler, which Python only permits
        # on the main thread.  Disabling that handler is exactly what its
        # `interruptible` option is for.
        gmsh.initialize(interruptible=threading.current_thread()
                        is threading.main_thread())
    except TypeError:                    # compatibility with older Gmsh wheels
        gmsh.initialize()


def list_specs():
    """Spec files available as geometry sources."""
    d = os.path.join(PROJECT_DIR, "mesh", "specs")
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d)
                  if f.endswith(".toml") and not f.endswith(".example.toml"))


def spec_param_defs(path: str) -> dict:
    """[params] normalised to {name: {value, min, max, step}}.

    Re-implemented here rather than imported from from_spec.py so the GUI does
    not pull gmsh in at import time -- gmsh is loaded lazily, only when a mesh
    is actually read.
    """
    try:
        spec = _toml_load(path)
    except Exception:
        return {}
    out = {}
    for name, entry in (spec.get("params") or {}).items():
        if isinstance(entry, dict) and "value" in entry:
            out[name] = dict(entry)
        elif isinstance(entry, (int, float)):
            out[name] = {"value": float(entry)}
    return out


def read_mesh_for_plot(path: str, max_elements: int = 4_000_000):
    """Read a .msh into plottable pieces.

    Returns {"polys": [...], "edges": {name: [((x0,y0),(x1,y1)), ...]},
             "points": {name: [(x, y), ...]}, "nnode", "nelem", "bbox"}.

    Elements come back as polygons rather than a node cloud so the mesh reads
    as a mesh. 1D groups are pulled as their LINE ELEMENTS, not merely as their
    nodes -- a set of nodes has no connectivity, so it can only be scattered as
    dots, and a boundary drawn as dots is unreadable at this element count.
    """
    empty = {"polys": [], "edges": {}, "points": {}, "nnode": 0, "nelem": 0,
             "bbox": (0, 0, 1, 1), "cent": None, "size": None}
    try:
        import gmsh
    except Exception:
        return empty
    if not path or not os.path.isfile(path):
        return empty

    # Re-parsing a 12k-element mesh takes ~0.15 s. That is nothing once, and
    # far too much on every keystroke in the BC table, so the result is cached
    # against the file's mtime -- edit the mesh on disk and it reloads, edit a
    # BC and it does not.
    mtime = os.path.getmtime(path)
    hit = _MESH_CACHE.get(path)
    if hit and hit[0] == mtime:
        return hit[1]

    try:
        _gmsh_initialize(gmsh)
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(path)

        tags, coords, _ = gmsh.model.mesh.getNodes()
        # Node lookup done entirely in numpy. Node tags are not guaranteed
        # contiguous (Plugin(Crack) duplicates some), so map through a sorted
        # array rather than assuming tag == index.
        #
        # There used to be a {tag: (x, y)} dict here. Building it is a Python
        # loop over EVERY node -- about 2 s on a 2M-node mesh, and pure
        # overhead, since every consumer below indexes in bulk.
        _t = np.asarray(tags, dtype=np.int64)
        tag_order = np.argsort(_t)
        tag_sorted = _t[tag_order]
        xy_all = np.asarray(coords, dtype=float).reshape(-1, 3)[:, :2][tag_order]

        def lookup(tag_array):
            """Coordinates for an array of node tags, and a validity mask."""
            a = np.asarray(tag_array, dtype=np.int64)
            idx = np.searchsorted(tag_sorted, a)
            idx = np.clip(idx, 0, len(tag_sorted) - 1)
            ok = tag_sorted[idx] == a
            return xy_all[idx], ok

        polys = []
        etypes, etags, enodes = gmsh.model.mesh.getElements(2, -1)
        nelem = sum(len(t) for t in etags)
        if nelem > max_elements:
            gmsh.finalize()
            out = dict(empty)
            out["nnode"], out["nelem"] = len(tags), nelem
            out["too_big"] = True
            return out
        # nnode per gmsh element type: 2=tri3, 3=quad4, 16=quad8 (corners only
        # for drawing -- the midside nodes would just add clutter).
        per = {2: 3, 3: 4, 16: 8}
        blocks = []
        for typ, els, nds in zip(etypes, etags, enodes):
            n = per.get(int(typ))
            if not n:
                continue
            corners = 4 if n == 8 else n
            # Whole element block at once. The per-element Python loop this
            # replaces was the single dominant cost of opening a mesh -- 11 s
            # on a 2M-element file, which is what made a big mesh feel like a
            # hang. searchsorted + fancy indexing does the same work in numpy.
            conn = np.asarray(nds, dtype=np.int64).reshape(-1, n)[:, :corners]
            pts = xy_all[np.searchsorted(tag_sorted, conn)]
            # Pad triangles to four vertices by repeating the last one. A quad
            # with two coincident corners draws as exactly that triangle, and
            # it keeps EVERY element the same width so the whole array stays
            # numpy-shaped. Without this a mesh of 340,195 quads + 358
            # triangles falls off the fast path -- precisely the mesh that
            # needs it.
            if corners == 3:
                pts = np.concatenate([pts, pts[:, 2:3, :]], axis=1)
            blocks.append(pts)
        if blocks:
            polys = (blocks[0] if len(blocks) == 1
                     else np.concatenate(blocks, axis=0))

        # polys is an (nelem, 4, 2) array. PolyCollection accepts that directly
        # and would otherwise have to convert a list of lists of tuples on
        # EVERY redraw -- which alone halved the Mesh tab's draw cost.
        cent = size = None
        if len(polys):
            # Centroid and mean edge length per element. Used to cull to the
            # visible region and to draw the size map -- both need a
            # per-element scalar, and computing it here (once, cached) is far
            # cheaper than per redraw.
            cent = polys.mean(axis=1)
            d = np.roll(polys, -1, axis=1) - polys
            L = np.hypot(d[:, :, 0], d[:, :, 1])
            # Ignore the zero-length edge on a padded triangle, or its size
            # would read 25% low against the quads around it.
            nz = L > 0.0
            size = L.sum(axis=1) / np.maximum(nz.sum(axis=1), 1)

        edges, points = {}, {}
        for dim, ptag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, ptag)
            if not name:
                continue
            if dim == 1:
                segs = []
                for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, ptag):
                    lt, lels, lnodes = gmsh.model.mesh.getElements(1, int(ent))
                    for typ, els, nds in zip(lt, lels, lnodes):
                        n = 2 if int(typ) == 1 else (3 if int(typ) == 8 else 0)
                        if not n:
                            continue
                        conn = np.asarray(nds, dtype=np.int64).reshape(-1, n)
                        pa, oka = lookup(conn[:, 0])
                        pb, okb = lookup(conn[:, 1])
                        good = oka & okb
                        segs.extend(zip(map(tuple, pa[good]),
                                        map(tuple, pb[good])))
                if segs:
                    edges[name] = segs
            elif dim == 0:
                ns, _ = gmsh.model.mesh.getNodesForPhysicalGroup(0, ptag)
                p0, ok0 = lookup(ns)
                pts = [tuple(v) for v in p0[ok0]]
                if pts:
                    points[name] = pts

        bbox = ((float(xy_all[:, 0].min()), float(xy_all[:, 1].min()),
                 float(xy_all[:, 0].max()), float(xy_all[:, 1].max()))
                if len(xy_all) else (0, 0, 1, 1))
        gmsh.finalize()
        data = {"polys": polys, "edges": edges, "points": points,
                "nnode": len(tags), "nelem": nelem, "bbox": bbox,
                "cent": cent, "size": size}
        _MESH_CACHE[path] = (mtime, data)
        return data
    except Exception:
        try:
            gmsh.finalize()
        except Exception:
            pass
        return empty


def groups_in_mesh(path: str):
    """Physical group names a mesh offers, as {name: dim}. Empty if gmsh is
    unavailable -- the GUI degrades to free-text BC names rather than failing."""
    try:
        import gmsh
    except Exception:
        return {}
    if not path or not os.path.isfile(path):
        return {}
    try:
        _gmsh_initialize(gmsh)
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(path)
        out = {gmsh.model.getPhysicalName(d, t): d
               for d, t in gmsh.model.getPhysicalGroups()}
        gmsh.finalize()
        return out
    except Exception:
        try:
            gmsh.finalize()
        except Exception:
            pass
        return {}


# ===========================================================================
#  Widgets
# ===========================================================================
class Tip:
    """Hover help for one widget.

    A tooltip rather than a permanent caption: the panel is tight vertically,
    and six tables of seven columns cannot each carry a line of prose. The text
    belongs where the mistake happens -- on the box itself, not only the header.
    """

    def __init__(self, widget, text, delay=400):
        self.widget, self.text, self.delay = widget, text, delay
        self.job = None
        self.win = None
        widget.bind("<Enter>", self._enter, add="+")
        widget.bind("<Leave>", self._leave, add="+")
        widget.bind("<ButtonPress>", self._leave, add="+")
        # Without this, deleting a row while its tooltip is pending fires the
        # timer against a destroyed widget. dup/x destroy widgets constantly.
        widget.bind("<Destroy>", self._leave, add="+")

    def _enter(self, _e=None):
        self._cancel()
        self.job = self.widget.after(self.delay, self._show)

    def _leave(self, _e=None):
        self._cancel()
        self._hide()

    def _cancel(self):
        if self.job is not None:
            try:
                self.widget.after_cancel(self.job)
            except Exception:
                pass
            self.job = None

    def _show(self):
        if self.win is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + px(12)
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + px(4)
        except Exception:
            return
        self.win = tk.Toplevel(self.widget)
        self.win.wm_overrideredirect(True)
        self.win.wm_geometry(f"+{x}+{y}")
        tk.Label(self.win, text=self.text, justify="left",
                 wraplength=px(380),
                 background=T["panel"], foreground=T["fg"], relief="solid",
                 borderwidth=1, padx=px(7), pady=px(5),
                 font=("TkDefaultFont", 9)).pack()

    def _hide(self):
        if self.win is not None:
            try:
                self.win.destroy()
            except Exception:
                pass
            self.win = None


def compile_template(tmpl: str):
    """One template drives BOTH directions of a repeatable CLI argument.

        "{x0},{y0},{x1},{y1}:{h}:{near}:{far}"
            -> format a table row into the argument
            -> and parse an argument back into a row

    Deriving them from a single string is the point: a hand-written
    format/parse pair drifts, and a dual-view editor whose views disagree is
    worse than either view alone.
    """
    keys = re.findall(r"\{(\w+)\}", tmpl)
    rx = re.compile("^" + re.sub(r"\\\{(\w+)\\\}", r"(?P<\1>[^,:=]+)",
                                 re.escape(tmpl)) + "$")

    def fmt(row):
        return tmpl.format(**row)

    def parse(line):
        m = rx.match(line.strip())
        return {k: v.strip() for k, v in m.groupdict().items()} if m else None

    return keys, fmt, parse


class SpecTable(ttk.Frame):
    """Rows of typed fields for a repeatable mesher argument, plus a text view.

    A table reads better; a raw line is better to REUSE -- duplicate it, paste
    it from notes, send it to someone. So both: `dup` clones a row, `text`
    swaps the table for its lines and back, `copy` puts them on the clipboard.

    columns: (key, heading, width, kind, help); kind is "num", "text", or a
             tuple of choices for a combobox.
    """

    def __init__(self, parent, columns, template, on_change=None, hint="",
                 example=None, legacy=()):
        super().__init__(parent)
        self.columns = columns
        self.template = template
        self.example = example or {}
        self.keys, self.fmt, self._parse_main = compile_template(template)
        # Older forms of the same argument, still accepted when reading. A
        # saved recipe outlives the panel that wrote it, and a stricter
        # template would drop those lines without a word.
        self._parse_legacy = [compile_template(t)[2] for t in legacy]
        self.on_change = on_change
        self.rows: list[dict] = []
        self.text_mode = False

        if hint:
            ttk.Label(self, text=hint, foreground=T["muted"],
                      wraplength=px(470)).pack(anchor="w", pady=(0, 2))

        self.grid_view = ttk.Frame(self)
        self.grid_view.pack(fill="x")
        # Headings live in the SAME grid as the cells. In two frames each grid
        # sizes its own columns, so a heading only lines up with its box by
        # luck -- and never after a column changes width.
        self.body = ttk.Frame(self.grid_view)
        self.body.pack(fill="x")
        for c, (key, heading, width, kind, help_) in enumerate(columns):
            lbl = ttk.Label(self.body, text=heading, foreground=T["muted"],
                            anchor="w")
            # 4 px = the cell's 1 px border + 3 px padding, so the heading sits
            # over the first character of the value, not the box edge.
            lbl.grid(row=0, column=c, padx=(4, 0), sticky="w")
            Tip(lbl, help_)

        self.fmt_hint = tk.Label(self, justify="left", anchor="w",
                                 background=T["panel"], foreground=T["muted"],
                                 font=("Consolas", 8), padx=px(6),
                                 pady=px(4))
        self.text_view = tk.Text(self, height=3, wrap="none",
                                 font=("Consolas", 9), background=T["cell"],
                                 foreground=T["fg"], insertbackground=T["fg"],
                                 relief="flat", borderwidth=0)

        # Kept as an attribute so both views can pack BEFORE it -- pack()
        # appends, so a view restored without an anchor reappears under the
        # buttons instead of above them.
        self.bar = bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(2, 0))
        b_add = ttk.Button(bar, text="+ add", width=6,
                           command=lambda: self.add_row())
        b_add.pack(side="left")
        Tip(b_add, "Add an empty row. Rows with any field left blank are "
                   "ignored when the command is built.")
        self.btn_text = ttk.Button(bar, text="text", width=5,
                                   command=self.toggle_text)
        self.btn_text.pack(side="left", padx=3)
        Tip(self.btn_text, "Swap this table for its raw lines -- paste lines "
                           "in, or copy them out, then press 'table' to go "
                           "back. Same data, two views.")
        b_copy = ttk.Button(bar, text="copy", width=5, command=self.copy)
        b_copy.pack(side="left")
        Tip(b_copy, "Copy every complete row to the clipboard, one line each.")
        self.status = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.status,
                  foreground=T["muted"]).pack(side="left", padx=6)

    def add_row(self, values=None, after=None):
        values = values or {}
        rec: dict = {"w": {}, "v": {}}
        for key, heading, width, kind, help_ in self.columns:
            # A choice column defaults to its FIRST option, never blank. A
            # blank field makes the whole row incomplete, so an unset dropdown
            # would drop the row from the command without saying anything --
            # you would add a crack, generate, and simply not get one.
            fallback = kind[0] if isinstance(kind, tuple) else ""
            v = tk.StringVar(value=str(values.get(key, fallback)))
            if isinstance(kind, tuple):
                w = ttk.Combobox(self.body, textvariable=v, values=list(kind),
                                 width=width - 2, state="readonly",
                                 style="Cell.TCombobox")
            else:
                w = ttk.Entry(self.body, textvariable=v, width=width,
                              style="Cell.TEntry")
            Tip(w, help_)
            # Enter walks to the next box. Filling a row is seven numbers in a
            # row; reaching for the mouse between each one is the slow part.
            for seq in ("<Return>", "<KP_Enter>"):
                w.bind(seq, lambda _e, rec=rec, key=key: self._focus_next(rec, key))
            if self.on_change:
                v.trace_add("write", lambda *_: self._changed())
            rec["w"][key] = w
            rec["v"][key] = v
        rec["dup"] = ttk.Button(self.body, text="dup", width=4,
                                command=lambda rec=rec: self.duplicate(rec))
        Tip(rec["dup"], "Clone this row directly below, then change the one "
                        "value that differs.")
        rec["del"] = ttk.Button(self.body, text="x", width=2,
                                command=lambda rec=rec: self.remove(rec))
        Tip(rec["del"], "Delete this row.")
        at = len(self.rows) if after is None else self.rows.index(after) + 1
        self.rows.insert(at, rec)
        self._regrid()
        self._changed()
        return rec

    def parse(self, line):
        """Current format first, then any legacy form. Extra fields a legacy
        line carries are dropped -- they are exactly the ones this table no
        longer offers."""
        for p in [self._parse_main] + self._parse_legacy:
            row = p(line)
            if row:
                return {k: row[k] for k in self.keys if k in row}
        return None

    def _focus_next(self, rec, key):
        """Enter -> the next box: across the row, then down to the next row.

        At the end of the LAST row it adds a new one, but only if the current
        row is complete -- otherwise holding Enter would fill the table with
        blank rows.
        """
        keys = [c[0] for c in self.columns]
        i = keys.index(key)
        if i + 1 < len(keys):
            self._focus(rec["w"][keys[i + 1]])
            return "break"
        if rec not in self.rows:
            return "break"
        r = self.rows.index(rec)
        if r + 1 < len(self.rows):
            nxt = self.rows[r + 1]
        else:
            vals = {k: rec["v"][k].get().strip() for k in rec["v"]}
            if not all(vals.values()):
                return "break"
            nxt = self.add_row()
        self._focus(nxt["w"][keys[0]])
        return "break"

    @staticmethod
    def _focus(widget):
        widget.focus_set()
        try:
            # Select what is there so the next keystroke replaces it, which is
            # what you want when stepping through an existing row.
            widget.select_range(0, "end")
            widget.icursor("end")
        except Exception:
            pass                      # readonly combobox has no selection

    def duplicate(self, rec):
        self.add_row({k: rec["v"][k].get() for k in rec["v"]}, after=rec)

    def remove(self, rec):
        for w in list(rec["w"].values()) + [rec["dup"], rec["del"]]:
            w.destroy()
        self.rows.remove(rec)
        self._regrid()
        self._changed()

    def _regrid(self):
        for r, rec in enumerate(self.rows):
            for c, (key, *_rest) in enumerate(self.columns):
                rec["w"][key].grid(row=r + 1, column=c, padx=0, pady=1,
                                   sticky="ew")
            rec["dup"].grid(row=r + 1, column=len(self.columns), padx=(8, 1))
            rec["del"].grid(row=r + 1, column=len(self.columns) + 1, padx=1)

    def _changed(self):
        n = len(self.rows)
        self.status.set("" if n == 0 else f"{n} row{'s' if n != 1 else ''}")
        if self.on_change:
            self.on_change()

    def toggle_text(self):
        if not self.text_mode:
            self.text_view.delete("1.0", "end")
            self.text_view.insert("end", "\n".join(self.lines()))
            self.fmt_hint.configure(text=self._format_help())
            self.grid_view.pack_forget()
            self.fmt_hint.pack(fill="x", before=self.bar, pady=(0, 2))
            self.text_view.pack(fill="x", before=self.bar)
            self.text_view.configure(height=max(3, len(self.rows) + 1))
            self.btn_text.configure(text="table")
            self.text_mode = True
            self.status.set("press 'table' to apply")
        else:
            raw = [l for l in self.text_view.get("1.0", "end").splitlines()
                   if l.strip()]
            good, bad = [], []
            for line in raw:
                row = self.parse(line)
                (good if row else bad).append(row if row else line)
            for rec in list(self.rows):
                self.remove(rec)
            for row in good:
                self.add_row(row)
            self.text_view.pack_forget()
            self.fmt_hint.pack_forget()
            self.grid_view.pack(fill="x", before=self.bar)
            self.btn_text.configure(text="text")
            self.text_mode = False
            if bad:
                # Reported, never dropped in silence -- losing a pasted line
                # without saying so is the worst possible behaviour here.
                self.status.set(f"{len(bad)} line(s) not understood: "
                                f"{bad[0][:32]}")

    def _format_help(self):
        # Headings, not internal keys: the box column is keyed "t" but reads
        # "thick", and the skeleton must match what the header row says.
        heads = {k: h.replace(" ", "_") for k, h, _w, _k, _t in self.columns}
        skeleton = re.sub(r"\{(\w+)\}",
                          lambda m: heads.get(m.group(1), m.group(1)),
                          self.template)
        src = None
        if self.rows:
            vals = {k: self.rows[0]["v"][k].get().strip()
                    for k in self.rows[0]["v"]}
            if all(vals.values()):
                src = vals
        src = src or self.example
        ex = self.fmt(src) if src and all(k in src for k in self.keys) else ""
        out = [f"one entry per line:   {skeleton}"]
        if ex:
            out.append(f"for example:          {ex}")
        out.append("blank lines ignored; a line that does not match is "
                   "reported, never dropped")
        return "\n".join(out)

    def lines(self):
        out = []
        for rec in self.rows:
            vals = {k: rec["v"][k].get().strip() for k in rec["v"]}
            if all(vals.values()):
                out.append(self.fmt(vals))
        return out

    def copy(self):
        self.clipboard_clear()
        self.clipboard_append("\n".join(self.lines()))
        self.status.set(f"{len(self.lines())} line(s) copied")

    def set_lines(self, lines):
        """Replace every row from a list of raw argument strings."""
        for rec in list(self.rows):
            self.remove(rec)
        for line in lines or []:
            row = self.parse(line)
            if row:
                self.add_row(row)


class ScrollFrame(ttk.Frame):
    """A frame scrollable in both directions.

    Horizontal matters here: the settings panel can be narrowed with the
    divider, and a table that no longer fits should be reachable by sliding
    rather than silently clipped off the edge -- which is exactly how the
    y-value column went missing.
    """

    def __init__(self, parent, width=0):
        super().__init__(parent)
        # Device pixels. The default is in design units so it still scales if
        # a caller ever omits it -- the one caller passes px() itself.
        width = width or px(500)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                width=width, background=T["bg"])
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Grid rather than pack, so the horizontal bar sits under the canvas
        # only -- not under the vertical bar as well.
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._req_w, self._req_at = width, 0.0
        self._live_resize = False
        self._pending_canvas_w = width
        self._applied_inner_w = 0
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner,
                                              anchor="nw")
        self.inner.bind("<Configure>", self._inner_configure)
        self.canvas.bind("<Configure>", self._canvas_configure)

        # Gate the wheel on the pointer being over the panel. Binding globally
        # would scroll the settings while you are zooming the mesh.
        self.canvas.bind("<Enter>",
                         lambda e: self.canvas.bind_all("<MouseWheel>", self._wheel))
        self.canvas.bind("<Leave>",
                         lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _inner_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._req_w, self._req_at = self.inner.winfo_reqwidth(), time.time()

    def _canvas_configure(self, event):
        # Stretch the inner frame to fill the canvas when there is room, but
        # never SHRINK it below what its children need -- forcing it to the
        # canvas width is what stops horizontal scrolling from ever engaging.
        #
        # winfo_reqwidth() forces a geometry recalculation of the entire form
        # below it, and this fires on every pixel of a sash drag. The answer
        # cannot change while dragging (the contents are not changing), so
        # cache it briefly. The TTL keeps it correct when the panel really is
        # rebuilt -- e.g. switching geometry source swaps the whole form.
        self._pending_canvas_w = event.width
        # During a sash drag, resizing this embedded window reflows every
        # section, table and validation label in the long form.  The canvas
        # itself already follows the pointer, so defer only the expensive
        # content reflow until release.
        if self._live_resize:
            return
        self._apply_canvas_width()

    def _apply_canvas_width(self):
        now = time.time()
        if now - getattr(self, "_req_at", 0.0) > 0.15:
            self._req_w, self._req_at = self.inner.winfo_reqwidth(), now
        target = max(self._pending_canvas_w, self._req_w)
        if target == self._applied_inner_w:
            return
        self._applied_inner_w = target
        self.canvas.itemconfig(self._win, width=target)

    def set_live_resize(self, active):
        self._live_resize = bool(active)
        if not active:
            self._apply_canvas_width()

    def _wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")


class LoadTable(ttk.Frame):
    """Restraints and loads in one table, one row per condition.

    Three kinds, because the solver reads three different config sections and
    each resolves against a different group dimension:

        displacement  ->  [[bcs]]        edge (1D) or point (0D)   mm
        force         ->  [[point_load]] point (0D) only           N
        traction      ->  [[neumann]]    edge (1D) only            N/mm

    Everything is laid out in ONE grid rather than a frame per row. Packing
    widgets of different widths row by row never lines up -- a shared grid is
    what makes the columns align.
    """

    KINDS = ["displacement", "force", "traction"]
    # Width in characters per column. These drive BOTH the header labels and
    # the grid minsize, so the header and the rows cannot drift apart.
    # "displacement" is 12 characters -- the type column must fit it or the
    # combobox shows "displacemen" and looks broken.
    COLS = [("type", 14), ("group", 16), ("fix x", 5), ("fix y", 5),
            ("x value", 9), ("y value", 9), ("", 3)]

    def __init__(self, parent, group_names_getter, on_change=None):
        super().__init__(parent)
        self._rows = []
        self._get_names = group_names_getter
        self._on_change = on_change
        self._loading = False

        self.body = ttk.Frame(self)
        self.body.pack(fill="x")
        # Column width in CHARACTERS -> pixels. Measured from the font that
        # will actually draw them rather than assumed at 7 px: the assumption
        # was only ever right at 96 dpi in the default face, so on a scaled
        # display the header row and the boxes under it drifted apart.
        try:
            ch = tkfont.nametofont("TkDefaultFont", root=self).measure("0")
        except (tk.TclError, KeyError):
            ch = 0
        ch = ch or px(7)
        for c, (text, w) in enumerate(self.COLS):
            ttk.Label(self.body, text=text, width=w, anchor="w",
                      foreground=T["muted"]).grid(row=0, column=c, padx=1,
                                                  sticky="w")
            self.body.grid_columnconfigure(c, minsize=w * ch)

        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=2)
        ttk.Button(bar, text="+ displacement", width=14,
                   command=lambda: self.add_row()).pack(side="left")
        ttk.Button(bar, text="+ force", width=9,
                   command=lambda: self.add_row({"kind": "force"})).pack(
                       side="left", padx=3)
        ttk.Button(bar, text="+ traction", width=11,
                   command=lambda: self.add_row({"kind": "traction"})).pack(
                       side="left")

    # -- rows -----------------------------------------------------------
    def add_row(self, bc=None):
        bc = {**{"kind": "displacement", "group": "", "fix_x": 1, "fix_y": 1,
                 "vx": 0.0, "vy": 0.0}, **(bc or {})}
        kind = tk.StringVar(value=bc["kind"])
        grp = tk.StringVar(value=bc["group"])
        fx = tk.IntVar(value=int(bc["fix_x"]))
        fy = tk.IntVar(value=int(bc["fix_y"]))
        vx = tk.StringVar(value=str(bc["vx"]))
        vy = tk.StringVar(value=str(bc["vy"]))

        w = {}
        w["kind"] = ttk.Combobox(self.body, textvariable=kind, values=self.KINDS,
                                 width=self.COLS[0][1] - 1, state="readonly")
        w["group"] = ttk.Combobox(self.body, textvariable=grp, width=self.COLS[1][1] - 2,
                                  values=sorted(self._get_names()))
        w["fx"] = ttk.Checkbutton(self.body, variable=fx)
        w["fy"] = ttk.Checkbutton(self.body, variable=fy)
        w["vx"] = ttk.Entry(self.body, textvariable=vx, width=self.COLS[4][1])
        w["vy"] = ttk.Entry(self.body, textvariable=vy, width=self.COLS[5][1])

        rec = {"kind": kind, "group": grp, "fix_x": fx, "fix_y": fy,
               "vx": vx, "vy": vy, "w": w}
        w["del"] = ttk.Button(self.body, text="x", width=2,
                              command=lambda r=rec: self.remove(r))
        self._rows.append(rec)

        w["kind"].bind("<<ComboboxSelected>>", lambda e, r=rec: self._kind_changed(r))
        for var in (kind, grp, fx, fy, vx, vy):
            var.trace_add("write", lambda *_: self._changed())
        self._kind_changed(rec)
        self._relayout()
        self._changed()

    def _kind_changed(self, rec):
        """Only a displacement BC has DOFs to fix; the other two carry a
        vector, so the checkboxes would be meaningless (and misleading) there."""
        disp = rec["kind"].get() == "displacement"
        for key in ("fx", "fy"):
            rec["w"][key].configure(state="normal" if disp else "disabled")

    def _relayout(self):
        for i, rec in enumerate(self._rows, start=1):
            for c, key in enumerate(("kind", "group", "fx", "fy", "vx", "vy",
                                     "del")):
                rec["w"][key].grid(row=i, column=c, padx=1, pady=1, sticky="w")

    def remove(self, rec):
        for widget in rec["w"].values():
            widget.destroy()
        self._rows.remove(rec)
        self._relayout()
        self._changed()

    def clear(self):
        for r in list(self._rows):
            self.remove(r)

    def _changed(self):
        if not self._loading and self._on_change:
            self._on_change()

    def refresh_names(self):
        names = sorted(self._get_names())
        for r in self._rows:
            r["w"]["group"].configure(values=names)

    # -- data -----------------------------------------------------------
    def get(self):
        out = []
        for r in self._rows:
            try:
                vx, vy = float(r["vx"].get() or 0), float(r["vy"].get() or 0)
            except ValueError:
                vx = vy = 0.0
            out.append({"kind": r["kind"].get(), "group": r["group"].get().strip(),
                        "fix_x": r["fix_x"].get(), "fix_y": r["fix_y"].get(),
                        "vx": vx, "vy": vy})
        return out

    def set(self, rows):
        """Bulk load: fire the change callback once, not once per row."""
        self._loading = True
        try:
            self.clear()
            for b in rows:
                self.add_row(b)
        finally:
            self._loading = False
        self._changed()


class App:
    def __init__(self, root: tk.Tk, initial_config: str = ""):
        self.root = root
        root.title("Phase-field solver")
        # The design size, scaled -- then clamped, because 1460x860 at 200% is
        # 2920x1720 and would open bigger than the screen it has to fit on.
        win_w = min(px(1460), root.winfo_screenwidth() - px(40))
        win_h = min(px(860), root.winfo_screenheight() - px(80))
        root.geometry(f"{win_w}x{win_h}")
        self.win_w = win_w              # the settings pane is sized against it
        apply_theme(root)
        self.proc = None
        self.proc_kind = ""          # "solve" | "check" | "mesh"
        self._on_success = None
        self.log_q: queue.Queue = queue.Queue()
        self.io_q: queue.Queue = queue.Queue()
        self.start_time = 0.0
        self.csv_path = ""
        self.last_rows = -1
        self.mesh_groups: dict = {}
        self.vars: dict = {}
        self.mesh_bbox = None
        self._mesh_dirty = True
        self._redraw_job = None
        self._zoom_job = None
        self._last_plot = 0.0
        # Progress / ETA state
        self._plan_total = 0        # planned steps for the running job
        self._step_times: list = []  # recent per-step wall times, seconds
        self._stage_times: list[list[float]] = [[], [], []]
        self._last_stage_rates = [0.0, 0.0, 0.0]
        self._run_form = None       # schedule used by the currently running job
        self._last_step = 0
        self._last_rate = 0.0       # s/step carried over from the previous run
        self._peak_reported = False
        self._laid_out: dict = {}   # id(figure) -> layout key, see _layout()
        self._dragging = False      # a pane sash is being dragged
        self._dragging_pane = None
        self._resize_pending: set = set()
        self._rows: dict = {}           # key -> [widgets], for show/hide
        self._sections: dict = {}       # title -> collapsible section state
        self._field_widgets: dict = {}  # config key -> entry/combobox
        self._field_feedback: dict = {} # config key -> (label, normal text)
        self._field_sections: dict = {} # config key -> owning section title
        self._validation_job = None
        self._charw = 0                 # width of "0" in the form font
        self._closing = False
        self._io_lock = threading.Lock()  # gmsh and large VTK reads, one at a time
        self._mesh_request = 0
        self._group_request = 0
        self._crack_request = 0
        self._crack_loading_path = ""
        self._curve_artists = None
        self._curve_annotations = []
        self._log_history = []
        self._log_history_chars = 0
        self.gui_state = self._load_gui_state()

        self._build()
        self._bind_shortcuts()
        self.set_form(dict(DEFAULTS))
        if initial_config:
            self.import_config(os.path.join(PROJECT_DIR, initial_config))
        self.root.after(POLL_MS, self._tick)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- construction ---------------------------------------------------
    def _build(self):
        # A split pane rather than two packed frames: the sash is draggable, so
        # the settings panel can be widened when a table needs it and narrowed
        # when the plot does.
        #
        # Classic tk.PanedWindow rather than ttk's, because it can be styled
        # (background = the sash colour) and takes minsize/stretch per pane.
        #
        # The horizontal split uses a preview sash: this panel can contain
        # hundreds of native controls and Windows repaints all of them when its
        # viewport changes.  Committing that once on release is much smoother.
        # The vertical split remains live because plots/log are cheap once the
        # matplotlib resize guard below defers figure rendering.
        outer = tk.PanedWindow(self.root, orient="horizontal",
                               # Repainting the long settings form at every
                               # mouse pixel remains expensive even when its
                               # geometry is frozen.  A preview sash moves at
                               # pointer speed and commits the layout once.
                               opaqueresize=False, background=T["grid"],
                               borderwidth=0, sashwidth=px(6), sashpad=0,
                               sashrelief="flat")
        outer.pack(fill="both", expand=True)

        panel_host = ScrollFrame(outer, width=px(500))
        # stretch="never": extra space from resizing the WINDOW goes to the
        # plot, not the form -- the form does not get more useful when wider.
        outer.add(panel_host, width=px(500), minsize=px(260),
                  stretch="never")
        self.panel = panel_host.inner
        self.panel_host = panel_host

        left = ttk.Frame(outer)          # results area: buttons, plot, log
        outer.add(left, minsize=px(420), stretch="always")
        self.paned = outer

        # Compact gui2-style results header.  Keep each line to one purpose:
        # solver, actions, state/time, progress, then the live numerical readout.
        bar = ttk.Frame(left, padding=px(8))
        bar.pack(fill="x")
        ttk.Label(bar, text="Solver").pack(side="left")
        self.exe_var = tk.StringVar(value=find_exe())
        ttk.Entry(bar, textvariable=self.exe_var, width=48).pack(
            side="left", padx=4)
        ttk.Button(bar, text="...", width=3, command=self._pick_exe).pack(side="left")

        bar2 = ttk.Frame(left, padding=(px(8), 0))
        bar2.pack(fill="x")
        self.btn_check = ttk.Button(bar2, text="Check mesh", command=self.check_mesh)
        self.btn_check.pack(side="left")
        self.btn_run = ttk.Button(bar2, text="Run", command=self.run,
                                  style="Primary.TButton")
        self.btn_run.pack(side="left", padx=px(6))
        self.btn_stop = ttk.Button(bar2, text="Stop", command=self.stop,
                                   state="disabled", style="Danger.TButton")
        self.btn_stop.pack(side="left")
        ttk.Button(bar2, text="Output folder",
                   command=self._open_output).pack(side="right")

        sbar = ttk.Frame(left, padding=(px(8), px(6), px(8), 0))
        sbar.pack(fill="x")
        sbar.grid_columnconfigure(0, weight=1)

        self.status = tk.StringVar(value="idle")
        self.state = self.status
        self.state_lbl = ttk.Label(sbar, textvariable=self.status, anchor="w",
                                   font=("TkDefaultFont", 10, "bold"),
                                   foreground=T["muted"])
        self.state_lbl.grid(row=0, column=0, sticky="w")
        self.elapsed_txt = tk.StringVar(value="")
        self.elapsed = self.elapsed_txt
        ttk.Label(sbar, textvariable=self.elapsed_txt, foreground=T["muted"],
                  anchor="e").grid(row=0, column=1, sticky="e")

        self.progress = ttk.Progressbar(sbar, mode="determinate", maximum=1000)
        self.progress.grid(row=1, column=0, columnspan=2, sticky="ew",
                           pady=(px(5), 0))
        self.progress_txt = tk.StringVar(value="")
        ttk.Label(sbar, textvariable=self.progress_txt, anchor="w",
                  foreground=T["muted"]).grid(
                      row=2, column=0, columnspan=2, sticky="ew",
                      pady=(px(2), 0))
        self.curve_status = tk.StringVar(value="")
        self.readout = self.curve_status
        ttk.Label(sbar, textvariable=self.curve_status, foreground=T["fg"],
                  font=("Consolas", 9), anchor="w").grid(
                      row=3, column=0, columnspan=2, sticky="ew",
                      pady=(px(2), 0))

        # Second splitter: plots above, log below.
        vpane = tk.PanedWindow(left, orient="vertical", opaqueresize=True,
                               background=T["grid"], borderwidth=0,
                               sashwidth=px(6), sashpad=0, sashrelief="flat")
        vpane.pack(fill="both", expand=True, padx=px(8), pady=(0, px(4)))
        self.vpaned = vpane

        nb = ttk.Notebook(vpane)
        vpane.add(nb, minsize=px(230), stretch="always")
        self.nb = nb

        curve_tab = ttk.Frame(nb)
        nb.add(curve_tab, text="Force - displacement")
        self.curve_tab = curve_tab
        fbar = ttk.Frame(curve_tab, padding=(0, px(4)))
        fbar.pack(fill="x")
        ttk.Button(fbar, text="Fit", width=5, style="Tool.TButton",
                   command=self.fit_curve).pack(side="left")
        self.follow_curve = tk.BooleanVar(value=True)
        ttk.Checkbutton(fbar, text="follow the data",
                        variable=self.follow_curve,
                        command=self._follow_curve_changed).pack(
                            side="left", padx=px(8))
        self.curve_hint = tk.StringVar(value="")
        ttk.Label(fbar, textvariable=self.curve_hint,
                  foreground=T["muted"]).pack(side="left", padx=px(4))
        self.fig = Figure(figsize=(7, 3.4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("applied displacement")
        self.ax.set_ylabel("reaction force")
        style_axes(self.fig, self.ax)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=curve_tab)
        # Reserve the fixed toolbar first.  Packing the expanding canvas first
        # lets it claim the whole parcel after a sash move, leaving a toolbar
        # that exists but is not mapped.
        self.curve_toolbar = add_toolbar(self.canvas, curve_tab)
        self.curve_toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.canvas.mpl_connect("button_release_event", self._curve_interacted)

        mesh_tab = ttk.Frame(nb)
        nb.add(mesh_tab, text="Mesh")
        self.mesh_tab = mesh_tab
        # Redrawing a tab nobody is looking at is wasted work, so edits made
        # while the curve is showing only set a flag; the redraw happens when
        # you switch to the Mesh tab.
        nb.bind("<<NotebookTabChanged>>", self._tab_changed)
        mbar = ttk.Frame(mesh_tab, padding=(0, px(4)))
        mbar.pack(fill="x")
        ttk.Button(mbar, text="Draw mesh", command=self.draw_mesh,
                   style="Tool.TButton").pack(side="left")
        ttk.Button(mbar, text="Fit", width=5, style="Tool.TButton",
                   command=self.fit_mesh).pack(side="left", padx=4)
        self.show_bc_only = tk.BooleanVar(value=True)
        ttk.Checkbutton(mbar, text="highlight BC groups only",
                        variable=self.show_bc_only,
                        command=self.draw_mesh).pack(side="left", padx=8)
        ttk.Label(mbar, text="view").pack(side="left", padx=(8, 2))
        self.mesh_view = tk.StringVar(value=MESH_VIEWS[0])
        mv = ttk.Combobox(mbar, textvariable=self.mesh_view, values=MESH_VIEWS,
                          width=11, state="readonly")
        mv.pack(side="left")
        mv.bind("<<ComboboxSelected>>",
                lambda _e: self.draw_mesh(preserve_view=True))
        self.mesh_stats = tk.StringVar(value="")
        ttk.Label(mbar, textvariable=self.mesh_stats,
                  foreground=T["muted"]).pack(side="left", padx=8)
        self.mfig = Figure(figsize=(7, 3.4), dpi=100)
        self.max_ = self.mfig.add_subplot(111)
        style_axes(self.mfig, self.max_)
        self.mfig.tight_layout()
        self.mcanvas = FigureCanvasTkAgg(self.mfig, master=mesh_tab)
        self.mesh_toolbar = add_toolbar(self.mcanvas, mesh_tab)
        self.mesh_toolbar.pack(side="bottom", fill="x")
        self.mcanvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        # Wheel-zoom about the cursor, which is what a mesh viewer should do.
        # The toolbar still offers rubber-band zoom and pan for fine control.
        self.mcanvas.mpl_connect("scroll_event", self._mesh_wheel_zoom)
        self.mesh_bbox = None

        # ---- Crack tab: phi from the solver's VTK snapshots ----------------
        crack_tab = ttk.Frame(nb)
        nb.add(crack_tab, text="Crack (phi)")
        self.crack_tab = crack_tab
        self._crack_step = -1
        self._crack_path = ""
        self._crack_snapshots = []
        cbar_ = ttk.Frame(crack_tab, padding=(0, px(4)))
        cbar_.pack(fill="x")

        # Compact snapshot history.  The states are saved VTK files rather
        # than every solver step, because vtk_every intentionally leaves gaps.
        nav = ttk.Frame(cbar_)
        self.crack_nav = nav
        ttk.Label(nav, text="State", font=("TkDefaultFont", 9, "bold")).pack(
            side="left", padx=(px(4), px(6)))
        self.btn_crack_first = ttk.Button(
            nav, text="First", width=5, style="Tool.TButton",
            command=self._show_first_crack, state="disabled")
        self.btn_crack_first.pack(side="left")
        Tip(self.btn_crack_first, "Show the first saved phase-field state.")
        self.btn_crack_prev = ttk.Button(
            nav, text="<", width=3, style="Tool.TButton",
            command=lambda: self._move_crack_state(-1), state="disabled")
        self.btn_crack_prev.pack(side="left", padx=(px(4), 0))
        Tip(self.btn_crack_prev, "Show the previous saved phase-field state.")
        self.crack_state_text = tk.StringVar(value="no states")
        state_label = ttk.Label(nav, textvariable=self.crack_state_text, width=14,
                                anchor="center", foreground=T["muted"])
        state_label.pack(side="left", padx=px(5))
        Tip(state_label, "Saved-state position and the corresponding solver step number.")
        self.btn_crack_next = ttk.Button(
            nav, text=">", width=3, style="Tool.TButton",
            command=lambda: self._move_crack_state(1), state="disabled")
        self.btn_crack_next.pack(side="left")
        Tip(self.btn_crack_next, "Show the next saved phase-field state.")
        self.btn_crack_latest = ttk.Button(
            nav, text="Latest", width=6, style="Tool.TButton",
            command=self._show_latest_crack, state="disabled")
        self.btn_crack_latest.pack(side="left", padx=(px(4), px(4)))
        Tip(self.btn_crack_latest,
            "Jump to the newest state and resume following the running solver.")

        options = ttk.Frame(cbar_)
        self.crack_options = options
        ttk.Button(options, text="Refresh", command=self.draw_crack,
                   style="Tool.TButton").pack(side="left")
        ttk.Button(options, text="Fit", width=5, style="Tool.TButton",
                   command=self.fit_crack).pack(side="left", padx=4)
        self.follow_crack = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="follow latest",
                        variable=self.follow_crack,
                        command=self._follow_crack_changed).pack(side="left", padx=8)
        self.show_notch = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="notch", variable=self.show_notch,
                        command=lambda: self.draw_crack(force=True)).pack(side="left")

        display = ttk.Frame(cbar_)
        self.crack_display = display
        ttk.Label(display, text="crack iso").pack(side="left", padx=(px(4), 2))
        self.iso_var = tk.StringVar(value="0.9")
        iso_entry = ttk.Entry(display, textvariable=self.iso_var, width=5)
        iso_entry.pack(side="left")
        iso_entry.bind("<Return>", lambda _e: self.draw_crack(force=True))
        ttk.Label(display, text="colours").pack(side="left", padx=(8, 2))
        self.cmap_var = tk.StringVar(value="inferno")
        cmb = ttk.Combobox(display, textvariable=self.cmap_var, values=CMAPS,
                           width=10, state="readonly")
        cmb.pack(side="left")
        cmb.bind("<<ComboboxSelected>>", lambda _e: self.draw_crack(force=True))
        self.crack_stats = tk.StringVar(value="no snapshots yet")
        ttk.Label(display, textvariable=self.crack_stats, anchor="w",
                  foreground=T["muted"]).pack(side="left", fill="x", expand=True,
                                                padx=(px(8), px(4)))
        self._crack_controls_mode = ""
        cbar_.bind("<Configure>", self._layout_crack_controls, add="+")
        # Give the frames an initial manager before the first Configure event.
        self._layout_crack_controls(width=max(1, self.win_w - px(520)))

        self.cfig = Figure(figsize=(7, 3.4), dpi=100)
        self.cax = self.cfig.add_subplot(111)
        style_axes(self.cfig, self.cax)
        self.cfig.tight_layout()
        self.ccanvas = FigureCanvasTkAgg(self.cfig, master=crack_tab)
        self.crack_toolbar = add_toolbar(self.ccanvas, crack_tab)
        self.crack_toolbar.pack(side="bottom", fill="x")
        self.ccanvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self._cbar = None          # colourbar, created once (levels are fixed)
        self._tri = None           # cached Triangulation
        self._tri_npoin = 0
        self._vtk_sizes: dict = {}

        # stretch="never" so growing the window grows the PLOT and leaves the
        # log at whatever height you dragged it to.
        lf = ttk.Frame(vpane, padding=(0, px(6), 0, 0))
        vpane.add(lf, height=px(210), minsize=px(60), stretch="never")

        logbar = ttk.Frame(lf)
        logbar.pack(fill="x", pady=(0, px(4)))
        ttk.Label(logbar, text="Log", font=("TkDefaultFont", 9, "bold")).pack(
            side="left", padx=(0, px(6)))
        self.log_search_var = tk.StringVar()
        search = ttk.Entry(logbar, textvariable=self.log_search_var, width=18)
        search.pack(side="left")
        search.bind("<Return>", lambda _e: self._find_log(1))
        search.bind("<Shift-Return>", lambda _e: self._find_log(-1))
        ttk.Button(logbar, text="Prev", width=5,
                   command=lambda: self._find_log(-1)).pack(side="left", padx=(px(3), 0))
        ttk.Button(logbar, text="Next", width=5,
                   command=lambda: self._find_log(1)).pack(side="left", padx=(px(3), px(7)))

        self.log_filter_var = tk.StringVar(value="All output")
        log_filter = ttk.Combobox(
            logbar, textvariable=self.log_filter_var,
            values=["All output", "Normal", "Warnings", "Convergence", "Errors"],
            width=12, state="readonly")
        log_filter.pack(side="left")
        log_filter.bind("<<ComboboxSelected>>", lambda _e: self._apply_log_filter())

        logactions = ttk.Frame(lf)
        logactions.pack(fill="x", pady=(0, px(4)))
        self.follow_log = tk.BooleanVar(value=True)
        ttk.Checkbutton(logactions, text="Follow tail", variable=self.follow_log,
                        command=self._follow_log_changed).pack(
            side="left")
        ttk.Button(logactions, text="Copy", width=5,
                   command=self._copy_log).pack(side="left", padx=(px(7), 0))
        ttk.Button(logactions, text="Clear", width=5, command=self._clear_log).pack(
            side="left", padx=px(3))
        ttk.Button(logactions, text="Save", width=5,
                   command=self._save_log).pack(side="left")

        logbody = ttk.Frame(lf)
        logbody.pack(fill="both", expand=True)
        self.log = tk.Text(logbody, height=6, wrap="none", font=("Consolas", 9),
                           background=T["panel"], foreground=T["fg"],
                           insertbackground=T["fg"], selectbackground=T["accent"],
                           selectforeground="#000000", relief="flat",
                           borderwidth=0)
        sb = ttk.Scrollbar(logbody, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.log.tag_configure("normal", foreground=T["fg"])
        self.log.tag_configure("warn", foreground=T["warn"])
        self.log.tag_configure("conv", foreground=T["accent"])
        self.log.tag_configure("err", foreground=T["err"])
        self.log.tag_configure("search_hit", background=T["accent"],
                               foreground="#101215")

        # Live pane resizing without the render storm -- see
        # _install_resize_guard. Must run AFTER every canvas exists.
        for c in (self.canvas, self.mcanvas, self.ccanvas):
            self._install_resize_guard(c)
        for p in (outer, vpane):
            self._bind_sash(p)

        self._build_panel()

        # Open the settings pane at the width the form actually asks for,
        # rather than a number typed in once against one font at one dpi.
        # A window that starts with its horizontal scrollbar already engaged
        # reads as broken, and the width that avoids it moves with the UI
        # font, the display scale and the length of the longest table.
        self.root.update_idletasks()
        want = self.panel.winfo_reqwidth() + px(26)      # + the scrollbar
        # Capped as a FRACTION of the window, not at a pixel count: the form
        # grows with the display scale and the window does not (it is clamped
        # to the screen), so at 200% an absolute cap let the settings take
        # nearly two thirds of the width and squeezed the plots into what was
        # left. Past this point the horizontal scrollbar is the better trade.
        try:
            outer.paneconfigure(
                panel_host,
                width=max(px(320), min(want, int(0.45 * self.win_w))))
        except tk.TclError:
            pass

        # Say what was detected. Scaling is the one setting here that cannot
        # be checked by looking at the window -- a display that is 25% out
        # just looks slightly wrong -- so it goes in the log where --scale
        # can be aimed at it.
        if SCALE_INFO:
            self._append_log(SCALE_INFO)

    def _load_gui_state(self):
        try:
            with open(GUI_STATE, "r", encoding="utf-8") as fh:
                state = json.load(fh)
            if not isinstance(state, dict):
                raise ValueError("state is not an object")
        except (OSError, ValueError, TypeError):
            state = {}
        state.setdefault("recent_configs", [])
        state.setdefault("presets", {})
        return state

    def _save_gui_state(self):
        tmp = GUI_STATE + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.gui_state, fh, indent=2)
            os.replace(tmp, GUI_STATE)
        except OSError as exc:
            if hasattr(self, "log"):
                self._append_log(f"[gui] could not save recent items: {exc}\n")

    @staticmethod
    def _display_config_path(path):
        try:
            rel = os.path.relpath(path, PROJECT_DIR)
            if not rel.startswith(".."):
                return rel
        except ValueError:
            pass
        return path

    def _refresh_recent_configs(self):
        if not hasattr(self, "recent_combo"):
            return
        self._recent_map = {
            self._display_config_path(p): p
            for p in self.gui_state.get("recent_configs", [])
        }
        self.recent_combo.configure(values=list(self._recent_map))
        if self.recent_var.get() not in self._recent_map:
            self.recent_var.set(next(iter(self._recent_map), ""))

    def _remember_config(self, path):
        path = os.path.abspath(path)
        recent = [p for p in self.gui_state.get("recent_configs", [])
                  if os.path.normcase(p) != os.path.normcase(path)]
        self.gui_state["recent_configs"] = [path] + recent[:MAX_RECENT_CONFIGS - 1]
        self._save_gui_state()
        self._refresh_recent_configs()
        self.recent_var.set(self._display_config_path(path))

    def _open_recent(self):
        path = getattr(self, "_recent_map", {}).get(self.recent_var.get())
        if not path:
            return
        if not os.path.isfile(path):
            self.gui_state["recent_configs"] = [
                p for p in self.gui_state.get("recent_configs", [])
                if os.path.normcase(p) != os.path.normcase(path)]
            self._save_gui_state()
            self._refresh_recent_configs()
            messagebox.showwarning("Configuration missing",
                                   f"This file is no longer available:\n{path}")
            return
        self.import_config(path)

    def _refresh_preset_values(self):
        if not hasattr(self, "preset_combo"):
            return
        saved = self.gui_state.get("presets", {})
        self._preset_map = dict(BUILTIN_PRESETS)
        self._preset_map.update({f"Saved: {name}": value
                                 for name, value in sorted(saved.items())})
        self.preset_combo.configure(values=list(self._preset_map))
        if self.preset_var.get() not in self._preset_map:
            self.preset_var.set(next(iter(self._preset_map), ""))

    def _apply_preset(self):
        name = self.preset_var.get()
        values = getattr(self, "_preset_map", {}).get(name)
        if not values:
            return
        if name.startswith("Saved: "):
            self.set_form(values)
        else:
            for key, value in values.items():
                var = self.vars.get(key)
                if var is not None:
                    var.set(value if isinstance(var, tk.BooleanVar) else str(value))
            self._apply_visibility()
            self._refresh_warnings()
        self._append_log(f"[gui] applied preset: {name}\n")

    def _save_current_preset(self):
        try:
            form = self.get_form()
        except ValueError as exc:
            messagebox.showerror("Cannot save preset", str(exc))
            return
        name = simpledialog.askstring("Save preset", "Preset name:", parent=self.root)
        if not name or not name.strip():
            return
        name = name.strip()
        self.gui_state.setdefault("presets", {})[name] = form
        self._save_gui_state()
        self._refresh_preset_values()
        self.preset_var.set(f"Saved: {name}")
        self._append_log(f"[gui] saved preset: {name}\n")

    # Match gui2's restrained but useful lifecycle colouring: idle is muted,
    # work is blue, and only an actual failure becomes red.
    STATE_COLOUR = {"idle": "muted", "busy": "accent",
                    "ok": "fg", "fail": "err"}

    def _set_state(self, text: str, kind: str = "idle") -> None:
        self.status.set(text)
        try:
            self.state_lbl.configure(
                foreground=T[self.STATE_COLOUR.get(kind, "fg")])
        except tk.TclError:
            pass

    def _bind_shortcuts(self):
        """A small, conventional shortcut set for repeated simulation work."""
        self.root.bind("<Control-o>", lambda _e: self.import_config())
        self.root.bind("<Control-s>", lambda _e: self.save_config())
        self.root.bind("<F5>", lambda _e: self.run())
        self.root.bind("<Control-F5>", lambda _e: self.check_mesh())
        self.root.bind("<Escape>", lambda _e: self.stop())

    # Every settings row is three columns: NAME | CONTROL | UNIT, in one grid
    # per section.
    #
    # It used to be a Frame per field with everything packed side by side.
    # pack sizes each row on its own, so a width=6 box and a width=14 box
    # started at the same x and ended in different places -- nine fields, nine
    # right edges, and the eye reads that ragged column as sloppiness before it
    # reads any of the values. A grid gives the section ONE control column, and
    # the controls fill it, so the edges line up by construction.
    #
    # It also makes the units honest. They were glued into the labels ("E
    # (MPa)", "gamma tol (deg)") on the fields that happened to have one, so
    # the names were different lengths for a reason that had nothing to do with
    # the names. In their own column they line up and the labels go back to
    # being the config keys you are actually setting.
    LABEL_COL = 15         # characters -- longest label is 13
    CTRL_COL = 13          # characters

    def _char_w(self) -> int:
        """Width of one digit in the form font, for sizing the grid columns.

        Measured, not assumed: the whole point of the column widths is that
        they hold a known number of characters, and at 150% scaling or in a
        different UI font a hard-coded pixel count holds a different number.
        """
        if not self._charw:
            try:
                self._charw = tkfont.nametofont("TkDefaultFont",
                                                root=self.root).measure("0")
            except (tk.TclError, KeyError):
                self._charw = 0
            self._charw = self._charw or px(7)
        return self._charw

    def _section(self, title, expanded=True):
        """Create a compact section with a useful summary while collapsed."""
        shell = ttk.Frame(self.panel)
        shell.pack(fill="x", padx=px(6), pady=px(3))
        button = ttk.Button(
            shell, style="Section.TButton",
            command=lambda t=title: self._set_section_expanded(
                t, not self._sections[t]["expanded"]))
        button.pack(fill="x")
        summary = ttk.Label(shell, foreground=T["muted"], anchor="w",
                            wraplength=px(400), justify="left",
                            padding=(px(11), px(1), px(4), px(3)))
        f = ttk.Frame(shell, padding=(px(7), px(4), px(7), px(7)))
        self._sections[title] = {
            "shell": shell, "button": button, "summary": summary,
            "body": f, "expanded": bool(expanded),
        }

        ch = self._char_w()
        f.grid_columnconfigure(0, minsize=self.LABEL_COL * ch)
        f.grid_columnconfigure(1, minsize=self.CTRL_COL * ch)
        # The slack goes to the unit column, so widening the panel opens up
        # the space AFTER the values instead of stretching every box.
        f.grid_columnconfigure(2, weight=1)
        f.next_row = 0
        f.section_title = title
        self._set_section_expanded(title, expanded)
        return f

    def _set_section_expanded(self, title, expanded):
        state = self._sections.get(title)
        if not state:
            return
        state["expanded"] = bool(expanded)
        state["button"].configure(text=("−  " if expanded else "+  ") + title)
        if expanded:
            state["summary"].pack_forget()
            if not state["body"].winfo_manager():
                state["body"].pack(fill="x")
        else:
            if state["body"].winfo_manager():
                state["body"].pack_forget()
            state["summary"].pack(fill="x")
            self._refresh_section_summaries()

    def _v(self, key, default=""):
        """Return a form value exactly as displayed, without raising."""
        var = self.vars.get(key)
        if var is None:
            return default
        try:
            return str(var.get())
        except tk.TclError:
            return default

    def _section_summary(self, title):
        """Describe a collapsed section without making the user reopen it."""
        try:
            if title.startswith("1."):
                src = self.geom_source.get() or "no source chosen"
                return f"{src} → {self.geom_out.get()}"
            if title.startswith("2."):
                if self._v("mesh_source") != "file":
                    return "built-in mesh"
                path = self._v("mesh_path")
                if not path:
                    return "no mesh selected"
                groups = f" · {len(self.mesh_groups)} groups" if self.mesh_groups else ""
                return f"{os.path.basename(path)}{groups}"
            if title.startswith("3."):
                hyb = " · hybrid" if self._v("hybrid") in ("1", "True") else ""
                return (f"{self._v('ntype_label')} · "
                        f"{self._v('energy_split')} split{hyb} · "
                        f"ngaus {self._v('ngaus')}")
            if title.startswith("4."):
                return (f"E {self._v('E')} MPa · nu {self._v('nu')} · "
                        f"Gc {self._v('Gc')} · l0 {self._v('l0')} mm")
            if title.startswith("5."):
                rows = [b for b in self.bctable.get() if b.get("group")]
                if not rows:
                    return "nothing restrained or loaded"
                driven = [b for b in rows
                          if float(b.get("vx") or 0) or float(b.get("vy") or 0)]
                lead = (f" · {driven[0]['group']} driven" if driven
                        else " · nothing driven")
                return f"{len(rows)} condition{'s' if len(rows) != 1 else ''}{lead}"
            if title.startswith("6."):
                sweeps = (f" · max {self._v('max_staggered')} sweeps"
                          if self._v("scheme") == "staggered" else "")
                return f"{self._v('scheme')}{sweeps} · tol_rel {self._v('tol_rel')}"
            if title.startswith("7."):
                mode = self._v("step_mode")
                try:
                    plan = planned_steps(self.get_form())
                except (ValueError, KeyError, TypeError):
                    plan = None
                count = f" · {plan[0]:,} steps" if plan else ""
                return f"{mode}{count}"
            if title.startswith("8."):
                return (f"{self._v('output_dir')}/{self._v('base_name')} · "
                        f"VTK every {self._v('vtk_every')}")
        except Exception:
            pass
        return ""

    def _refresh_section_summaries(self):
        for title, state in self._sections.items():
            if not state["expanded"]:
                state["summary"].configure(text=self._section_summary(title))

    def _field(self, parent, label, key, kind="entry", values=None,
               on_change=None, unit="", help=""):
        r = parent.next_row
        parent.next_row = r + 1
        cells = [ttk.Label(parent, text=label, anchor="w")]
        cells[0].grid(row=r, column=0, sticky="w", pady=px(2))
        if kind == "check":
            v = tk.BooleanVar()
            w = ttk.Checkbutton(parent, variable=v)
            w.grid(row=r, column=1, sticky="w", pady=px(2))
        elif kind == "combo":
            v = tk.StringVar()
            # An explicit width, because ttk defaults BOTH Combobox and
            # Entry to 20 characters -- and a 20-character request quietly
            # overrides the 13-character column minsize, which is how the
            # form ended up wider than the pane it lives in. sticky="ew"
            # stretches them back out to the column, so this caps the
            # request without narrowing anything on screen.
            w = ttk.Combobox(parent, textvariable=v, values=values,
                             width=self.CTRL_COL - 2, state="readonly")
            w.grid(row=r, column=1, sticky="ew", pady=px(2))
            if on_change:
                w.bind("<<ComboboxSelected>>", lambda _e: on_change())
        else:
            v = tk.StringVar()
            w = ttk.Entry(parent, textvariable=v, width=self.CTRL_COL)
            w.grid(row=r, column=1, sticky="ew", pady=px(2))
        cells.append(w)
        self.vars[key] = v
        if kind != "check":
            self._field_widgets[key] = w
            self._field_sections[key] = getattr(parent, "section_title", "")
        if unit:
            u = ttk.Label(parent, text=unit, foreground=T["muted"], anchor="w",
                          wraplength=px(170))
            u.grid(row=r, column=2, sticky="w", padx=(px(6), 0))
            cells.append(u)
        else:
            # Normally empty; becomes the short, field-specific explanation
            # when validation fails.
            u = ttk.Label(parent, text="", foreground=T["muted"], anchor="w",
                          wraplength=px(170))
            u.grid(row=r, column=2, sticky="w", padx=(px(6), 0))
            cells.append(u)
        self._field_feedback[key] = (u, unit)
        v.trace_add("write", lambda *_args: self._schedule_validation())
        if help:
            for c in cells:
                Tip(c, help)
        # Every widget of the row, so _apply_visibility can take the whole
        # thing out and put it back exactly where it was.
        self._rows[key] = cells
        return v

    def _note(self, parent, text="", var=None, warn=False):
        """A full-width line of explanation under a section's fields."""
        r = parent.next_row
        parent.next_row = r + 1
        kw = {"textvariable": var} if var is not None else {"text": text}
        lbl = ttk.Label(parent, foreground=T["warn"] if warn else T["muted"],
                        wraplength=px(400), justify="left", **kw)
        lbl.grid(row=r, column=0, columnspan=3, sticky="w", pady=(px(2), 0))
        return lbl

    def _wide(self, parent, widget, pady=0):
        """Grid a whole widget across the three columns -- a table, a button."""
        r = parent.next_row
        parent.next_row = r + 1
        widget.grid(row=r, column=0, columnspan=3, sticky="ew", pady=pady)
        return widget

    # Which fields are MEANINGFUL for a given choice. A field that the solver
    # ignores is worse than clutter: it invites you to set du_fine in uniform
    # mode, see nothing change, and go looking for the bug somewhere else.
    def _visible_keys(self):
        hide = set()
        mode = self.vars["step_mode"].get() if "step_mode" in self.vars else ""
        if mode == "uniform":
            hide |= {"du_coarse", "u_switch", "du_fine", "u_switch2", "du_final"}
        elif mode == "two_stage":
            hide |= {"du", "N_steps", "u_switch2", "du_final"}
        elif mode == "three_stage":
            hide |= {"du", "N_steps"}

        if self.vars.get("mesh_source") and \
                self.vars["mesh_source"].get() != "file":
            hide.add("mesh_path")

        # A monolithic solve has no staggered loop to bound or to stop.
        if self.vars.get("scheme") and self.vars["scheme"].get() != "staggered":
            hide |= {"max_staggered", "stagger_stop", "gamma_tol"}
        elif self.vars.get("stagger_stop") and \
                self.vars["stagger_stop"].get() != "energy":
            hide.add("gamma_tol")        # the angle tolerance IS the criterion
        return hide

    def _apply_visibility(self):
        """Show only the fields the current mode actually uses.

        grid_remove() rather than forget-and-re-add: it REMEMBERS the row and
        column, so bare grid() puts the field back exactly where it belongs
        instead of at the bottom of its section. That is most of the reason
        these rows are gridded rather than packed -- the old version had to
        rebuild an entire section to restore one field to its place.
        """
        hide = self._visible_keys()
        for key, cells in self._rows.items():
            for w in cells:
                w.grid_remove() if key in hide else w.grid()

    def _build_panel(self):
        # Keep gui3's recent-project and preset tools, but present them with
        # gui2's compact, utility-first hierarchy.
        top = ttk.Frame(self.panel, padding=(px(6), px(6), px(6), 0))
        top.pack(fill="x")
        btns = ttk.Frame(top)
        btns.pack(fill="x")
        ttk.Button(btns, text="Import config...",
                   command=lambda: self.import_config()).pack(side="left")
        ttk.Button(btns, text="Save as...",
                   command=self.save_config).pack(side="left", padx=px(4))
        ttk.Label(top, text="drag the dividers to resize the panel and the log",
                  foreground=T["muted"], wraplength=px(400),
                  justify="left").pack(anchor="w", pady=(px(2), 0))

        recent = ttk.Frame(top)
        recent.pack(fill="x", pady=(px(8), 0))
        ttk.Label(recent, text="Recent", width=9, anchor="w").pack(side="left")
        self.recent_var = tk.StringVar()
        self.recent_combo = ttk.Combobox(recent, textvariable=self.recent_var,
                                         width=27, state="readonly")
        self.recent_combo.pack(side="left", fill="x", expand=True)
        self.recent_combo.bind("<<ComboboxSelected>>",
                               lambda _e: self._open_recent())
        ttk.Button(recent, text="Open", width=6,
                   command=self._open_recent).pack(side="left", padx=(px(4), 0))

        presets = ttk.Frame(top)
        presets.pack(fill="x", pady=(px(4), 0))
        ttk.Label(presets, text="Preset", width=9, anchor="w").pack(side="left")
        self.preset_var = tk.StringVar(value="Robust staggered")
        self.preset_combo = ttk.Combobox(presets, textvariable=self.preset_var,
                                         width=20, state="readonly")
        self.preset_combo.pack(side="left", fill="x", expand=True)
        ttk.Button(presets, text="Apply", width=6,
                   command=self._apply_preset).pack(side="left", padx=(px(4), 0))
        ttk.Button(presets, text="Save current", width=11,
                   command=self._save_current_preset).pack(side="left", padx=(px(4), 0))
        self._refresh_recent_configs()
        self._refresh_preset_values()

        summary = ttk.Frame(self.panel, padding=(px(7), px(3)))
        summary.pack(fill="x", padx=px(6), pady=(px(2), px(1)))
        self.summary_var = tk.StringVar(value="Complete the setup to see a run summary.")
        ttk.Label(summary, textvariable=self.summary_var, foreground=T["muted"],
                  justify="left", wraplength=px(430)).pack(anchor="w")

        # ---- Geometry: build a mesh, rather than only pointing at one ------
        g = self._section("1. Geometry", expanded=False)
        ttk.Label(g, text="source", anchor="w").grid(row=0, column=0,
                                                     sticky="w", pady=px(2))
        self.geom_source = tk.StringVar()
        sources = list_specs() + ["STEP / IGES file..."]
        self.geom_combo = ttk.Combobox(g, textvariable=self.geom_source,
                                       values=sources, width=self.CTRL_COL - 2,
                                       state="readonly")
        self.geom_combo.grid(row=0, column=1, columnspan=2, sticky="ew",
                             pady=px(2))
        self.geom_combo.bind("<<ComboboxSelected>>", self._geom_source_changed)
        g.next_row = 1

        # Parameter fields are REBUILT from the chosen spec's [params]. The
        # specs already declare value/min/max/step, which is exactly what a
        # form needs -- so a new spec file gets a working UI for free.
        #
        # Its own frame, so it can be emptied and refilled without disturbing
        # the rows around it, and so what it packs inside cannot collide with
        # the section's grid.
        self.geom_param_frame = ttk.Frame(g)
        self._wide(g, self.geom_param_frame)
        self.geom_vars = {}
        self.geom_tables = {}
        self.geom_cad = ""

        r = g.next_row; g.next_row = r + 1
        ttk.Label(g, text="write to", anchor="w").grid(row=r, column=0,
                                                       sticky="w", pady=px(2))
        self.geom_out = tk.StringVar(value="meshes/generated.msh")
        ttk.Entry(g, textvariable=self.geom_out,
                  width=self.CTRL_COL).grid(row=r, column=1, columnspan=2,
                                            sticky="ew", pady=px(2))

        self.btn_gen = ttk.Button(g, text="Generate mesh",
                                  command=self.generate_mesh)
        r = g.next_row; g.next_row = r + 1
        self.btn_gen.grid(row=r, column=0, sticky="w", pady=(px(4), 0))
        self._note(g, "runs the mesher as a separate process; the log is "
                      "below and the result loads into section 2")

        s = self._section("2. Mesh", expanded=True)
        self._field(s, "source", "mesh_source", kind="combo",
                    values=["file", "builtin"],
                    on_change=self._apply_visibility)
        # Built by hand rather than through _field because it carries a Browse
        # button -- but registered in _rows the same way, so the source combo
        # can hide it like any other field.
        r = s.next_row; s.next_row = r + 1
        lbl = ttk.Label(s, text="mesh file", anchor="w")
        lbl.grid(row=r, column=0, sticky="w", pady=px(2))
        self.vars["mesh_path"] = tk.StringVar()
        ent = ttk.Entry(s, textvariable=self.vars["mesh_path"],
                        width=self.CTRL_COL)
        ent.grid(row=r, column=1, sticky="ew", pady=px(2))
        btn = ttk.Button(s, text="...", width=3, command=self._pick_mesh)
        btn.grid(row=r, column=2, sticky="w", padx=(px(6), 0))
        er = s.next_row; s.next_row = er + 1
        mesh_feedback = ttk.Label(s, text="", foreground=T["muted"], anchor="w",
                                  wraplength=px(230))
        mesh_feedback.grid(row=er, column=1, columnspan=2, sticky="w")
        self._rows["mesh_path"] = [lbl, ent, btn, mesh_feedback]
        self._field_widgets["mesh_path"] = ent
        self._field_feedback["mesh_path"] = (mesh_feedback, "")
        self._field_sections["mesh_path"] = getattr(s, "section_title", "")
        self.vars["mesh_path"].trace_add(
            "write", lambda *_args: self._schedule_validation())
        self.mesh_info = tk.StringVar(value="")
        self._note(s, var=self.mesh_info)
        self._field(s, "base_name", "base_name",
                    unit="output folder")

        s = self._section("3. Model", expanded=False)
        self._field(s, "ntype", "ntype_label", kind="combo",
                    values=list(NTYPES.keys()))
        self._field(s, "energy_split", "energy_split", kind="combo",
                    values=SPLITS)
        self._field(s, "hybrid", "hybrid", kind="check")
        self._field(s, "ngaus", "ngaus", unit="per direction")

        s = self._section("4. Material", expanded=True)
        self._field(s, "name", "mat_name")
        self._field(s, "E", "E", unit="MPa")
        self._field(s, "nu", "nu")
        self._field(s, "Gc", "Gc", unit="N/mm")
        self._field(s, "l0", "l0", unit="mm")
        self._field(s, "k", "k", unit="residual stiffness")
        self._field(s, "domain group", "domain")

        s = self._section("5. Restraints and loads", expanded=True)
        self.bctable = LoadTable(s, lambda: self.mesh_groups.keys(),
                                 on_change=self._bcs_changed)
        self._wide(s, self.bctable)
        self._note(s,
                   "displacement: mm, tick fix x / fix y for the DOFs held\n"
                   "force: N at a point (0D) group\n"
                   "traction: N per mm of edge, on an edge (1D) group\n"
                   "all are scaled by the load factor each step")

        s = self._section("6. Solver", expanded=False)
        self._field(s, "scheme", "scheme", kind="combo", values=SCHEMES,
                    on_change=self._apply_visibility)
        self._field(s, "stagger_stop", "stagger_stop", kind="combo",
                    values=STAGGER_STOPS, on_change=self._apply_visibility)
        self._field(s, "max_staggered", "max_staggered", unit="sweeps")
        self._field(s, "gamma_tol", "gamma_tol", unit="deg")
        self._field(s, "tol_rel", "tol_rel")
        self._field(s, "tol_abs", "tol_abs")
        self._field(s, "max_iter", "max_iter")

        # The qualifiers that used to sit in these labels -- "(uniform)",
        # "(3-stage)" -- are gone because _visible_keys already hides each
        # field in the modes where it does nothing. A field you can see is a
        # field this mode uses; that is a stronger statement than a bracket.
        s = self._section("7. Load stepping", expanded=True)
        self._field(s, "step_mode", "step_mode", kind="combo",
                    values=STEP_MODES, on_change=self._apply_visibility)
        self._field(s, "du", "du", unit="mm")
        self._field(s, "N_steps", "N_steps", unit="used when du = 0")
        self._field(s, "du_coarse", "du_coarse", unit="mm")
        self._field(s, "u_switch", "u_switch", unit="mm")
        self._field(s, "du_fine", "du_fine", unit="mm")
        self._field(s, "u_switch2", "u_switch2", unit="mm")
        self._field(s, "du_final", "du_final", unit="mm")
        self._field(s, "max_subdivs", "max_subdivs", unit="halvings")

        s = self._section("8. Output", expanded=False)
        self._field(s, "output_dir", "output_dir")
        self._field(s, "vtk_every", "vtk_every", unit="steps")
        self._field(s, "profile_every", "profile_every", unit="steps, 0 = off")
        self._field(s, "write_log", "write_log", kind="check")

        self.validation_state = tk.StringVar(value="Checking configuration...")
        self.validation_label = ttk.Label(self.panel,
                                          textvariable=self.validation_state,
                                          style="Warning.TLabel",
                                          padding=(px(8), px(4)))
        self.validation_label.pack(fill="x", padx=px(6), pady=(px(4), 0))
        self.warnbox = tk.Text(self.panel, height=7, width=20, wrap="word",
                               font=("TkDefaultFont", 8), foreground=T["warn"],
                               background=T["warnbg"], relief="flat",
                               insertbackground=T["fg"], borderwidth=0)
        self.warnbox.pack(fill="x", padx=px(6), pady=(px(4), px(8)))
        self._apply_visibility()

    # -- mesh view ------------------------------------------------------
    HILITE = ["#4a9eff", "#ff9f43", "#2ed573", "#ff6b81", "#c56cf0",
              "#ffd32a", "#7bed9f", "#eccc68"]

    # -- geometry -------------------------------------------------------
    def _geom_source_changed(self, _event=None, cad_path=None):
        """Rebuild the parameter fields for the newly chosen source."""
        for w in self.geom_param_frame.winfo_children():
            w.destroy()
        self.geom_vars = {}
        self.geom_tables = {}      # only the STEP source builds tables
        src = self.geom_source.get()

        if src.startswith("STEP"):
            path = cad_path or filedialog.askopenfilename(
                initialdir=PROJECT_DIR, title="Select a CAD file",
                filetypes=[("CAD", "*.step *.stp *.iges *.igs *.brep"),
                           ("All files", "*.*")])
            if not path:
                self.geom_source.set("")
                return
            self.geom_cad = path

            cad = ttk.LabelFrame(self.geom_param_frame, text="CAD file",
                                 padding=(px(7), px(6)))
            cad.pack(fill="x", pady=(px(2), px(3)))
            cad.grid_columnconfigure(0, weight=1)
            cad_name = tk.StringVar(value=path)
            cad_entry = ttk.Entry(cad, textvariable=cad_name, state="readonly")
            cad_entry.grid(row=0, column=0, sticky="ew")
            ttk.Button(cad, text="Change...", style="Tool.TButton",
                       command=self._change_step_cad).grid(
                           row=0, column=1, padx=(px(5), 0))
            Tip(cad_entry, path)

            def prepare_grid(parent):
                parent.grid_columnconfigure(0, minsize=self.LABEL_COL * self._char_w())
                parent.grid_columnconfigure(1, minsize=self.CTRL_COL * self._char_w())
                parent.grid_columnconfigure(2, weight=1)
                parent.next_row = 0

            def field(parent, label, key, default, unit="", help=""):
                row = parent.next_row
                parent.next_row += 1
                lbl = ttk.Label(parent, text=label, anchor="w")
                lbl.grid(row=row, column=0, sticky="w", pady=px(2))
                v = tk.StringVar(value=default)
                ent = ttk.Entry(parent, textvariable=v, width=self.CTRL_COL)
                ent.grid(row=row, column=1, sticky="ew", pady=px(2))
                if unit:
                    ttk.Label(parent, text=unit, foreground=T["muted"],
                              anchor="w").grid(row=row, column=2, sticky="w",
                                               padx=(px(6), 0), pady=px(2))
                if help:
                    Tip(lbl, help); Tip(ent, help)
                self.geom_vars[key] = v

            settings = ttk.LabelFrame(self.geom_param_frame, text="Mesh settings",
                                      padding=(px(7), px(5)))
            settings.pack(fill="x", pady=px(3))
            prepare_grid(settings)
            field(settings, "bulk size h", "h", "0.15", "CAD units",
                  "Target element size away from locally refined regions.")

            row = settings.next_row
            settings.next_row += 1
            ttk.Label(settings, text="elements", anchor="w").grid(
                row=row, column=0, sticky="w", pady=px(2))
            ev = tk.StringVar(value="quad4")
            cb = ttk.Combobox(settings, textvariable=ev, values=ELEMENTS,
                              width=self.CTRL_COL - 2, state="readonly")
            cb.grid(row=row, column=1, sticky="ew", pady=px(2))
            Tip(cb, "The only three element types the solver's reader accepts. "
                    "quad8 needs fem.ngaus = 3. Changing element type changes "
                    "the ANSWER, not just the discretisation -- tri3 is "
                    "noticeably stiffer than quad4 at the same size.")
            self.geom_vars["elements"] = ev

            row = settings.next_row
            settings.next_row += 1
            v = tk.BooleanVar(value=True)
            ttk.Label(settings, text="boundary groups", anchor="w").grid(
                row=row, column=0, sticky="w", pady=px(2))
            sides = ttk.Checkbutton(settings,
                                    text="Left / Right / Top / Bottom",
                                    variable=v)
            sides.grid(row=row, column=1, columnspan=2, sticky="w", pady=px(2))
            Tip(sides, "Create named physical groups for the four outer sides.")
            self.geom_vars["auto_sides"] = v

            ttk.Label(self.geom_param_frame,
                      text="Coordinates, sizes, and distances use the CAD file's "
                           "units (normally millimetres).",
                      foreground=T["muted"], wraplength=px(470),
                      justify="left").pack(anchor="w", padx=px(7),
                                           pady=(px(2), px(4)))

            # One table per repeatable argument, each with the plain fields
            # that belong to it. Cracks and refinement are not optional extras:
            # without them an imported part can be meshed and looked at but not
            # run, since a phase-field study needs a sharp crack and l0
            # resolved in the band.
            self.geom_tables = {}
            for title, flag, hint, tmpl, cols, *rest in GEOM_TABLES:
                box = ttk.LabelFrame(self.geom_param_frame, text=title,
                                     padding=(px(7), px(5)))
                box.pack(fill="x", pady=3)
                t = SpecTable(box, cols, tmpl, hint=hint,
                              legacy=(rest[0] if rest else ()))
                t.pack(fill="x")
                t.flag = flag
                self.geom_tables[title] = t
                extras = GEOM_EXTRAS.get(title, ())
                if extras:
                    extra_grid = ttk.Frame(box)
                    extra_grid.pack(fill="x", pady=(px(4), 0))
                    prepare_grid(extra_grid)
                    for label, key, default, help_ in extras:
                        field(extra_grid, label, key, default,
                              "CAD units", help_)

            self.geom_out.set("meshes/" +
                              os.path.splitext(os.path.basename(path))[0] + ".msh")
            # A mesh remembers how it was made: if a recipe sits next to the
            # output path, reload it so the panel comes back as you left it.
            self._load_recipe()
            return

        if not src:
            return
        spec_path = os.path.join(PROJECT_DIR, "mesh", "specs", src)
        defs = spec_param_defs(spec_path)
        if not defs:
            ttk.Label(self.geom_param_frame,
                      text="no [params] in this spec - it builds one fixed shape",
                      foreground=T["muted"], wraplength=px(430)).pack(anchor="w")
        for name, d in defs.items():
            r = ttk.Frame(self.geom_param_frame); r.pack(fill="x", pady=1)
            ttk.Label(r, text=name, width=15).pack(side="left")
            v = tk.StringVar(value=repr(float(d["value"])))
            ttk.Entry(r, textvariable=v, width=12).pack(side="left")
            if "min" in d and "max" in d:
                ttk.Label(r, text=f"[{d['min']:g}, {d['max']:g}]",
                          foreground=T["muted"]).pack(side="left", padx=4)
            self.geom_vars[name] = v
        self.geom_out.set("meshes/" + os.path.splitext(src)[0] + ".msh")

    def _change_step_cad(self):
        """Choose another CAD file while keeping the STEP source selected."""
        path = filedialog.askopenfilename(
            initialdir=os.path.dirname(self.geom_cad) or PROJECT_DIR,
            title="Select a CAD file",
            filetypes=[("CAD", "*.step *.stp *.iges *.igs *.brep"),
                       ("All files", "*.*")])
        if path:
            self._geom_source_changed(cad_path=path)

    def generate_mesh(self):
        src = self.geom_source.get()
        if not src:
            messagebox.showwarning("No source", "Pick a geometry source first.")
            return
        out = self.geom_out.get().strip()
        if not out:
            messagebox.showwarning("No output", "Set an output .msh path.")
            return

        if src.startswith("STEP"):
            cmd = [sys.executable, os.path.join("mesh", "from_step.py"),
                   self.geom_cad, "--out", out]
            if self.geom_vars.get("auto_sides") and self.geom_vars["auto_sides"].get():
                cmd.append("--auto-sides")
            for key, flag in (("h", "--h"), ("h_fine", "--h-fine"),
                              ("r_fine", "--r-fine"), ("h_crack", "--h-crack"),
                              ("r_crack", "--r-crack"),
                              ("h_pad", "--h-pad"), ("elements", "--elements")):
                v = self.geom_vars.get(key)
                if v is not None and v.get().strip():
                    cmd += [flag, v.get().strip()]
            for title, flag, *_rest in GEOM_TABLES:
                t = self.geom_tables.get(title)
                if t is None:
                    continue
                if t.text_mode:
                    # Unapplied text edits would be silently discarded.
                    t.toggle_text()
                for line in t.lines():
                    cmd += [flag, line]
        else:
            cmd = [sys.executable, os.path.join("mesh", "from_spec.py"),
                   os.path.join("mesh", "specs", src), "--out", out]
            for name, v in self.geom_vars.items():
                val = v.get().strip()
                if val:
                    cmd += ["--set", f"{name}={val}"]

        # On success the new mesh becomes the one section 2 points at, its
        # groups are rescanned so the BC dropdowns update, and the view
        # switches to the Mesh tab -- which is the checkpoint that matters
        # before committing to a solve.
        def done():
            self._save_recipe(out, cmd)
            self.vars["mesh_path"].set(out)
            self.vars["mesh_source"].set("file")
            self._scan_mesh()
            self.draw_mesh()
            try:
                self.nb.select(self.mesh_tab)
            except Exception:
                pass
            self._refresh_warnings()

        self._run_tool(cmd, "meshing", "mesh", on_success=done)

    # -- mesh recipe ----------------------------------------------------
    #
    # A .msh carries no record of how it was made: close the GUI and the CAD
    # file, sizes, pads, cracks and refinements are gone, and a mesh you cannot
    # regenerate is a mesh you cannot tweak. The recipe is written next to the
    # mesh on every successful generate and reloaded when you point at that
    # output again.
    @staticmethod
    def recipe_path(msh: str) -> str:
        return os.path.splitext(msh)[0] + ".recipe.json"

    def _save_recipe(self, out: str, cmd) -> None:
        if not self.geom_source.get().startswith("STEP"):
            return
        data = {
            "tool": "from_step",
            "cad": self.geom_cad,
            "out": out,
            "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fields": {k: (bool(v.get()) if isinstance(v, tk.BooleanVar)
                           else v.get())
                       for k, v in self.geom_vars.items()},
            "tables": {title: t.lines()
                       for title, t in self.geom_tables.items()},
            # The exact command, so the mesh can be rebuilt from a shell
            # without the GUI at all.
            "command": [str(c) for c in cmd],
        }
        path = os.path.join(PROJECT_DIR, self.recipe_path(out))
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            self._append_log(f"[gui] recipe saved to {self.recipe_path(out)}\n")
        except OSError as exc:
            self._append_log(f"[gui] could not save the recipe: {exc}\n")

    def _load_recipe(self, out: str = "") -> bool:
        """Repopulate the STEP panel from the recipe beside `out`, if any."""
        out = out or self.geom_out.get().strip()
        if not out or not getattr(self, "geom_tables", None):
            return False
        path = os.path.join(PROJECT_DIR, self.recipe_path(out))
        if not os.path.isfile(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            self._append_log(f"[gui] recipe unreadable: {exc}\n")
            return False
        for key, val in (data.get("fields") or {}).items():
            v = self.geom_vars.get(key)
            if v is None:
                continue
            try:
                v.set(bool(val) if isinstance(v, tk.BooleanVar) else str(val))
            except Exception:
                pass
        for title, lines in (data.get("tables") or {}).items():
            t = self.geom_tables.get(title)
            if t is not None:
                t.set_lines(lines)
        self._append_log(f"[gui] loaded the recipe saved with "
                         f"{os.path.basename(out)}"
                         f" ({data.get('saved', 'unknown date')})\n")
        return True

    def _bcs_changed(self):
        """A BC row was added, removed or edited."""
        self._mesh_dirty = True
        self._refresh_warnings()
        # Debounce: typing a group name fires once per KEYSTROKE, and each
        # redraw is ~0.3 s. Coalesce into one redraw 300 ms after you stop.
        if self._redraw_job is not None:
            try:
                self.root.after_cancel(self._redraw_job)
            except Exception:
                pass
        self._redraw_job = self.root.after(300, self._redraw_if_visible)

    def _redraw_if_visible(self):
        self._redraw_job = None
        if not self._mesh_dirty:
            return
        try:
            showing = self.nb.select() == str(self.mesh_tab)
        except Exception:
            showing = False
        if showing:
            # preserve_view: adding a BC must not throw away a zoom you set up
            # to inspect the very region you are restraining.
            self.draw_mesh(preserve_view=True)

    # -- smooth sash dragging -------------------------------------------
    class _FakeConfigure:
        """Minimal stand-in for a Tk <Configure> event.

        FigureCanvasTkAgg.resize() only reads .width/.height/.widget, so this
        is enough to replay a resize that was skipped during a drag.
        """
        def __init__(self, widget, width, height):
            self.widget, self.width, self.height = widget, width, height

    def _install_resize_guard(self, canvas):
        """Take over <Configure> for a matplotlib canvas.

        matplotlib binds resize() to <Configure> on its Tk widget, and resize()
        ends in a full render. During a sash drag that fires on every pixel, so
        the pane geometry updates at mouse rate but the window spends all its
        time re-rendering a figure nobody has finished resizing yet.

        We unbind matplotlib's handler and install our own: pass the event
        straight through normally, and while a sash is being dragged just
        remember the canvas and replay one resize on release. The plot stays
        visible at its old size during the drag (it is never blanked) and
        re-renders once, at the end.

        Wrapped in try/except: if a future matplotlib changes this, we fall
        back to its own behaviour rather than breaking the window.
        """
        try:
            w = canvas.get_tk_widget()
            w.unbind("<Configure>")
            w.bind("<Configure>", lambda e, c=canvas: self._canvas_resize(e, c))
        except Exception:
            pass

    def _canvas_resize(self, event, canvas):
        if self._dragging:
            self._resize_pending.add(canvas)
            return
        try:
            canvas.resize(event)
        except Exception:
            pass

    def _bind_sash(self, pane):
        pane.bind("<ButtonPress-1>",
                  lambda e, p=pane: self._sash_press(e, p), add="+")
        pane.bind("<ButtonRelease-1>", self._sash_release, add="+")

    def _sash_press(self, event, pane):
        # identify() returns a non-empty result only over a sash; a click that
        # lands on a pane's contents goes to the child widget, not here, but
        # check anyway so a stray click cannot leave rendering frozen.
        try:
            on_sash = bool(pane.identify(event.x, event.y))
        except Exception:
            on_sash = True
        if on_sash:
            self._dragging = True
            self._dragging_pane = pane
            if pane is self.paned:
                self.panel_host.set_live_resize(True)

    def _sash_release(self, _event=None):
        if not self._dragging:
            return
        pane = self._dragging_pane
        self._dragging = False
        self._dragging_pane = None
        if pane is self.paned:
            self.panel_host.set_live_resize(False)
        # Anything the drag suppressed catches up now.
        self._last_plot = 0.0
        pending, self._resize_pending = self._resize_pending, set()
        for canvas in pending:
            try:
                w = canvas.get_tk_widget()
                canvas.resize(self._FakeConfigure(w, w.winfo_width(),
                                                  w.winfo_height()))
            except Exception:
                try:
                    canvas.draw_idle()
                except Exception:
                    pass

    def _layout(self, fig, key=""):
        """tight_layout, but only when the result could actually differ.

        tight_layout measures every tick label and title to solve for margins.
        It costs 25-40 ms -- on the curve tab it QUADRUPLES the redraw (12 ms
        -> 49 ms) -- and it was running on every refresh, twice a second during
        a solve and on every button press that redraws.

        The margins only change when the axis labels change or the figure is
        resized, so key on exactly those and skip otherwise. Everything else
        reuses the subplot params already in place.
        """
        w, h = fig.get_size_inches()
        full = (key, round(float(w), 2), round(float(h), 2))
        if self._laid_out.get(id(fig)) == full:
            return
        fig.tight_layout()
        self._laid_out[id(fig)] = full

    def _curve_interacted(self, _event=None):
        """A toolbar pan or zoom holds the chosen curve view."""
        mode = str(getattr(self.curve_toolbar, "mode", "") or "")
        if mode and self.follow_curve.get():
            self.follow_curve.set(False)
            self._follow_curve_changed()

    def _follow_curve_changed(self):
        if self.follow_curve.get():
            self.curve_hint.set("")
            self._refresh_plot(force=True)
        else:
            self.curve_hint.set("view held - press Fit to follow again")

    def fit_curve(self):
        """Fit the complete curve and resume following new data."""
        self.follow_curve.set(True)
        self.curve_hint.set("")
        self._refresh_plot(force=True)

    def _tab_changed(self, _event=None):
        try:
            if self.nb.select() == str(self.mesh_tab) and self._mesh_dirty:
                self.draw_mesh(preserve_view=True)
            elif self.nb.select() == str(self.crack_tab):
                self.draw_crack()
        except Exception:
            pass

    # -- crack view -----------------------------------------------------
    def _layout_crack_controls(self, event=None, width=None):
        """Keep the crack toolbar to one row whenever its width permits it."""
        if width is None:
            width = getattr(event, "width", self.crack_nav.master.winfo_width())
        if width >= px(840):
            mode = "one"
        elif width >= px(570):
            mode = "two"
        else:
            mode = "three"
        if mode == self._crack_controls_mode:
            return
        self._crack_controls_mode = mode
        for frame in (self.crack_nav, self.crack_options, self.crack_display):
            frame.pack_forget()

        if mode == "one":
            self.crack_nav.pack(side="left")
            self.crack_options.pack(side="left", padx=(px(5), 0))
            self.crack_display.pack(side="left", fill="x", expand=True)
        elif mode == "two":
            self.crack_nav.pack(side="top", fill="x")
            self.crack_options.pack(side="left", padx=(0, px(5)),
                                    pady=(px(3), 0))
            self.crack_display.pack(side="left", fill="x", expand=True,
                                    pady=(px(3), 0))
        else:
            self.crack_nav.pack(side="top", fill="x")
            self.crack_options.pack(side="top", fill="x", pady=(px(3), 0))
            self.crack_display.pack(side="top", fill="x", pady=(px(3), 0))

    def _vtk_dir(self) -> str:
        """Snapshot folder for the CURRENT form, so an old run can be opened
        without launching anything."""
        try:
            f = self.get_form()
        except ValueError:
            return ""
        d = vtk_dir_for(f)
        return os.path.join(PROJECT_DIR, d) if d else ""

    def _stable(self, path: str) -> bool:
        """True once a file's size has stopped changing.

        The solver may be part-way through writing the newest snapshot when we
        notice it. Rather than parse a truncated file and raise, wait until the
        size is the same on two consecutive looks.
        """
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        was = self._vtk_sizes.get(path)
        self._vtk_sizes[path] = size
        return was == size and size > 0

    def _sync_crack_navigator(self, snapshots, selected_step=None):
        """Update state label and navigation buttons without loading a file."""
        self._crack_snapshots = snapshots
        n = len(snapshots)
        if not n:
            self.crack_state_text.set("no states")
            for button in (self.btn_crack_first, self.btn_crack_prev,
                           self.btn_crack_next,
                           self.btn_crack_latest):
                button.configure(state="disabled")
            return -1

        steps = [item[0] for item in snapshots]
        if selected_step in steps:
            index = steps.index(selected_step)
        else:
            index = n - 1
        self.crack_state_text.set(f"{index + 1}/{n}  #{steps[index]:,}")
        self.btn_crack_first.configure(state="normal" if index > 0 else "disabled")
        self.btn_crack_prev.configure(state="normal" if index > 0 else "disabled")
        self.btn_crack_next.configure(
            state="normal" if index < n - 1 else "disabled")
        self.btn_crack_latest.configure(
            state="normal" if index < n - 1 else "disabled")
        return index

    def _select_crack_state(self, index):
        snapshots = vtk_snapshots(self._vtk_dir())
        if not snapshots:
            self._sync_crack_navigator([])
            return
        index = min(max(int(index), 0), len(snapshots) - 1)
        self.follow_crack.set(False)
        self._sync_crack_navigator(snapshots, snapshots[index][0])
        self.draw_crack(target_index=index)

    def _move_crack_state(self, delta):
        snapshots = vtk_snapshots(self._vtk_dir())
        if not snapshots:
            self._sync_crack_navigator([])
            return
        current = self._sync_crack_navigator(snapshots, self._crack_step)
        self._select_crack_state(min(max(current + delta, 0), len(snapshots) - 1))

    def _show_first_crack(self):
        snapshots = vtk_snapshots(self._vtk_dir())
        if snapshots:
            self._select_crack_state(0)
        else:
            self._sync_crack_navigator([])

    def _show_latest_crack(self):
        self.follow_crack.set(True)
        snapshots = vtk_snapshots(self._vtk_dir())
        if snapshots:
            self.draw_crack(target_index=len(snapshots) - 1)
        else:
            self._sync_crack_navigator([])

    def _follow_crack_changed(self):
        if self.follow_crack.get():
            self._show_latest_crack()

    def draw_crack(self, force=False, target_index=None):
        vdir = self._vtk_dir()
        snapshots = vtk_snapshots(vdir)
        if not snapshots:
            self._sync_crack_navigator([])
            self.crack_stats.set(
                "no snapshots yet" if not vdir else f"no .vtk files in {vdir}")
            return

        if target_index is None:
            steps = [item[0] for item in snapshots]
            if not self.follow_crack.get() and self._crack_step in steps:
                target_index = steps.index(self._crack_step)
            else:
                target_index = len(snapshots) - 1
        target_index = min(max(int(target_index), 0), len(snapshots) - 1)
        step, path = snapshots[target_index]
        self._sync_crack_navigator(snapshots, step)

        # Only the newest file can still be in the middle of being written.
        if target_index == len(snapshots) - 1 and not self._stable(path) and not force:
            return                       # still being written; try again shortly
        if self._crack_loading_path == path:
            return
        self._crack_request += 1
        request = self._crack_request
        self._crack_loading_path = path
        self.crack_stats.set(f"loading step {step}...")
        threading.Thread(
            target=self._read_crack_worker,
            args=(request, path, step, force, self._tri, self._tri_npoin),
            daemon=True).start()

    def _read_crack_worker(self, request, path, step, force, tri, npoin):
        try:
            with self._io_lock:
                if tri is None:
                    tri, npoin = read_vtk_geometry(path)
                phi = read_vtk_phi(path, npoin)
            error = ""
        except (OSError, ValueError, IndexError) as exc:
            phi, error = None, str(exc)
        self.io_q.put((self._crack_loaded,
                       (request, path, step, force, tri, npoin, phi, error)))

    def _crack_loaded(self, request, path, step, force, tri, npoin, phi, error):
        if self._crack_loading_path == path:
            self._crack_loading_path = ""
        if request != self._crack_request:
            return
        if error:
            # A torn read is normal and self-correcting; only say so once.
            if step != self._crack_step:
                self.crack_stats.set(f"waiting for step {step} ({error})")
            return
        self._tri, self._tri_npoin = tri, npoin
        self._render_crack(path, step, phi, force)

    def _render_crack(self, path, step, phi, force=False):
        """Draw VTK data that was parsed away from Tk's event thread."""

        try:
            iso = float(self.iso_var.get())
        except ValueError:
            iso = 0.9

        keep = self.cax.get_xlim(), self.cax.get_ylim()
        had_view = self._crack_step >= 0
        self.cax.clear(); style_axes(self.cfig, self.cax)
        # Fixed 0..1 levels: phi IS a 0..1 field, and rescaling per frame would
        # make an early run of tiny damage look identical to a broken specimen.
        #
        # extend="both" is NOT cosmetic. The discretised phase field overshoots
        # its bounds slightly -- real snapshots from this solver reach phi =
        # 1.009 and -0.063 -- and with extend="neither" every node outside
        # [0, 1] falls into no level and is left UNPAINTED. That punches a hole
        # straight through the middle of the crack, which is exactly the region
        # being looked at. "both" clamps them to the end colours instead.
        cmap = self.cmap_var.get() or "inferno"
        cs = self.cax.tricontourf(self._tri, phi,
                                  levels=np.linspace(0.0, 1.0, 11),
                                  cmap=cmap, extend="both")
        if phi.max() >= iso:
            # The iso-line has to stay legible against whatever map is chosen.
            # The theme accent is blue, which vanishes into viridis/cividis and
            # into the blue end of jet/turbo, so pick per map rather than
            # assuming a dark background behind the line.
            self.cax.tricontour(self._tri, phi, levels=[iso],
                                colors=[ISO_COLOR.get(cmap, T["accent"])],
                                linewidths=1.4)
        # ---- the pre-existing notch ---------------------------------------
        #
        # Plugin(Crack) duplicates the nodes along the notch, so the
        # triangulation has NO elements spanning it -- tricontourf leaves a
        # zero-width gap, which draws as nothing. Without this overlay the
        # machined notch and the damage that actually grew are both just red,
        # and you cannot see where one ends and the other begins.
        #
        # Colour is taken from the ACTIVE colourmap so it belongs to the plot:
        # the mid-tone at 0.55 (distinct from both the phi=0 background and the
        # phi=1 crack), cased in the map's darkest colour so it stays legible
        # wherever it crosses.
        if self.show_notch.get():
            segs = self._notch_segments()
            if segs:
                cm = matplotlib.colormaps[cmap]
                self.cax.add_collection(LineCollection(
                    segs, colors=[cm(0.55)], linewidths=2.0, zorder=4,
                    path_effects=[mpe.withStroke(linewidth=3.6,
                                                 foreground=cm(0.0))]))

        self.cax.set_aspect("equal", adjustable="datalim")
        if self._cbar is None:
            self._cbar = self.cfig.colorbar(cs, ax=self.cax, label="phi")
            self._cbar.ax.yaxis.label.set_color(T["muted"])
            self._cbar.ax.tick_params(colors=T["muted"])
        else:
            # Without this the bar keeps the colours of the PREVIOUS map and
            # silently mislabels the plot.
            self._cbar.update_normal(cs)
        if had_view and not force:
            self.cax.set_xlim(*keep[0]); self.cax.set_ylim(*keep[1])
        # Key on the colourbar: adding it changes the margins, so the first
        # draw after it appears must re-solve the layout.
        self._layout(self.cfig, "cbar" if self._cbar is not None else "")
        self.ccanvas.draw_idle()

        self._crack_step = step
        self._crack_path = path
        snapshots = vtk_snapshots(self._vtk_dir())
        if snapshots:
            self._sync_crack_navigator(snapshots, step)
        n_dam = int((phi >= iso).sum())
        self.crack_stats.set(
            f"step {step}   max phi = {phi.max():.3f}   "
            f"nodes above {iso:g}: {n_dam}   {os.path.basename(path)}")

    # Any 1D physical group that names a pre-existing flaw. Matched loosely so
    # it works across specimens without configuration: the STEP-built meshes
    # use lowercase "crack", the spec-built ones "Crack" or "NotchFaces".
    _NOTCH_RE = re.compile(r"crack|notch", re.IGNORECASE)

    def _notch_segments(self):
        """Line segments of the notch, from the mesh the solver was given.

        Deliberately NOT from the VTK: the snapshot carries the field, not the
        named groups. read_mesh_for_plot already extracts every named 1D group
        and is cached on the file's mtime, so this is free after the first call
        -- and the Mesh tab has usually paid for it already.
        """
        path = self.vars["mesh_path"].get().strip()
        if not path:
            return []
        full = path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)
        # Never make the crack render wait on gmsh.  The Mesh tab/background
        # loader populates this cache; until then the field is still useful and
        # the notch overlay appears on the next refresh.
        hit = _MESH_CACHE.get(full)
        try:
            current_mtime = os.path.getmtime(full)
        except OSError:
            return []
        if not hit or hit[0] != current_mtime:
            return []
        m = hit[1]
        segs = []
        for name, s in m.get("edges", {}).items():
            # A crack MOUTH/TIP is a 0D group and never lands here; the mouth
            # group would only be two coincident points anyway.
            if self._NOTCH_RE.search(name):
                segs.extend(s)
        return segs

    def fit_crack(self):
        # Force a fitted redraw of the CURRENT historical state.  Resetting the
        # view must not unexpectedly jump an intentionally paused browser back
        # to the newest snapshot.
        snapshots = vtk_snapshots(self._vtk_dir())
        steps = [item[0] for item in snapshots]
        index = steps.index(self._crack_step) if self._crack_step in steps else None
        self._crack_step = -1        # makes _render_crack discard saved limits
        self._crack_path = ""
        self.draw_crack(force=True, target_index=index)

    def _maybe_follow_crack(self):
        """Redraw only when a NEW snapshot exists and the tab is visible.

        Both guards matter: snapshots appear every vtk_every steps (tens of
        seconds apart), and redrawing a tab nobody is looking at is pure waste.
        """
        if not self.follow_crack.get():
            return
        try:
            if self.nb.select() != str(self.crack_tab):
                return
        except Exception:
            return
        _, step = newest_vtk(self._vtk_dir())
        if step >= 0 and step != self._crack_step:
            self.draw_crack()

    def _mesh_wheel_zoom(self, event):
        """Zoom about the pointer: the point under the cursor stays put."""
        if event.inaxes is not self.max_ or event.xdata is None:
            return
        scale = 0.82 if event.button == "up" else 1.0 / 0.82
        x, y = event.xdata, event.ydata
        x0, x1 = self.max_.get_xlim()
        y0, y1 = self.max_.get_ylim()
        self.max_.set_xlim(x - (x - x0) * scale, x + (x1 - x) * scale)
        self.max_.set_ylim(y - (y - y0) * scale, y + (y1 - y) * scale)
        self.mcanvas.draw_idle()
        # Re-cull to the new view, so zooming in swaps the size map for the
        # real wireframe. Debounced: a wheel spin is many events, and each
        # redraw rebuilds a collection.
        if self._zoom_job is not None:
            try:
                self.root.after_cancel(self._zoom_job)
            except Exception:
                pass
        self._zoom_job = self.root.after(
            220, lambda: self.draw_mesh(preserve_view=True))

    def fit_mesh(self):
        """Back to the whole specimen."""
        if not self.mesh_bbox:
            return
        x0, y0, x1, y1 = self.mesh_bbox
        pad = 0.03 * max(x1 - x0, y1 - y0, 1e-9)
        self.max_.set_xlim(x0 - pad, x1 + pad)
        self.max_.set_ylim(y0 - pad, y1 + pad)
        self.mcanvas.draw_idle()

    def draw_mesh(self, preserve_view: bool = False):
        """Render the .msh currently selected, with BC groups picked out."""
        path = self.vars["mesh_path"].get().strip()
        full = path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)
        keep = None
        if preserve_view and self.mesh_bbox is not None:
            keep = (self.max_.get_xlim(), self.max_.get_ylim())
        else:
            self.mesh_stats.set("reading...")
        self._mesh_request += 1
        request = self._mesh_request
        threading.Thread(target=self._read_mesh_worker,
                         args=(request, full, keep), daemon=True).start()

    def _read_mesh_worker(self, request, full, keep):
        try:
            with self._io_lock:
                mesh = read_mesh_for_plot(full)  # cached unless the file changed
            error = ""
        except Exception as exc:
            mesh, error = {}, str(exc)
        self.io_q.put((self._mesh_loaded, (request, mesh, keep, error)))

    def _mesh_loaded(self, request, m, keep, error):
        if request != self._mesh_request:
            return                         # a newer selection superseded this one
        if error:
            self.mesh_stats.set(f"could not read mesh: {error}")
            return
        self._render_mesh(m, keep)

    def _render_mesh(self, m, keep):
        """Render already-loaded mesh data; this is the Tk-only half."""
        self._mesh_dirty = False

        self.max_.clear()
        style_axes(self.mfig, self.max_)
        self.max_.set_facecolor(T["mesh_face"])
        self.max_.grid(False)          # a grid behind a wireframe is just noise
        if m.get("too_big"):
            self.mesh_stats.set(
                f"{m['nelem']:,} elements - too many to draw")
            self.max_.text(0.5, 0.5,
                           f"{m['nelem']:,} elements\ntoo many to draw quickly",
                           ha="center", va="center", color=T["muted"],
                           transform=self.max_.transAxes)
            self.mcanvas.draw_idle()
            return
        # len(), not truthiness: polys is a numpy array now and `not array`
        # raises "truth value of an array is ambiguous".
        if len(m["polys"]) == 0:
            self.mesh_stats.set("nothing to draw")
            self.max_.text(0.5, 0.5, "no mesh loaded\n(pick a .msh on the left)",
                           ha="center", va="center", color=T["muted"],
                           transform=self.max_.transAxes)
            self.mcanvas.draw_idle()
            return

        # ---- what to draw at this element count / zoom -------------------
        #
        # Drawing 350k elements as a wireframe is pointless as well as slow:
        # on a ~700 px plot each element is a fraction of a pixel, so it
        # renders as a solid grey rectangle. Two modes instead:
        #
        #   many elements in view  -> ELEMENT SIZE MAP. Bin the domain and
        #                             colour by the finest element in each bin.
        #                             O(n) in numpy, instant, and it answers
        #                             the actual question -- is the refinement
        #                             where I asked for it.
        #   few elements in view   -> the real wireframe, culled to the view.
        #
        # Culling is what makes a 350k mesh usable: zoomed in, only the
        # elements on screen are drawn, so detail costs the same as a small
        # mesh.
        note = ""
        polys, cent, size = m["polys"], m.get("cent"), m.get("size")
        vis = None
        if keep is not None and cent is not None:
            (vx0, vx1), (vy0, vy1) = keep
            vis = ((cent[:, 0] >= vx0) & (cent[:, 0] <= vx1) &
                   (cent[:, 1] >= vy0) & (cent[:, 1] <= vy1))
            n_vis = int(vis.sum())
        else:
            n_vis = len(polys)

        mode = self.mesh_view.get()
        as_map = (cent is not None
                  and (mode == "size map"
                       or (mode == "auto" and n_vis > MESH_DETAIL_LIMIT)))
        if not as_map and n_vis > MESH_DETAIL_LIMIT:
            # Forced wireframe on a mesh where it is genuinely slow. Say so
            # before the window locks up, rather than after.
            self.mesh_stats.set(f"drawing {n_vis:,} elements...")
            self.root.update_idletasks()

        if as_map:
            x0b, y0b, x1b, y1b = m["bbox"]
            if keep is not None:
                (x0b, x1b), (y0b, y1b) = keep
            nb = 420
            sel = vis if vis is not None else slice(None)
            cx, cy, cs = cent[sel, 0], cent[sel, 1], size[sel]
            ix = np.clip(((cx - x0b) / max(x1b - x0b, 1e-12) * nb).astype(int),
                         0, nb - 1)
            iy = np.clip(((cy - y0b) / max(y1b - y0b, 1e-12) * nb).astype(int),
                         0, nb - 1)
            grid = np.full((nb, nb), np.nan)
            flat = iy * nb + ix
            order = np.argsort(-cs)            # finest written last => wins
            np.put(grid, flat[order], cs[order])

            # Fill empty bins from their neighbours. Element density varies by
            # 100x across a refined mesh, so at any binning fine enough to show
            # the refined zone the COARSE region has far fewer elements than
            # bins and comes out as speckle on the background -- unreadable
            # exactly where you want to confirm "this part is coarse, and
            # that's fine". A few nearest-neighbour passes close it up; the
            # values are still measured sizes, only spread to bins that held
            # no centroid.
            for _ in range(8):
                holes = np.isnan(grid)
                if not holes.any():
                    break
                filled = grid
                for sh, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
                    nbr = np.roll(grid, sh, axis=axis)
                    take = holes & ~np.isnan(nbr)
                    if take.any():
                        filled = np.where(take, nbr, filled)
                if filled is grid:
                    break
                grid = filled

            self.max_.imshow(grid, origin="lower", extent=(x0b, x1b, y0b, y1b),
                             cmap="viridis_r", interpolation="nearest",
                             aspect="auto", zorder=0)
            note = (f"  |  size map, {n_vis:,} in view"
                    + ("" if mode == "size map"
                       else " - zoom in, or view=elements, for the wireframe"))
        else:
            shown = polys[vis] if vis is not None else polys
            self.max_.add_collection(PolyCollection(
                shown, facecolors=T["mesh_face"], edgecolors=T["mesh_edge"],
                linewidths=0.45))
            if vis is not None and n_vis < len(polys):
                note = f"  |  {n_vis:,} of {len(polys):,} elements in view"
            elif n_vis:
                note = f"  |  {n_vis:,} elements drawn"

        wanted = {b["group"] for b in self.bctable.get() if b.get("group")}
        i = 0
        for name, segs in sorted(m["edges"].items()):
            hot = name in wanted
            if self.show_bc_only.get() and not hot:
                self.max_.add_collection(LineCollection(
                    segs, colors=T["muted"], linewidths=0.8, alpha=0.35))
                continue
            colour = self.HILITE[i % len(self.HILITE)] if hot else T["muted"]
            self.max_.add_collection(LineCollection(
                segs, colors=colour, linewidths=2.2 if hot else 0.9,
                label=name if hot else None, alpha=1.0 if hot else 0.5))
            if hot:
                i += 1
        for name, pts in sorted(m["points"].items()):
            hot = name in wanted
            self.max_.plot([p[0] for p in pts], [p[1] for p in pts], "o",
                           ms=8 if hot else 4,
                           color=self.HILITE[i % len(self.HILITE)] if hot
                           else T["muted"],
                           label=name if hot else None, zorder=5)
            if hot:
                i += 1

        self.mesh_bbox = m["bbox"]
        if keep:
            self.max_.set_xlim(*keep[0])
            self.max_.set_ylim(*keep[1])
        else:
            x0, y0, x1, y1 = m["bbox"]
            pad = 0.03 * max(x1 - x0, y1 - y0, 1e-9)
            self.max_.set_xlim(x0 - pad, x1 + pad)
            self.max_.set_ylim(y0 - pad, y1 + pad)
        # Equal aspect or the specimen is silently distorted -- an 8x2 beam
        # stretched to fill the pane looks like a different problem.
        self.max_.set_aspect("equal", adjustable="box")
        handles, labels = self.max_.get_legend_handles_labels()
        if handles:
            leg = self.max_.legend(loc="upper right", fontsize=8,
                                   facecolor=T["panel"], edgecolor=T["grid"])
            for t in leg.get_texts():
                t.set_color(T["fg"])
        rng = ""
        if m.get("size") is not None and len(m["size"]):
            rng = f", h {m['size'].min():.4g}..{m['size'].max():.4g}"
        self.mesh_stats.set(f"{m['nnode']:,} nodes, {m['nelem']:,} elements"
                            f"{rng}, "
                            f"{len(m['edges']) + len(m['points'])} groups{note}")
        self._layout(self.mfig)
        self.mcanvas.draw_idle()

    # -- form <-> dict --------------------------------------------------
    def set_form(self, f: dict):
        for key, v in self.vars.items():
            if key == "ntype_label":
                v.set(next(k for k, n in NTYPES.items() if n == int(f["ntype"])))
            elif key in f:
                v.set(f[key] if isinstance(v, tk.BooleanVar) else str(f[key]))
        self.bctable.set(f.get("bcs", []))
        # An imported config can switch step_mode or mesh source, so the field
        # set has to follow it.
        self._apply_visibility()
        self._scan_mesh()

    def get_form(self) -> dict:
        f = dict(DEFAULTS)
        for key, v in self.vars.items():
            if key == "ntype_label":
                f["ntype"] = NTYPES.get(v.get(), 2)
            else:
                f[key] = v.get()
        # Numeric fields arrive as strings from Entry widgets.
        for key in ("E", "nu", "Gc", "l0", "k", "gamma_tol", "tol_rel",
                    "tol_abs", "du", "du_coarse", "u_switch", "du_fine",
                    "u_switch2", "du_final"):
            try:
                f[key] = float(str(f[key]).strip())
            except ValueError:
                raise ValueError(f"'{key}' is not a number: {f[key]!r}")
        for key in ("ngaus", "max_staggered", "max_iter", "N_steps",
                    "max_subdivs", "vtk_every", "profile_every"):
            try:
                f[key] = int(float(str(f[key]).strip()))
            except ValueError:
                raise ValueError(f"'{key}' is not an integer: {f[key]!r}")
        f["bcs"] = self.bctable.get()
        return f

    # -- actions --------------------------------------------------------
    def _pick_exe(self):
        p = filedialog.askopenfilename(initialdir=PROJECT_DIR,
                                       title="Select phasefield_1")
        if p:
            self.exe_var.set(p)

    def _pick_mesh(self):
        p = filedialog.askopenfilename(
            initialdir=PROJECT_DIR, title="Select a mesh",
            filetypes=[("Gmsh mesh", "*.msh *.geo"), ("All files", "*.*")])
        if p:
            try:
                p = os.path.relpath(p, PROJECT_DIR)
            except ValueError:
                pass                       # different drive: keep it absolute
            self.vars["mesh_path"].set(p)
            self.vars["mesh_source"].set("file")
            self._scan_mesh()
            self.draw_mesh()

    def _scan_mesh(self, blocking=False):
        """Read physical groups, normally away from Tk's event thread."""
        path = self.vars["mesh_path"].get().strip()
        full = path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)
        self._group_request += 1
        request = self._group_request
        if not path:
            self._mesh_groups_loaded(request, path, {}, "")
            return
        if blocking:
            # Launch validation needs the answer before it can safely start.
            with self._io_lock:
                groups = groups_in_mesh(full)
            self._mesh_groups_loaded(request, path, groups, "")
            return
        self.mesh_info.set("reading physical groups...")
        threading.Thread(target=self._scan_mesh_worker,
                         args=(request, path, full), daemon=True).start()

    def _scan_mesh_worker(self, request, path, full):
        try:
            with self._io_lock:
                groups = groups_in_mesh(full)
            error = ""
        except Exception as exc:
            groups, error = {}, str(exc)
        self.io_q.put((self._mesh_groups_loaded, (request, path, groups, error)))

    def _mesh_groups_loaded(self, request, path, groups, error):
        if request != self._group_request:
            return
        self.mesh_groups = groups
        self._mesh_dirty = True
        if self.mesh_groups:
            dims = {0: "point", 1: "edge", 2: "surface"}
            names = ", ".join(f"{n} ({dims.get(d, d)})"
                              for n, d in sorted(self.mesh_groups.items(),
                                                 key=lambda kv: (kv[1], kv[0])))
            self.mesh_info.set(f"{len(self.mesh_groups)} groups: {names}")
        elif path:
            self.mesh_info.set(error or
                               "no groups read (file missing, or gmsh not installed)")
        else:
            self.mesh_info.set("")
        self.bctable.refresh_names()
        self._schedule_validation()

    def import_config(self, path: str = ""):
        if not path:
            path = filedialog.askopenfilename(
                initialdir=PROJECT_DIR, title="Import a run config",
                filetypes=[("TOML config", "*.toml"), ("All files", "*.*")])
            if not path:
                return
        try:
            self.set_form(config_to_form(_toml_load(path)))
            self._remember_config(path)
            self._append_log(f"[gui] imported {os.path.basename(path)}\n")
            self._refresh_warnings()
        except Exception as exc:
            messagebox.showerror("Import failed", f"{path}\n\n{exc}")

    def save_config(self):
        try:
            text = form_to_toml(self.get_form())
        except ValueError as exc:
            messagebox.showerror("Bad value", str(exc))
            return
        p = filedialog.asksaveasfilename(
            initialdir=PROJECT_DIR, defaultextension=".toml",
            initialfile=self.vars["base_name"].get() + ".toml",
            filetypes=[("TOML config", "*.toml")])
        if p:
            open(p, "w", encoding="utf-8").write(text)
            self._remember_config(p)
            self._append_log(f"[gui] saved {p}\n")

    def _schedule_validation(self):
        """Coalesce edits so validation runs once after the user pauses typing."""
        if self._validation_job is not None:
            try:
                self.root.after_cancel(self._validation_job)
            except tk.TclError:
                pass
        self._validation_job = self.root.after(250, self._refresh_warnings)

    def _raw_field_errors(self):
        """Find conversion errors without aborting on the first bad field."""
        errors = {}
        float_keys = ("E", "nu", "Gc", "l0", "k", "gamma_tol", "tol_rel",
                      "tol_abs", "du", "du_coarse", "u_switch", "du_fine",
                      "u_switch2", "du_final")
        int_keys = ("ngaus", "max_staggered", "max_iter", "N_steps",
                    "max_subdivs", "vtk_every", "profile_every")
        for key in float_keys:
            try:
                float(self.vars[key].get().strip())
            except (KeyError, ValueError):
                errors[key] = "Enter a number"
        for key in int_keys:
            try:
                value = float(self.vars[key].get().strip())
                if not value.is_integer():
                    raise ValueError
            except (KeyError, ValueError):
                errors[key] = "Enter a whole number"
        return errors

    def _logical_field_errors(self, f):
        """Map blocking configuration rules back to the controls that own them."""
        errors = {}
        if not f["base_name"].strip():
            errors["base_name"] = "Required"
        if f["mesh_source"] == "file":
            path = f["mesh_path"].strip()
            full = path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)
            if not path:
                errors["mesh_path"] = "Choose a mesh file"
            elif not os.path.isfile(full):
                errors["mesh_path"] = "File not found"
        for key in ("E", "Gc", "l0"):
            if float(f[key]) <= 0:
                errors[key] = "Must be positive"
        if not 0.0 < float(f["nu"]) < 0.5:
            errors["nu"] = "Use a value from 0 to 0.5"
        if f["step_mode"] == "three_stage":
            if float(f["du_final"]) <= 0:
                errors["du_final"] = "Must be positive"
            if float(f["u_switch2"]) <= float(f["u_switch"]):
                errors["u_switch2"] = "Must exceed the first switch"
        if (f["step_mode"] in ("two_stage", "three_stage") and
                float(f["du_fine"]) > float(f["du_coarse"])):
            errors["du_fine"] = "Must not exceed coarse step"
        return errors

    def _set_inline_errors(self, errors):
        """Apply red outlines and short explanations beside invalid controls."""
        for key, widget in self._field_widgets.items():
            normal = "TCombobox" if isinstance(widget, ttk.Combobox) else "TEntry"
            invalid = "Invalid.TCombobox" if normal == "TCombobox" else "Invalid.TEntry"
            try:
                widget.configure(style=invalid if key in errors else normal)
            except tk.TclError:
                pass
            feedback = self._field_feedback.get(key)
            if feedback:
                label, normal_text = feedback
                label.configure(text=errors.get(key, normal_text),
                                foreground=T["err"] if key in errors else T["muted"])

        # Do not leave the offending control hidden inside a collapsed group.
        for key in errors:
            title = self._field_sections.get(key)
            if title:
                self._set_section_expanded(title, True)

    def _update_run_summary(self, f=None):
        if f is None:
            self.summary_var.set("Complete the highlighted fields to calculate the run.")
            self._refresh_section_summaries()
            return
        mesh = (os.path.basename(f["mesh_path"]) if f["mesh_source"] == "file"
                else "built-in mesh")
        groups = (f" · {len(self.mesh_groups)} physical groups"
                  if self.mesh_groups else "")
        model = (f"{next((k for k, v in NTYPES.items() if v == f['ntype']), f['ntype'])}"
                 f" · {f['energy_split']} split · {f['scheme']}")

        loads = []
        for b in f["bcs"]:
            vals = []
            if b.get("kind") == "displacement":
                if b.get("fix_x") and float(b.get("vx", 0)):
                    vals.append(f"ux {float(b['vx']):g} mm")
                if b.get("fix_y") and float(b.get("vy", 0)):
                    vals.append(f"uy {float(b['vy']):g} mm")
            elif float(b.get("vx", 0)) or float(b.get("vy", 0)):
                vals.append(f"{b['kind']} ({float(b['vx']):g}, {float(b['vy']):g})")
            if vals and b.get("group"):
                loads.append(f"{b['group']}: {', '.join(vals)}")
        load_text = loads[0] + (f" · +{len(loads)-1} more" if len(loads) > 1 else "") \
            if loads else "no non-zero load"

        plan = planned_steps(f)
        plan_text = f"{plan[0]:,} planned steps" if plan else "step count unknown"
        out = os.path.join(f["output_dir"] or ".", f["base_name"] or "run")
        self.summary_var.set(
            f"{mesh}{groups} · {model} · {plan_text}\n"
            f"{load_text} · VTK every {f['vtk_every']} · {out}")
        self._refresh_section_summaries()

    def _refresh_warnings(self):
        self._validation_job = None
        self.warnbox.delete("1.0", "end")
        raw_errors = self._raw_field_errors()
        if raw_errors:
            self._set_inline_errors(raw_errors)
            self._update_run_summary()
            self.validation_state.set(
                f"{len(raw_errors)} field{'s' if len(raw_errors) != 1 else ''} need attention")
            self.validation_label.configure(style="Warning.TLabel")
            self.warnbox.insert("end", "\n".join(
                f"{key}: {message}" for key, message in raw_errors.items()))
            if not self.warnbox.winfo_manager():
                self.warnbox.pack(fill="x", padx=px(6), pady=(px(4), px(8)))
            self.progress_txt.set("")
            return
        try:
            f = self.get_form()
        except ValueError as exc:
            self.warnbox.insert("end", str(exc))
            self.validation_state.set("Configuration needs attention")
            self.validation_label.configure(style="Warning.TLabel")
            if not self.warnbox.winfo_manager():
                self.warnbox.pack(fill="x", padx=px(6), pady=(px(4), px(8)))
            return
        field_errors = self._logical_field_errors(f)
        self._set_inline_errors(field_errors)
        self._update_run_summary(f)
        msgs = ([("PROBLEM: " + m) for m in validate_form(f)]
                + [("PROBLEM: " + m)
                   for m in check_group_dims(f["bcs"], self.mesh_groups)]
                + warn_form(f))
        if msgs:
            self.validation_state.set(
                f"{len(msgs)} configuration note{'s' if len(msgs) != 1 else ''}")
            self.validation_label.configure(style="Warning.TLabel")
            self.warnbox.insert("end", "\n\n".join(msgs))
            if not self.warnbox.winfo_manager():
                self.warnbox.pack(fill="x", padx=px(6), pady=(px(4), px(8)))
        else:
            self.validation_state.set("Configuration ready")
            self.validation_label.configure(style="Good.TLabel")
            if self.warnbox.winfo_manager():
                self.warnbox.pack_forget()
        self._show_plan(f)

    def _update_progress(self):
        """Progress and a stage-aware ETA based on actual displacement.

        Accepted step numbers are not progress: adaptive half-stepping can add
        accepted increments.  Reading the latest already-cached CSV point lets
        the bar continue to represent the requested load schedule accurately.
        """
        if self._last_step <= 0:
            return
        total = self._plan_total
        done = self._last_step
        rate = (sum(self._step_times) / len(self._step_times)
                if self._step_times else 0.0)
        if rate:
            self._last_rate = rate

        staged = load_stages(self._run_form or {})
        u_now = None
        if staged:
            u, _F, _phi, _conv, _labels = read_curve(self.csv_path)
            if u:
                u_now = abs(u[-1])

        if staged and u_now is not None:
            stages, _u_ref = staged
            remaining = remaining_stage_steps(stages, u_now)
            nominal_left = sum(remaining)
            nominal_done = max(0, total - nominal_left)
            frac = min(nominal_done / total, 1.0) if total else 0.0
            self.progress["value"] = int(1000 * frac)
            active = next((i for i, n in enumerate(remaining) if n),
                          len(stages) - 1)
            bits = [f"step {done:,}", f"{100.0 * frac:.1f}%",
                    f"stage {active + 1}/{len(stages)} {stages[active][0]}"]
        else:
            remaining = None
            frac = min(done / total, 1.0) if total else 0.0
            self.progress["value"] = int(1000 * frac)
            bits = [f"step {done:,}" + (f" / {total:,}" if total else "")]

        if rate:
            bits.append(f"{rate:.2f} s/step")
        if remaining is not None and nominal_left and rate:
            eta = 0.0
            for i, count in enumerate(remaining):
                if not count:
                    continue
                samples = self._stage_times[i]
                stage_rate = (sum(samples) / len(samples) if samples
                              else self._last_stage_rates[i] or rate)
                eta += count * stage_rate
            bits.append(f"ETA ~{fmt_duration(eta)}")
        elif total and rate and done < total:
            bits.append(f"ETA ~{fmt_duration((total - done) * rate)}")
        self.progress_txt.set("   ".join(bits))

    def _record_step_timings(self, samples):
        """Associate accepted-step wall times with their displacement stage."""
        if not samples:
            return
        staged = load_stages(self._run_form or {})
        if not staged:
            return
        stages, _u_ref = staged
        u, _F, _phi, _conv, _labels = read_curve(self.csv_path)
        for step, seconds in samples:
            # The CSV begins with step 0, so accepted step N is at index N.
            if step >= len(u):
                continue
            i = stage_for_displacement(stages, u[step])
            bucket = self._stage_times[i]
            bucket.append(seconds)
            del bucket[:-RATE_WINDOW]
            self._last_stage_rates[i] = sum(bucket) / len(bucket)

    def _show_plan(self, f=None):
        """Idle text: what this config will cost, before committing to it.

        The step count is spread over four [run] fields plus a BC value, so it
        is not something you can read off the form -- which is exactly why a
        schedule of 8,000 steps could sit here unnoticed.
        """
        if self.proc is not None:
            return                       # a run owns the label; leave it alone
        try:
            f = f if f is not None else self.get_form()
        except ValueError:
            self.progress_txt.set("")
            return
        plan = planned_steps(f)
        if not plan:
            self.progress_txt.set("no prescribed displacement - step count unknown")
            self.progress["value"] = 0
            return
        total, n_c, n_f, u_ref = plan
        self.progress["value"] = 0
        mode = f.get("step_mode")
        staged = load_stages(f)
        if staged:
            counts = [s[4] for s in staged[0]]
            detail = "  (" + " + ".join(
                f"{n:,} {stage[0]}" for n, stage in zip(counts, staged[0])) + ")"
        else:
            detail = "  (uniform)"
        rate = self._sec_per_step_hint(total)
        self.progress_txt.set(f"plan: {total:,} steps{detail}"
                              + (f"   ~{rate}" if rate else ""))

    def _sec_per_step_hint(self, total):
        """Wall-time guess from the LAST run's measured rate, if there was one.

        Deliberately not a guess from element count: seconds per step depends on
        the solver settings and how many staggered sweeps the crack needs, which
        no formula here can predict. Better to say nothing than to invent it.
        """
        if not self._last_rate:
            return ""
        return f"{fmt_duration(total * self._last_rate)} at last run's rate"

    # -- peak detection -------------------------------------------------
    def _schedule_info(self):
        """(two_stage, u_switch, u_ref) for the CURRENT form, or (False, 0, 0)."""
        try:
            f = self.get_form()
            if f.get("step_mode") != "two_stage":
                return False, 0.0, 0.0
            return True, float(f["u_switch"]), u_ref_of(f)
        except (ValueError, KeyError, TypeError):
            return False, 0.0, 0.0

    def _annotate_peak(self, u, F):
        """Mark the peak load and say whether it was resolved finely enough.

        Only meaningful under two_stage: with a uniform schedule there is no
        coarse/fine boundary for the peak to fall on the wrong side of, so the
        u_switch line and the warning are both suppressed.
        """
        for artist in self._curve_annotations:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._curve_annotations = []

        two_stage, u_sw, _ = self._schedule_info()
        i = peak_index(F)

        # The u_switch guide is drawn ONLY once the run has reached it.
        #
        # axvline lives in DATA coordinates, so a line at u_switch while the
        # curve has covered a hundredth of that makes matplotlib autoscale the
        # x-axis out to include it -- the actual response is squashed into a
        # sliver at the origin and you cannot see what the current step is
        # doing. That is worst in the early steps, which is when the live plot
        # is most worth watching.
        #
        # Once the curve arrives, the line is inside the data range and costs
        # nothing. Before that it would only be marking empty space.
        if two_stage and u_sw > 0 and u and abs(u[-1]) >= u_sw:
            # u is SIGNED (load_factor * u_full, and u_full is -0.12 here)
            # while u_switch is a magnitude, so the guide line has to take the
            # sign of the curve or it lands off-screen on the wrong side.
            x_sw = math.copysign(u_sw, u[-1] if u[-1] else 1.0)
            self._curve_annotations.append(
                self.ax.axvline(x_sw, color=T["muted"], lw=1.0, ls="--", zorder=0))
            self._curve_annotations.append(
                self.ax.annotate("u_switch", xy=(x_sw, 0),
                                 xycoords=("data", "axes fraction"),
                                 xytext=(3, 4), textcoords="offset points",
                                 color=T["muted"], fontsize=8))

        if i is None:
            return
        up, Fp = u[i], F[i]
        coarse = two_stage and u_sw > 0 and abs(up) < u_sw
        col = T["err"] if coarse else T["warn"]
        peak_artist, = self.ax.plot([up], [Fp], marker="v", ms=9, color=col,
                                    linestyle="none", zorder=4)
        self._curve_annotations.append(peak_artist)
        self._curve_annotations.append(
            self.ax.annotate(f"peak {abs(Fp):.4g}\n@ u = {abs(up):.4g}",
                             xy=(up, Fp), xytext=(6, -22),
                             textcoords="offset points", fontsize=8, color=col))

        # Report once per run, not once per redraw (twice a second).
        if self._peak_reported:
            return
        self._peak_reported = True
        if coarse:
            self._append_log(
                f"[gui] WARNING: peak load reached at u = {abs(up):.5g}, which is "
                f"BEFORE u_switch = {u_sw:g}.\n"
                f"       The peak was resolved with du_coarse, so its value is "
                f"unreliable and any\n"
                f"       snap-back may have been stepped straight over. Lower "
                f"u_switch below {abs(up):.3g}\n"
                f"       and re-run if the peak load matters.\n")
        else:
            self._append_log(
                f"[gui] peak load {abs(Fp):.5g} at u = {abs(up):.5g} "
                f"(inside the fine stage; u_switch = {u_sw:g}). "
                f"Resolution OK.\n")

        # Once the response has decayed, u_max is knowable rather than guessed.
        if abs(F[-1]) < 0.05 * abs(Fp):
            self._append_log(
                f"[gui] the response has fallen below 5% of peak by u = "
                f"{abs(u[-1]):.5g};\n"
                f"       u_max beyond about {abs(u[-1]) * 1.2:.3g} is spent on an "
                f"already-broken specimen.\n")

    def _open_output(self):
        try:
            f = self.get_form()
        except ValueError:
            return
        d = os.path.join(PROJECT_DIR, f["output_dir"] or ".", f["base_name"])
        if os.path.isdir(d):
            os.startfile(d) if os.name == "nt" else subprocess.run(["xdg-open", d])
        else:
            messagebox.showinfo("Not there yet", f"No output folder yet:\n{d}")

    # -- launching ------------------------------------------------------
    def _run_tool(self, cmd, label, kind, on_success=None):
        """Run any command as a subprocess, streaming into the log pane.

        Meshing runs OUT OF PROCESS deliberately. The GUI already calls gmsh
        in-process to draw the mesh, which is fine at ~140 ms -- but meshing
        can take minutes and gmsh keeps global state, so a failed mesh could
        leave the library wedged and break the display too. A subprocess
        cannot do that, and the progress log comes for free.
        """
        if self.proc is not None:
            messagebox.showwarning("Busy", "Something is already running.")
            return False
        self._append_log(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        try:
            self.proc = subprocess.Popen(
                [str(c) for c in cmd], cwd=PROJECT_DIR, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace",
                env=child_env(), **process_group_kwargs())
        except OSError as exc:
            messagebox.showerror("Could not start", str(exc))
            self.proc = None
            return False
        self.proc_kind = kind
        self._on_success = on_success
        self.start_time = time.time()
        self._set_state(f"{label}...", "busy")
        self.elapsed_txt.set("0s")
        self.curve_status.set("")
        for b in (self.btn_run, self.btn_check, self.btn_gen):
            b.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        return True

    def _launch(self, extra, label, check_only=False):
        if self.proc is not None:
            messagebox.showwarning("Busy", "Something is already running.")
            return
        try:
            f = self.get_form()
        except ValueError as exc:
            messagebox.showerror("Bad value", str(exc))
            return
        self._scan_mesh(blocking=True)
        self._refresh_warnings()

        problems = validate_form(f) + check_group_dims(f["bcs"], self.mesh_groups)
        if problems:
            messagebox.showerror("Fix these first", "\n\n".join(problems))
            return
        warns = warn_form(f)
        if warns and not check_only:
            if not messagebox.askyesno(
                    "Worth a look",
                    "\n\n".join(warns) + "\n\nRun anyway?"):
                return

        exe = self.exe_var.get().strip()
        if not exe or not os.path.isfile(exe):
            messagebox.showerror("Solver not found",
                                 "Point me at phasefield_1.exe.\n\n"
                                 "Not built yet?   cmake --build build")
            return

        # Is another window already writing into this run folder? Two solves in
        # parallel are fine and useful -- but only into DIFFERENT base_names.
        # Sharing one means both write the same CSV and the same numbered VTKs,
        # and the check below would delete a live run's CSV out from under it.
        # Checked BEFORE anything is written or removed.
        self.csv_path = os.path.join(PROJECT_DIR, csv_path_for(f))
        reset_curve_cache(self.csv_path)
        if not check_only and os.path.isfile(self.csv_path):
            age = time.time() - os.path.getmtime(self.csv_path)
            if age < LIVE_RUN_SECONDS:
                if not messagebox.askyesno(
                        "Already running?",
                        f"{f['base_name']} was written {age:.0f}s ago, so another "
                        "run is probably still using it.\n\n"
                        "Two runs sharing one base_name overwrite each other's "
                        "CSV and VTK snapshots, and both curves will be wrong.\n\n"
                        "Give this run its own base_name instead.\n\n"
                        "Start anyway?"):
                    return

        cfg_path = os.path.join(PROJECT_DIR, GENERATED)
        open(cfg_path, "w", encoding="utf-8").write(form_to_toml(f))

        self.last_rows = -1
        # A new run may use a different mesh, so the cached triangulation and
        # the last-seen step must not carry over -- otherwise phi from the new
        # run gets drawn on the old geometry, or silently rejected for having
        # the wrong length.
        self._tri = None
        self._tri_npoin = 0
        self._crack_step = -1
        self._crack_path = ""
        self._crack_snapshots = []
        self._sync_crack_navigator([])
        self._vtk_sizes.clear()

        # Progress: a mesh-only check has no load steps, so leave the bar blank
        # rather than showing a plan that will never advance.
        plan = planned_steps(f)
        self._plan_total = 0 if check_only else (plan[0] if plan else 0)
        self._step_times = []
        self._stage_times = [[], [], []]
        self._run_form = None if check_only else f
        self._last_step = 0
        self._peak_reported = False
        self.progress["value"] = 0
        self.progress_txt.set(
            "checking mesh..." if check_only
            else (f"0 / {self._plan_total:,} steps" if self._plan_total
                  else "running (step count unknown)"))
        if not check_only and os.path.isfile(self.csv_path):
            # A leftover CSV would be drawn instantly and look like progress.
            try:
                os.remove(self.csv_path)
            except OSError:
                pass
        self.follow_curve.set(True)
        self.curve_hint.set("")
        self.ax.clear(); style_axes(self.fig, self.ax); self.canvas.draw_idle()
        self._curve_artists = None
        self._curve_annotations = []
        self.curve_status.set("")

        cmd = [exe, GENERATED] + extra
        self._append_log(f"\n$ {' '.join(cmd)}\n")
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=PROJECT_DIR, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace",
                env=child_env(), **process_group_kwargs())
        except OSError as exc:
            messagebox.showerror("Could not start", str(exc))
            self.proc = None
            return

        # Keep the exact config beside the results.
        if not check_only:
            outdir = os.path.join(PROJECT_DIR, f["output_dir"] or ".",
                                  f["base_name"])
            try:
                os.makedirs(outdir, exist_ok=True)
                shutil.copyfile(cfg_path, os.path.join(outdir, "config_used.toml"))
            except OSError:
                pass

        self.proc_kind = "check" if check_only else "solve"
        self._on_success = None
        self.start_time = time.time()
        self._set_state(f"{label}...", "busy")
        self.elapsed_txt.set("0s")
        for b in (self.btn_run, self.btn_check, self.btn_gen):
            b.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()

    def _pump(self, proc):
        for line in proc.stdout:
            self.log_q.put(line)
        proc.wait()
        self.log_q.put(None)

    def check_mesh(self):
        self._launch(["--mesh-only", "--no-gui", "--no-preview"],
                     "checking mesh", check_only=True)

    def run(self):
        self._launch(["--no-gui", "--no-preview"], "solving")

    def stop(self):
        if self.proc is None:
            return
        if messagebox.askyesno("Stop", "Terminate the running solver?"):
            proc = self.proc
            self._set_state("stopping...", "busy")
            self.btn_stop.configure(state="disabled")
            threading.Thread(target=self._stop_worker, args=(proc,),
                             daemon=True).start()

    def _stop_worker(self, proc):
        signal_process_tree(proc, hard=False)
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self.log_q.put("[gui] graceful stop timed out; forcing process tree shutdown\n")
            signal_process_tree(proc, hard=True)

    # -- periodic -------------------------------------------------------
    @staticmethod
    def _tag_for(text):
        low = text.lower()
        return ("err" if ("error" in low or "diverged" in low
                          or "non-converged" in low or "failed" in low
                          or "fatal" in low)
                else "warn" if ("warning" in low or low.startswith("[bc]"))
                else "conv" if ("converg" in low or "residual" in low
                                 or "iteration" in low or "stagger" in low)
                else "normal")

    def _append_log(self, text):
        self._append_chunks([(self._tag_for(text), text)])

    def _append_chunks(self, chunks):
        """Insert several (tag, text) pieces as one update.

        One insert per LINE was the thing that made a fast poll interval
        expensive: each one re-wraps the widget and each see("end") forces a
        scroll. Consecutive lines sharing a tag are merged, so a 500-line burst
        is typically three inserts and a single scroll.
        """
        if not chunks:
            return
        for _tag, text in chunks:
            self._log_history.append(text)
            self._log_history_chars += len(text)
        while (self._log_history_chars > LOG_HISTORY_CHARS and
               len(self._log_history) > 1):
            self._log_history_chars -= len(self._log_history.pop(0))
        # Only follow the tail if the view is ALREADY at the bottom. Otherwise
        # scrolling back to read something gets yanked away every 60 ms, which
        # a faster refresh would make unbearable.
        try:
            at_bottom = self.follow_log.get() and self.log.yview()[1] > 0.999
        except Exception:
            at_bottom = True

        merged = []
        for tag, text in chunks:
            if merged and merged[-1][0] == tag:
                merged[-1][1].append(text)
            else:
                merged.append((tag, [text]))
        for tag, parts in merged:
            self.log.insert("end", "".join(parts), tag)

        if int(self.log.index("end-1c").split(".")[0]) > LOG_LINES:
            self.log.delete("1.0", f"end-{LOG_LINES}l")
        if at_bottom:
            self.log.see("end")

    def _follow_log_changed(self):
        if self.follow_log.get():
            self.log.see("end")

    def _apply_log_filter(self):
        wanted = {
            "Normal": "normal", "Warnings": "warn",
            "Convergence": "conv", "Errors": "err",
        }.get(self.log_filter_var.get())
        for tag in ("normal", "warn", "conv", "err"):
            try:
                self.log.tag_configure(tag, elide=bool(wanted and tag != wanted))
            except tk.TclError:
                pass
        if self.follow_log.get():
            self.log.see("end")

    def _find_log(self, direction=1):
        term = self.log_search_var.get()
        if not term:
            return
        self.log.tag_remove("search_hit", "1.0", "end")
        if direction >= 0:
            start = self.log.index("insert +1c")
            idx = self.log.search(term, start, stopindex="end", nocase=True)
            if not idx:
                idx = self.log.search(term, "1.0", stopindex=start, nocase=True)
        else:
            start = self.log.index("insert -1c")
            idx = self.log.search(term, start, stopindex="1.0", nocase=True,
                                  backwards=True)
            if not idx:
                idx = self.log.search(term, "end", stopindex=start, nocase=True,
                                      backwards=True)
        if not idx:
            self.root.bell()
            return
        end = f"{idx}+{len(term)}c"
        self.log.tag_add("search_hit", idx, end)
        self.log.mark_set("insert", end)
        self.log.see(idx)

    def _copy_log(self):
        try:
            text = self.log.get("sel.first", "sel.last")
        except tk.TclError:
            text = self.log.get("1.0", "end-1c")
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _clear_log(self):
        self.log.delete("1.0", "end")
        self._log_history.clear()
        self._log_history_chars = 0

    def _save_log(self):
        path = filedialog.asksaveasfilename(
            parent=self.root, initialdir=PROJECT_DIR,
            initialfile="solver_log.txt", defaultextension=".txt",
            filetypes=[("Text log", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("".join(self._log_history))
        except OSError as exc:
            messagebox.showerror("Could not save log", str(exc))

    def _tick(self):
        if self.proc is not None and self.start_time:
            self.elapsed_txt.set(fmt_duration(time.time() - self.start_time))
        # Background file readers return plain data here.  Every Tk and
        # matplotlib operation remains on the UI thread.
        for _ in range(8):
            try:
                callback, args = self.io_q.get_nowait()
            except queue.Empty:
                break
            if self._closing:
                break
            try:
                callback(*args)
            except Exception as exc:
                self._append_log(f"[gui] background result failed: {exc}\n")

        finished = False
        pending = []
        timing_samples = []
        deadline = time.time() + DRAIN_MS / 1000.0
        while time.time() < deadline:
            try:
                item = self.log_q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                finished = True
                break
            pending.append((self._tag_for(item), item))
            m = _TIME_RE.search(item)
            if m:
                self._last_step = int(m.group(1))
                seconds = float(m.group(2))
                self._step_times.append(seconds)
                del self._step_times[:-RATE_WINDOW]
                timing_samples.append((self._last_step, seconds))
        self._append_chunks(pending)
        if pending and self.proc_kind == "solve":
            self._record_step_timings(timing_samples)
            self._update_progress()

        if self.proc is not None:
            # Throttled independently of the log: see PLOT_MS.
            #
            # Suppressed entirely while a sash is being dragged. This refresh
            # re-reads the whole force-displacement CSV (19 ms at 10.5k rows)
            # and redraws the curve, and it lands every 500 ms -- so during a
            # drag it injects a ~135 ms freeze into an otherwise 8 ms frame,
            # twice a second. That, not the pane geometry, is what still felt
            # like glitching once the render guard was in. It is purely a
            # display update, so deferring it to the end of the drag loses
            # nothing.
            if self.proc_kind == "solve" and not self._dragging and (
                    time.time() - self._last_plot >= PLOT_MS / 1000.0):
                self._last_plot = time.time()
                self._refresh_plot()
                self._maybe_follow_crack()
            if finished:
                code = self.proc.poll()
                elapsed = time.time() - self.start_time
                kind, hook = self.proc_kind, self._on_success
                self.proc = None
                self.proc_kind = ""
                self._on_success = None
                for b in (self.btn_run, self.btn_check, self.btn_gen):
                    b.configure(state="normal")
                self.btn_stop.configure(state="disabled")
                if code == 0:
                    self._set_state("finished", "ok")
                else:
                    self._set_state(f"FAILED (exit {code})", "fail")
                self.elapsed_txt.set(fmt_duration(elapsed))
                self._append_log(f"--- exit code {code} after {elapsed:.1f}s ---\n")
                if code and code != 0:
                    note = explain_exit_code(code)
                    if note:
                        self._append_log(note)
                if kind == "solve":
                    self._refresh_plot(force=True)
                # Only run the follow-on when the tool actually succeeded --
                # loading a mesh that failed to write would show a stale file.
                if kind == "solve" and self._last_step:
                    # Final tally, and note whether it actually got there --
                    # a run that stops early because the crack finished is the
                    # normal outcome, not a failure.
                    short = (self._plan_total and
                             self._last_step < self._plan_total)
                    self.progress_txt.set(
                        f"{self._last_step:,} steps in {fmt_duration(elapsed)}"
                        + (f"  (stopped before the planned "
                           f"{self._plan_total:,})" if short else ""))
                if code == 0 and hook is not None:
                    try:
                        hook()
                    except Exception as exc:
                        self._append_log(f"[gui] post-step failed: {exc}\n")
                if kind != "solve":
                    self._show_plan()

        self.root.after(POLL_MS, self._tick)

    def _refresh_plot(self, force=False):
        u, F, phi, conv, labels = read_curve(self.csv_path)
        if not u or (len(u) == self.last_rows and not force):
            return
        hold_view = self._curve_artists is not None and not self.follow_curve.get()
        held_limits = (self.ax.get_xlim(), self.ax.get_ylim())
        self.last_rows = len(u)

        ok = [(x, y) for x, y, c in zip(u, F, conv) if c != 0]
        bad = [(x, y) for x, y, c in zip(u, F, conv) if c == 0]

        # Converged steps are drawn HOLLOW in the line colour, non-converged
        # SOLID red on top. Same shape, so the eye reads the difference as
        # status rather than as two unrelated series.
        #
        # Thinned, because this config can produce several thousand steps
        # (du_fine = 1e-5 over u_switch..u_max) and a marker per step is both
        # slow to draw at every refresh and unreadable -- the curve turns into
        # a solid bar. NON-CONVERGED STEPS ARE NEVER THINNED: they are the
        # reason to look at the plot at all, and dropping one would hide it.
        stride = max(1, len(ok) // MAX_MARKERS) if ok else 1
        shown = ok[::stride]
        ok_label = "converged" + (f" (every {stride})" if stride > 1 else "")

        # Keep the three line artists and only replace their data.  Clearing
        # the axes rebuilt ticks, spines, legend and every marker twice a
        # second; on long runs that was substantially more work than drawing
        # the newly appended point.
        if self._curve_artists is None or getattr(self, "_curve_labels", None) != labels:
            self.ax.clear(); style_axes(self.fig, self.ax)
            curve, = self.ax.plot([], [], "-", lw=1.4, color=T["line"], zorder=1)
            good, = self.ax.plot([], [], "o", ms=3.5, mfc="none", mew=1.1,
                                 mec=T["line"], linestyle="none", zorder=2)
            failed, = self.ax.plot([], [], "o", ms=4.5, color=T["err"],
                                   linestyle="none", zorder=3)
            self._curve_artists = (curve, good, failed)
            self._curve_labels = labels
            self._curve_legend_sig = None
            self._curve_annotations = []
        curve, good, failed = self._curve_artists
        curve.set_data(u, F)
        good.set_data([p[0] for p in shown], [p[1] for p in shown])
        failed.set_data([p[0] for p in bad], [p[1] for p in bad])
        good.set_label(ok_label if ok else "_nolegend_")
        failed.set_label("not converged" if bad else "_nolegend_")
        self.ax.relim()
        self.ax.autoscale_view()
        if hold_view:
            self.ax.set_xlim(*held_limits[0])
            self.ax.set_ylim(*held_limits[1])

        legend_sig = (bool(ok), bool(bad), stride)
        if legend_sig != self._curve_legend_sig:
            old_legend = self.ax.get_legend()
            if old_legend is not None:
                old_legend.remove()
            if ok or bad:
                leg = self.ax.legend(loc="best", fontsize=8,
                                     facecolor=T["panel"], edgecolor=T["grid"])
                for txt in leg.get_texts():
                    txt.set_color(T["fg"])
            self._curve_legend_sig = legend_sig
        self._annotate_peak(u, F)
        self.ax.set_xlabel(labels[0]); self.ax.set_ylabel(labels[1])
        # The axis labels come from the CSV header (applied_uy vs applied_ux),
        # so they are part of the layout key.
        self._layout(self.fig, f"{labels[0]}|{labels[1]}")
        self.canvas.draw_idle()

        peak = max((abs(x) for x in F), default=0.0)
        self.curve_status.set(
            f"step {len(u) - 1}   u = {u[-1]:.5g}   F = {F[-1]:.5g}   "
            f"peak |F| = {peak:.5g}   max phi = {phi[-1]:.3f}"
            + (f"   [{len(bad)} non-converged]" if bad else ""))

    def _on_close(self):
        if self.proc is not None:
            if not messagebox.askyesno("Quit", "A solve is running. Kill it?"):
                return
            signal_process_tree(self.proc, hard=True)
        self._closing = True
        # Our per-process scratch config. Only ours -- another window may still
        # be running and needs its own.
        try:
            os.remove(os.path.join(PROJECT_DIR, GENERATED))
        except OSError:
            pass
        self.root.destroy()


def main() -> int:
    global T
    args = [a for a in sys.argv[1:]]
    if "--light" in args:
        args.remove("--light")
        T = LIGHT

    # --scale 1.5 / --scale=1.5: override the detected display scale, for a
    # screen that reports its dpi wrongly (common over remote desktop and on
    # Linux) or simply to taste.
    forced = 0.0
    for a in list(args):
        val = None
        if a.startswith("--scale="):
            val = a.split("=", 1)[1]
        elif a == "--scale":
            i = args.index(a)
            val = args[i + 1] if i + 1 < len(args) else None
            if val is not None:
                args.remove(val)
        else:
            continue
        args.remove(a)
        try:
            forced = float(val)
        except (TypeError, ValueError):
            print("--scale needs a number, e.g. --scale 1.5", file=sys.stderr)
            return 2

    initial = args[0] if args else ""
    if not initial:
        for guess in ("config_ambati_3pb.toml", "config.toml"):
            if os.path.isfile(os.path.join(PROJECT_DIR, guess)):
                initial = guess
                break

    # Before the first Tk window exists: awareness is a property of the
    # PROCESS, and Windows fixes it at the point the first window is created.
    enable_dpi_awareness()
    root = tk.Tk()
    init_scaling(root, forced)
    print(SCALE_INFO, end="")
    App(root, initial)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
