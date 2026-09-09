#!/usr/bin/env python3
"""
Set up, run and watch a phase-field solve.

WORKING COPY. run_gui.py beside this file is the untouched original and
stays that way -- every change from here lands in run_gui2.py, so the
two can always be diffed and the original re-run if something regresses.

    pip install matplotlib
    python run_gui2.py
    python run_gui2.py config_ambati_3pb.toml

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
import shutil
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
import matplotlib.tri as mtri
import matplotlib.patheffects as mpe
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


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

# What to draw over the phi contour on the Crack tab.
#
# Everything is CULLED TO THE VISIBLE AXES and skipped entirely above the
# limits below. Drawing 340k element outlines over a contour is not a
# comparison, it is a grey rectangle on top of the answer -- and it is the
# zoomed-in view, at the crack tip, where the question is actually asked.
CRACK_OVERLAYS = ["none", "nodes", "elements", "nodes + elements"]
# Elements in view above which the outlines are dropped (they would be
# sub-pixel), and nodes in view above which the dots are dropped.
CRACK_ELEM_LIMIT = 25_000
CRACK_NODE_LIMIT = 12_000

# Everything writeVTK() puts in a snapshot, and how each one has to be drawn.
#
#   where   "point" values live on nodes and are CONTOURED; "cell" values are
#           one number per element and are drawn FLAT, one colour per element.
#           That distinction is not cosmetic. Output.cpp integrates the
#           stresses over each element's Gauss points and says so: "in ParaView
#           each cell shows a single (flat) value, which matches how the
#           quantity is integrated." Smoothing them onto nodes to get a pretty
#           contour would invent a continuity the solver never claimed.
#   signed  fields that cross zero. They get a DIVERGING map centred on zero,
#           which is the exact opposite of the rule for phi below -- for a
#           quantity that goes negative, the middle of the range IS the
#           meaningful place to put the neutral colour.
CRACK_FIELDS = {
    "phi":       {"block": "SCALARS phi",         "where": "point",
                  "comp": None,  "signed": False, "unit": ""},
    "|u|":       {"block": "VECTORS displacement", "where": "point",
                  "comp": "mag", "signed": False, "unit": "mm"},
    "ux":        {"block": "VECTORS displacement", "where": "point",
                  "comp": 0,     "signed": True,  "unit": "mm"},
    "uy":        {"block": "VECTORS displacement", "where": "point",
                  "comp": 1,     "signed": True,  "unit": "mm"},
    "sigma_xx":  {"block": "SCALARS sigma_xx",    "where": "cell",
                  "comp": None,  "signed": True,  "unit": "MPa"},
    "sigma_yy":  {"block": "SCALARS sigma_yy",    "where": "cell",
                  "comp": None,  "signed": True,  "unit": "MPa"},
    "sigma_xy":  {"block": "SCALARS sigma_xy",    "where": "cell",
                  "comp": None,  "signed": True,  "unit": "MPa"},
    "von_mises": {"block": "SCALARS von_mises",   "where": "cell",
                  "comp": None,  "signed": False, "unit": "MPa"},
}

# Displacement magnification for the Crack view.
#
# "auto" scales so the largest displacement is AUTO_DEFORM_FRACTION of the
# specimen diagonal, which is the only setting that works without knowing the
# specimen: u_max here is 0.12 mm on a part measured in millimetres, so x1 is
# invisible and x100 is a cartoon, and which of those you want depends
# entirely on the geometry you loaded.
CRACK_SHAPES = ["undeformed", "auto", "x1", "x2", "x5", "x10", "x20", "x50",
                "x100", "x500"]
AUTO_DEFORM_FRACTION = 0.06

# Diverging maps, offered only for the signed fields. Their neutral colour
# sits in the MIDDLE, which is wrong for phi and right for a stress.
DIVERGING = ["coolwarm", "RdBu_r", "seismic", "bwr", "PuOr_r", "BrBG_r",
             "Spectral_r"]

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
    # accent_hi is the accent under the pointer; onaccent is what stays
    # readable ON the accent -- picked per theme because a light accent needs
    # dark text and a dark one needs light, and getting that pair wrong is how
    # a primary button ends up unreadable in one theme only.
    "accent_hi": "#8ab4e8", "onaccent": "#0d1117", "errbg": "#2e2022",
    # Softer lines. "grid" was doing double duty as the section outline AND
    # the field border, which made a row of seven entries read as a cage.
    "edge": "#33363b", "outline": "#2c2f34", "cell": "#2b2e33",
    # Mesh view: near-white wireframe on a face darker than the panel, so the
    # elements read as a mesh rather than as texture.
    "mesh_edge": "#dfe6ee", "mesh_face": "#15171a",
}
LIGHT = {
    "bg": "#f0f0f0", "panel": "#e4e4e4", "field": "#ffffff", "hover": "#d8d8d8",
    "fg": "#101010", "muted": "#555555", "accent": "#0a5fb4",
    "warn": "#b36b00", "err": "#b00020", "grid": "#cccccc", "line": "#1f77b4",
    "warnbg": "#fffaf0",
    "accent_hi": "#0b6ac9", "onaccent": "#ffffff", "errbg": "#fdeaea",
    "edge": "#c4c4c4", "outline": "#d2d2d2", "cell": "#fbfbfb",
    "mesh_edge": "#222222", "mesh_face": "#ffffff",
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

    # The one button that commits you to an hours-long solve should not be
    # drawn identically to the "..." that opens a file dialog beside it. Run
    # takes the accent as a fill; everything else stays the flat grey above,
    # so there is exactly one loud control in the window.
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

    # Stop kills a solve that may be hours in, so it is marked -- but in its
    # TEXT only. A red fill sitting next to the blue Run would read as an
    # error state permanently present in the toolbar; the fill appears only
    # under the pointer, at the moment it is about to be pressed.
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

    # A section heading that is also its own toggle. Left-aligned and flat,
    # so it reads as a heading rather than as another control competing with
    # Run -- the only thing that marks it as pressable is that it lights up
    # under the pointer.
    style.configure("Section.TButton", background=T["panel"],
                    foreground=T["fg"], bordercolor=T["outline"],
                    lightcolor=T["outline"], darkcolor=T["outline"],
                    anchor="w", relief="flat", padding=[px(9), px(6)],
                    font=("TkDefaultFont", 9, "bold"))
    style.map("Section.TButton",
              background=[("active", T["hover"])],
              foreground=[("active", T["fg"])])

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
    # Tabs are navigation, not a call to action. The selected one used to be
    # a solid accent fill with black text at 10pt bold -- which made the
    # loudest object in the window a label telling you which view you are
    # already looking at, competing with Run for the same attention.
    #
    # Now the selected tab takes the CONTENT's background, so it reads as the
    # front sheet of a stack rather than a highlighted button, and only its
    # text is accented. Unselected tabs sit back on the panel colour.
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

    # THE CHECKERBOARD SQUARE in the middle of the toolbar. matplotlib
    # DISABLES Back and Forward whenever there is no view history to move
    # through, and Tk draws a disabled image button by masking the image with
    # a 50% stipple. On the light default theme that reads as a greyed-out
    # arrow; against this dark strip it is every other pixel of a white icon
    # on near-black, which is a checkerboard.
    #
    # There is no Tk option for the stipple, so the choice is between the
    # artefact and the affordance. Both buttons are no-ops on an empty
    # history, so nothing can go wrong by pressing one, and a blank-looking
    # button is a worse signal than an inert one.
    try:
        tb.set_history_buttons = lambda: None      # stop it re-disabling them
        for name in ("Back", "Forward"):
            button = getattr(tb, "_buttons", {}).get(name)
            if button is not None:
                button.configure(state="normal")
    except (AttributeError, tk.TclError):
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
    L = ["# Generated by run_gui2.py -- edit the GUI, not this file.",
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


ARCHIVE_MIN_ROWS = 20     # below this a CSV is a false start, not a result


def archive_curve(path: str) -> str:
    """Move a finished run's force-displacement CSV aside. Returns the new name.

    THE BUG THIS FIXES: re-running with a base_name that already exists simply
    DELETED the previous curve. The only guard was the liveness check, which
    asks whether another window is writing the file RIGHT NOW -- twenty
    seconds of protection for a result that took eleven hours to produce. Once
    the run had finished, starting another one with the same name destroyed it
    without a word.

    The VTK snapshots are overwritten by index and the log is rewritten, but
    those are reproducible from the config. The force-displacement curve IS
    the result, so it is the one file worth keeping, and keeping it costs a
    rename and a few hundred kilobytes.

    A file with almost nothing in it is a false start -- launch, notice a bad
    setting, stop, launch again -- and archiving those would bury the real
    ones, so they are still removed.
    """
    try:
        with open(path, "r", errors="replace") as fh:
            rows = sum(1 for _ in fh) - 1        # minus the header
    except OSError:
        rows = 0
    if rows < ARCHIVE_MIN_ROWS:
        try:
            os.remove(path)
        except OSError:
            pass
        return ""
    stamp = time.strftime("%Y%m%d-%H%M%S",
                          time.localtime(os.path.getmtime(path)))
    root_, ext = os.path.splitext(path)
    dest = f"{root_}.{stamp}{ext}"
    i = 2
    while os.path.exists(dest):
        dest, i = f"{root_}.{stamp}-{i}{ext}", i + 1
    try:
        os.replace(path, dest)
    except OSError:
        return ""
    return dest


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


def list_vtk(vdir: str):
    """Every snapshot as (step, path), ascending by STEP NUMBER.

    Numerically, not asciibetically: setw(3) stops zero-padding at step 999,
    so "_step_1000.vtk" sorts before "_step_999.vtk" as text -- which would
    drop the whole end of a long run into the middle of the list, and step
    backwards from frame 1000 to frame 100.
    """
    try:
        entries = os.listdir(vdir)
    except OSError:
        return []
    out = [(int(m.group(1)), os.path.join(vdir, name))
           for name, m in ((e, _STEP_RE.search(e)) for e in entries) if m]
    out.sort(key=lambda kv: kv[0])
    return out


def newest_vtk(vdir: str):
    """(path, step) of the highest-numbered snapshot, or (None, -1)."""
    snaps = list_vtk(vdir)
    return (snaps[-1][1], snaps[-1][0]) if snaps else (None, -1)


def _header(lines, keyword):
    """Index of the line beginning with `keyword`, or -1."""
    for i, ln in enumerate(lines):
        if ln.startswith(keyword):
            return i
    return -1


def _floats(lines, a, b):
    return np.array(" ".join(lines[a:b]).split(), dtype=np.float64)


def read_vtk_geometry(path: str):
    """-> (Triangulation, npoin, polys). Quads are split into two triangles.

    tricontourf needs a triangulation; the solver's elements are quads. Corner
    nodes only for Q8 -- VTK's Q8 ordering puts the four corners first, so
    idx[:4] is correct and the midside nodes are simply not used for contouring.

    `cells` is the same elements as NODE INDICES, (nelem, 4). polys is the
    undeformed shape and is what the overlay draws by default; the indices are
    what lets the deformed shape be rebuilt, since displacing an element means
    moving its nodes and coordinates alone have forgotten which nodes those
    were.

    `polys` is the ORIGINAL cell outline, (nelem, 4, 2), kept alongside the
    triangulation so the mesh overlay can draw real elements rather than the
    triangulation's edges. Those are not the same picture: splitting a quad
    adds a diagonal that is not in the mesh, and a wireframe full of invented
    diagonals is exactly the thing you must not show someone who is checking
    their discretisation. Triangles are padded to four vertices by repeating
    the last one, so the array stays rectangular -- the same trick
    read_mesh_for_plot uses, and it draws as the triangle it is.

    It comes from the VTK rather than from the .msh on purpose: this is the
    geometry the phi values on screen were computed on, needs no mesh path to
    be set, and cannot disagree with the field it is drawn over.
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
    cells = []
    for ln in lines[j + 1:j + 1 + nelem]:
        v = ln.split()
        nn = int(v[0])
        idx = [int(t) for t in v[1:1 + nn]]
        if nn == 3:
            tris.append(idx)
            cells.append([idx[0], idx[1], idx[2], idx[2]])
        elif nn in (4, 8):
            q = idx[:4]
            tris.append([q[0], q[1], q[2]])
            tris.append([q[0], q[2], q[3]])
            cells.append(q)
    if not tris:
        raise ValueError("no supported cells")
    idx = np.asarray(cells, dtype=np.int64) if cells else None
    polys = pts[idx, :2] if idx is not None else None
    return (mtri.Triangulation(pts[:, 0], pts[:, 1], np.array(tris)),
            npoin, polys, idx)


def read_vtk_field(path: str, name: str, npoin: int, nelem: int):
    """One named field out of a snapshot. Deliberately skips POINTS/CELLS.

    Returns (values, where) with one value per NODE or per ELEMENT depending
    on the field -- the caller has to know which, because a nodal field is
    contoured and an element field is not.

    The block names are unique across the file ("SCALARS phi" vs "SCALARS
    sigma_xx"), so each is found by its own header and the point/cell split in
    the file never has to be parsed.
    """
    spec = CRACK_FIELDS.get(name)
    if spec is None:
        raise ValueError(f"unknown field {name!r}")
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()
    i = _header(lines, spec["block"])
    if i < 0:
        raise ValueError(f"no '{spec['block']}' block")

    count = npoin if spec["where"] == "point" else nelem
    if spec["block"].startswith("VECTORS"):
        # No LOOKUP_TABLE after a VECTORS header, and three floats per node
        # (writeVTK pads the z component with a literal 0).
        vals = _floats(lines, i + 1, i + 1 + count)
        if vals.size != 3 * count:
            raise ValueError(f"{name} has {vals.size} values, "
                             f"expected {3 * count}")
        vec = vals.reshape(count, 3)[:, :2]
        out = (np.hypot(vec[:, 0], vec[:, 1]) if spec["comp"] == "mag"
               else vec[:, int(spec["comp"])])
    else:
        # LOOKUP_TABLE always follows SCALARS in what writeVTK emits.
        out = _floats(lines, i + 2, i + 2 + count)
        if out.size != count:
            raise ValueError(f"{name} has {out.size} values, "
                             f"expected {count}")
    return out, spec["where"]


def read_vtk_displacement(path: str, npoin: int):
    """(npoin, 2) nodal displacement, for drawing the deformed shape."""
    ux, _ = read_vtk_field(path, "ux", npoin, 0)
    uy, _ = read_vtk_field(path, "uy", npoin, 0)
    return np.column_stack([ux, uy])


def read_vtk_phi(path: str, npoin: int):
    """phi alone, kept as its own name because most callers only want that."""
    return read_vtk_field(path, "phi", npoin, 0)[0]


_CURVE_CACHE = {}


def reset_curve_cache(path: str) -> None:
    """Forget the incremental reader state before a new solve reuses a path."""
    if path:
        _CURVE_CACHE.pop(os.path.abspath(path), None)


def _curve_snapshot(state):
    """Hand back COPIES of the accumulating lists.

    The incremental reader owns these lists and appends to them on every call.
    Returning them directly means a caller that keeps `u` across two reads
    finds it has silently grown -- the value it measured and the value it
    plotted are no longer the same object's contents. Nothing in the GUI is
    bitten by that today because every reader runs on the Tk thread, but it is
    exactly the aliasing that turns into a real fault the moment one of them
    moves to a worker.

    The copy is not the cost that mattered: re-PARSING the whole CSV was
    ~19 ms at 10.5k rows, and copying four built lists is a couple of hundred
    microseconds. Keeping the parse incremental is what bought the speed;
    handing out references bought nothing.
    """
    return (list(state["u"]), list(state["F"]), list(state["phi"]),
            list(state["conv"]), state["labels"])


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
        return _curve_snapshot(state)

    try:
        with open(path, "rb") as fh:
            fh.seek(state["offset"])
            chunk = fh.read()
            state["offset"] = fh.tell()
    except OSError:
        return _curve_snapshot(state)

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
    return _curve_snapshot(state)


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


class FlowBar(ttk.Frame):
    """A row of controls that WRAPS onto more rows instead of being clipped.

    Tk has no flow layout. pack() puts everything on one line and silently
    cuts off whatever does not fit, which is how a toolbar loses its
    right-hand end the moment the window is not maximised -- and the
    right-hand end is where the read-outs are. Splitting a bar into two fixed
    rows by hand only moves the width at which it happens.

    So: measure what each control asks for, and lay them out into as many rows
    as it takes. Children are created with the FlowBar as their parent and are
    then packed INTO row frames, which Tk allows because the rows are its
    descendants.
    """

    def __init__(self, parent, gap=6, **kw):
        super().__init__(parent, **kw)
        self._items = []
        self._gap = gap
        self._rows = []
        self._width = -1
        self.bind("<Configure>", self._reflow)

    def add(self, widget, padx=0):
        self._items.append((widget, padx))
        return widget

    def _row(self, i):
        while len(self._rows) <= i:
            self._rows.append(ttk.Frame(self))
        return self._rows[i]

    def _reflow(self, event=None):
        width = event.width if event is not None else self.winfo_width()
        # Re-packing changes this frame's HEIGHT, which fires <Configure>
        # again with the same width. Without this guard that is an infinite
        # loop, not a slow redraw.
        if width <= 1 or abs(width - self._width) < 2:
            return
        self._width = width
        for row in self._rows:
            row.pack_forget()
        last, used = 0, 0
        for widget, padx in self._items:
            # padx is Tk's, so it is either a number or a (left, right) pair.
            pad = (sum(padx) if isinstance(padx, (tuple, list)) else 2 * padx)
            need = widget.winfo_reqwidth() + pad + self._gap
            if used and used + need > width:
                last += 1
                used = 0
            row = self._row(last)
            widget.pack(in_=row, side="left", padx=padx)
            # AND RAISE IT. Packing a widget `in_` something that is not its
            # parent only decides where it is POSITIONED; the stacking order
            # is untouched, and these rows are siblings of the controls that
            # were created before them. So the row's own background was drawn
            # on top and the entire bar came out as an empty strip -- present,
            # correctly sized, invisible. Nothing that inspects geometry can
            # see that, which is why it survived a test suite.
            widget.lift(row)
            used += need
        for i, row in enumerate(self._rows):
            if i <= last:
                row.pack(fill="x")


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
        now = time.time()
        if now - getattr(self, "_req_at", 0.0) > 0.15:
            self._req_w, self._req_at = self.inner.winfo_reqwidth(), now
        self.canvas.itemconfig(self._win,
                               width=max(event.width, self._req_w))

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
        # Results from the background file readers come back here as
        # (callback, args) and are applied on the Tk thread by _tick. Tk and
        # matplotlib are touched from nowhere else.
        self.io_q: queue.Queue = queue.Queue()
        self._closing = False
        # gmsh keeps global state, so exactly one reader may be inside it at a
        # time -- a lock, not a thread pool.
        self._io_lock = threading.Lock()
        # Monotonic per-kind request ids. A read that has been superseded is
        # DISCARDED on arrival rather than cancelled: gmsh cannot be
        # interrupted part-way through an open(), so the only safe thing is to
        # let it finish and ignore what it produced.
        self._mesh_request = 0
        self._group_request = 0
        self._crack_request = 0
        self._crack_loading_path = ""
        self._crack_snaps: list = []    # (step, path) for the current run
        self._notch_tried: set = set()  # (mesh path, mtime) already read
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
        # Per-STAGE step times. A coarse step and a step during crack growth
        # cost very different amounts, so one pooled rate predicts neither.
        self._stage_times: list = [[], [], []]
        self._last_stage_rates = [0.0, 0.0, 0.0]
        self._run_form = None       # the schedule the running job was given
        self._last_step = 0
        self._last_rate = 0.0       # s/step carried over from the previous run
        self._peak_reported = False
        self._last_elapsed = -1     # whole seconds already shown on the clock
        self._curve_drawn = False   # has the curve been drawn at least once
        self._laid_out: dict = {}   # id(figure) -> layout key, see _layout()
        self._dragging = False      # a pane sash is being dragged
        self._resize_pending: set = set()
        self._rows: dict = {}           # key -> [widgets], for show/hide
        self._charw = 0                 # width of "0" in the form font
        self._sections: dict = {}       # title -> its widgets and open state
        self._field_section: dict = {}  # field key -> the section holding it

        self._build()
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
        # opaqueresize=True: the panes follow the mouse. What made that stutter
        # before is NOT the pane geometry -- it is that every <Configure> makes
        # FigureCanvasTkAgg re-render the whole figure (80 ms on the Mesh tab).
        # _install_resize_guard below intercepts that event and defers the
        # render until the sash is released, so the panes track the pointer at
        # full rate and the plot re-renders exactly once.
        outer = tk.PanedWindow(self.root, orient="horizontal",
                               opaqueresize=True, background=T["grid"],
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

        # --- top bar ---
        bar = ttk.Frame(left, padding=px(8))
        bar.pack(fill="x")
        ttk.Label(bar, text="Solver").pack(side="left")
        # The BUTTON is packed first, against the right, and the entry takes
        # whatever is left. Packed the other way round a 48-character entry
        # claims its full width and pushes the browse button off the edge --
        # and an entry you can scroll is a far smaller loss than a button you
        # cannot reach.
        ttk.Button(bar, text="...", width=3,
                   command=self._pick_exe).pack(side="right", padx=(px(4), 0))
        self.exe_var = tk.StringVar(value=find_exe())
        ttk.Entry(bar, textvariable=self.exe_var).pack(
            side="left", padx=px(4), fill="x", expand=True)

        bar2 = ttk.Frame(left, padding=(px(8), 0))
        bar2.pack(fill="x")
        # Left to right in the order you use them -- check, then run, then
        # stop -- with the weight on Run rather than on its position.
        # Also first, for the same reason: packed last it was the control
        # that vanished when the results column got narrow.
        ttk.Button(bar2, text="Output folder",
                   command=self._open_output).pack(side="right")
        self.btn_check = ttk.Button(bar2, text="Check mesh",
                                    command=self.check_mesh)
        self.btn_check.pack(side="left")
        self.btn_run = ttk.Button(bar2, text="Run", command=self.run,
                                  style="Primary.TButton")
        self.btn_run.pack(side="left", padx=px(6))
        self.btn_stop = ttk.Button(bar2, text="Stop", command=self.stop,
                                   state="disabled", style="Danger.TButton")
        self.btn_stop.pack(side="left")

        # ONE status area, and every slot in it has exactly one writer:
        #
        #   state     what the program is doing      lifecycle, colour-coded
        #   elapsed   how long it has been doing it  the clock
        #   progress  how far through                bar + step / rate / ETA
        #   readout   what the solve currently says  the physics numbers
        #
        # They were two labels before, and _refresh_plot rewrote one of them
        # twice a second with the live numbers -- the same label that carried
        # "FAILED (exit 3221225781)". So the one message you most needed to
        # read survived until the next plot refresh, and the two writers had
        # to be kept apart by a comment rather than by structure.
        sbar = ttk.Frame(left, padding=(px(8), px(6), px(8), 0))
        sbar.pack(fill="x")
        sbar.grid_columnconfigure(0, weight=1)

        self.state = tk.StringVar(value="idle")
        self.state_lbl = ttk.Label(sbar, textvariable=self.state, anchor="w",
                                   font=("TkDefaultFont", 10, "bold"))
        self.state_lbl.grid(row=0, column=0, sticky="w")
        self.elapsed = tk.StringVar(value="")
        ttk.Label(sbar, textvariable=self.elapsed, foreground=T["muted"],
                  anchor="e").grid(row=0, column=1, sticky="e")

        self.progress = ttk.Progressbar(sbar, mode="determinate", maximum=1000)
        self.progress.grid(row=1, column=0, columnspan=2, sticky="ew",
                           pady=(px(5), 0))
        self.progress_txt = tk.StringVar(value="")
        self._wrapping(ttk.Label(sbar, textvariable=self.progress_txt,
                                 foreground=T["muted"], anchor="w",
                                 justify="left"),
                       sbar).grid(row=2, column=0, columnspan=2, sticky="ew",
                                  pady=(px(2), 0))
        # Monospaced: these are numbers that change in place every half
        # second, and in a proportional face the whole line jitters sideways
        # every time a digit changes width.
        self.readout = tk.StringVar(value="")
        self._wrapping(ttk.Label(sbar, textvariable=self.readout,
                                 foreground=T["fg"], font=("Consolas", 9),
                                 anchor="w", justify="left"),
                       sbar).grid(row=3, column=0, columnspan=2, sticky="ew",
                                  pady=(px(2), 0))

        # Second splitter: plots above, log below.
        vpane = tk.PanedWindow(left, orient="vertical", opaqueresize=True,
                               background=T["grid"], borderwidth=0,
                               sashwidth=px(6), sashpad=0, sashrelief="flat")
        vpane.pack(fill="both", expand=True, padx=px(8), pady=(0, px(4)))
        self.vpaned = vpane

        nb = ttk.Notebook(vpane)
        # The floor has to fit the tab strip, the tab's own control bar and
        # the matplotlib toolbar, with something left for the plot. At 180 it
        # was about equal to that chrome, which leaves the packer deciding
        # what to drop when the sash is dragged to the bottom -- and what it
        # drops is whatever was packed last.
        vpane.add(nb, minsize=px(230), stretch="always")
        self.nb = nb

        curve_tab = ttk.Frame(nb)
        nb.add(curve_tab, text="Force - displacement")
        self.curve_tab = curve_tab
        # Same idiom as the Mesh and Crack tabs: a Fit button and a follow
        # toggle. Those two tabs already preserved a view you had set up; this
        # one threw it away on the next refresh, twice a second.
        fbar = FlowBar(curve_tab, padding=(0, px(4)))
        fbar.pack(fill="x")
        fbar.add(ttk.Button(fbar, text="Fit", width=5, command=self.fit_curve))
        self.follow_curve = tk.BooleanVar(value=True)
        fbar.add(ttk.Checkbutton(fbar, text="follow the data",
                                 variable=self.follow_curve,
                                 command=self._follow_curve_changed),
                 padx=px(8))
        self.curve_hint = tk.StringVar(value="")
        chint = ttk.Frame(curve_tab)
        chint.pack(fill="x")
        self._wrapping(ttk.Label(chint, textvariable=self.curve_hint,
                                 foreground=T["muted"], anchor="w",
                                 justify="left"), chint).pack(fill="x",
                                                              padx=px(2))
        self.fig = Figure(figsize=(7, 3.4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("applied displacement")
        self.ax.set_ylabel("reaction force")
        style_axes(self.fig, self.ax)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=curve_tab)
        # Toolbar FIRST, against the bottom. Tk's packer hands out space in
        # packing order, so a canvas packed first with expand=True claims the
        # lot and anything packed after it is left with nothing -- which is
        # why the toolbar vanished as soon as the sash made the plot pane
        # short. Reserving its height first and letting the canvas expand into
        # the remainder is the fix, and it costs nothing when there is room.
        self.curve_toolbar = add_toolbar(self.canvas, curve_tab)
        self.curve_toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        # Reaching for the toolbar's zoom or pan IS the instruction to stop
        # following. Asking for it twice -- zoom, then also remember to untick
        # a box -- would just be the same bug with an extra step.
        self.canvas.mpl_connect("button_release_event", self._curve_interacted)

        mesh_tab = ttk.Frame(nb)
        nb.add(mesh_tab, text="Mesh")
        self.mesh_tab = mesh_tab
        # Redrawing a tab nobody is looking at is wasted work, so edits made
        # while the curve is showing only set a flag; the redraw happens when
        # you switch to the Mesh tab.
        nb.bind("<<NotebookTabChanged>>", self._tab_changed)
        mbar = FlowBar(mesh_tab, padding=(0, px(4)))
        mbar.pack(fill="x")
        mbar.add(ttk.Button(mbar, text="Draw mesh", command=self.draw_mesh))
        mbar.add(ttk.Button(mbar, text="Fit", width=5, command=self.fit_mesh),
                 padx=px(4))
        self.show_bc_only = tk.BooleanVar(value=True)
        bcchk = mbar.add(ttk.Checkbutton(mbar, text="BC groups only",
                                         variable=self.show_bc_only,
                                         command=self.draw_mesh), padx=px(8))
        Tip(bcchk, "Pick out only the physical groups a boundary condition "
                   "refers to. Off, every named group in the mesh is drawn.")
        mbar.add(ttk.Label(mbar, text="view"), padx=(px(8), px(2)))
        self.mesh_view = tk.StringVar(value=MESH_VIEWS[0])
        mv = mbar.add(ttk.Combobox(mbar, textvariable=self.mesh_view,
                                   values=MESH_VIEWS, width=11,
                                   state="readonly"))
        mv.bind("<<ComboboxSelected>>",
                lambda _e: self.draw_mesh(preserve_view=True))
        # Its own row. On one line with the controls it was the last thing
        # packed, so it was the first thing clipped -- and it is the only part
        # of that row carrying numbers.
        self.mesh_stats = tk.StringVar(value="")
        mstat = ttk.Frame(mesh_tab)
        mstat.pack(fill="x")
        self._wrapping(ttk.Label(mstat, textvariable=self.mesh_stats,
                                 foreground=T["muted"], anchor="w",
                                 justify="left"), mstat).pack(fill="x",
                                                              padx=px(2))
        self.mfig = Figure(figsize=(7, 3.4), dpi=100)
        self.max_ = self.mfig.add_subplot(111)
        style_axes(self.mfig, self.max_)
        self.mfig.tight_layout()
        self.mcanvas = FigureCanvasTkAgg(self.mfig, master=mesh_tab)
        add_toolbar(self.mcanvas, mesh_tab).pack(side="bottom", fill="x")
        self.mcanvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        # Wheel-zoom about the cursor, which is what a mesh viewer should do.
        # The toolbar still offers rubber-band zoom and pan for fine control.
        self.mcanvas.mpl_connect("scroll_event", self._mesh_wheel_zoom)
        self.mesh_bbox = None

        # ---- Crack tab: phi from the solver's VTK snapshots ----------------
        crack_tab = ttk.Frame(nb)
        nb.add(crack_tab, text="Crack (phi)")
        self.crack_tab = crack_tab
        cbar_ = FlowBar(crack_tab, padding=(0, px(4)))
        cbar_.pack(fill="x")
        # Step through the snapshots. The solver writes one every vtk_every
        # accepted steps, so the folder is already a filmstrip of the crack --
        # there was just no way to look at any frame but the last.
        #
        # "|<" and ">|" rather than arrow glyphs: this bar has to render in
        # whatever UI font the machine has, and a missing glyph draws as a box
        # on the two buttons you reach for most.
        nav = cbar_.add(ttk.Frame(cbar_))
        for text, cmd, tip in (
                ("|<", self.crack_first, "First snapshot."),
                ("<", self.crack_prev, "Previous snapshot. Stops following "
                                       "the live run."),
                (">", self.crack_next, "Next snapshot."),
                (">|", self.crack_last, "Latest snapshot, and follow the run "
                                        "again from here.")):
            b = ttk.Button(nav, text=text, width=3, command=cmd)
            b.pack(side="left", padx=1)
            Tip(b, tip)
        self.crack_pos = tk.StringVar(value="")
        cbar_.add(ttk.Label(cbar_, textvariable=self.crack_pos,
                            foreground=T["muted"], width=11, anchor="w"),
                  padx=px(6))
        cbar_.add(ttk.Button(cbar_, text="Refresh", command=self.draw_crack))
        cbar_.add(ttk.Button(cbar_, text="Fit", width=5,
                             command=self.fit_crack), padx=px(4))
        self.follow_crack = tk.BooleanVar(value=True)
        fchk = cbar_.add(ttk.Checkbutton(cbar_, text="follow",
                                         variable=self.follow_crack),
                         padx=px(8))
        Tip(fchk, "Jump to each new snapshot as the solver writes it. Turned "
                  "off automatically when you step back through the frames.")
        self.show_notch = tk.BooleanVar(value=True)
        cbar_.add(ttk.Checkbutton(cbar_, text="notch",
                                  variable=self.show_notch,
                                  command=self._redraw_crack_overlay))

        # Everything below is on the SAME bar -- FlowBar decides how many rows
        # that needs at the width it actually has. All of it on one line comes
        # to roughly 1200 px, which no results column here is ever going to
        # be, but a hand-split into two fixed rows only moves the width at
        # which the end gets cut off.
        cbar2 = cbar_
        cbar2.add(ttk.Label(cbar2, text="field"), padx=(px(8), px(2)))
        self.field_var = tk.StringVar(value="phi")
        fld = cbar2.add(ttk.Combobox(cbar2, textvariable=self.field_var,
                                     values=list(CRACK_FIELDS), width=10,
                                     state="readonly"))
        Tip(fld, "Which quantity to colour. phi and the displacements live on "
                 "NODES and are contoured; the stresses are one value per "
                 "ELEMENT and are drawn flat, because that is how the solver "
                 "integrates them.")
        fld.bind("<<ComboboxSelected>>", lambda _e: self._field_changed())
        cbar2.add(ttk.Label(cbar2, text="crack iso"), padx=(px(8), px(2)))
        self.iso_var = tk.StringVar(value="0.9")
        self.iso_entry = cbar2.add(ttk.Entry(cbar2, textvariable=self.iso_var,
                                             width=5))
        cbar2.add(ttk.Label(cbar2, text="shape"), padx=(px(8), px(2)))
        self.deform_var = tk.StringVar(value=CRACK_SHAPES[0])
        dfm = cbar2.add(ttk.Combobox(cbar2, textvariable=self.deform_var,
                                     values=CRACK_SHAPES, width=11,
                                     state="readonly"))
        Tip(dfm, "Draw the specimen at x + scale*u instead of at its "
                 "reference position, so you see the crack physically open "
                 "rather than only a field that says it did. 'auto' picks a "
                 "magnification that makes the largest displacement about 6% "
                 "of the specimen -- visible without being a cartoon.")
        dfm.bind("<<ComboboxSelected>>", lambda _e: self._deform_changed())
        cbar2.add(ttk.Label(cbar2, text="mesh"), padx=(px(8), px(2)))
        self.crack_overlay = tk.StringVar(value=CRACK_OVERLAYS[0])
        ov = cbar2.add(ttk.Combobox(cbar2, textvariable=self.crack_overlay,
                                    values=CRACK_OVERLAYS, width=14,
                                    state="readonly"))
        Tip(ov, "Draw the mesh over the phi field. The question this answers "
                "is whether l0 is resolved WHERE THE CRACK IS -- so zoom into "
                "the tip first; the overlay is culled to what is on screen "
                "and switches itself off when the elements would be "
                "sub-pixel.")
        # Redrawn from the phi already in hand -- no file is re-read, so this
        # is instant even on a mesh whose first load took seconds.
        ov.bind("<<ComboboxSelected>>", lambda _e: self._redraw_crack_overlay())
        cbar2.add(ttk.Label(cbar2, text="colours"), padx=(px(8), px(2)))
        self.cmap_var = tk.StringVar(value="inferno")
        # Two remembered choices, one combo. A sequential map is right for phi
        # and a diverging one for a signed stress, so switching field swaps the
        # list -- and swapping back has to return the map you had, not a
        # default.
        self._cmap_seq, self._cmap_div = "inferno", "coolwarm"
        self.cmap_combo = cbar2.add(
            ttk.Combobox(cbar2, textvariable=self.cmap_var, values=CMAPS,
                         width=10, state="readonly"))
        self.cmap_combo.bind("<<ComboboxSelected>>",
                             lambda _e: self._cmap_changed())
        # And the stats on a row of their own, reflowing. This line now
        # carries the field range, the deformation scale and the overlay
        # counts, so it is both the longest string in the window and the one
        # you are actually reading.
        self.crack_stats = tk.StringVar(value="no snapshots yet")
        cstat = ttk.Frame(crack_tab)
        cstat.pack(fill="x")
        self._wrapping(ttk.Label(cstat, textvariable=self.crack_stats,
                                 foreground=T["muted"], anchor="w",
                                 justify="left"), cstat).pack(fill="x",
                                                              padx=px(2))

        self.cfig = Figure(figsize=(7, 3.4), dpi=100)
        self.cax = self.cfig.add_subplot(111)
        style_axes(self.cfig, self.cax)
        self.cfig.tight_layout()
        self.ccanvas = FigureCanvasTkAgg(self.cfig, master=crack_tab)
        add_toolbar(self.ccanvas, crack_tab).pack(side="bottom", fill="x")
        self.ccanvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        # Arrow keys, bound to the CANVAS rather than the window: Left and
        # Right belong to whatever Entry has focus everywhere else in this
        # GUI, and stealing them globally would break typing in the form.
        # matplotlib gives the canvas focus when it is clicked.
        _cw = self.ccanvas.get_tk_widget()
        _cw.bind("<Left>", lambda _e: self.crack_prev())
        _cw.bind("<Right>", lambda _e: self.crack_next())
        _cw.bind("<Home>", lambda _e: self.crack_first())
        _cw.bind("<End>", lambda _e: self.crack_last())
        self._cbar = None          # colourbar, created once (levels are fixed)
        self._tri = None           # cached Triangulation
        self._tri_npoin = 0
        self._tri_polys = None     # real cell outlines, for the mesh overlay
        self._tri_cells = None     # the same elements as node indices
        self._crack_last = None    # the frame on screen, for a cheap re-render
        self._crack_step = -1
        self._vtk_sizes: dict = {}

        # stretch="never" so growing the window grows the PLOT and leaves the
        # log at whatever height you dragged it to.
        lf = ttk.Frame(vpane, padding=(0, px(6), 0, 0))
        vpane.add(lf, height=px(210), minsize=px(60), stretch="never")
        self.log = tk.Text(lf, height=6, wrap="none", font=("Consolas", 9),
                           background=T["panel"], foreground=T["fg"],
                           insertbackground=T["fg"], selectbackground=T["accent"],
                           selectforeground="#000000", relief="flat",
                           borderwidth=0)
        sb = ttk.Scrollbar(lf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.log.tag_configure("warn", foreground=T["warn"])
        self.log.tag_configure("err", foreground=T["err"])

        # Live pane resizing without the render storm -- see
        # _install_resize_guard. Must run AFTER every canvas exists.
        for c in (self.canvas, self.mcanvas, self.ccanvas):
            self._install_resize_guard(c)
        for p in (outer, vpane):
            self._bind_sash(p)
        outer.bind("<Configure>", self._clamp_panes, add="+")

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
        self._bind_shortcuts()

        if SCALE_INFO:
            self._append_log(SCALE_INFO)

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

    # What each lifecycle state should look like. Colour is the point: a
    # failed run and an idle one used to be the same shade of grey text.
    STATE_COLOUR = {"idle": "muted", "busy": "accent",
                    "ok": "fg", "fail": "err"}

    def _set_state(self, text: str, kind: str = "idle") -> None:
        self.state.set(text)
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

    @staticmethod
    def _wrapping(label, parent, slack=20):
        """Make a status Label reflow at its parent's width.

        A Tk label CLIPS; it does not wrap unless told a pixel width, and a
        fixed one is wrong at every size but the one it was chosen at. So the
        width is re-applied whenever the parent changes size.

        This is what "information disappears when the window is not maximised"
        actually was: every status line in here ends with its numbers, so the
        part that gets cut off is the part worth reading.
        """
        def _resize(event, label=label, slack=slack):
            try:
                label.configure(wraplength=max(px(140), event.width - slack))
            except tk.TclError:
                pass
        parent.bind("<Configure>", _resize, add="+")
        return label

    def _section(self, title, expanded=True):
        """A section whose heading opens and closes it.

        THE RISK OF COLLAPSING A FORM is that you hide a setting you got
        wrong, so a closed section here still shows a one-line summary of what
        is inside it. Nothing about the run becomes invisible; it becomes
        smaller. That is also why the summary is generated from the live
        variables rather than written once -- a stale summary would be worse
        than no summary at all.
        """
        shell = ttk.Frame(self.panel)
        shell.pack(fill="x", padx=px(6), pady=px(3))
        button = ttk.Button(shell, style="Section.TButton",
                            command=lambda t=title: self._set_section_expanded(
                                t, not self._sections[t]["expanded"]))
        button.pack(fill="x")
        summary = ttk.Label(shell, foreground=T["muted"], anchor="w",
                            wraplength=px(400), justify="left",
                            padding=(px(11), px(1), px(4), px(3)))
        body = ttk.Frame(shell, padding=(px(7), px(4), px(7), px(7)))
        self._sections[title] = {"shell": shell, "button": button,
                                 "summary": summary, "body": body,
                                 "expanded": bool(expanded)}

        ch = self._char_w()
        body.grid_columnconfigure(0, minsize=self.LABEL_COL * ch)
        body.grid_columnconfigure(1, minsize=self.CTRL_COL * ch)
        # The slack goes to the unit column, so widening the panel opens up
        # the space AFTER the values instead of stretching every box.
        body.grid_columnconfigure(2, weight=1)
        body.next_row = 0
        body.section_title = title
        self._set_section_expanded(title, expanded)
        return body

    def _set_section_expanded(self, title, expanded):
        state = self._sections.get(title)
        if not state:
            return
        state["expanded"] = bool(expanded)
        state["button"].configure(text=("\u2212  " if expanded else "+  ") + title)
        if expanded:
            state["summary"].pack_forget()          # the fields say it better
            if not state["body"].winfo_manager():
                state["body"].pack(fill="x")
        else:
            if state["body"].winfo_manager():
                state["body"].pack_forget()
            state["summary"].pack(fill="x")
            self._refresh_section_summaries()

    def _v(self, key, default=""):
        """A form value as the user typed it, never raising."""
        var = self.vars.get(key)
        if var is None:
            return default
        try:
            return str(var.get())
        except tk.TclError:
            return default

    def _section_summary(self, title) -> str:
        """One line describing a closed section, from the live variables."""
        try:
            if title.startswith("1."):
                src = self.geom_source.get() or "no source chosen"
                return f"{src} \u2192 {self.geom_out.get()}"
            if title.startswith("2."):
                if self._v("mesh_source") != "file":
                    return "built-in mesh"
                path = self._v("mesh_path")
                if not path:
                    return "no mesh selected"
                groups = (f" \u00b7 {len(self.mesh_groups)} groups"
                          if self.mesh_groups else "")
                return f"{os.path.basename(path)}{groups}"
            if title.startswith("3."):
                hyb = " \u00b7 hybrid" if self._v("hybrid") in ("1", "True") else ""
                return (f"{self._v('ntype_label')} \u00b7 "
                        f"{self._v('energy_split')} split{hyb} \u00b7 "
                        f"ngaus {self._v('ngaus')}")
            if title.startswith("4."):
                return (f"E {self._v('E')} MPa \u00b7 nu {self._v('nu')} \u00b7 "
                        f"Gc {self._v('Gc')} \u00b7 l0 {self._v('l0')} mm")
            if title.startswith("5."):
                rows = [b for b in self.bctable.get() if b.get("group")]
                if not rows:
                    return "nothing restrained or loaded"
                driven = [b for b in rows
                          if float(b.get("vx") or 0) or float(b.get("vy") or 0)]
                lead = (f" \u00b7 {driven[0]['group']} driven"
                        if driven else " \u00b7 nothing driven")
                return (f"{len(rows)} condition{'s' if len(rows) != 1 else ''}"
                        f"{lead}")
            if title.startswith("6."):
                sweeps = (f" \u00b7 max {self._v('max_staggered')} sweeps"
                          if self._v("scheme") == "staggered" else "")
                return (f"{self._v('scheme')}{sweeps} \u00b7 "
                        f"tol_rel {self._v('tol_rel')}")
            if title.startswith("7."):
                mode = self._v("step_mode")
                try:
                    plan = planned_steps(self.get_form())
                except (ValueError, KeyError, TypeError):
                    plan = None
                count = f" \u00b7 {plan[0]:,} steps" if plan else ""
                return f"{mode}{count}"
            if title.startswith("8."):
                return (f"{self._v('output_dir')}/{self._v('base_name')} \u00b7 "
                        f"VTK every {self._v('vtk_every')}")
        except Exception:
            pass
        return ""

    def _refresh_section_summaries(self):
        for title, state in self._sections.items():
            if not state["expanded"]:
                state["summary"].configure(text=self._section_summary(title))

    def _reveal_field(self, key):
        """Open whichever section holds `key`.

        A validation message about a field you cannot see is a dead end, so
        anything the form complains about opens itself.
        """
        title = self._field_section.get(key)
        if title and title in self._sections \
                and not self._sections[title]["expanded"]:
            self._set_section_expanded(title, True)

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
        if unit:
            u = ttk.Label(parent, text=unit, foreground=T["muted"], anchor="w")
            u.grid(row=r, column=2, sticky="w", padx=(px(6), 0))
            cells.append(u)
        if help:
            for c in cells:
                Tip(c, help)
        # Every widget of the row, so _apply_visibility can take the whole
        # thing out and put it back exactly where it was.
        self._rows[key] = cells
        self._field_section[key] = getattr(parent, "section_title", "")
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

        # ---- Geometry: build a mesh, rather than only pointing at one ------
        # Open by default only what changes from run to run. The mesher in
        # section 1 builds a mesh once and is then done with; the model,
        # solver tolerances and output stride are study-level settings. Each
        # closed section still states its contents in one line, so nothing is
        # hidden -- only made smaller.
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

        s = self._section("2. Mesh")
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
        self._rows["mesh_path"] = [lbl, ent, btn]
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

        s = self._section("4. Material")
        self._field(s, "name", "mat_name")
        self._field(s, "E", "E", unit="MPa")
        self._field(s, "nu", "nu")
        self._field(s, "Gc", "Gc", unit="N/mm")
        self._field(s, "l0", "l0", unit="mm")
        self._field(s, "k", "k", unit="residual stiffness")
        self._field(s, "domain group", "domain")

        s = self._section("5. Restraints and loads")
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
        s = self._section("7. Load stepping")
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
    def _geom_source_changed(self, _event=None):
        """Rebuild the parameter fields for the newly chosen source."""
        for w in self.geom_param_frame.winfo_children():
            w.destroy()
        self.geom_vars = {}
        self.geom_tables = {}      # only the STEP source builds tables
        src = self.geom_source.get()

        if src.startswith("STEP"):
            path = filedialog.askopenfilename(
                initialdir=PROJECT_DIR, title="Select a CAD file",
                filetypes=[("CAD", "*.step *.stp *.iges *.igs *.brep"),
                           ("All files", "*.*")])
            if not path:
                self.geom_source.set("")
                return
            self.geom_cad = path
            ttk.Label(self.geom_param_frame, text=os.path.basename(path),
                      foreground=T["muted"], wraplength=px(430)).pack(anchor="w")
            def field(label, key, default, hint="", help="", parent=None):
                r = ttk.Frame(parent or self.geom_param_frame)
                r.pack(fill="x", pady=1)
                lbl = ttk.Label(r, text=label, width=17)
                lbl.pack(side="left")
                v = tk.StringVar(value=default)
                ent = ttk.Entry(r, textvariable=v, width=10,
                                style="Cell.TEntry")
                ent.pack(side="left")
                if hint:
                    ttk.Label(r, text=hint, foreground=T["muted"]).pack(
                        side="left", padx=4)
                if help:
                    Tip(lbl, help); Tip(ent, help)
                self.geom_vars[key] = v

            ttk.Label(self.geom_param_frame,
                      text="all coordinates, sizes and distances below are in "
                           "MILLIMETRES - the same units as the CAD file and "
                           "the config",
                      foreground=T["warn"], wraplength=px(470)).pack(anchor="w",
                                                                 pady=(2, 4))
            field("bulk size h", "h", "0.15")
            row = ttk.Frame(self.geom_param_frame); row.pack(fill="x", pady=1)
            ttk.Label(row, text="elements", width=17).pack(side="left")
            ev = tk.StringVar(value="quad4")
            cb = ttk.Combobox(row, textvariable=ev, values=ELEMENTS, width=8,
                              state="readonly", style="Cell.TCombobox")
            cb.pack(side="left")
            Tip(cb, "The only three element types the solver's reader accepts. "
                    "quad8 needs fem.ngaus = 3. Changing element type changes "
                    "the ANSWER, not just the discretisation -- tri3 is "
                    "noticeably stiffer than quad4 at the same size.")
            self.geom_vars["elements"] = ev
            v = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.geom_param_frame,
                            text="name outer edges Left/Right/Top/Bottom",
                            variable=v).pack(anchor="w", pady=(2, 0))
            self.geom_vars["auto_sides"] = v

            # One table per repeatable argument, each with the plain fields
            # that belong to it. Cracks and refinement are not optional extras:
            # without them an imported part can be meshed and looked at but not
            # run, since a phase-field study needs a sharp crack and l0
            # resolved in the band.
            self.geom_tables = {}
            for title, flag, hint, tmpl, cols, *rest in GEOM_TABLES:
                box = ttk.LabelFrame(self.geom_param_frame, text=title,
                                     padding=px(5))
                box.pack(fill="x", pady=3)
                t = SpecTable(box, cols, tmpl, hint=hint,
                              legacy=(rest[0] if rest else ()))
                t.pack(fill="x")
                t.flag = flag
                self.geom_tables[title] = t
                for label, key, default, help_ in GEOM_EXTRAS.get(title, ()):
                    field(label, key, default, help=help_, parent=box)

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
        self._refresh_section_summaries()
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

    def _clamp_panes(self, _event=None):
        """Stop the settings pane eating the results column.

        A PanedWindow remembers its sash in PIXELS. A pane sized against a
        maximised window keeps exactly that width when the window is made
        small, so ALL of the shrinkage lands on the column to its right -- at
        860 px the results side was down to about 280 px and its control rows
        were simply cut off. That is what "information disappears when the
        window is not fully opened" was.

        Fires only when the PANED WINDOW itself is resized, not when a sash is
        dragged, so it never fights you: drag the settings pane as wide as you
        like and it stays there until the window itself gets smaller.
        """
        try:
            total = self.paned.winfo_width()
            if total <= 1:
                return
            cap = max(px(300), int(0.45 * total))
            x, _y = self.paned.sash_coord(0)
            if x > cap:
                self.paned.sash_place(0, cap, 0)
        except (tk.TclError, ValueError, IndexError):
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

    def _sash_release(self, _event=None):
        if not self._dragging:
            return
        self._dragging = False
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
        """A toolbar zoom or pan on the curve turns following off."""
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
        """Back to the whole curve, and following again."""
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

    def draw_crack(self, force=False, target=None):
        """Draw one snapshot. `target` is a step number; None means the newest.

        The geometry is identical in every snapshot and is parsed once (see
        read_vtk_geometry), so moving between frames costs one phi block --
        which is what makes stepping through a run feel immediate on a mesh
        where the first load took seconds.
        """
        vdir = self._vtk_dir()
        snaps = list_vtk(vdir)
        self._crack_snaps = snaps
        if not snaps:
            self.crack_stats.set(
                "no snapshots yet" if not vdir else f"no .vtk files in {vdir}")
            self.crack_pos.set("")
            return
        if target is None:
            index = len(snaps) - 1
        else:
            # Nearest available step, not an exact match: vtk_every can change
            # between runs and a remembered step may no longer exist.
            index = min(range(len(snaps)),
                        key=lambda i: abs(snaps[i][0] - target))
        step, path = snaps[index]
        # Only the NEWEST file can be half-written, and only the AUTOMATIC
        # path has to wait for it: _maybe_follow_crack looks again twice a
        # second, so a snapshot the solver is still writing costs nothing to
        # skip. An explicit request reads now -- older frames finished long
        # ago, and a frame that HAS just appeared would otherwise need the
        # button pressed twice, silently doing nothing the first time. A torn
        # read on that path is reported by _crack_loaded and self-corrects.
        if (target is None and index == len(snaps) - 1
                and not self._stable(path) and not force):
            return
        if self._crack_loading_path == path:
            return
        self._crack_request += 1
        request = self._crack_request
        self._crack_loading_path = path
        self.crack_stats.set(f"loading step {step}...")
        threading.Thread(
            target=self._read_crack_worker,
            args=(request, path, step, force, self._tri, self._tri_npoin,
                  self._tri_polys, self._tri_cells,
                  self.field_var.get() or "phi",
                  self.deform_var.get() != "undeformed"),
            daemon=True).start()

    # -- moving through the snapshots -----------------------------------
    def _crack_index(self) -> int:
        """Where the frame on screen sits in the snapshot list."""
        for i, (st, _p) in enumerate(self._crack_snaps):
            if st == self._crack_step:
                return i
        return len(self._crack_snaps) - 1

    def _crack_go(self, delta=None, index=None):
        # Re-listed on every move, not reused from the last draw: during a
        # solve the folder grows underneath you, and a cached list meant ">|"
        # took you to whatever was newest when you last looked rather than to
        # what is newest now.
        snaps = list_vtk(self._vtk_dir()) or self._crack_snaps
        self._crack_snaps = snaps
        if not snaps:
            self.draw_crack()
            return
        if index is None:
            index = self._crack_index() + (delta or 0)
        index = max(0, min(index, len(snaps) - 1))
        # Stepping back IS the instruction to stop following, the same way
        # reaching for the zoom is on the curve tab -- otherwise the next
        # snapshot the solver writes would yank you to the end again, which is
        # the one thing you were trying to get away from. Landing on the last
        # frame puts you at the live end, so following resumes there.
        self.follow_crack.set(index == len(snaps) - 1)
        self.draw_crack(target=snaps[index][0])

    def crack_first(self):
        self._crack_go(index=0)

    def crack_prev(self):
        self._crack_go(delta=-1)

    def crack_next(self):
        self._crack_go(delta=+1)

    def crack_last(self):
        self._crack_go(index=1 << 30)      # clamped to the end of the list

    def _read_crack_worker(self, request, path, step, force, tri, npoin,
                           polys, cells, field, want_disp):
        where, disp = "point", None
        try:
            with self._io_lock:
                if tri is None:
                    tri, npoin, polys, cells = read_vtk_geometry(path)
                nelem = 0 if polys is None else len(polys)
                values, where = read_vtk_field(path, field, npoin, nelem)
                # Read here, on the worker, rather than lazily on the Tk
                # thread the moment someone picks a magnification.
                if want_disp:
                    disp = read_vtk_displacement(path, npoin)
            error = ""
        except (OSError, ValueError, IndexError, KeyError) as exc:
            values, error = None, str(exc)
        self.io_q.put((self._crack_loaded,
                       (request, path, step, force, tri, npoin, polys, cells,
                        values, where, field, disp, error)))

    def _crack_loaded(self, request, path, step, force, tri, npoin, polys,
                      cells, values, where, field, disp, error):
        if self._crack_loading_path == path:
            self._crack_loading_path = ""
        if request != self._crack_request:
            return
        if error:
            # A torn read is normal and self-correcting; only say so once.
            if step != self._crack_step:
                self.crack_stats.set(f"waiting for step {step} ({error})")
            return
        self._tri, self._tri_npoin, self._tri_polys = tri, npoin, polys
        self._tri_cells = cells
        self._crack_last = (path, step, values, where, field, disp)
        self._render_crack(path, step, values, where, field, disp, force)

    def _render_crack(self, path, step, values, where="point", field="phi",
                      disp=None, force=False):
        """Draw VTK data that was parsed away from Tk's event thread."""

        try:
            iso = float(self.iso_var.get())
        except ValueError:
            iso = 0.9
        spec = CRACK_FIELDS.get(field, CRACK_FIELDS["phi"])

        # The geometry everything below is drawn on. Zero means the reference
        # shape and nothing changes; otherwise every node moves to x + s*u and
        # the CONNECTIVITY is reused untouched -- deformation moves nodes, it
        # does not re-mesh, so the triangles and the elements are the same
        # ones and only their corners have moved.
        scale = self._deform_scale(disp)
        dx, dy = self._deformed_xy(disp, scale)
        tri = (self._tri if scale <= 0.0
               else mtri.Triangulation(dx, dy, self._tri.triangles))
        self._crack_scale = scale

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
        cmap = self.cmap_var.get() or ("coolwarm" if spec["signed"]
                                       else "inferno")
        if field == "phi":
            cs = self.cax.tricontourf(tri, values,
                                      levels=np.linspace(0.0, 1.0, 11),
                                      cmap=cmap, extend="both")
        else:
            # Everything else has no natural fixed range, so it is scaled to
            # the frame.
            #
            # A SIGNED field is CENTRED on zero, not forced symmetric about
            # it. Symmetric was the obvious thing and it was wrong: uy on this
            # run spans -0.126 to +0.0046, so padding it out to +/-0.126 threw
            # away HALF the colourmap -- 51.8% of the bar carried data and the
            # structure worth seeing was squeezed into the bottom half. A
            # two-slope norm keeps zero on the neutral colour, which is the
            # whole point of a diverging map, AND runs the full colour range
            # out to each end of the real data. The two sides then have
            # different value-per-colour, which is exactly what you are asking
            # for when you say "put zero in the middle".
            lo, hi = float(np.min(values)), float(np.max(values))
            norm = None
            if hi <= lo:
                # A frame where the field is uniform -- step 0 of a
                # displacement run is all zeros. contourf cannot take a
                # degenerate range, so open it slightly rather than raise.
                lo, hi = lo - 0.5, hi + 0.5
            elif spec["signed"] and lo < 0.0 < hi:
                half = 7
                levels = np.unique(np.concatenate(
                    [np.linspace(lo, 0.0, half + 1),
                     np.linspace(0.0, hi, half + 1)]))
                norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
            if norm is None:
                # Includes a signed field that never crosses zero: everything
                # in tension, or a displacement that only went one way. Half a
                # diverging map is then the honest picture -- it says so.
                levels = np.linspace(lo, hi, 15)
                norm = Normalize(vmin=lo, vmax=hi)
            if where == "point":
                # extend="neither", unlike phi. phi genuinely overshoots its
                # bounds (this run reaches 1.008) so its end colours are for
                # real data; here the levels ARE the data range, so reserving
                # an under- and an over-colour just moves both ends of the
                # colourmap somewhere nothing is ever drawn.
                cs = self.cax.tricontourf(tri, values, levels=levels,
                                          cmap=cmap, norm=norm,
                                          extend="neither")
            else:
                # ONE COLOUR PER ELEMENT, not a contour. Output.cpp integrates
                # these over each element's Gauss points and writes a single
                # value per cell; drawing them smooth would show a continuity
                # that was never computed.
                cs = PolyCollection(self._crack_polys(scale, dx, dy),
                                    array=np.asarray(values), cmap=cmap,
                                    norm=norm, edgecolors="none", zorder=0)
                self.cax.add_collection(cs)
                # A Collection carries no data limits of its own, so without
                # this the axes stay at their 0..1 default and the field is
                # drawn off screen.
                self.cax.set_xlim(float(dx.min()), float(dx.max()))
                self.cax.set_ylim(float(dy.min()), float(dy.max()))
        # The iso line is a contour of PHI. On a stress field it would be a
        # line of constant stress that happens to be numbered 0.9, which is
        # worse than no line at all.
        if field == "phi" and values.max() >= iso:
            # The iso-line has to stay legible against whatever map is chosen.
            # The theme accent is blue, which vanishes into viridis/cividis and
            # into the blue end of jet/turbo, so pick per map rather than
            # assuming a dark background behind the line.
            self.cax.tricontour(tri, values, levels=[iso],
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
        # NOT drawn on a deformed shape. These segments come from the .msh
        # as coordinates, and a coordinate on the notch maps to TWO nodes --
        # Plugin(Crack) splits them -- which move apart as the crack opens.
        # There is no honest single place to put the line, and the ambiguity
        # is worst exactly where the picture is being read. Better to show
        # nothing than to draw the notch where neither face is.
        if self.show_notch.get() and scale <= 0.0:
            segs = self._notch_segments()
            if segs:
                cm = matplotlib.colormaps[cmap]
                self.cax.add_collection(LineCollection(
                    segs, colors=[cm(0.55)], linewidths=2.0, zorder=4,
                    path_effects=[mpe.withStroke(linewidth=3.6,
                                                 foreground=cm(0.0))]))

        self.cax.set_aspect("equal", adjustable="datalim")
        label = field + (f"  [{spec['unit']}]" if spec["unit"] else "")
        if self._cbar is None:
            self._cbar = self.cfig.colorbar(cs, ax=self.cax, label=label)
            self._cbar.ax.yaxis.label.set_color(T["muted"])
            self._cbar.ax.tick_params(colors=T["muted"])
        else:
            self._cbar.set_label(label)
            self._cbar.ax.yaxis.label.set_color(T["muted"])
            # Without this the bar keeps the colours of the PREVIOUS map and
            # silently mislabels the plot.
            self._cbar.update_normal(cs)
        if had_view and not force:
            self.cax.set_xlim(*keep[0]); self.cax.set_ylim(*keep[1])
            box = (min(keep[0]), max(keep[0]), min(keep[1]), max(keep[1]))
        else:
            box = None                    # the whole specimen
        # LAST, and with the box handed to it. Two reasons it cannot go
        # earlier: set_aspect and the restore above both move the limits, and
        # asking matplotlib for them before a draw returns whatever they were
        # BEFORE autoscaling -- so the overlay would cull against the wrong
        # rectangle and, zoomed in, draw nothing at all.
        self._draw_crack_mesh(box)
        # Key on the colourbar: adding it changes the margins, so the first
        # draw after it appears must re-solve the layout.
        self._layout(self.cfig, "cbar" if self._cbar is not None else "")
        self.ccanvas.draw_idle()

        self._crack_step = step
        idx = next((i for i, (st, _p) in enumerate(self._crack_snaps)
                    if st == step), None)
        if idx is not None:
            self.crack_pos.set(f"{idx + 1} / {len(self._crack_snaps)}"
                               + ("" if self.follow_crack.get() else "  held"))
        if field == "phi":
            n_dam = int((values >= iso).sum())
            head = (f"max phi = {values.max():.3f}   "
                    f"nodes above {iso:g}: {n_dam}")
        else:
            unit = f" {spec['unit']}" if spec["unit"] else ""
            head = (f"{field} = {float(np.min(values)):.4g} .. "
                    f"{float(np.max(values)):.4g}{unit}"
                    f"   ({'per element' if where == 'cell' else 'per node'})")
        shape = ""
        if scale > 0.0:
            umax = float(np.hypot(disp[:, 0], disp[:, 1]).max())
            shape = (f"   deformed x{scale:.4g}"
                     f" (max |u| = {umax:.4g} mm)")
        self.crack_stats.set(
            f"step {step}   {head}   {os.path.basename(path)}{shape}"
            + getattr(self, "_crack_mesh_note", ""))

    def _cmap_changed(self):
        """Remember the choice against the KIND of field it was made for."""
        spec = CRACK_FIELDS.get(self.field_var.get(), CRACK_FIELDS["phi"])
        if spec["signed"]:
            self._cmap_div = self.cmap_var.get()
        else:
            self._cmap_seq = self.cmap_var.get()
        self._redraw_crack_overlay()

    def _deform_scale(self, disp):
        """The magnification actually in force, 0.0 meaning undeformed."""
        choice = self.deform_var.get()
        if choice == "undeformed" or disp is None or self._tri is None:
            return 0.0
        if choice != "auto":
            try:
                return float(choice.lstrip("x"))
            except ValueError:
                return 0.0
        umax = float(np.hypot(disp[:, 0], disp[:, 1]).max())
        if umax <= 0.0:
            return 0.0
        diag = float(np.hypot(self._tri.x.max() - self._tri.x.min(),
                              self._tri.y.max() - self._tri.y.min()))
        return AUTO_DEFORM_FRACTION * diag / umax

    def _deformed_xy(self, disp, scale):
        """Node coordinates at x + scale*u, or the reference ones."""
        if scale <= 0.0 or disp is None:
            return self._tri.x, self._tri.y
        return self._tri.x + scale * disp[:, 0], self._tri.y + scale * disp[:, 1]

    def _deform_changed(self):
        """Switch between the reference and the deformed shape.

        Only goes back to the file when it has to: the displacement is read
        alongside the field, so once it is in hand every other magnification
        is a redraw. Turning deformation ON for the first time on a frame that
        was loaded without it is the one case that needs a reload.
        """
        need = self.deform_var.get() != "undeformed"
        have = self._crack_last is not None and self._crack_last[5] is not None
        if need and not have:
            self.draw_crack(target=self._crack_step if self._crack_step >= 0
                            else None)
        else:
            self._redraw_crack_overlay()

    def _field_changed(self):
        """Switch the quantity being drawn.

        This one DOES have to go back to the file -- a different field is a
        different block of the snapshot, and it is not in memory. It reloads
        the frame you are on rather than jumping to the newest, so changing
        what you are looking at does not also change when you are looking at.
        """
        spec = CRACK_FIELDS.get(self.field_var.get(), CRACK_FIELDS["phi"])
        self.cmap_combo.configure(values=DIVERGING if spec["signed"] else CMAPS)
        self.cmap_var.set(self._cmap_div if spec["signed"] else self._cmap_seq)
        self.iso_entry.configure(
            state="normal" if self.field_var.get() == "phi" else "disabled")
        # A different field means a different range, so the saved view is kept
        # but the colour scale is rebuilt from scratch.
        self.draw_crack(target=self._crack_step if self._crack_step >= 0
                        else None)

    def _redraw_crack_overlay(self):
        """Re-render the frame already on screen with the overlay changed.

        Deliberately NOT draw_crack(): that would go back to the disk for a
        file whose contents are already in memory, and on the newest frame it
        would also have to wait for the stability check. Switching an overlay
        is a drawing decision, not a reason to re-read anything.
        """
        if self._crack_last is None:
            self.draw_crack()
            return
        path, step, values, where, field, disp = self._crack_last
        self._render_crack(path, step, values, where, field, disp)

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
        try:
            current_mtime = os.path.getmtime(full)
        except OSError:
            return []                        # no mesh to read: nothing to draw
        hit = _MESH_CACHE.get(full)
        if not hit or hit[0] != current_mtime:
            # NOT a silent give-up, which is what this was: the render must not
            # block on gmsh, but returning [] and waiting for "the next
            # refresh" meant the notch appeared only if you had already opened
            # the Mesh tab and pressed Draw mesh. Nothing else ever filled this
            # cache, so on a fresh session the overlay simply never worked.
            #
            # Ask for the mesh in the background instead and redraw when it
            # lands -- the same worker and queue every other read here uses.
            self._request_notch_mesh(full, current_mtime)
            return []
        m = hit[1]
        segs = []
        for name, s in m.get("edges", {}).items():
            # A crack MOUTH/TIP is a 0D group and never lands here; the mouth
            # group would only be two coincident points anyway.
            if self._NOTCH_RE.search(name):
                segs.extend(s)
        return segs

    def _request_notch_mesh(self, full, mtime):
        """Read the mesh off the Tk thread, once, then re-render.

        Keyed on (path, mtime) so a mesh that fails to read is not retried on
        every frame -- _render_crack calls this, and _notch_loaded triggers a
        render, so without that guard a broken mesh would spin between the
        two forever. Editing the mesh changes the mtime and it tries again.
        """
        key = (full, mtime)
        if key in self._notch_tried:
            return
        self._notch_tried.add(key)
        threading.Thread(target=self._notch_worker, args=(full,),
                         daemon=True).start()

    def _notch_worker(self, full):
        try:
            with self._io_lock:              # gmsh, one caller at a time
                read_mesh_for_plot(full)     # its only job is to fill the cache
        except Exception:
            pass
        self.io_q.put((self._notch_loaded, ()))

    def _notch_loaded(self):
        if self.show_notch.get() and self._crack_last is not None:
            self._redraw_crack_overlay()

    def _crack_polys(self, scale, dx, dy):
        """Element outlines on the current geometry.

        The undeformed set is cached from the parse; the deformed one is
        rebuilt from the node indices, which is why read_vtk_geometry keeps
        them. One fancy-index over an (nelem, 4) array -- cheap enough to do
        per frame, and it cannot drift from the nodes it was built from.
        """
        if scale <= 0.0 or self._tri_cells is None:
            return self._tri_polys
        return np.stack([dx, dy], axis=1)[self._tri_cells]

    def _draw_crack_mesh(self, box=None):
        """The mesh over the contour: nodes, element outlines, or both.

        `box` is (x0, x1, y0, y1) to cull against -- the view that is about to
        be shown. None means the whole specimen.

        Culled to the visible axes, because that is the only view in which the
        question makes sense. Zoomed out, 340k elements are sub-pixel and the
        overlay is a grey wash over the field it is supposed to be explaining;
        zoomed into the tip, the same code draws a few hundred and answers
        "is l0 resolved here" at a glance.
        """
        mode = self.crack_overlay.get()
        if mode == "none" or self._tri is None:
            self._crack_mesh_note = ""
            return
        want_nodes = "nodes" in mode
        want_elems = "element" in mode
        # Whatever geometry the field was just drawn on -- the overlay has to
        # sit on the SAME nodes, or a deformed contour would be wearing an
        # undeformed wireframe. Hoisted above the box branch below, which
        # falls back to these coordinates for its extent.
        scale = getattr(self, "_crack_scale", 0.0)
        disp = self._crack_last[5] if self._crack_last else None
        dx, dy = self._deformed_xy(disp, scale)

        if box is not None:
            x0, x1, y0, y1 = box
            # A small margin, because set_aspect("equal", adjustable="datalim")
            # is allowed to widen the limits after they are set -- matplotlib
            # says so out loud ("Ignoring fixed x limits to fulfil fixed data
            # aspect"). Culling against the REQUESTED box would then leave a
            # thin band at the edge of the view with no overlay on it.
            mx, my = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
            x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
        else:
            x0, x1 = float(dx.min()), float(dx.max())
            y0, y1 = float(dy.min()), float(dy.max())
        notes = []

        # White with a dark edge rather than a single flat colour: a
        # sequential map runs from near-black to near-white, so ANY one colour
        # disappears at one end of it. The pairing survives both ends.
        if want_elems and self._tri_polys is not None:
            polys = self._crack_polys(scale, dx, dy)
            cent = polys.mean(axis=1)
            vis = ((cent[:, 0] >= x0) & (cent[:, 0] <= x1) &
                   (cent[:, 1] >= y0) & (cent[:, 1] <= y1))
            n_vis = int(vis.sum())
            if n_vis == 0:
                pass
            elif n_vis > CRACK_ELEM_LIMIT:
                notes.append(f"{n_vis:,} elements in view - zoom in to draw them")
            else:
                self.cax.add_collection(PolyCollection(
                    polys[vis], facecolors="none", edgecolors="#ffffff",
                    linewidths=0.4, alpha=0.45, zorder=3))
                notes.append(f"{n_vis:,} elements")

        if want_nodes:
            xs, ys = dx, dy
            vis = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
            n_vis = int(vis.sum())
            if n_vis == 0:
                pass
            elif n_vis > CRACK_NODE_LIMIT:
                notes.append(f"{n_vis:,} nodes in view - zoom in to draw them")
            else:
                # Size backs off as they crowd together, so a refined patch
                # reads as a mesh instead of as one solid blob of markers.
                ms = 3.4 if n_vis < 1500 else (2.4 if n_vis < 5000 else 1.6)
                self.cax.plot(xs[vis], ys[vis], "o", ms=ms, mfc="#ffffff",
                              mec="#101010", mew=0.35, linestyle="none",
                              alpha=0.9, zorder=5)
                notes.append(f"{n_vis:,} nodes")
        self._crack_mesh_note = ("   mesh: " + ", ".join(notes)) if notes else ""

    def fit_crack(self):
        """Refit the axes to the specimen, WITHOUT jumping to the newest
        frame -- fitting the view and choosing which snapshot to look at are
        two different intentions."""
        here = self._crack_step
        self._crack_step = -1        # forget the saved view
        self.draw_crack(force=True,
                        target=None if self.follow_crack.get() else here)

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
        self._refresh_warnings()

    def import_config(self, path: str = ""):
        if not path:
            path = filedialog.askopenfilename(
                initialdir=PROJECT_DIR, title="Import a run config",
                filetypes=[("TOML config", "*.toml"), ("All files", "*.*")])
            if not path:
                return
        try:
            self.set_form(config_to_form(_toml_load(path)))
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
            self._append_log(f"[gui] saved {p}\n")

    def _refresh_warnings(self):
        self.warnbox.delete("1.0", "end")
        try:
            f = self.get_form()
        except ValueError as exc:
            self.warnbox.insert("end", str(exc))
            for key in self._field_section:
                if f"'{key}'" in str(exc):
                    self._reveal_field(key)
            self._refresh_section_summaries()
            return
        msgs = ([("PROBLEM: " + m) for m in validate_form(f)]
                + [("PROBLEM: " + m)
                   for m in check_group_dims(f["bcs"], self.mesh_groups)]
                + warn_form(f))
        self.warnbox.insert("end", "\n\n".join(msgs) if msgs else "")
        # A complaint about a field you cannot see is a dead end, so anything
        # named in a PROBLEM opens the section holding it. Only problems --
        # warn_form's notes are advice, not a reason to rearrange the panel
        # under someone's hands.
        problems = " ".join(msgs[:len(msgs) - len(warn_form(f))])
        for key in self._field_section:
            if key in problems:
                self._reveal_field(key)
        self._refresh_section_summaries()
        self._show_plan(f)

    def _update_progress(self):
        """Progress bar + a stage-aware ETA.

        ACCEPTED STEP NUMBERS ARE NOT PROGRESS. Two separate things break that
        assumption: adaptive half-stepping inserts steps that were never
        planned, and a staged schedule can fail to reach a later stage at all.
        So the bar is driven by the DISPLACEMENT actually reached, read from
        the CSV the solver is already flushing, and only the labels talk about
        steps.

        The ETA is summed per remaining stage, each at its own measured rate
        (`_stage_times`), because a coarse step and a step during crack growth
        cost very different amounts and one pooled average predicts neither.
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
        u_now = du_actual = None
        if staged:
            u, _F, _phi, _conv, _labels = read_curve(self.csv_path)
            if u:
                u_now = abs(u[-1])
            if len(u) >= 2:
                step = abs(u[-1] - u[-2])
                if step > 0.0:
                    du_actual = step

        stuck = ""
        if staged and u_now is not None:
            stages, _u_ref = staged
            remaining = remaining_stage_steps(stages, u_now)
            # Steps already behind us, measured in the schedule's own terms.
            # Taken BEFORE the correction below, because what is done is done
            # -- only the estimate of what is left can change.
            done_nominal = max(0, total - sum(remaining))
            active = next((i for i, n in enumerate(remaining) if n),
                          len(stages) - 1)

            # THE INCREMENT THE SOLVER IS REALLY TAKING, not the one the
            # config asked for.
            #
            # main.cpp used to clamp lf_inc DOWN at a stage boundary and never
            # up, so a stage COARSER than the one before it was reachable only
            # through the adaptive re-growth -- two consecutive CONVERGED
            # steps -- which during crack growth never happens. One measured
            # run planned 600 steps for its final stage and executed 6,000.
            #
            # A boundary now SETS the increment in both directions, so this
            # should no longer fire. It is kept because it is a few lines and
            # still catches the case that matters: an exe built before that
            # change. Nothing else here can tell you which solver you just
            # launched.
            nom = stages[active][3]
            if du_actual and du_actual < 0.8 * nom:
                end = stages[active][2]
                width = max(0.0, end - u_now)
                remaining[active] = max(0, math.ceil(width / du_actual - 1e-7))
                stuck = (f"increment {du_actual:g}, schedule says {nom:g}"
                         f" - stage not entered")

            nominal_left = sum(remaining)
            # Against a REVISED total, not the original plan. Once the
            # correction above has decided the run needs 5,600 more steps
            # where the plan allowed 560, dividing by the plan gives a
            # fraction below zero and a bar that reads 0% two thirds of the
            # way through the loading. The denominator has to grow with the
            # estimate.
            revised = done_nominal + nominal_left
            frac = min(done_nominal / revised, 1.0) if revised else 0.0
            self.progress["value"] = int(1000 * frac)
            bits = [f"step {done:,}", f"{100.0 * frac:.1f}%",
                    f"stage {active + 1}/{len(stages)} {stages[active][0]}"]
            if rate:
                bits.append(f"{rate:.2f} s/step")
            if nominal_left and rate:
                eta = 0.0
                for i, count in enumerate(remaining):
                    if not count:
                        continue
                    samples = self._stage_times[i]
                    stage_rate = (sum(samples) / len(samples) if samples
                                  else self._last_stage_rates[i] or rate)
                    eta += count * stage_rate
                bits.append(f"ETA ~{fmt_duration(eta)}")
            if stuck:
                bits.append(stuck)
            self.progress_txt.set("   ".join(bits))
            return

        # Unstaged, or the CSV has not produced a point yet: fall back to
        # counting steps against the plan. Passing the plan is REPORTED rather
        # than fallen off the end of -- the old `done < total` guard made the
        # bar pin at 100% and the ETA vanish, which reads as finished.
        over = bool(total) and done > total
        frac = 1.0 if over else (min(done / total, 1.0) if total else 0.0)
        self.progress["value"] = int(1000 * frac)
        if not total:
            bits = [f"step {done:,}"]
        elif over:
            bits = [f"step {done:,}   {done - total:,} past the planned "
                    f"{total:,}"]
        else:
            bits = [f"step {done:,} / {total:,}"]
        if rate:
            bits.append(f"{rate:.2f} s/step")
        if over:
            bits.append("ETA unknown - past the planned schedule")
        elif total and rate:
            bits.append(f"ETA {fmt_duration((total - done) * rate)}")
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
        detail = (f"  ({n_c} coarse + {n_f:,} fine)" if mode == "two_stage"
                  else f"  ({n_c} + {n_f:,} over 2 finer stages)"
                  if mode == "three_stage" else "  (uniform)")
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
            self.ax.axvline(x_sw, color=T["muted"], lw=1.0, ls="--", zorder=0)
            self.ax.annotate("u_switch", xy=(x_sw, 0), xycoords=("data", "axes fraction"),
                             xytext=(3, 4), textcoords="offset points",
                             color=T["muted"], fontsize=8)

        if i is None:
            return
        up, Fp = u[i], F[i]
        coarse = two_stage and u_sw > 0 and abs(up) < u_sw
        col = T["err"] if coarse else T["warn"]
        self.ax.plot([up], [Fp], marker="v", ms=9, color=col,
                     linestyle="none", zorder=4)
        self.ax.annotate(f"peak {abs(Fp):.4g}\n@ u = {abs(up):.4g}",
                         xy=(up, Fp), xytext=(6, -22),
                         textcoords="offset points", fontsize=8, color=col)

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
        self._last_elapsed = -1
        self.readout.set("")
        self._set_state(f"{label}...", "busy")
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
        # Blocking here on purpose: validation below needs the group
        # dimensions before it can decide whether this run is safe to start.
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
        # The incremental CSV reader keys on the path, and this run is about
        # to overwrite that path with a different curve.
        reset_curve_cache(self.csv_path)
        self._stage_times = [[], [], []]
        self._last_stage_rates = [0.0, 0.0, 0.0]
        self._run_form = None if check_only else f
        # A new run may use a different mesh, so the cached triangulation and
        # the last-seen step must not carry over -- otherwise phi from the new
        # run gets drawn on the old geometry, or silently rejected for having
        # the wrong length.
        self._tri = None
        self._tri_npoin = 0
        self._tri_polys = None
        self._tri_cells = None
        self._crack_last = None
        self._crack_step = -1
        self._vtk_sizes.clear()

        # Progress: a mesh-only check has no load steps, so leave the bar blank
        # rather than showing a plan that will never advance.
        plan = planned_steps(f)
        self._plan_total = 0 if check_only else (plan[0] if plan else 0)
        self._step_times = []
        self._last_step = 0
        self._peak_reported = False
        self.progress["value"] = 0
        self.progress_txt.set(
            "checking mesh..." if check_only
            else (f"0 / {self._plan_total:,} steps" if self._plan_total
                  else "running (step count unknown)"))
        if not check_only and os.path.isfile(self.csv_path):
            # A leftover CSV would be drawn instantly and look like progress,
            # so it has to go -- but MOVED, not deleted. See archive_curve.
            kept = archive_curve(self.csv_path)
            if kept:
                self._append_log(
                    f"[gui] {os.path.basename(self.csv_path)} already held a "
                    f"finished run; kept as {os.path.basename(kept)}\n")
        # A new run is a new curve, so any held view belongs to the old one.
        self.follow_curve.set(True)
        self.curve_hint.set("")
        self._curve_drawn = False
        self.ax.clear(); style_axes(self.fig, self.ax); self.canvas.draw_idle()

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
        self._last_elapsed = -1
        self.readout.set("")
        self._set_state(f"{label}...", "busy")
        for b in (self.btn_run, self.btn_check, self.btn_gen):
            b.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()

    def _stop_worker(self, proc):
        signal_process_tree(proc, hard=False)
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self.log_q.put("[gui] graceful stop timed out; forcing process tree shutdown\n")
            signal_process_tree(proc, hard=True)

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
            # Off the UI thread: taskkill /T walks the process tree and can
            # take a second or two, and freezing the window while stopping a
            # run is precisely the wrong moment to freeze it.
            proc = self.proc
            self._set_state("stopping...", "busy")
            self.btn_stop.configure(state="disabled")
            threading.Thread(target=self._stop_worker, args=(proc,),
                             daemon=True).start()

    # -- periodic -------------------------------------------------------
    @staticmethod
    def _tag_for(text):
        low = text.lower()
        return ("err" if ("error" in low or "diverged" in low
                          or "non-converged" in low)
                else "warn" if ("warning" in low or low.startswith("[bc]"))
                else "")

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
        # Only follow the tail if the view is ALREADY at the bottom. Otherwise
        # scrolling back to read something gets yanked away every 60 ms, which
        # a faster refresh would make unbearable.
        try:
            at_bottom = self.log.yview()[1] > 0.999
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

    def _tick(self):
        # Results from the background readers land here as plain data. Bounded
        # per tick so a burst cannot monopolise the frame, exactly like the
        # log drain below.
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
            # The clock. Once a SECOND, not once a tick: a StringVar write
            # redraws the label, and at POLL_MS that is 40 redraws a second to
            # show a number that changes once.
            secs = int(time.time() - self.start_time)
            if secs != self._last_elapsed:
                self._last_elapsed = secs
                self.elapsed.set(fmt_duration(secs))

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
                self.elapsed.set(fmt_duration(elapsed))
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
                    plan = self._plan_total
                    if plan and self._last_step < plan:
                        note = f"  (stopped before the planned {plan:,})"
                    elif plan and self._last_step > plan:
                        # Worth saying out loud: it is the fingerprint of a
                        # stage schedule the solver never actually entered.
                        note = (f"  ({self._last_step - plan:,} more than the "
                                f"planned {plan:,})")
                    else:
                        note = ""
                    self.progress_txt.set(
                        f"{self._last_step:,} steps in {fmt_duration(elapsed)}"
                        + note)
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
        # ax.clear() resets the limits to autoscale. That is what you want
        # while the curve is still growing, and exactly what you do not want
        # the moment you have zoomed in to look at something -- which is when
        # this runs most often. Capture the view BEFORE clearing and put it
        # back afterwards whenever following is off.
        hold = self._curve_drawn and not self.follow_curve.get()
        keep = (self.ax.get_xlim(), self.ax.get_ylim())
        self.last_rows = len(u)
        self.ax.clear(); style_axes(self.fig, self.ax)
        self.ax.plot(u, F, "-", lw=1.4, color=T["line"], zorder=1)

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
        if ok:
            stride = max(1, len(ok) // MAX_MARKERS)
            shown = ok[::stride]
            lbl = "converged" + (f" (every {stride})" if stride > 1 else "")
            self.ax.plot([p[0] for p in shown], [p[1] for p in shown], "o",
                         ms=3.5, mfc="none", mew=1.1, mec=T["line"],
                         linestyle="none", label=lbl, zorder=2)
        if bad:
            self.ax.plot([b[0] for b in bad], [b[1] for b in bad], "o", ms=4.5,
                         color=T["err"], linestyle="none",
                         label="not converged", zorder=3)
        if ok or bad:
            leg = self.ax.legend(loc="best", fontsize=8,
                                 facecolor=T["panel"], edgecolor=T["grid"])
            for txt in leg.get_texts():
                txt.set_color(T["fg"])
        self._annotate_peak(u, F)
        self.ax.set_xlabel(labels[0]); self.ax.set_ylabel(labels[1])
        # After the annotations, not before: _annotate_peak draws the u_switch
        # guide with axvline, which is in DATA coordinates and widens the
        # x range to include it.
        if hold:
            self.ax.set_xlim(*keep[0])
            self.ax.set_ylim(*keep[1])
        self._curve_drawn = True
        # The axis labels come from the CSV header (applied_uy vs applied_ux),
        # so they are part of the layout key.
        self._layout(self.fig, f"{labels[0]}|{labels[1]}")
        self.canvas.draw_idle()

        peak = max((abs(x) for x in F), default=0.0)
        # No elapsed time here any more -- the clock has its own slot, which
        # is updated once a SECOND rather than being rewritten as a side
        # effect of every plot refresh.
        self.readout.set(
            f"u = {u[-1]:.5g}   F = {F[-1]:.5g}   "
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
