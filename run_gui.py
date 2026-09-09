#!/usr/bin/env python3
"""
Set up, run and watch a phase-field solve.

    pip install matplotlib
    python run_gui.py
    python run_gui.py config_ambati_3pb.toml

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
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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
    "edge": "#33363b", "outline": "#2c2f34", "cell": "#2b2e33",
    # Mesh view: near-white wireframe on a face darker than the panel, so the
    # elements read as a mesh rather than as texture.
    "mesh_edge": "#dfe6ee", "mesh_face": "#15171a", "tabsel": "#4a9eff",
}
LIGHT = {
    "bg": "#f0f0f0", "panel": "#e4e4e4", "field": "#ffffff", "hover": "#d8d8d8",
    "fg": "#101010", "muted": "#555555", "accent": "#0a5fb4",
    "warn": "#b36b00", "err": "#b00020", "grid": "#cccccc", "line": "#1f77b4",
    "warnbg": "#fffaf0",
    "edge": "#c4c4c4", "outline": "#d2d2d2", "cell": "#fbfbfb",
    "mesh_edge": "#222222", "mesh_face": "#ffffff", "tabsel": "#0a5fb4",
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
                    relief="flat", borderwidth=1, padding=4)
    style.map("TButton",
              background=[("active", T["hover"]), ("disabled", T["bg"])],
              foreground=[("active", T["fg"]), ("disabled", T["muted"])],
              bordercolor=[("active", T["grid"])],
              lightcolor=[("active", T["grid"])],
              darkcolor=[("active", T["grid"])])

    # Table cells: flush, flat, dim border. Same reasoning as the buttons --
    # the bevel is what made a row of entries look like a cage.
    style.configure("Cell.TEntry", fieldbackground=T["cell"],
                    foreground=T["fg"], bordercolor=T["edge"],
                    lightcolor=T["edge"], darkcolor=T["edge"],
                    insertcolor=T["fg"], borderwidth=1, relief="flat",
                    padding=3)
    style.map("Cell.TEntry",
              bordercolor=[("focus", T["accent"])],
              lightcolor=[("focus", T["accent"])],
              darkcolor=[("focus", T["accent"])])
    style.configure("Cell.TCombobox", fieldbackground=T["cell"],
                    foreground=T["fg"], background=T["panel"],
                    arrowcolor=T["muted"], bordercolor=T["edge"],
                    lightcolor=T["edge"], darkcolor=T["edge"],
                    borderwidth=1, relief="flat", padding=2)
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
    # The two views are the main navigation, so the tabs are made loud: the
    # selected one takes the accent colour outright rather than a subtle tint.
    style.configure("TNotebook", background=T["bg"], bordercolor=T["grid"],
                    tabmargins=[2, 6, 2, 0])
    style.configure("TNotebook.Tab", background=T["panel"],
                    foreground=T["muted"], bordercolor=T["grid"],
                    padding=[18, 8], font=("TkDefaultFont", 10, "bold"))
    style.map("TNotebook.Tab",
              background=[("selected", T["tabsel"]), ("active", T["hover"])],
              foreground=[("selected", "#000000"), ("active", T["fg"])],
              expand=[("selected", [1, 1, 1, 0])])

    # The sash is invisible at its default 2 px on a dark theme; make it wide
    # enough to grab and light enough to see.
    style.configure("TPanedwindow", background=T["grid"])
    style.configure("Sash", sashthickness=7, gripcount=12,
                    background=T["grid"], lightcolor=T["hover"],
                    bordercolor=T["bg"], handlepad=60)

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
    L = ["# Generated by run_gui.py -- edit the GUI, not this file.",
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


def planned_steps(f: dict):
    """(n_total, n_first, n_rest, u_ref) or None if it cannot be determined."""
    try:
        if f.get("step_mode") == "three_stage":
            u_ref = u_ref_of(f)
            du1, u1 = float(f["du_coarse"]), float(f["u_switch"])
            du2, u2 = float(f["du_fine"]), float(f["u_switch2"])
            du3 = float(f["du_final"])
            if min(u_ref, du1, du2, du3) <= 0.0 or u2 <= u1:
                return None
            u1 = min(u1, u_ref)
            u2 = min(u2, u_ref)
            n1 = math.ceil(u1 / du1 - 1e-12)
            n2 = math.ceil((u2 - u1) / du2 - 1e-12)
            n3 = math.ceil((u_ref - u2) / du3 - 1e-12)
            return n1 + n2 + n3, n1, n2 + n3, u_ref
        if f.get("step_mode") != "two_stage":
            u_ref = u_ref_of(f)
            du = float(f.get("du", 0.0) or 0.0)
            # Mirrors main.cpp: run.du wins when set, N_steps is the fallback.
            if du > 0.0 and u_ref > 0.0:
                n = math.ceil(u_ref / du - 1e-12)
                return n, n, 0, u_ref
            n = int(f.get("N_steps", 0))
            return (n, n, 0, u_ref) if n > 0 else None
        u_ref = u_ref_of(f)
        du_c = float(f["du_coarse"])
        du_f = float(f["du_fine"])
        u_sw = float(f["u_switch"])
    except (KeyError, TypeError, ValueError):
        return None
    if u_ref <= 0.0 or du_c <= 0.0 or du_f <= 0.0:
        return None
    u_sw = max(0.0, min(u_sw, u_ref))          # clamped exactly as the solver does
    n_coarse = math.ceil(u_sw / du_c - 1e-12)
    n_fine = math.ceil((u_ref - u_sw) / du_f - 1e-12)
    return n_coarse + n_fine, n_coarse, n_fine, u_ref


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
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            rows = list(csv.reader(fh))
    except OSError:
        return [], [], [], [], ("u", "F")
    if len(rows) < 2:
        return [], [], [], [], ("u", "F")

    header = [h.strip() for h in rows[0]]
    def col(prefix):
        return next((i for i, h in enumerate(header) if h.startswith(prefix)), None)
    iu, iF, iphi, iconv = (col("applied_u"), col("reaction_F"),
                           col("max_phi"), col("converged"))
    if iu is None or iF is None:
        return [], [], [], [], ("u", "F")

    u, F, phi, conv = [], [], [], []
    for r in rows[1:]:
        try:
            vu, vF = float(r[iu]), float(r[iF])
            vphi = float(r[iphi]) if iphi is not None else 0.0
            vconv = int(float(r[iconv])) if iconv is not None else 1
        except (ValueError, IndexError):
            continue
        u.append(vu); F.append(vF); phi.append(vphi); conv.append(vconv)
    return u, F, phi, conv, (header[iu], header[iF])


_MESH_CACHE = {}          # path -> (mtime, data)


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
        gmsh.initialize()
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
        gmsh.initialize()
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
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        except Exception:
            return
        self.win = tk.Toplevel(self.widget)
        self.win.wm_overrideredirect(True)
        self.win.wm_geometry(f"+{x}+{y}")
        tk.Label(self.win, text=self.text, justify="left", wraplength=380,
                 background=T["panel"], foreground=T["fg"], relief="solid",
                 borderwidth=1, padx=7, pady=5,
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
                      wraplength=470).pack(anchor="w", pady=(0, 2))

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
                                 font=("Consolas", 8), padx=6, pady=4)
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

    def __init__(self, parent, width=500):
        super().__init__(parent)
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
        for c, (text, w) in enumerate(self.COLS):
            ttk.Label(self.body, text=text, width=w, anchor="w",
                      foreground=T["muted"]).grid(row=0, column=c, padx=1,
                                                  sticky="w")
            self.body.grid_columnconfigure(c, minsize=w * 7)

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
        root.geometry("1460x860")
        apply_theme(root)
        self.proc = None
        self.proc_kind = ""          # "solve" | "check" | "mesh"
        self._on_success = None
        self.log_q: queue.Queue = queue.Queue()
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
        self._last_step = 0
        self._last_rate = 0.0       # s/step carried over from the previous run
        self._peak_reported = False
        self._laid_out: dict = {}   # id(figure) -> layout key, see _layout()
        self._dragging = False      # a pane sash is being dragged
        self._resize_pending: set = set()
        self._rows: dict = {}           # key -> row frame, for show/hide
        self._section_rows: dict = {}   # id(section) -> [(key, row, section)]

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
                               borderwidth=0, sashwidth=6, sashpad=0,
                               sashrelief="flat")
        outer.pack(fill="both", expand=True)

        panel_host = ScrollFrame(outer, width=500)
        # stretch="never": extra space from resizing the WINDOW goes to the
        # plot, not the form -- the form does not get more useful when wider.
        outer.add(panel_host, width=500, minsize=260, stretch="never")
        self.panel = panel_host.inner
        self.panel_host = panel_host

        left = ttk.Frame(outer)          # results area: buttons, plot, log
        outer.add(left, minsize=420, stretch="always")
        self.paned = outer

        # --- top bar ---
        bar = ttk.Frame(left, padding=8)
        bar.pack(fill="x")
        ttk.Label(bar, text="Solver").pack(side="left")
        self.exe_var = tk.StringVar(value=find_exe())
        ttk.Entry(bar, textvariable=self.exe_var, width=48).pack(
            side="left", padx=4)
        ttk.Button(bar, text="...", width=3, command=self._pick_exe).pack(side="left")

        bar2 = ttk.Frame(left, padding=(8, 0))
        bar2.pack(fill="x")
        self.btn_check = ttk.Button(bar2, text="Check mesh", command=self.check_mesh)
        self.btn_check.pack(side="left")
        self.btn_run = ttk.Button(bar2, text="Run", command=self.run)
        self.btn_run.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(bar2, text="Stop", command=self.stop,
                                   state="disabled")
        self.btn_stop.pack(side="left")
        ttk.Button(bar2, text="Output folder",
                   command=self._open_output).pack(side="right")

        # Progress row. Deliberately separate from `status`, which _refresh_plot
        # rewrites every 500 ms -- sharing one label would make them clobber
        # each other.
        prow = ttk.Frame(left, padding=(8, 4, 8, 0))
        prow.pack(fill="x")
        self.progress = ttk.Progressbar(prow, mode="determinate", maximum=1000)
        self.progress.pack(side="left", fill="x", expand=True)
        self.progress_txt = tk.StringVar(value="")
        ttk.Label(prow, textvariable=self.progress_txt, width=46,
                  anchor="w").pack(side="left", padx=8)

        self.status = tk.StringVar(value="idle")
        ttk.Label(left, textvariable=self.status, padding=(8, 6),
                  font=("TkDefaultFont", 10, "bold")).pack(fill="x")

        # Second splitter: plots above, log below.
        vpane = tk.PanedWindow(left, orient="vertical", opaqueresize=True,
                               background=T["grid"], borderwidth=0,
                               sashwidth=6, sashpad=0, sashrelief="flat")
        vpane.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self.vpaned = vpane

        nb = ttk.Notebook(vpane)
        vpane.add(nb, minsize=180, stretch="always")
        self.nb = nb

        curve_tab = ttk.Frame(nb)
        nb.add(curve_tab, text="Force - displacement")
        self.fig = Figure(figsize=(7, 3.4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("applied displacement")
        self.ax.set_ylabel("reaction force")
        style_axes(self.fig, self.ax)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=curve_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        add_toolbar(self.canvas, curve_tab).pack(fill="x")

        mesh_tab = ttk.Frame(nb)
        nb.add(mesh_tab, text="Mesh")
        self.mesh_tab = mesh_tab
        # Redrawing a tab nobody is looking at is wasted work, so edits made
        # while the curve is showing only set a flag; the redraw happens when
        # you switch to the Mesh tab.
        nb.bind("<<NotebookTabChanged>>", self._tab_changed)
        mbar = ttk.Frame(mesh_tab, padding=(0, 4))
        mbar.pack(fill="x")
        ttk.Button(mbar, text="Draw mesh", command=self.draw_mesh).pack(side="left")
        ttk.Button(mbar, text="Fit", width=5,
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
        self.mcanvas.get_tk_widget().pack(fill="both", expand=True)
        add_toolbar(self.mcanvas, mesh_tab).pack(fill="x")
        # Wheel-zoom about the cursor, which is what a mesh viewer should do.
        # The toolbar still offers rubber-band zoom and pan for fine control.
        self.mcanvas.mpl_connect("scroll_event", self._mesh_wheel_zoom)
        self.mesh_bbox = None

        # ---- Crack tab: phi from the solver's VTK snapshots ----------------
        crack_tab = ttk.Frame(nb)
        nb.add(crack_tab, text="Crack (phi)")
        self.crack_tab = crack_tab
        cbar_ = ttk.Frame(crack_tab, padding=(0, 4))
        cbar_.pack(fill="x")
        ttk.Button(cbar_, text="Refresh", command=self.draw_crack).pack(side="left")
        ttk.Button(cbar_, text="Fit", width=5,
                   command=self.fit_crack).pack(side="left", padx=4)
        self.follow_crack = tk.BooleanVar(value=True)
        ttk.Checkbutton(cbar_, text="follow latest snapshot",
                        variable=self.follow_crack).pack(side="left", padx=8)
        self.show_notch = tk.BooleanVar(value=True)
        ttk.Checkbutton(cbar_, text="notch", variable=self.show_notch,
                        command=self.draw_crack).pack(side="left")
        ttk.Label(cbar_, text="crack iso").pack(side="left", padx=(8, 2))
        self.iso_var = tk.StringVar(value="0.9")
        ttk.Entry(cbar_, textvariable=self.iso_var, width=5).pack(side="left")
        ttk.Label(cbar_, text="colours").pack(side="left", padx=(8, 2))
        self.cmap_var = tk.StringVar(value="inferno")
        cmb = ttk.Combobox(cbar_, textvariable=self.cmap_var, values=CMAPS,
                           width=10, state="readonly")
        cmb.pack(side="left")
        cmb.bind("<<ComboboxSelected>>", lambda _e: self.draw_crack(force=True))
        self.crack_stats = tk.StringVar(value="no snapshots yet")
        ttk.Label(cbar_, textvariable=self.crack_stats,
                  foreground=T["muted"]).pack(side="left", padx=8)

        self.cfig = Figure(figsize=(7, 3.4), dpi=100)
        self.cax = self.cfig.add_subplot(111)
        style_axes(self.cfig, self.cax)
        self.cfig.tight_layout()
        self.ccanvas = FigureCanvasTkAgg(self.cfig, master=crack_tab)
        self.ccanvas.get_tk_widget().pack(fill="both", expand=True)
        add_toolbar(self.ccanvas, crack_tab).pack(fill="x")
        self._cbar = None          # colourbar, created once (levels are fixed)
        self._tri = None           # cached Triangulation
        self._tri_npoin = 0
        self._crack_step = -1
        self._vtk_sizes: dict = {}

        # stretch="never" so growing the window grows the PLOT and leaves the
        # log at whatever height you dragged it to.
        lf = ttk.Frame(vpane, padding=(0, 6, 0, 0))
        vpane.add(lf, height=210, minsize=60, stretch="never")
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

        self._build_panel()

    def _section(self, title):
        f = ttk.LabelFrame(self.panel, text=title, padding=6)
        f.pack(fill="x", padx=6, pady=4)
        return f

    def _field(self, parent, label, key, width=14, kind="entry", values=None,
               on_change=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=15).pack(side="left")
        if kind == "check":
            v = tk.BooleanVar()
            ttk.Checkbutton(row, variable=v).pack(side="left")
        elif kind == "combo":
            v = tk.StringVar()
            cb = ttk.Combobox(row, textvariable=v, values=values,
                              width=width - 2, state="readonly")
            cb.pack(side="left")
            if on_change:
                cb.bind("<<ComboboxSelected>>", lambda _e: on_change())
        else:
            v = tk.StringVar()
            ttk.Entry(row, textvariable=v, width=width).pack(side="left")
        self.vars[key] = v
        # Remember the row and its position, so _apply_visibility can hide the
        # fields that the current mode ignores and put them back in order.
        self._rows[key] = row
        self._section_rows.setdefault(id(parent), []).append((key, row, parent))
        return v

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
        """Re-pack each managed section so only usable fields are shown.

        Forget-all-then-repack rather than hiding rows in place: pack() appends
        to the end, so a row put back individually would jump to the bottom of
        its section. Re-packing the whole section in creation order keeps them
        where they belong. It is a handful of widgets, so the cost is nil.
        """
        hide = self._visible_keys()
        for rows in self._section_rows.values():
            for _, row, _ in rows:
                row.pack_forget()
            for key, row, _ in rows:
                if key not in hide:
                    row.pack(fill="x", pady=1)

    def _build_panel(self):
        top = ttk.Frame(self.panel, padding=(6, 6, 6, 0))
        top.pack(fill="x")
        ttk.Button(top, text="Import config...", command=lambda: self.import_config()
                   ).pack(side="left")
        ttk.Button(top, text="Save as...", command=self.save_config).pack(
            side="left", padx=4)
        ttk.Label(top, text="drag the dividers to resize the panel and the log",
                  foreground=T["muted"]).pack(side="left", padx=6)

        # ---- Geometry: build a mesh, rather than only pointing at one ------
        g = self._section("1. Geometry")
        row = ttk.Frame(g); row.pack(fill="x", pady=1)
        ttk.Label(row, text="source", width=15).pack(side="left")
        self.geom_source = tk.StringVar()
        sources = list_specs() + ["STEP / IGES file..."]
        self.geom_combo = ttk.Combobox(row, textvariable=self.geom_source,
                                       values=sources, width=20,
                                       state="readonly")
        self.geom_combo.pack(side="left")
        self.geom_combo.bind("<<ComboboxSelected>>", self._geom_source_changed)

        # Parameter fields are REBUILT from the chosen spec's [params]. The
        # specs already declare value/min/max/step, which is exactly what a
        # form needs -- so a new spec file gets a working UI for free.
        self.geom_param_frame = ttk.Frame(g)
        self.geom_param_frame.pack(fill="x")
        self.geom_vars = {}
        self.geom_tables = {}
        self.geom_cad = ""

        row = ttk.Frame(g); row.pack(fill="x", pady=1)
        ttk.Label(row, text="write to", width=15).pack(side="left")
        self.geom_out = tk.StringVar(value="meshes/generated.msh")
        ttk.Entry(row, textvariable=self.geom_out, width=18).pack(side="left")

        self.btn_gen = ttk.Button(g, text="Generate mesh",
                                  command=self.generate_mesh)
        self.btn_gen.pack(anchor="w", pady=(4, 0))
        ttk.Label(g, text="runs the mesher as a separate process; the log is "
                          "below and the result loads into section 2",
                  foreground=T["muted"], wraplength=430).pack(anchor="w")

        s = self._section("2. Mesh")
        self._field(s, "source", "mesh_source", kind="combo",
                    values=["file", "builtin"],
                    on_change=self._apply_visibility)
        row = ttk.Frame(s); row.pack(fill="x", pady=1)
        ttk.Label(row, text="mesh file", width=15).pack(side="left")
        self.vars["mesh_path"] = tk.StringVar()
        ttk.Entry(row, textvariable=self.vars["mesh_path"], width=12).pack(side="left")
        ttk.Button(row, text="...", width=3, command=self._pick_mesh).pack(side="left")
        # Registered by hand: this row is built inline rather than through
        # _field (it carries a Browse button), so _apply_visibility would
        # never see it otherwise.
        self._rows["mesh_path"] = row
        self._section_rows.setdefault(id(s), []).append(("mesh_path", row, s))
        self.mesh_info = tk.StringVar(value="")
        ttk.Label(s, textvariable=self.mesh_info, foreground=T["muted"],
                  wraplength=430).pack(anchor="w")
        self._field(s, "base_name", "base_name")

        s = self._section("3. Model")
        self._field(s, "ntype", "ntype_label", kind="combo",
                    values=list(NTYPES.keys()))
        self._field(s, "energy_split", "energy_split", kind="combo", values=SPLITS)
        self._field(s, "hybrid", "hybrid", kind="check")
        self._field(s, "ngaus", "ngaus", width=6)

        s = self._section("4. Material")
        self._field(s, "name", "mat_name")
        self._field(s, "E  (MPa)", "E")
        self._field(s, "nu", "nu")
        self._field(s, "Gc (N/mm)", "Gc")
        self._field(s, "l0 (mm)", "l0")
        self._field(s, "k", "k")
        self._field(s, "domain group", "domain")

        s = self._section("5. Restraints and loads")
        self.bctable = LoadTable(s, lambda: self.mesh_groups.keys(),
                                 on_change=self._bcs_changed)
        self.bctable.pack(fill="x")
        ttk.Label(s, wraplength=430, foreground=T["muted"],
                  text="displacement: mm, tick fix x / fix y for the DOFs held\n"
                       "force: N at a point (0D) group\n"
                       "traction: N per mm of edge, on an edge (1D) group\n"
                       "all are scaled by the load factor each step").pack(
                           anchor="w", pady=(2, 0))

        s = self._section("6. Solver")
        self._field(s, "scheme", "scheme", kind="combo", values=SCHEMES,
                    on_change=self._apply_visibility)
        self._field(s, "stagger stop", "stagger_stop", kind="combo",
                    values=STAGGER_STOPS, on_change=self._apply_visibility)
        self._field(s, "max_staggered", "max_staggered", width=8)
        self._field(s, "gamma tol (deg)", "gamma_tol", width=8)
        self._field(s, "tol_rel", "tol_rel", width=10)
        self._field(s, "tol_abs", "tol_abs", width=10)
        self._field(s, "max_iter", "max_iter", width=8)

        s = self._section("7. Load stepping")
        self._field(s, "step_mode", "step_mode", kind="combo",
                    values=STEP_MODES, on_change=self._apply_visibility)
        self._field(s, "du (uniform)", "du", width=10)
        self._field(s, "N_steps (fallback)", "N_steps", width=8)
        self._field(s, "du_coarse", "du_coarse", width=10)
        self._field(s, "u_switch", "u_switch", width=10)
        self._field(s, "du_fine", "du_fine", width=10)
        self._field(s, "u_switch2 (3-stage)", "u_switch2", width=10)
        self._field(s, "du_final (3-stage)", "du_final", width=10)
        self._field(s, "max_subdivs", "max_subdivs", width=8)

        s = self._section("8. Output")
        self._field(s, "output_dir", "output_dir")
        self._field(s, "vtk_every", "vtk_every", width=8)
        self._field(s, "profile_every", "profile_every", width=8)
        self._field(s, "write_log", "write_log", kind="check")

        self.warnbox = tk.Text(self.panel, height=7, wrap="word",
                               font=("TkDefaultFont", 8), foreground=T["warn"],
                               background=T["warnbg"], relief="flat",
                               insertbackground=T["fg"], borderwidth=0)
        self.warnbox.pack(fill="x", padx=6, pady=(4, 8))
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
                      foreground=T["muted"], wraplength=430).pack(anchor="w")
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
                      foreground=T["warn"], wraplength=470).pack(anchor="w",
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
                                     padding=5)
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
                      foreground=T["muted"], wraplength=430).pack(anchor="w")
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

    def draw_crack(self, force=False):
        vdir = self._vtk_dir()
        path, step = newest_vtk(vdir)
        if path is None:
            self.crack_stats.set(
                "no snapshots yet" if not vdir else f"no .vtk files in {vdir}")
            return
        if not self._stable(path) and not force:
            return                       # still being written; try again shortly

        try:
            if self._tri is None:
                self._tri, self._tri_npoin = read_vtk_geometry(path)
            phi = read_vtk_phi(path, self._tri_npoin)
        except (OSError, ValueError, IndexError) as exc:
            # A torn read is normal and self-correcting; only say so once.
            if step != self._crack_step:
                self.crack_stats.set(f"waiting for step {step} ({exc})")
            return

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
        m = read_mesh_for_plot(full)
        segs = []
        for name, s in m.get("edges", {}).items():
            # A crack MOUTH/TIP is a 0D group and never lands here; the mouth
            # group would only be two coincident points anyway.
            if self._NOTCH_RE.search(name):
                segs.extend(s)
        return segs

    def fit_crack(self):
        self._crack_step = -1        # forget the saved view
        self.draw_crack(force=True)

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
            self.root.update_idletasks()
        m = read_mesh_for_plot(full)          # cached unless the file changed
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

    def _scan_mesh(self):
        """Read the mesh's physical groups so BC names can be picked, not typed."""
        path = self.vars["mesh_path"].get().strip()
        full = path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)
        self.mesh_groups = groups_in_mesh(full)
        self._mesh_dirty = True
        if self.mesh_groups:
            dims = {0: "point", 1: "edge", 2: "surface"}
            names = ", ".join(f"{n} ({dims.get(d, d)})"
                              for n, d in sorted(self.mesh_groups.items(),
                                                 key=lambda kv: (kv[1], kv[0])))
            self.mesh_info.set(f"{len(self.mesh_groups)} groups: {names}")
        elif path:
            self.mesh_info.set("no groups read (file missing, or gmsh not installed)")
        else:
            self.mesh_info.set("")
        self.bctable.refresh_names()

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
            return
        msgs = ([("PROBLEM: " + m) for m in validate_form(f)]
                + [("PROBLEM: " + m)
                   for m in check_group_dims(f["bcs"], self.mesh_groups)]
                + warn_form(f))
        self.warnbox.insert("end", "\n\n".join(msgs) if msgs else "")
        self._show_plan(f)

    def _update_progress(self):
        """Progress bar + ETA from the steps the solver has actually reported."""
        if self._last_step <= 0:
            return
        total = self._plan_total
        done = self._last_step
        frac = min(done / total, 1.0) if total else 0.0
        self.progress["value"] = int(1000 * frac)

        rate = (sum(self._step_times) / len(self._step_times)
                if self._step_times else 0.0)
        if rate:
            self._last_rate = rate
        bits = [f"step {done:,}" + (f" / {total:,}" if total else "")]
        if rate:
            bits.append(f"{rate:.2f} s/step")
        if total and rate and done < total:
            # Steps remaining x the RECENT rate. Two known biases, both small
            # and both in the optimistic direction: adaptive subdivisions add
            # steps that were never planned, and steps get slower as the crack
            # grows. Treat it as a lower bound, not a promise.
            bits.append(f"ETA {fmt_duration((total - done) * rate)}")
        self.progress_txt.set("   ".join(bits))

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
                env=child_env())
        except OSError as exc:
            messagebox.showerror("Could not start", str(exc))
            self.proc = None
            return False
        self.proc_kind = kind
        self._on_success = on_success
        self.start_time = time.time()
        self.status.set(f"{label}...")
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
        self._scan_mesh()
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
        # A new run may use a different mesh, so the cached triangulation and
        # the last-seen step must not carry over -- otherwise phi from the new
        # run gets drawn on the old geometry, or silently rejected for having
        # the wrong length.
        self._tri = None
        self._tri_npoin = 0
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
            # A leftover CSV would be drawn instantly and look like progress.
            try:
                os.remove(self.csv_path)
            except OSError:
                pass
        self.ax.clear(); style_axes(self.fig, self.ax); self.canvas.draw_idle()

        cmd = [exe, GENERATED] + extra
        self._append_log(f"\n$ {' '.join(cmd)}\n")
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=PROJECT_DIR, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace",
                env=child_env())
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
        self.status.set(f"{label}...")
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
            try:
                self.proc.terminate()
            except OSError:
                pass
            self.status.set("stopping...")

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
        finished = False
        pending = []
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
                self._step_times.append(float(m.group(2)))
                del self._step_times[:-RATE_WINDOW]
        self._append_chunks(pending)
        if pending and self.proc_kind == "solve":
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
                self.status.set(("finished" if code == 0 else f"FAILED (exit {code})")
                                + f" in {elapsed:.0f}s")
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
        # The axis labels come from the CSV header (applied_uy vs applied_ux),
        # so they are part of the layout key.
        self._layout(self.fig, f"{labels[0]}|{labels[1]}")
        self.canvas.draw_idle()

        peak = max((abs(x) for x in F), default=0.0)
        self.status.set(
            f"step {len(u) - 1}   u = {u[-1]:.5g}   F = {F[-1]:.5g}   "
            f"peak |F| = {peak:.5g}   max phi = {phi[-1]:.3f}"
            + (f"   [{len(bad)} non-converged]" if bad else "")
            + f"   {time.time() - self.start_time:.0f}s")

    def _on_close(self):
        if self.proc is not None:
            if not messagebox.askyesno("Quit", "A solve is running. Kill it?"):
                return
            try:
                self.proc.terminate()
            except OSError:
                pass
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
    initial = args[0] if args else ""
    if not initial:
        for guess in ("config_ambati_3pb.toml", "config.toml"):
            if os.path.isfile(os.path.join(PROJECT_DIR, guess)):
                initial = guess
                break
    root = tk.Tk()
    App(root, initial)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
