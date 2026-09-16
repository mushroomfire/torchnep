# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# TorchNEP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with TorchNEP.  If not, see <http://www.gnu.org/licenses/>.

"""Figures from training and prediction outputs (needs matplotlib).

Everything is read from the files torchnep (and GPUMD) write:
``loss.out`` and ``energy_train.out`` / ``force_train.out`` /
``virial_train.out`` / ``stress_train.out`` (``*_test.out`` for the
validation set; ``predict_dataset`` writes the same files). Optionally the
extended-XYZ the outputs were computed for, to break errors down by element
and ``config_type`` and to shift energies per element.

    from torchnep.plot import NEPPlotter
    p = NEPPlotter()                                  # Arial if installed
    p.loss("run/loss.out", out="loss.png")
    p.parity("run", split="train", kind="density", out="parity.png")
    p.dashboard("run", out="dashboard.png")           # loss + train/test parity
    p.prediction("pred", xyz="test.xyz", out="pred.png")

Every method returns the matplotlib Figure; with ``out`` it is also saved
(and closed). Axis ranges always cover the full data, so outliers stay
visible.
"""
import os
import re
import warnings
from collections import Counter

import numpy as np

_NO_LABEL = -1e5          # reference virial / stress below this: no label (GPUMD -1e6 sentinel)


# ---------------------------------------------------------------------------
# file readers
# ---------------------------------------------------------------------------

def _read_table(path):
    """Whitespace table -> float64 (rows, cols); '#' lines ignored. polars
    when installed (multi-threaded, ~20x faster than numpy on the big force
    files), else numpy."""
    try:
        import polars as pl
    except ImportError:
        return np.loadtxt(path, comments="#", ndmin=2)
    with open(path) as fh:
        first = fh.readline()
        while first.startswith("#"):
            first = fh.readline()
    ncol = len(first.split())
    try:
        df = pl.read_csv(path, separator=" ", has_header=False, comment_prefix="#",
                         schema_overrides=[pl.Float64] * ncol, truncate_ragged_lines=True,
                         infer_schema_length=0)
        t = df.to_numpy()
        if t.shape[1] == ncol and not np.isnan(t).any():
            return t
    except Exception:
        pass
    # multiple spaces between columns (GPUMD's fixed-width output): normalise
    # the whitespace first, then let polars parse the single-space table
    import io
    buf = io.StringIO()
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                buf.write(" ".join(line.split()) + "\n")
    buf.seek(0)
    return pl.read_csv(buf, separator=" ", has_header=False,
                       schema_overrides=[pl.Float64] * ncol).to_numpy()


def read_loss(path="loss.out"):
    """``loss.out`` -> dict of 1-D arrays: ``epoch``, ``loss``, ``e``, ``f``,
    ``v``, ``s`` (train RMSEs) and, when the run had a validation set,
    ``e_test``, ``f_test``, ``v_test``, ``s_test``."""
    if os.path.isdir(path):
        path = os.path.join(path, "loss.out")
    t = _read_table(path)
    keys = ["epoch", "loss", "e", "f", "v", "s", "e_test", "f_test", "v_test", "s_test"]
    return {k: t[:, i] for i, k in enumerate(keys) if i < t.shape[1]}


def read_outputs(path=".", split="train"):
    """The four ``*_<split>.out`` files of a directory -> dict:
    ``E`` (pred, ref per frame, eV/atom), ``F`` ((N,3) pred, ref, eV/A),
    ``V`` ((M,6) pred, ref, eV/atom) and ``S`` (GPa), each with a ``mask``
    of the frames that carry a label. Missing files are simply absent."""
    out = {}
    for key, name in (("E", "energy"), ("F", "force"), ("V", "virial"), ("S", "stress")):
        f = os.path.join(path, f"{name}_{split}.out")
        if not os.path.exists(f):
            continue
        t = _read_table(f)
        half = t.shape[1] // 2
        pred, ref = t[:, :half], t[:, half:2 * half]
        if key == "E":
            pred, ref = pred[:, 0], ref[:, 0]
        mask = (ref[:, 0] > _NO_LABEL) if key in ("V", "S") else np.ones(len(ref), bool)
        out[key] = {"pred": pred, "ref": ref, "mask": mask}
    if not out:
        raise FileNotFoundError(f"no *_{split}.out files in {path}")
    return out


def exclude_frames(d, exclude, natoms):
    """Copy of a :func:`read_outputs` dict without the frames in ``exclude``
    (frame indices); ``natoms`` (atoms per frame) maps them to force rows."""
    natoms = np.asarray(natoms, dtype=np.int64)
    n = len(natoms)
    keep = np.ones(n, bool)
    keep[np.asarray(list(exclude), dtype=np.int64)] = False
    out = {}
    for key, v in d.items():
        if key == "F":
            if len(v["ref"]) != natoms.sum():
                raise ValueError(f"force rows {len(v['ref'])} != sum of natoms {natoms.sum()}")
            m = np.repeat(keep, natoms)
        else:
            if len(v["ref"]) != n:
                raise ValueError(f"{key} rows {len(v['ref'])} != frames {n}")
            m = keep
        out[key] = {"pred": v["pred"][m], "ref": v["ref"][m], "mask": v["mask"][m]}
    return out


def stage2_epoch(path="."):
    """Epoch at which stage 2 started, from ``output.log`` ("Stage 2 from
    epoch N") or ``nep.in`` (``start_stage2``, else half of ``epoch`` when
    ``stage2 1``) in ``path``; None when not found / no stage 2."""
    log = os.path.join(path, "output.log")
    if os.path.exists(log):
        with open(log, errors="replace") as fh:
            for line in fh:
                m = re.search(r"Stage 2 from epoch (\d+)", line)
                if m:
                    return int(m.group(1))
    nep_in = os.path.join(path, "nep.in")
    if os.path.exists(nep_in):
        kv = {}
        for line in open(nep_in):
            parts = line.split("#")[0].split()
            if len(parts) >= 2:
                kv[parts[0].lower()] = parts[1]
        if kv.get("stage2", "0") not in ("0", "false"):
            if "start_stage2" in kv:
                return int(float(kv["start_stage2"]))
            if "epoch" in kv:
                return int(float(kv["epoch"]) * 0.5)
    return None


def frame_meta(xyz):
    """Per frame of an extended-XYZ: atom count, element symbols per atom and
    the ``config_type`` tag. Reads only the header and the species column."""
    natoms, species, ctype = [], [], []
    with open(xyz, buffering=1 << 22) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            n = int(line)
            hdr = fh.readline()
            m = re.search(r"config_type=(\S+)", hdr)
            ctype.append(m.group(1).strip('"') if m else "-")
            species.append([fh.readline().split(None, 1)[0] for _ in range(n)])
            natoms.append(n)
    return np.asarray(natoms), species, ctype


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))) if len(a) else float("nan")


def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float)))) if len(a) else float("nan")


def _font_family(font, font_dir=None):
    """``font`` if matplotlib can find it, else None (matplotlib's default).
    ``font_dir`` (or ``$TORCHNEP_FONT_DIR``): a folder of .ttf/.otf files
    registered first — for machines without the font installed."""
    if not font:
        return None
    from matplotlib import font_manager as fm
    font_dir = font_dir or os.environ.get("TORCHNEP_FONT_DIR")
    if font_dir and os.path.isdir(font_dir):
        for name in sorted(os.listdir(font_dir)):
            if name.lower().endswith((".ttf", ".otf")):
                try:
                    fm.fontManager.addfont(os.path.join(font_dir, name))
                except Exception:
                    pass
    try:
        fm.findfont(fm.FontProperties(family=font), fallback_to_default=False)
        return font
    except Exception:
        warnings.warn(f"font {font!r} not found; using matplotlib's default",
                      stacklevel=3)
        return None


def shift_energy(d, mode=None, meta=None):
    """Energies of a different DFT reference: return ``pred`` shifted so that
    the constant (``mode="mean"``) or per-element (``mode="element"``, needs
    ``meta = frame_meta(xyz)``) offset against ``ref`` is removed. The
    per-element shift is the least-squares fit of the per-frame TOTAL energy
    difference to the element counts, like a reference-energy correction."""
    pred, ref = d["E"]["pred"], d["E"]["ref"]
    if mode is None:
        return pred
    if mode == "mean":
        return pred - float(np.mean(pred - ref))
    if mode == "element":
        if meta is None:
            raise ValueError("shift_energy='element' needs the xyz (element counts)")
        natoms, species, _ = meta
        els = sorted({e for s in species for e in s})
        A = np.array([[s.count(e) for e in els] for s in species], float)
        coef, *_ = np.linalg.lstsq(A, (pred - ref) * natoms, rcond=None)
        return pred - A @ coef / natoms
    raise ValueError(f"shift_energy: {mode!r} (None, 'mean' or 'element')")


def _dark(cmap_name):
    """A dark colour of a sequential colormap (for text over its hexbins)."""
    import matplotlib
    return matplotlib.colormaps[cmap_name](0.85)


def _hex_counts(x, y, extent, sx, sy, chunk=20_000_000):
    """Counts of (x, y) in the regular hexagonal lattice with spacing ``sx``
    (x) and ``sy`` (y) over ``extent`` — the two offset rectangular lattices
    of matplotlib's hexbin, binned in chunks (O(N), little memory). Returns
    (centres (M, 2), counts (M,)) of the occupied cells."""
    x0, x1, y0, y1 = extent
    nx, ny = int(np.ceil((x1 - x0) / sx)), int(np.ceil((y1 - y0) / sy))
    n1 = (nx + 1) * (ny + 1)
    counts = np.zeros(n1 + nx * ny, dtype=np.int64)
    x, y = np.ravel(x), np.ravel(y)
    for i in range(0, len(x), chunk):
        u = (np.asarray(x[i:i + chunk], float) - x0) / sx
        v = (np.asarray(y[i:i + chunk], float) - y0) / sy
        ok = (u >= 0) & (u <= nx) & (v >= 0) & (v <= ny)
        u, v = u[ok], v[ok]
        i1, j1 = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
        i2 = np.clip(np.floor(u).astype(np.int64), 0, nx - 1)
        j2 = np.clip(np.floor(v).astype(np.int64), 0, ny - 1)
        d1 = (u - i1) ** 2 + 3.0 * (v - j1) ** 2
        d2 = (u - i2 - 0.5) ** 2 + 3.0 * (v - j2 - 0.5) ** 2
        idx = np.where(d1 < d2, i1 * (ny + 1) + j1, n1 + i2 * ny + j2)
        counts += np.bincount(idx, minlength=len(counts))
    occ = np.nonzero(counts)[0]
    first = occ < n1
    cx = np.where(first, occ // (ny + 1), (occ - n1) // ny + 0.5)
    cy = np.where(first, occ % (ny + 1), (occ - n1) % ny + 0.5)
    return np.column_stack([x0 + cx * sx, y0 + cy * sy]), counts[occ]


def _density(ax, x, y, extent, cmap, zorder=2, bins=80, cmap_range=(0.1, 0.9), cell="hex",
             cmap_reverse=True):
    """Log-count density of (x, y) over ``extent`` (x0, x1, y0, y1), binned
    in O(N) with little memory (hundreds of millions of points are fine).
    ``cell="hex"``: regular hexagons (regular on screen, whatever the axes'
    aspect), ``bins`` of them across the x range; ``"square"``: a 2-D
    histogram with ``bins`` cells per axis (or ``(nx, ny)``). Empty cells
    are transparent. Returns the mappable (for a colorbar)."""
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.collections import PolyCollection
    x0, x1, y0, y1 = extent
    # cmap_range: the part of the colormap used (clear of its white end);
    # cmap_reverse: sparse cells (the outliers one looks for) get the end of
    # the range that is normally used for crowded cells
    lo_c, hi_c = sorted(float(c) for c in cmap_range)
    base = plt.get_cmap(cmap)
    stops = np.linspace(hi_c, lo_c, 256) if cmap_reverse else np.linspace(lo_c, hi_c, 256)
    cm = mcolors.ListedColormap(base(stops), name=f"{base.name}_{'rev' if cmap_reverse else 'part'}")
    if cell == "square":
        nb = (bins, bins) if np.isscalar(bins) else bins
        h, xe, ye = np.histogram2d(np.ravel(x).astype(float), np.ravel(y).astype(float), bins=nb,
                                   range=[[x0, x1], [y0, y1]])
        h = np.ma.masked_less(h.T, 1)
        return ax.pcolormesh(xe, ye, h, cmap=cm, norm=mcolors.LogNorm(vmin=1, vmax=max(float(h.max()), 2)),
                             rasterized=True, zorder=zorder, shading="flat")
    if cell != "hex":
        raise ValueError(f"cell must be 'hex' or 'square', got {cell!r}")
    nx = int(bins if np.isscalar(bins) else bins[0])
    sx = (x1 - x0) / nx
    # y spacing for hexagons that are regular in display space
    fw, fh = ax.figure.get_size_inches()
    pos = ax.get_position()
    w_in, h_in = pos.width * fw, pos.height * fh
    sy = np.sqrt(3.0) * sx * ((y1 - y0) / h_in) / ((x1 - x0) / w_in)
    centres, c = _hex_counts(x, y, extent, sx, sy)
    hexagon = np.array([[0.5, -0.5], [0.5, 0.5], [0.0, 1.0], [-0.5, 0.5], [-0.5, -0.5], [0.0, -1.0]]) \
        * [sx, sy / 3.0]
    verts = centres[:, None, :] + hexagon[None, :, :]
    col = PolyCollection(verts, array=c.astype(float), cmap=cm,
                         norm=mcolors.LogNorm(vmin=1, vmax=max(float(c.max()) if len(c) else 2.0, 2.0)),
                         edgecolors="face", linewidths=0.2, rasterized=True, zorder=zorder)
    ax.add_collection(col)
    return col


def _kde(x, grid):
    """Gaussian kernel density of ``x`` on the uniform ``grid`` (Silverman
    bandwidth, at least one grid step). Every point is used: the data are
    binned on the grid and the histogram is smoothed with the kernel, so the
    cost is O(N) and rare outliers are not lost to subsampling."""
    x = np.asarray(x, float).ravel()
    if len(x) == 0:
        return np.zeros_like(grid)
    sd = float(x.std())
    step = float(grid[1] - grid[0])
    if sd <= 0:
        return np.zeros_like(grid)
    h = max(1.06 * sd * len(x) ** (-0.2), step)
    edges = np.concatenate([grid - step / 2, [grid[-1] + step / 2]])
    counts, _ = np.histogram(x, bins=edges)
    half = int(np.ceil(4 * h / step))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) * step / h) ** 2)
    out = np.convolve(counts.astype(float), k, mode="same")[:len(grid)] if len(k) <= len(grid) \
        else np.convolve(counts.astype(float), k, mode="full")[half:half + len(grid)]
    return out / (len(x) * h * np.sqrt(2 * np.pi))


_QTY = {   # key: (title, unit, error unit, error scale)
    "E": ("Energy", "eV/atom", "meV/atom", 1e3),
    "F": ("Force", "eV/Å", "meV/Å", 1e3),
    "V": ("Virial", "eV/atom", "meV/atom", 1e3),
    "S": ("Stress", "GPa", "GPa", 1.0),
}


# default palette (the paper template): pink / blue / yellow + two greys
PALETTE = {"pink": "#FE218B", "blue": "#21B0FE", "yellow": "#FED700",
           "grey": "#7f7f7f", "lightgrey": "#bfbfbf"}
DEFAULT_COLORS = {"train": PALETTE["blue"], "valid": PALETTE["pink"],
                  "energy": PALETTE["blue"], "force": PALETTE["pink"], "virial": PALETTE["yellow"],
                  "stress": PALETTE["grey"]}
DEFAULT_CMAPS = {"train": "Blues", "valid": "Reds"}


# Periodic-table layout: symbol -> (row, column), 18 columns, lanthanides and
# actinides in two extra rows (rows 9 and 10, columns 4-18 in the usual way).
_PT_ROWS = [
    "H . . . . . . . . . . . . . . . . He",
    "Li Be . . . . . . . . . . B C N O F Ne",
    "Na Mg . . . . . . . . . . Al Si P S Cl Ar",
    "K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr",
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe",
    "Cs Ba * Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn",
    "Fr Ra ** Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og",
    ". . . La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu",
    ". . . Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr",
]
PT_POSITIONS = {}
for _r, _row in enumerate(_PT_ROWS):
    for _c, _sym in enumerate(_row.split()):
        if _sym not in (".", "*", "**"):
            PT_POSITIONS[_sym] = (_r + (0.5 if _r >= 7 else 0), _c)   # half a row of gap before the f-block


# Chemical families (for the periodic-table outlines and legend)
PT_FAMILIES = [
    ("Alkali metals", "Li Na K Rb Cs Fr", "#1f77b4"),
    ("Alkaline-earth metals", "Be Mg Ca Sr Ba Ra", "#17becf"),
    ("3d transition metals", "Sc Ti V Cr Mn Fe Co Ni Cu Zn", "#2ca02c"),
    ("4d transition metals", "Y Zr Nb Mo Tc Ru Rh Pd Ag Cd", "#98df8a"),
    ("5d transition metals", "Hf Ta W Re Os Ir Pt Au Hg", "#bcbd22"),
    ("Post-transition metals", "Al Ga In Sn Tl Pb Bi", "#ff7f0e"),
    ("Metalloids", "B Si Ge As Sb Te", "#9467bd"),
    ("Non-metals", "H C N O F P S Cl Se Br I", "#d62728"),
    ("Noble gases", "He Ne Ar Kr Xe Rn", "#7f7f7f"),
    ("Lanthanides (4f)", "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu", "#e377c2"),
    ("Actinides (5f)", "Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr", "#c49c94"),
]


def element_errors(path, xyz, split="train"):
    """Per-element RMSEs of the ``*_<split>.out`` outputs of ``path`` for the
    frames of ``xyz``: ``{"E": {el: meV/atom}, "F": {el: meV/A},
    "n_atoms": {el: count}, "n_frames": {el: count}}``. The force RMSE is
    over the atoms of the element; the energy RMSE over the frames that
    contain it (per-atom energies, every frame weighted once)."""
    d = read_outputs(path, split)
    natoms, species, _ = frame_meta(xyz)
    out = {"E": {}, "F": {}, "n_atoms": {}, "n_frames": {}}
    if "F" in d:
        sym = np.concatenate([np.asarray(s) for s in species])
        if len(sym) != len(d["F"]["ref"]):
            raise ValueError(f"{xyz} has {len(sym)} atoms, the force file {len(d['F']['ref'])} rows")
        err2 = ((d["F"]["pred"] - d["F"]["ref"]) ** 2).sum(1)
        for e in np.unique(sym):
            m = sym == e
            out["F"][str(e)] = float(np.sqrt(err2[m].sum() / (3 * m.sum())) * 1e3)
            out["n_atoms"][str(e)] = int(m.sum())
    if "E" in d:
        if len(natoms) != len(d["E"]["ref"]):
            raise ValueError(f"{xyz} has {len(natoms)} frames, the energy file {len(d['E']['ref'])} rows")
        de2 = (d["E"]["pred"] - d["E"]["ref"]) ** 2
        frames_of = {}
        for i, s in enumerate(species):
            for e in set(s):
                frames_of.setdefault(e, []).append(i)
        for e, idx in frames_of.items():
            out["E"][e] = float(np.sqrt(de2[idx].mean()) * 1e3)
            out["n_frames"][e] = len(idx)
    return out


def _cm(*v):
    return tuple(x / 2.54 for x in v)


class NEPPlotter:
    """Figures for torchnep / GPUMD training and prediction outputs.

    Parameters
    ----------
    font : str
        Font family (default ``"Arial"``); matplotlib's default when absent.
    fontsize : float
        Base font size in points (labels; ticks and legends one point smaller).
    dpi : int
        Figure / saved-file resolution (default 300).
    cmaps : dict
        Colormaps of the density (hexbin) panels per set, overrides of
        ``DEFAULT_CMAPS`` (``train``: Blues, ``valid``: Reds).
    cmap_range : (float, float)
        Part of each density colormap used (default ``(0.1, 0.9)``), kept
        clear of the colormap's white end.
    cmap_reverse : bool
        ``True`` (default): density colormaps run reversed, so cells with few
        points (the outliers) take the dark end and crowded cells the light
        end; ``False``: the usual direction (crowded cells dark).
    max_points : int
        Scatter panels draw at most this many points (a fixed random
        subsample); metrics always use every point.
    colors : dict
        Overrides of ``DEFAULT_COLORS`` — keys ``train``, ``valid`` (parity
        sets) and ``energy``, ``force``, ``virial``, ``stress`` (curves).
    panel_labels : str or sequence
        Labels of the panels of multi-panel figures (default ``"abcd..."``);
        ``label_format`` wraps them (``"{}"`` -> ``a``, ``"({})"`` -> ``(a)``),
        ``label_weight`` is their font weight. ``panel_labels=None`` for none.
    frame : bool
        ``False`` (default): only the left and bottom spines; ``True``: the
        full box with ticks on all four sides.
    font_dir : str
        Folder of .ttf/.otf files to register before looking up ``font``
        (default: the ``TORCHNEP_FONT_DIR`` environment variable).
    rc : dict
        Extra matplotlib rcParams applied on top of the built-in style.
    """

    def __init__(self, font="Arial", fontsize=7, dpi=300, cmaps=None,
                 max_points=300_000, colors=None, panel_labels="abcdefghijkl",
                 label_format="{}", label_weight="bold", frame=False, rc=None,
                 font_dir=None, cmap_range=(0.1, 0.9), cmap_reverse=True):
        import matplotlib  # noqa: F401  (fail early with a clear message)
        fam = _font_family(font, font_dir)
        fs = float(fontsize)
        self.rc = {
            "font.family": fam or "sans-serif",
            "font.size": fs, "axes.labelsize": fs, "axes.titlesize": fs,
            "legend.fontsize": fs - 1, "xtick.labelsize": fs, "ytick.labelsize": fs,
            "axes.spines.right": bool(frame), "axes.spines.top": bool(frame),
            "xtick.top": bool(frame), "ytick.right": bool(frame),
            "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6,
            "ytick.major.width": 0.6, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
            "xtick.minor.width": 0.5, "ytick.minor.width": 0.5, "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5, "axes.labelpad": 1.5,
            "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3,
            "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.4,
            "legend.labelspacing": 0.2, "legend.borderaxespad": 0.2,
            "mathtext.fontset": "stix", "axes.formatter.use_mathtext": True,
            "figure.dpi": dpi, "savefig.dpi": dpi,
        }
        self.rc.update(rc or {})
        self.colors = dict(DEFAULT_COLORS, **(colors or {}))
        self.cmaps = dict(DEFAULT_CMAPS, **(cmaps or {}))
        self.cmap = self.cmaps["train"]
        self.cmap_range = (float(cmap_range[0]), float(cmap_range[1]))
        self.cmap_reverse = bool(cmap_reverse)
        self.max_points = int(max_points)
        self.panel_labels = list(panel_labels) if panel_labels else []
        self.label_format = label_format
        self.label_weight = label_weight
        self._rng = np.random.default_rng(0)

    # kept for the simpler figures below
    @property
    def c_train(self):
        return self.colors["train"]

    @property
    def c_test(self):
        return self.colors["valid"]

    def _label_panels(self, axes, dx=-22, dy=1):
        """Panel labels at the top-left corner outside every axes, offset in
        points so they line up whatever the panel's axis type."""
        for ax, lab in zip(axes, self.panel_labels):
            ax.annotate(self.label_format.format(lab), xy=(0, 1), xycoords="axes fraction",
                        xytext=(dx, dy), textcoords="offset points", ha="left", va="bottom",
                        fontsize=self.rc["font.size"] + 3, fontweight=self.label_weight)

    # ---- generic panels ---------------------------------------------------
    def _finish(self, fig, out):
        """Save (inside the style context: fonts resolve at draw time) and
        close when ``out`` is given; always return the figure."""
        import matplotlib.pyplot as plt
        if out:
            with plt.rc_context(self.rc):
                fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
        return fig

    def _small(self, n=1):
        return self.rc["font.size"] - n

    def _parity_panel(self, ax, ref, pred, key, kind="scatter", color=None,
                      label=None, note=None):
        """One parity panel over the FULL data range with the diagonal and
        RMSE / MAE / N annotated. ``ref``/``pred`` are flattened."""
        ref = np.asarray(ref, float).ravel()
        pred = np.asarray(pred, float).ravel()
        title, unit, eunit, scale = _QTY[key]
        color = color or self.c_train
        ax.set_title(title + (f" ({label})" if label else "") + (f", {note}" if note else ""))
        if len(ref) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            return
        lo = float(min(ref.min(), pred.min()))
        hi = float(max(ref.max(), pred.max()))
        pad = 0.03 * (hi - lo) + 1e-9
        lo, hi = lo - pad, hi + pad
        if kind == "density":
            ax.hexbin(ref, pred, gridsize=120, extent=(lo, hi, lo, hi), mincnt=1,
                      bins="log", cmap=self.cmap, rasterized=True)
        else:
            if len(ref) > self.max_points:
                sel = self._rng.choice(len(ref), self.max_points, replace=False)
                r, p = ref[sel], pred[sel]
            else:
                r, p = ref, pred
            ax.scatter(r, p, s=4, alpha=0.5, color=color, edgecolors="none",
                       rasterized=True)
        ax.plot([lo, hi], [lo, hi], "--", color="0.3", lw=0.8, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"DFT {title.lower()} ({unit})")
        ax.set_ylabel(f"NEP {title.lower()} ({unit})")
        ax.text(0.04, 0.96,
                f"RMSE {_rmse(ref, pred) * scale:.1f} {eunit}\n"
                f"MAE {_mae(ref, pred) * scale:.1f} {eunit}\nN = {len(ref):,}",
                transform=ax.transAxes, ha="left", va="top", fontsize=self._small(),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))

    def _hist_panel(self, ax, err, key, color=None, bins=120):
        """Histogram of ``pred - ref`` in the error unit, with RMSE / MAE / max."""
        err = np.asarray(err, float).ravel() * _QTY[key][3]
        title, _, eunit, _ = _QTY[key]
        ax.set_title(f"{title} error")
        if len(err) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            return
        lo, hi = float(err.min()), float(err.max())
        if hi - lo < 1e-9:                      # constant error: one finite bin
            lo, hi = lo - 0.5, hi + 0.5
        ax.hist(err, bins=bins, range=(lo, hi), color=color or self.c_train, alpha=0.75)
        ax.axvline(0.0, color="0.3", lw=0.8, ls="--")
        ax.set_xlabel(f"NEP − DFT {title.lower()} ({eunit})")
        ax.set_ylabel("count")
        ax.set_yscale("log")
        ax.text(0.97, 0.96,
                f"RMSE {np.sqrt(np.mean(err ** 2)):.1f}\nMAE {np.mean(np.abs(err)):.1f}\n"
                f"max {np.abs(err).max():.1f} {eunit}",
                transform=ax.transAxes, ha="right", va="top", fontsize=self._small(),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))

    def _force_vs_magnitude(self, ax, d):
        fref = np.linalg.norm(d["F"]["ref"], axis=1)
        ferr = np.linalg.norm(d["F"]["pred"] - d["F"]["ref"], axis=1)
        ax.hexbin(fref, ferr, gridsize=100, mincnt=1, bins="log", cmap=self.cmap,
                  rasterized=True)
        ax.set_xlabel("|DFT force| (eV/Å)")
        ax.set_ylabel("|NEP − DFT force| (eV/Å)")
        ax.set_title("Force error vs magnitude")

    def _bar_panel(self, ax, names, values, counts, ylabel, title, color):
        ax.bar(range(len(names)), values, color=color, alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90 if len(names) > 8 else 0)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        top = max(values) if len(values) else 1.0
        ax.set_ylim(0, top * 1.25)
        for i, (v, c) in enumerate(zip(values, counts)):
            ax.text(i, v + 0.02 * top, f"{c:,}", ha="center", va="bottom",
                    fontsize=self._small(3), rotation=90 if len(names) > 8 else 0)

    def _breakdown_panels(self, d, meta, e_pred=None):
        """(name, draw_fn) list: force RMSE per element and, when the xyz has
        several config_types, energy RMSE per config_type."""
        natoms, species, ctypes = meta
        panels = []
        if "F" in d:
            sym = np.concatenate([np.asarray(s) for s in species])
            if len(sym) != len(d["F"]["ref"]):
                raise ValueError(f"xyz has {len(sym)} atoms, the force file {len(d['F']['ref'])}")
            err2 = ((d["F"]["pred"] - d["F"]["ref"]) ** 2).sum(1)
            els = sorted(set(sym))
            rm = np.array([np.sqrt(err2[sym == e].sum() / (3 * (sym == e).sum())) * 1e3 for e in els])
            cnt = [int((sym == e).sum()) for e in els]
            order = np.argsort(rm)[::-1]
            names_e = [els[i] for i in order]
            vals_e, cnt_e = list(rm[order]), [cnt[i] for i in order]
            panels.append(lambda ax, n=names_e, v=vals_e, c=cnt_e: self._bar_panel(
                ax, n, v, c, "force RMSE (meV/Å)", "Force RMSE per element (atom count)",
                self.c_train))
        if "E" in d and len(set(ctypes)) > 1:
            if len(natoms) != len(d["E"]["ref"]):
                raise ValueError(f"xyz has {len(natoms)} frames, the energy file {len(d['E']['ref'])}")
            ct = np.asarray(ctypes)
            pred = d["E"]["pred"] if e_pred is None else e_pred
            de = (pred - d["E"]["ref"]) * 1e3
            names = [c for c, _ in Counter(ctypes).most_common(30)]
            rm = [float(np.sqrt(np.mean(de[ct == c] ** 2))) for c in names]
            cnt = [int((ct == c).sum()) for c in names]
            panels.append(lambda ax, n=names, v=rm, c=cnt: self._bar_panel(
                ax, n, v, c, "energy RMSE (meV/atom)",
                "Energy RMSE per config_type (frame count)", self.c_test))
        return panels

    def loss(self, path, out=None, stage2="auto", stress=False, figsize=None):
        """Training curves from ``loss.out``: the E / F / V RMSEs against the
        epoch on a log scale, training solid and validation faint (same
        colour), a dashed line at the start of stage 2 — the panel of the
        dashboard on its own. ``stress=True`` adds the stress RMSE.
        ``path``: the run directory or the loss.out file. ``figsize`` in cm."""
        import matplotlib.pyplot as plt
        run_dir = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
        d = read_loss(path)
        if stage2 == "auto":
            stage2 = stage2_epoch(run_dir)
        keys = [("e", "e_test", "energy", "E", "eV/atom"), ("f", "f_test", "force", "F", "eV/Å"),
                ("v", "v_test", "virial", "V", "eV/atom")]
        if stress:
            keys.append(("s", "s_test", "stress", "S", "GPa"))
        with plt.rc_context(self.rc):
            fig, ax = plt.subplots(figsize=_cm(*(figsize or (6.5, 5.6))), constrained_layout=True)
            self._rmse_curves(ax, d, stage2, keys)
        return self._finish(fig, out)

    def _block(self, fig, box, sets, q, kind, margins, bins=80, cell="hex"):
        """One quantity: square parity panel at ``box = (x0, y0, L)`` (cm) and,
        with ``margins``, the error against the DFT value on top and the
        error density on the right. Returns (main axes, top axes or None,
        hexbin handles)."""
        W, H = fig.get_size_inches() * 2.54
        x0, y0, L = box
        gap, strip = 0.12, 0.3 * L

        def add(x, y, w, h):
            return fig.add_axes([x / W, y / H, w / W, h / H])
        ax = add(x0, y0, L, L)
        handles = self._parity_overlay(ax, sets, q, kind, bins=bins, cell=cell)
        if not margins:
            return ax, None, handles
        ax_top = add(x0, y0 + L + gap, L, strip)
        ax_right = add(x0 + L + gap, y0, strip, L)
        title, unit, _, _ = _QTY[q]
        errs = []
        for k, (label, ref, pred, sk) in enumerate(sets):
            ref, pred = np.ravel(ref).astype(float), np.ravel(pred).astype(float)
            err = pred - ref
            errs.append((err, sk))
            color = self.colors[sk] if kind != "density" else _dark(self.cmaps[sk])
            if kind == "density":
                pass                                    # drawn below, once the error range is known
            else:
                if len(ref) > self.max_points:
                    sel = self._rng.choice(len(ref), self.max_points, replace=False)
                    ref, err = ref[sel], err[sel]
                ax_top.scatter(ref, err, s=2, alpha=0.5, color=color, edgecolors="none",
                               rasterized=True, zorder=2 + k)
        ax_top.axhline(0.0, color="grey", lw=0.7, ls="--", zorder=1)
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_ylabel("Error")               # same unit as the axes below
        ax_top.tick_params(labelbottom=False)
        if errs:
            allerr = np.concatenate([e for e, _ in errs])
            lo, hi = float(allerr.min()), float(allerr.max())
            if hi - lo < 1e-12:
                lo, hi = lo - 0.5, hi + 0.5
            lo, hi = lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo)
            grid = np.linspace(lo, hi, 300)
            if kind == "density":
                xl = ax.get_xlim()
                for k, (label, ref, pred, sk) in enumerate(sets):
                    ref, pred = np.ravel(ref).astype(float), np.ravel(pred).astype(float)
                    _density(ax_top, ref, pred - ref, (xl[0], xl[1], lo, hi), self.cmaps[sk],
                             zorder=2 + k, cell=cell, cmap_range=self.cmap_range, cmap_reverse=self.cmap_reverse,
                             bins=bins if cell == "hex" else (bins, max(10, bins // 3)))
            for err, sk in errs:
                color = self.colors[sk] if kind != "density" else _dark(self.cmaps[sk])
                dens = _kde(err, grid)
                ax_right.fill_betweenx(grid, 0, dens, color=color, alpha=0.35, lw=0)
                ax_right.plot(dens, grid, color=color, lw=0.9)
            ax_right.axhline(0.0, color="grey", lw=0.7, ls="--", zorder=1)
            ax_top.set_ylim(lo, hi)
            ax_right.set_ylim(lo, hi)
            ax_right.set_xlim(0, None)
        ax_right.set_xlabel("Density")
        ax_right.set_xticks([])
        # the error axis of the density strip is labelled on its own right side (the left side faces the parity
        # panel, whose y axis is the NEP value, not the error): same range and ticks as the error strip on top
        ax_right.spines["left"].set_visible(False)
        ax_right.spines["right"].set_visible(True)
        ax_right.yaxis.tick_right()
        ax_right.yaxis.set_label_position("right")
        ax_right.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
        if ax_top is not None:
            ax_right.set_yticks(ax_top.get_yticks())
            ax_right.set_ylim(ax_top.get_ylim())
        ax_right.set_ylabel("Error", rotation=270, labelpad=8)
        return ax, ax_top, handles

    def parity(self, path, kind="scatter", margins=False, virial=False, out=None,
               quantities=None, size=4.0, shift_energy=None, xyz=None, title=None,
               exclude=None, natoms=None, bins=80, cell="hex"):
        """Parity plots of the run in ``path``: energy, force and stress (the
        last only with stress labels) with training and validation overlaid
        and R^2 / RMSE / MAE per set. ``kind="density"``: log-count 2-D histograms
        (training Blues, validation Reds) with a small horizontal colorbar
        under every panel. ``margins=True`` adds the error distribution
        around each panel: NEP − DFT against the DFT value on top, a kernel
        density estimate of the error on the right. ``virial=True`` shows the
        virial (eV/atom) instead of the stress in the third panel;
        ``quantities`` overrides the panels altogether (any of ``E F V S``);
        ``shift_energy`` (``"mean"`` /
        ``"element"``, needs ``xyz``) removes a reference offset from the
        predicted energies; ``size`` is the side of one panel in cm.
        ``exclude``: frame indices (of the training split) to leave out, e.g.
        an outlier list — needs ``natoms`` (atoms per frame, or the ``xyz``)
        to drop their force rows too. ``bins``: density cells across the x
        axis (default 80; fewer = larger cells); ``cell``: ``"hex"`` (default)
        or ``"square"``."""
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LogLocator, NullLocator
        splits = [sp for sp in ("train", "test")
                  if os.path.exists(os.path.join(path, f"energy_{sp}.out"))]
        if not splits:
            raise FileNotFoundError(f"no energy_train.out / energy_test.out in {path}")
        data = {sp: read_outputs(path, sp) for sp in splits}
        if exclude is not None and "train" in data:
            if natoms is None:
                if xyz is None:
                    raise ValueError("exclude= needs natoms= (atoms per frame) or xyz=")
                natoms = frame_meta(xyz)[0]
            data["train"] = exclude_frames(data["train"], exclude, natoms)
        if shift_energy:
            meta = frame_meta(xyz) if xyz else None
            for sp in splits:
                data[sp]["E"]["pred"] = shift_energy_fn(data[sp], shift_energy, meta)
        if quantities is None:
            third = "V" if virial else "S"
            has_3 = all(third in data[sp] and data[sp][third]["mask"].any() for sp in splits)
            quantities = ["E", "F"] + ([third] if has_3 else [])
        qs = [q for q in quantities if any(q in data[sp] for sp in splits)]
        n = len(qs)
        L = float(size)
        left, top, gap = 1.25, 0.55, 0.12                  # margins in cm
        strip = 0.3 * L if margins else 0.0
        right = 0.35 + (strip + gap + 0.9 if margins else 0.0)     # + room for the error ticks/label of the density strip
        cbar_h = 0.75 if kind == "density" else 0.0         # bar + its label
        bottom = 0.95 + cbar_h
        block = left + L + right
        W = n * block
        H = bottom + L + (strip + gap if margins else 0.0) + top + (0.4 if title else 0.0)
        with plt.rc_context(self.rc):
            fig = plt.figure(figsize=_cm(W, H))
            mains, tops, per_block = [], [], []
            for i, q in enumerate(qs):
                ax, ax_top, h = self._block(fig, (i * block + left, bottom, L),
                                            self._sets(data, q), q, kind, margins, bins=bins, cell=cell)
                mains.append(ax)
                tops.append(ax_top)
                per_block.append(h)
            self._label_panels(tops if margins else mains,
                               dx=-int(left * 72 / 2.54) + 2, dy=2)
            if kind == "density":
                for i, h in enumerate(per_block):
                    keys = [k for k in ("train", "valid") if k in h]
                    bw = (L - 0.6 * (len(keys) - 1)) / max(1, len(keys))
                    for j, k in enumerate(keys):
                        x = (i * block + left + j * (bw + 0.6)) / W
                        cax = fig.add_axes([x, 0.5 / H, bw / W, 0.14 / H])
                        cb = fig.colorbar(h[k], cax=cax, orientation="horizontal")
                        cb.set_label("Training count" if k == "train" else "Validation count",
                                     labelpad=1)
                        cb.ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
                        cb.ax.xaxis.set_minor_locator(NullLocator())
                        cb.ax.tick_params(length=1.5, pad=1, labelsize=self.rc["font.size"] - 1)
            if title:
                fig.suptitle(title)
        return self._finish(fig, out)

    def periodic_table(self, values=None, path=None, xyz=None, split="train",
                       quantities=("E", "F"), out=None, cmap="YlOrRd", vmax=None,
                       title=None, size=0.62, families=False):
        """Periodic table coloured by a per-element number: either the
        per-element RMSEs of a run (``path`` + ``xyz`` -> :func:`element_errors`,
        one table per entry of ``quantities``) or your own ``values``
        (``{label: {element: value}}``). Elements without a value are grey.
        ``vmax``: colour-scale top (default: the largest value); ``size``: cell
        side in cm; ``families=True`` outlines the chemical families of the
        elements that carry a value (``PT_FAMILIES``) with a legend below."""
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
        from matplotlib.patches import Rectangle
        if values is None:
            if path is None or xyz is None:
                raise ValueError("periodic_table needs values= or path= and xyz=")
            errs = element_errors(path, xyz, split)
            units = {"E": "Energy RMSE (meV/atom)", "F": "Force RMSE (meV/Å)"}
            values = {units[q]: errs[q] for q in quantities if errs.get(q)}
        panels = list(values.items())
        ncol, nrow = 18, 9.5
        w = ncol * size + 0.4
        h = len(panels) * (nrow * size + 1.2 + (1.4 if families else 0.0))
        with plt.rc_context(self.rc):
            fig, axes = plt.subplots(len(panels), 1, figsize=_cm(w, h), squeeze=False)
            for ax, (label, vals) in zip(axes.ravel(), panels):
                vmax_ = vmax or (max(vals.values()) if vals else 1.0)
                norm = mcolors.Normalize(vmin=0.0, vmax=vmax_)
                cm_ = plt.get_cmap(cmap)
                for sym, (r, c) in PT_POSITIONS.items():
                    x, y = c, nrow - 1 - r
                    if sym in vals:
                        v = vals[sym]
                        fc = cm_(norm(v))
                        lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
                        tc = "white" if lum < 0.5 else "black"
                        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=fc, edgecolor="white", lw=0.6))
                        ax.text(x + 0.5, y + 0.62, sym, ha="center", va="center", color=tc,
                                fontsize=self.rc["font.size"], fontweight="bold")
                        ax.text(x + 0.5, y + 0.24, f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                                ha="center", va="center", color=tc, fontsize=self.rc["font.size"] - 1.5)
                    else:
                        ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#ececec", edgecolor="white", lw=0.6))
                        ax.text(x + 0.5, y + 0.5, sym, ha="center", va="center", color="#9a9a9a",
                                fontsize=self.rc["font.size"] - 1)
                if families:
                    handles = []
                    for name, syms, color in PT_FAMILIES:
                        members = [t for t in syms.split() if t in vals]
                        if not members:
                            continue
                        for t in members:
                            r, c = PT_POSITIONS[t]
                            ax.add_patch(Rectangle((c + 0.06, nrow - 1 - r + 0.06), 0.88, 0.88, fill=False,
                                                   edgecolor=color, lw=1.6, zorder=5))
                        handles.append(Rectangle((0, 0), 1, 1, fill=False, edgecolor=color, lw=1.6, label=name))
                    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.01), ncol=4,
                              frameon=False, handlelength=1.2, labelspacing=0.4, columnspacing=1.2)
                ax.set_xlim(0, ncol)
                ax.set_ylim(0, nrow)
                ax.set_aspect("equal")
                ax.axis("off")
                sm = plt.cm.ScalarMappable(norm=norm, cmap=cm_)
                cax = ax.inset_axes([3.2 / ncol, (nrow - 1.9) / nrow, 8.0 / ncol, 0.32 / nrow])
                cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
                cb.set_label(label, labelpad=2)
                cb.ax.tick_params(length=1.5, pad=1)
                if vmax is not None and max(vals.values(), default=0) > vmax:
                    cb.ax.set_title(f"> {vmax:g} saturated", fontsize=self.rc["font.size"] - 1, pad=1)
            if title:
                fig.suptitle(title)
            fig.tight_layout(pad=0.3)
        return self._finish(fig, out)

    def errors(self, path, split="train", out=None, xyz=None,
               quantities=("E", "F", "V"), shift_energy=None):
        """Error distributions of the ``*_<split>.out`` files: histograms of
        NEP − DFT for E / F / V, the force error against the force magnitude
        and — with the ``xyz`` the outputs belong to — the force RMSE per
        element and the energy RMSE per ``config_type``."""
        import matplotlib.pyplot as plt
        d = read_outputs(path, split)
        qs = [q for q in quantities if q in d]
        meta = frame_meta(xyz) if xyz else None
        e_pred = shift_energy_fn(d, shift_energy, meta) if "E" in d else None
        extra = self._breakdown_panels(d, meta, e_pred) if meta else []
        n_cols = len(qs) + (1 if "F" in d else 0)
        n_rows = 1 + (1 if extra else 0)
        with plt.rc_context(self.rc):
            fig = plt.figure(figsize=(3.7 * n_cols, 3.3 * n_rows))
            gs = fig.add_gridspec(n_rows, n_cols)
            col = 0
            for q in qs:
                m = d[q]["mask"]
                pred = e_pred if q == "E" else d[q]["pred"]
                self._hist_panel(fig.add_subplot(gs[0, col]), pred[m] - d[q]["ref"][m], q)
                col += 1
            if "F" in d:
                self._force_vs_magnitude(fig.add_subplot(gs[0, col]), d)
            if extra:
                bounds = np.linspace(0, n_cols, len(extra) + 1).astype(int)
                for k, draw in enumerate(extra):
                    draw(fig.add_subplot(gs[1, bounds[k]:bounds[k + 1]]))
            fig.tight_layout()
        return self._finish(fig, out)

    def _parity_overlay(self, ax, sets, key, kind="scatter", annotate=True, bins=80, cell="hex"):
        """Training and validation points of one quantity in one panel (full
        range, diagonal). ``sets``: ``(label, ref, pred, set_key)`` tuples,
        ``set_key`` in {"train", "valid"}. ``kind="density"`` draws every set
        as a log-count hexbin with the set's colormap. Each set is annotated
        with R^2 / RMSE / MAE in its own corner (train top-left, valid
        bottom-right: off the diagonal, so the text never covers the data).
        Returns the hexbin handles (density) for colorbars."""
        title, unit, eunit, scale = _QTY[key]
        allv = np.concatenate([np.concatenate([np.ravel(r), np.ravel(p)]) for _, r, p, _ in sets])
        lo, hi = float(allv.min()), float(allv.max())
        pad = 0.04 * (hi - lo) + 1e-9
        lo, hi = lo - pad, hi + pad
        corners = {"train": dict(x=0.03, y=0.97, ha="left", va="top"),
                   "valid": dict(x=0.97, y=0.03, ha="right", va="bottom")}
        handles = {}
        for k, (label, ref, pred, sk) in enumerate(sets):
            ref, pred = np.ravel(ref).astype(float), np.ravel(pred).astype(float)
            if len(ref) == 0:
                continue
            color = self.colors[sk]
            if kind == "density":
                handles[sk] = _density(ax, ref, pred, (lo, hi, lo, hi), self.cmaps[sk], zorder=2 + k,
                                       bins=bins, cell=cell, cmap_range=self.cmap_range,
                                       cmap_reverse=self.cmap_reverse)
            else:
                if len(ref) > self.max_points:
                    sel = self._rng.choice(len(ref), self.max_points, replace=False)
                    r, pp = ref[sel], pred[sel]
                else:
                    r, pp = ref, pred
                ax.scatter(r, pp, s=3, alpha=0.6, color=color, edgecolors="none",
                           rasterized=True, label=label, zorder=2 + k)
            if annotate:
                ss_res = float(np.sum((pred - ref) ** 2))
                ss_tot = float(np.sum((ref - ref.mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                c = corners[sk]
                fmt = ".2f" if key == "S" else ".1f"
                tcolor = color if kind != "density" else _dark(self.cmaps[sk])
                ax.text(c["x"], c["y"],
                        f"{label}\n$R^2$ = {r2:.4f}\nRMSE = {_rmse(ref, pred) * scale:{fmt}} {eunit}\n"
                        f"MAE = {_mae(ref, pred) * scale:{fmt}} {eunit}",
                        transform=ax.transAxes, ha=c["ha"], va=c["va"], color=tcolor,
                        fontsize=self.rc["font.size"], linespacing=1.3, zorder=10)
        ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=0.8, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"DFT {title.lower()} ({unit})")
        ax.set_ylabel(f"NEP {title.lower()} ({unit})")
        return handles

    def _sets(self, data, q):
        """``(label, ref, pred, set_key)`` for the splits that carry ``q``."""
        out = []
        for split, sk, label in (("train", "train", "Training"), ("test", "valid", "Validation")):
            if split in data and q in data[split]:
                m = data[split][q]["mask"]
                if m.any():
                    out.append((label, data[split][q]["ref"][m], data[split][q]["pred"][m], sk))
        return out

    def _rmse_curves(self, ax, d, stage2=None, keys=None):
        """Training (solid) and validation (same colour, alpha 0.5) RMSE of
        E / F / V against the epoch on a log scale; a dashed line marks the
        start of stage 2."""
        tr, va = [], []
        keys = keys or [("e", "e_test", "energy", "E", ""), ("f", "f_test", "force", "F", ""),
                        ("v", "v_test", "virial", "V", "")]
        for k, kt, name, short, _unit in keys:
            if k not in d:
                continue
            color = self.colors[name]
            tr.append(ax.plot(d["epoch"], d[k], color=color, label=f"Training-{short}")[0])
            if kt in d:
                va.append(ax.plot(d["epoch"], d[kt], color=color, alpha=0.5,
                                  label=f"Validation-{short}")[0])
        if stage2 is not None:
            ax.axvline(stage2, color="0.4", lw=0.7, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        handles = tr + va
        leg = ax.legend(handles, [h.get_label() for h in handles], loc="upper right",
                        ncol=2 if va else 1, columnspacing=0.8, frameon=True,
                        facecolor="white", edgecolor="none", framealpha=1.0,
                        borderpad=0.3)
        leg.set_zorder(20)                       # above the stage-2 line

    def dashboard(self, path, out=None, virial=False, title=None, stage2="auto", figsize=None):
        """One figure per training run: (a) the training / validation RMSE
        curves of E, F, V from ``loss.out`` and (b)(c)(d) the energy / force /
        stress parity plots (scatter) with both sets overlaid — 2 x 2 panels
        of equal size, or one row of three when the data carry no stress
        labels. ``virial=True``: the virial (eV/atom) instead of the stress.
        ``stage2``: epoch where stage 2 started (dashed line); ``"auto"``
        reads it from ``output.log`` / ``nep.in`` in ``path``, None for
        none. ``figsize`` in cm."""
        import matplotlib.pyplot as plt
        d = read_loss(os.path.join(path, "loss.out"))
        if stage2 == "auto":
            stage2 = stage2_epoch(path)
        splits = [s for s in ("train", "test")
                 if os.path.exists(os.path.join(path, f"energy_{s}.out"))]
        data = {s: read_outputs(path, s) for s in splits}
        third = "V" if virial else "S"
        has_s = bool(splits) and all(third in data[s] and data[s][third]["mask"].any()
                                     for s in splits)
        qs = ["E", "F"] + ([third] if has_s else [])
        with plt.rc_context(self.rc):
            if has_s:
                fig, axes = plt.subplots(2, 2, figsize=_cm(*(figsize or (12, 11.5))),
                                         constrained_layout=True)
            else:
                fig, axes = plt.subplots(1, 3, figsize=_cm(*(figsize or (17, 5.6))),
                                         constrained_layout=True)
            fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.04)
            axes = axes.ravel()
            self._rmse_curves(axes[0], d, stage2)
            axes[0].set_box_aspect(1)                  # same square as the parity panels
            for ax, q in zip(axes[1:], qs):
                self._parity_overlay(ax, self._sets(data, q), q, "scatter")
            self._label_panels(axes)
            if title:
                fig.suptitle(title)
        return self._finish(fig, out)

def _shift_note(q, mode):
    return None if (q != "E" or mode is None) else f"shifted ({mode})"


shift_energy_fn = shift_energy   # keyword ``shift_energy`` shadows the helper inside methods
