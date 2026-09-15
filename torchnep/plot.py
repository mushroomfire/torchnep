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
    """Whitespace table -> float64 (rows, cols); '#' lines ignored. pandas
    when available (C parser, ~10x faster on the big force files)."""
    try:
        import pandas as pd
        return pd.read_csv(path, sep=r"\s+", header=None, comment="#",
                           dtype=np.float64).to_numpy()
    except ImportError:
        return np.loadtxt(path, comments="#", ndmin=2)


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


def _font_family(font):
    """``font`` if matplotlib can find it, else None (matplotlib's default)."""
    if not font:
        return None
    from matplotlib import font_manager as fm
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


_QTY = {   # key: (title, unit, error unit, error scale)
    "E": ("Energy", "eV/atom", "meV/atom", 1e3),
    "F": ("Force", "eV/Å", "meV/Å", 1e3),
    "V": ("Virial", "eV/atom", "meV/atom", 1e3),
    "S": ("Stress", "GPa", "GPa", 1.0),
}


class NEPPlotter:
    """Figures for torchnep / GPUMD training and prediction outputs.

    Parameters
    ----------
    font : str
        Font family (default ``"Arial"``); matplotlib's default when absent.
    fontsize, dpi : int
    cmap : str
        Colormap of the density (hexbin) panels.
    max_points : int
        Scatter panels draw at most this many points (a fixed random
        subsample); metrics always use every point.
    colors : (train, test) line / marker colours.
    """

    def __init__(self, font="Arial", fontsize=10, dpi=150, cmap="viridis",
                 max_points=300_000, colors=("#1f77b4", "#d62728")):
        import matplotlib  # noqa: F401  (fail early with a clear message)
        fam = _font_family(font)
        self.rc = {"font.size": fontsize, "axes.labelsize": fontsize,
                   "axes.titlesize": fontsize + 1, "legend.fontsize": fontsize - 1,
                   "xtick.labelsize": fontsize - 1, "ytick.labelsize": fontsize - 1,
                   "figure.dpi": dpi, "savefig.dpi": dpi, "axes.linewidth": 0.8,
                   "xtick.direction": "in", "ytick.direction": "in",
                   "legend.frameon": False, "mathtext.default": "regular"}
        if fam:
            self.rc["font.family"] = fam
        self.cmap = cmap
        self.max_points = int(max_points)
        self.c_train, self.c_test = colors
        self._rng = np.random.default_rng(0)

    # ---- generic panels ---------------------------------------------------
    def _finish(self, fig, out):
        import matplotlib.pyplot as plt
        if out:
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

    def _loss_panels(self, fig, gs_row, d):
        for i, (k, kt, name, unit) in enumerate(
                [("loss", None, "loss", ""), ("e", "e_test", "RMSE E", "eV/atom"),
                 ("f", "f_test", "RMSE F", "eV/Å"), ("v", "v_test", "RMSE V", "eV/atom")]):
            ax = fig.add_subplot(gs_row[i])
            if k == "loss":
                ax.plot(d["epoch"], d["loss"], color=self.c_train, lw=1.2,
                        label="valid" if "e_test" in d else "train")
                ax.set_ylabel("loss")
            else:
                ax.plot(d["epoch"], d[k], color=self.c_train, lw=1.2, label="train")
                if kt in d:
                    ax.plot(d["epoch"], d[kt], color=self.c_test, lw=1.2, ls="--", label="test")
                ax.set_ylabel(f"{name} ({unit})")
                last = f"{d[k][-1]:.4g}" + (f" / {d[kt][-1]:.4g}" if kt in d else "")
                ax.set_title(f"{name}: last {last}", fontsize=self.rc["font.size"])
            ax.set_yscale("log")
            ax.set_xlabel("epoch")
            ax.legend(loc="upper right")

    # ---- figures ----------------------------------------------------------
    def loss(self, path="loss.out", out=None):
        """Training curves from ``loss.out``: the loss and the E / F / V
        RMSEs per epoch, train (solid) and test (dashed) when present."""
        import matplotlib.pyplot as plt
        d = read_loss(path)
        with plt.rc_context(self.rc):
            fig = plt.figure(figsize=(13.6, 3.1))
            self._loss_panels(fig, fig.add_gridspec(1, 4)[0, :].subgridspec(1, 4), d)
            fig.tight_layout()
        return self._finish(fig, out)

    def parity(self, path=".", split="train", kind="scatter", out=None,
               quantities=("E", "F", "V"), title=None, shift_energy=None, xyz=None):
        """Parity plots (DFT vs NEP) of the ``*_<split>.out`` files in
        ``path``. ``kind``: ``"scatter"`` or ``"density"`` (log-count hexbin).
        ``quantities``: any of ``E F V S``. ``shift_energy``: None, ``"mean"``
        or ``"element"`` (needs ``xyz``) for data of another DFT reference."""
        import matplotlib.pyplot as plt
        d = read_outputs(path, split)
        qs = [q for q in quantities if q in d]
        meta = frame_meta(xyz) if xyz else None
        e_pred = shift_energy_fn(d, shift_energy, meta) if "E" in d else None
        with plt.rc_context(self.rc):
            fig, axes = plt.subplots(1, len(qs), figsize=(3.9 * len(qs), 3.7))
            axes = np.atleast_1d(axes)
            color = self.c_test if split == "test" else self.c_train
            for ax, q in zip(axes, qs):
                m = d[q]["mask"]
                pred = e_pred if q == "E" else d[q]["pred"]
                self._parity_panel(ax, d[q]["ref"][m], pred[m], q, kind, color,
                                   note=_shift_note(q, shift_energy))
            if title:
                fig.suptitle(title, fontsize=self.rc["font.size"] + 1)
            fig.tight_layout()
        return self._finish(fig, out)

    def parity_train_test(self, path=".", kind="scatter", out=None,
                          quantities=("E", "F", "V")):
        """Train (top row) and test (bottom row) parity plots."""
        import matplotlib.pyplot as plt
        splits = [s for s in ("train", "test")
                  if os.path.exists(os.path.join(path, f"energy_{s}.out"))]
        data = {s: read_outputs(path, s) for s in splits}
        qs = [q for q in quantities if all(q in data[s] for s in splits)]
        with plt.rc_context(self.rc):
            fig, axes = plt.subplots(len(splits), len(qs),
                                     figsize=(3.9 * len(qs), 3.7 * len(splits)), squeeze=False)
            for i, s in enumerate(splits):
                color = self.c_test if s == "test" else self.c_train
                for ax, q in zip(axes[i], qs):
                    m = data[s][q]["mask"]
                    self._parity_panel(ax, data[s][q]["ref"][m], data[s][q]["pred"][m],
                                       q, kind, color, label=s)
            fig.tight_layout()
        return self._finish(fig, out)

    def errors(self, path=".", split="train", out=None, xyz=None,
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

    def prediction(self, path=".", split="train", kind="density", out=None, xyz=None,
                   quantities=("E", "F", "V"), shift_energy=None, title=None):
        """The ``predict_dataset`` view: parity plots (top) and error
        distributions (middle) of the outputs in ``path``; with ``xyz`` also
        the per-element / per-config_type breakdown (bottom)."""
        import matplotlib.pyplot as plt
        d = read_outputs(path, split)
        qs = [q for q in quantities if q in d]
        meta = frame_meta(xyz) if xyz else None
        e_pred = shift_energy_fn(d, shift_energy, meta) if "E" in d else None
        extra = self._breakdown_panels(d, meta, e_pred) if meta else []
        n_cols = max(len(qs), 2)
        n_rows = 2 + (1 if extra else 0)
        with plt.rc_context(self.rc):
            fig = plt.figure(figsize=(3.9 * n_cols, 3.6 * n_rows))
            gs = fig.add_gridspec(n_rows, n_cols)
            for i, q in enumerate(qs):
                m = d[q]["mask"]
                pred = e_pred if q == "E" else d[q]["pred"]
                self._parity_panel(fig.add_subplot(gs[0, i]), d[q]["ref"][m], pred[m], q, kind,
                                   note=_shift_note(q, shift_energy))
                self._hist_panel(fig.add_subplot(gs[1, i]), pred[m] - d[q]["ref"][m], q)
            if extra:
                bounds = np.linspace(0, n_cols, len(extra) + 1).astype(int)
                for k, draw in enumerate(extra):
                    draw(fig.add_subplot(gs[2, bounds[k]:bounds[k + 1]]))
            if title:
                fig.suptitle(title, fontsize=self.rc["font.size"] + 1)
            fig.tight_layout()
        return self._finish(fig, out)

    def dashboard(self, path=".", kind="scatter", out=None, quantities=("E", "F", "V"),
                  title=None):
        """One figure per training run: the loss curves (top row) and the
        train / test parity plots (one row each) from ``path``."""
        import matplotlib.pyplot as plt
        d = read_loss(os.path.join(path, "loss.out"))
        splits = [s for s in ("train", "test")
                  if os.path.exists(os.path.join(path, f"energy_{s}.out"))]
        data = {s: read_outputs(path, s) for s in splits}
        qs = [q for q in quantities if all(q in data[s] for s in splits)] if splits else []
        n_cols = max(4, len(qs))
        with plt.rc_context(self.rc):
            fig = plt.figure(figsize=(3.6 * n_cols, 3.3 * (1 + len(splits))))
            gs = fig.add_gridspec(1 + len(splits), n_cols)
            self._loss_panels(fig, gs[0, :4].subgridspec(1, 4), d)
            for r, s in enumerate(splits, start=1):
                color = self.c_test if s == "test" else self.c_train
                for i, q in enumerate(qs):
                    m = data[s][q]["mask"]
                    self._parity_panel(fig.add_subplot(gs[r, i]), data[s][q]["ref"][m],
                                       data[s][q]["pred"][m], q, kind, color, label=s)
            fig.suptitle(title or os.path.basename(os.path.abspath(path)),
                         fontsize=self.rc["font.size"] + 1)
            fig.tight_layout()
        return self._finish(fig, out)


def _shift_note(q, mode):
    return None if (q != "E" or mode is None) else f"shifted ({mode})"


shift_energy_fn = shift_energy   # keyword ``shift_energy`` shadows the helper inside methods
