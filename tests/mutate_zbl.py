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
"""Mutation audit of the ZBL tests: plant a bug, run the ZBL tests, expect
a failure. Every mutant must be killed; a survivor is a hole in the tests.

    python tests/mutate_zbl.py [-j 6] [name ...]

Each mutant runs in a scratch copy of torchnep/ and tests/ (the working
tree is never touched). Not part of the pytest run (minutes); rerun it
after changing the ZBL code or its tests.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_FILES = ["test_zbl_matrix.py", "test_typewise_zbl_cap.py", "test_zbl_flexible.py",
              "test_gpumd_parity.py", "test_slim.py", "test_cutoff_rules.py"]

M, O, N, D, C = ("torchnep/model.py", "torchnep/ops.py", "torchnep/nep.py",
                 "torchnep/data.py", "torchnep/constants.py")

# name: (file, old, new) — old must occur exactly once
MUTANTS = {
    # the typewise cutoff and its cap
    "pr29_cap_from_radii": (M, "self.zbl_rc_outer = self.zbl        #",
                            "self.zbl_rc_outer = max(2.0 * r for r in rc_i)        #"),
    "table_cap_removed": (M, "max=self.zbl_rc_outer)", "max=1e9)"),
    "eager_cap_removed": (O, "rc_outer = torch.clamp(rc_outer_pair, max=rc_outer_default)",
                          "rc_outer = rc_outer_pair"),
    "eager_coarse_mask_short": (O, "max_rc = min(float(rc_outer_per_type.max().item()), rc_outer_default)",
                                "max_rc = 0.9 * min(float(rc_outer_per_type.max().item()), rc_outer_default)"),
    "table_pair_cutoff_one_side": (M, "0.5 * (rt.view(-1, 1) + rt.view(1, -1))",
                                   "(rt.view(-1, 1) + 0 * rt.view(1, -1))"),
    "eager_pair_cutoff_one_side": (O, "rc_outer_pair = (rc_outer_per_type[t1] + rc_outer_per_type[t2]) * 0.5",
                                   "rc_outer_pair = rc_outer_per_type[t1] * 1.0"),
    "table_typewise_inner_nonzero": (M, "rc_i_pair = torch.zeros_like(rc_o_pair)",
                                     "rc_i_pair = torch.full_like(rc_o_pair, self.zbl_rc_inner)"),
    "eager_typewise_inner_nonzero": (O, "rc_inner = torch.zeros_like(rc_outer)",
                                     "rc_inner = torch.full_like(rc_outer, rc_inner_default)"),
    "typewise_inner_radii_float32": (M, "torch.tensor(rc_i, dtype=torch.float64))", "torch.tensor(rc_i))"),
    "typewise_outer_radii_float32": (M, "torch.tensor([2.0 * r for r in rc_i], dtype=torch.float64))",
                                     "torch.tensor([2.0 * r for r in rc_i]))"),
    "calc_typewise_radii_scaled": (N, "self.zbl_rc_outer_per_type = 2.0 * self.zbl_rc_inner_per_type",
                                   "self.zbl_rc_outer_per_type = 1.99 * self.zbl_rc_inner_per_type"),
    "radius_table_entry": (C, "2.78667, 2.34667,", "2.7867, 2.34667,"),
    # universal
    "universal_inner_cutoff": (M, "self.zbl_rc_inner = self.zbl / 2.0", "self.zbl_rc_inner = self.zbl / 2.1"),
    # the screening function and the switching function
    "eager_a_inv_constant": (O, "a_inv = (zi ** 0.23 + zj ** 0.23) * 2.134563",
                             "a_inv = (zi ** 0.23 + zj ** 0.23) * 2.1344717"),
    "eager_switch_linear": (O, "    fc = 0.5 * torch.cos(PI * t) + 0.5\n\n    e_pair",
                            "    fc = 1.0 - t\n\n    e_pair"),
    "pair_switch_overshoot": (O, "t = torch.clamp((d - rc_i) * inv_w, 0.0, 1.0)",
                              "t = torch.clamp((d - rc_i) * inv_w, 0.0, 1.02)"),
    "pair_dfc_sign": (O, "dfc = -0.5 * PI * torch.sin(PI * t) * inv_w",
                      "dfc = 0.5 * PI * torch.sin(PI * t) * inv_w"),
    "pair_dfc_dropped": (O, "de = zizj * (dphi * fc + phi * dfc - phi * fc / d) / d",
                         "de = zizj * (dphi * fc - phi * fc / d) / d"),
    "energy_to_neighbour": (O, "e_atom.scatter_add_(0, pi, 0.5 * e_pair)",
                            "e_atom.scatter_add_(0, pj, 0.5 * e_pair)"),
    "pair_table_index_i_only": (O, "idx = atom_types[pair_i] * T + atom_types[pair_j]",
                                "idx = atom_types[pair_i] * T + atom_types[pair_i]"),
    "per_atom_virial_on_centre": (O, "virial.scatter_add_(0, pj.unsqueeze(-1).expand_as(v9), v9)",
                                  "virial.scatter_add_(0, pi.unsqueeze(-1).expand_as(v9), v9)"),
    # flexible (zbl.in)
    "flex_index_formula": (D, "return t1 * num_types - (t1 * (t1 - 1)) // 2 + (t2 - t1)",
                           "return t1 * num_types - (t1 * (t1 + 1)) // 2 + (t2 - t1)"),
    "flex_model_row_diagonal": (M, "                row = tab[zbl_pair_index(t1, t2, T)]",
                                "                row = tab[zbl_pair_index(t1, t1, T)]"),
    "flex_calc_row_diagonal": (N, "                    row = tab[zbl_pair_index(t1, t2, T)]",
                               "                    row = tab[zbl_pair_index(t2, t2, T)]"),
    "flex_eager_coef_diagonal": (O, "coef = phi_pair[t1, t2].to(dtype)", "coef = phi_pair[t2, t2].to(dtype)"),
    "flex_compiled_phi_dropped": (M, "phi_tab=self.zbl_phi_pair if self.zbl_flexible is not None else None)",
                                  "phi_tab=None)"),
    "flex_compiled_autograd_phi_dropped": ("torchnep/compiled_autograd.py",
                                           "phi_tab=(pd[\"zbl_phi_pair\"]", "phi_tab=(None"),
    "flex_nep_txt_not_adopted": (M, "if file_flexible and self.zbl is not None and self.zbl_flexible is None:",
                                 "if False:"),
    # nep.in / nep.txt
    "nep_in_factor_misread": (D, "params[\"typewise_cutoff_zbl_factor\"] = float(parts[1])",
                              "params[\"typewise_cutoff_zbl_factor\"] = float(parts[1]) * 1.01"),
    "nep_txt_factor_not_written": (M, "lines.append(f\"zbl {rc_inner_out} {rc_outer_out} {tw}\")",
                                   "lines.append(f\"zbl {rc_inner_out} {rc_outer_out}\")"),
    "nep_txt_inner_written_wrong": (M, "rc_inner_out = self.zbl / 2.0", "rc_inner_out = self.zbl_rc_inner"),
    "calc_universal_read_as_typewise": (N, "float(parts[3]) if len(parts) > 3 else None",
                                        "float(parts[3]) if len(parts) > 3 else 0.7"),
    # inference paths
    "tiled_zbl_energy_dropped": (N, "E_atom[blk] = E_atom[blk] + Ei_zbl[blk].detach()",
                                 "E_atom[blk] = E_atom[blk]"),
    # slimming
    "slim_typewise_dropped": (M, "new_config[\"typewise_cutoff_zbl_factor\"] = model.zbl_typewise_factor",
                              "pass"),
    "slim_config_flex_rows": (M, "out[\"zbl_flexible\"] = [list(rows[zbl_pair_index(keep_idx[a], keep_idx[b], T)])",
                              "out[\"zbl_flexible\"] = [list(rows[zbl_pair_index(a, b, T)])"),
    # precision of the constant tables
    "coefficients_float32": (M, "torch.tensor(C3B[:self.num_lm], dtype=torch.float64))",
                             "torch.tensor(C3B[:self.num_lm]))"),
    "checkpoint_constants_kept": (M, "        self.register_load_state_dict_post_hook(NEPModel._restore_constants)\n",
                                  "\n"),
}


def run(name, verbose):
    rel, old, new = MUTANTS[name]
    tmp = Path(tempfile.mkdtemp(prefix=f"mut_{name}_"))
    try:
        shutil.copytree(ROOT / "torchnep", tmp / "torchnep",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "tests", tmp / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        f = tmp / rel
        text = f.read_text()
        if text.count(old) != 1:
            return name, "BAD", f"pattern found {text.count(old)} times in {rel}"
        f.write_text(text.replace(old, new))
        env = dict(os.environ, PYTHONPATH=str(tmp), TEST_DEVICE="cpu", PYTHONDONTWRITEBYTECODE="1")
        check = subprocess.run([sys.executable, "-c", "import torchnep; print(torchnep.__file__)"],
                               cwd=tmp / "tests", env=env, capture_output=True, text=True)
        if not check.stdout.startswith(str(tmp)):
            return name, "BAD", f"imports torchnep from {check.stdout.strip()}"
        r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
                            "-W", "ignore", *TEST_FILES], cwd=tmp / "tests", env=env,
                           capture_output=True, text=True)
        failed = [ln for ln in r.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))]
        if r.returncode == 0:
            return name, "SURVIVED", r.stdout.strip().splitlines()[-1]
        return name, "killed", (failed[0] if failed else r.stdout.strip().splitlines()[-1])[:150]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=6)
    ap.add_argument("-v", action="store_true")
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    names = a.names or list(MUTANTS)
    with ThreadPoolExecutor(a.j) as ex:
        results = list(ex.map(lambda n: run(n, a.v), names))
    width = max(len(n) for n in names)
    for name, status, info in results:
        print(f"{name:{width}s}  {status:8s}  {info}")
    bad = [n for n, s, _ in results if s != "killed"]
    print(f"\n{len(results) - len(bad)} / {len(results)} mutants killed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
