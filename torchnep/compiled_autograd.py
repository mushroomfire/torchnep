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

"""torch.compile for the AUTOGRAD force path (the DeepMD/DPA make_fx route).

The autograd path computes forces as F = -dE/drij by a nested
``autograd.grad(create_graph=True)``, which torch.compile cannot lower
directly — that is why ``use_compile`` used to be ignored for
``use_autograd_forces=True``. The workaround (used by DeepMD-kit / DPA and
by the ctp project this port follows):

  1. write (energy, forces) as a PURE function of (params+buffers, batch
     geometry) — parameters enter as function inputs, so a normal
     ``loss.backward()`` still fills ``model.parameters().grad``;
  2. ``make_fx`` trace it symbolically: the inner first-order gradient is
     materialized as ordinary FX ops (no runtime double-backward left);
  3. ``strip_detach``: make_fx wraps saved activations in aten.detach chains
     that would silently CUT the second-order path from the force loss back
     to the parameters (param grads come out wrong, rel err ~1) — the traced
     leaves are already detached, so every detach node is safe to drop;
  4. ``torch.compile`` the resulting graph with dynamic shapes, so ONE graph
     serves every batch size.

The traced energy re-implements the per-type NN dispatch with
``torch.where`` over all types (NEPModel.forward's ``mask.any()`` branches
are data-dependent and cannot be traced symbolically), and uses the "bmm"
contraction backend (the "loop" backend's per-type-pair ``.any()`` masks are
data-dependent too). ZBL stays OUTSIDE the compiled graph (its typewise
cutoffs use ``.item()``) and is added eagerly, exactly like the cached
analytical path does.
"""

import os
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from . import ops


_PATCHED = False


def apply_compile_patches():
    """Process-global adjustments for a robust dynamic-shape compile.

    Idempotent. The key one is disabling opt_einsum: it flattens contraction
    operands and bakes the pair/atom count into a constant, forcing a
    recompile for EVERY distinct batch shape. The einsums here contract tiny
    angular dims, so the naive path costs nothing at runtime.
    """
    global _PATCHED
    if _PATCHED:
        return
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    torch.backends.opt_einsum.enabled = False
    try:
        torch._dynamo.config.recompile_limit = 64
        torch._dynamo.config.accumulated_recompile_limit = 256
    except Exception:
        pass
    _patch_cantsplit()
    _PATCHED = True


def _patch_cantsplit():
    """torch 2.11: the Triton tiler does ``raise CantSplit`` with no args while
    CantSplit.__init__ requires them — a TypeError escapes instead of the
    catchable control-flow exception. Give it an arg-free constructor."""
    try:
        from torch._inductor.codegen.simd import CantSplit
    except Exception:
        try:
            from torch._inductor.exc import CantSplit
        except Exception:
            return
    if getattr(CantSplit, "_torchnep_patched", False):
        return

    def __init__(self, expr=None, remaining=None):
        Exception.__init__(self)
        self.expr, self.remaining = expr, remaining

    CantSplit.__init__ = __init__
    CantSplit._torchnep_patched = True


def strip_detach(gm):
    """Remove every aten.detach node from a make_fx graph (see module doc)."""
    det = torch.ops.aten.detach.default
    for n in list(gm.graph.nodes):
        if n.op == "call_function" and n.target == det:
            n.replace_all_uses_with(n.args[0])
            gm.graph.erase_node(n)
    gm.graph.lint()
    gm.recompile()
    return gm


def inductor_options():
    """Conservative Inductor options (ported from DPA's compile options via
    ctp): the default pipeline runs heavy fusion/autotune on the large
    materialized second-order graph — minutes of codegen, and small GPUs
    can't even run max_autotune_gemm. This cuts compile to seconds."""
    opts = {
        "max_autotune": False,
        "shape_padding": True,
        "epilogue_fusion": False,
        "triton.cudagraphs": False,
        "max_fusion_size": 8,
        "triton.persistent_reductions": False,
        "triton.mix_order_reduction": False,
        "triton.max_tiles": 1,
    }
    try:
        from torch._inductor import config as ic
        valid = ic.get_config_copy()
        opts = {k: v for k, v in opts.items() if k.replace("-", "_") in valid}
    except Exception:
        pass
    return opts


class CompiledAutogradForce:
    """Lazily make_fx-traces + compiles the autograd force on first use, then
    one dynamic-shape graph serves all batch sizes.

    ``compute_properties`` mirrors ``NEPModel.compute_properties`` (same
    signature, same result dict), so the training/eval loops can use it as a
    drop-in replacement for the eager autograd path. Trace lazily AFTER
    q_scaler is set — buffers are graph inputs, but tracing needs the model
    fully initialized.
    """

    def __init__(self, model):
        apply_compile_patches()
        self.model = model
        pd = {**dict(model.named_parameters()), **dict(model.named_buffers())}
        self.pkeys = list(pd)
        self._compiled = None

    def _pvals(self):
        d = {**dict(self.model.named_parameters()),
             **dict(self.model.named_buffers())}
        return tuple(d[k] for k in self.pkeys)

    # -- pure traced function -------------------------------------------------

    def _energy(self, pd, rij_rad, rij_ang, pi_rad, pj_rad, pi_ang, pj_ang,
                atom_types):
        """Per-atom NN energy as a pure function of (params, geometry).

        Re-implements NEPModel.forward without data-dependent branches:
        every type's net runs on all atoms, torch.where selects. All
        parameters appear in the graph, so DDP-style full-graph gradient
        coverage is preserved too.
        """
        m = self.model
        N = atom_types.shape[0]
        q = ops.compute_descriptors(
            rij_rad, rij_ang, pi_rad, pj_rad, pi_ang, pj_ang,
            atom_types, N, pd["c_param_2"], pd.get("c_param_3"),
            m.rc_radial, m.rc_angular,
            m.basis_size_radial, m.basis_size_angular,
            m.n_max_radial, m.n_max_angular,
            m.l_max_3b,
            m.has_q_222, m.has_q_1111, m.has_q_112,
            m.num_lm, pd["_c3b"], pd["_c4b"], pd["_c5b"],
            pd["_c4b2"],
            rij_rad.dtype, rij_rad.device,
            backend="bmm",
            has_q_123=m.has_q_123, has_q_233=m.has_q_233,
            has_q_134=m.has_q_134,
        )
        q_scaled = q * pd["q_scaler"]
        Ei = torch.zeros(N, dtype=q.dtype, device=q.device)
        for t in range(m.num_types):
            w0 = pd[f"fitting_nets.{t}.w0"]
            b0 = pd[f"fitting_nets.{t}.b0"]
            w1 = pd[f"fitting_nets.{t}.w1"]
            e_t = torch.tanh(q_scaled @ w0 - b0) @ w1
            Ei = Ei + torch.where(atom_types == t, e_t,
                                  torch.zeros((), dtype=q.dtype,
                                              device=q.device))
        return Ei - pd["b1"]

    def _raw(self, pvals, rij_rad, rij_ang, pi_rad, pj_rad, pi_ang, pj_ang,
             atom_types):
        pd = dict(zip(self.pkeys, pvals))
        N = atom_types.shape[0]
        rr = rij_rad.detach().requires_grad_(True)
        ra = rij_ang.detach().requires_grad_(True)
        Ei = self._energy(pd, rr, ra, pi_rad, pj_rad, pi_ang, pj_ang,
                          atom_types)
        gr, ga = torch.autograd.grad(Ei.sum(), [rr, ra], create_graph=True,
                                     allow_unused=True)
        if gr is None:
            gr = torch.zeros_like(rr)
        if ga is None:
            ga = torch.zeros_like(ra)
        forces, virial = ops.accumulate_forces_virial(
            N, pi_rad, pj_rad, rr, gr, pi_ang, pj_ang, ra, ga,
            rr.dtype, rr.device)
        return Ei, forces, virial

    # -- tracing --------------------------------------------------------------

    def _prime_args(self, args):
        """Synthetic tracing inputs with PRIME atom/pair counts (the DPA
        trick): make_fx then derives clean symbolic dims, so no batch dim can
        accidentally equal a channel/basis count and get mis-specialized as
        a constant."""
        pvals = args[0]
        dev = next(iter(pvals)).device if pvals else args[1].device
        dtype = args[1].dtype
        m = self.model
        na, npr, npa = 53, 101, 89          # distinct primes >> any model dim
        g = torch.Generator(device="cpu").manual_seed(0)

        def _rij(n, rc):
            v = torch.randn(n, 3, generator=g, dtype=torch.float64)
            v = v / v.norm(dim=1, keepdim=True)
            r = 0.5 + torch.rand(n, 1, generator=g, dtype=torch.float64) \
                * (0.8 * rc - 0.5)
            return (v * r).to(device=dev, dtype=dtype)

        s_types = torch.arange(na, device=dev) % m.num_types
        s_pir = torch.randint(0, na, (npr,), generator=g).to(dev)
        s_pjr = torch.randint(0, na, (npr,), generator=g).to(dev)
        s_pia = torch.randint(0, na, (npa,), generator=g).to(dev)
        s_pja = torch.randint(0, na, (npa,), generator=g).to(dev)
        return (pvals, _rij(npr, m.rc_radial), _rij(npa, m.rc_angular),
                s_pir, s_pjr, s_pia, s_pja, s_types)

    def _ensure_compiled(self, args):
        if self._compiled is not None:
            return
        # Trace a plain closure — make_fx on the bound method would count
        # ``self`` as an input.
        fn = lambda *a: self._raw(*a)
        gm = make_fx(fn, tracing_mode="symbolic",
                     _allow_non_fake_inputs=True)(*self._prime_args(args))
        strip_detach(gm)
        self._compiled = torch.compile(gm, dynamic=True,
                                       options=inductor_options())

    # -- public: drop-in for NEPModel.compute_properties ----------------------

    def compute_properties(self, rij_rad, rij_ang, pi_rad, pj_rad,
                           pi_ang, pj_ang, atom_types, N,
                           struct_idx, num_structures,
                           need_forces=True, need_virial=False,
                           backend: str = "bmm"):
        """Same signature/result dict as ``NEPModel.compute_properties``.

        The compiled graph always evaluates forces+virial (the graph is
        traced once); energy-only calls fall back to the eager model —
        they are rare (b1/eval passes) and cheap.
        """
        if not need_forces:
            return self.model.compute_properties(
                rij_rad, rij_ang, pi_rad, pj_rad, pi_ang, pj_ang,
                atom_types, N, struct_idx, num_structures,
                need_forces=False, need_virial=need_virial, backend=backend)

        args = (self._pvals(), rij_rad, rij_ang, pi_rad, pj_rad,
                pi_ang, pj_ang, atom_types)
        self._ensure_compiled(args)
        Ei, forces, virial = self._compiled(*args)

        m = self.model
        dtype, device = rij_rad.dtype, rij_rad.device
        if m.zbl is not None:
            # ZBL outside the compiled graph (typewise cutoffs use .item()).
            # No trainable parameters — energies/forces detach, same as the
            # cached analytical path.
            with torch.enable_grad():
                rz = rij_ang.detach().requires_grad_(True)
                Ei_zbl = ops.compute_zbl(
                    atom_types, pi_ang, pj_ang, rz, N,
                    m.atomic_numbers.tolist(),
                    m.zbl_rc_inner, m.zbl_rc_outer, m.zbl_typewise_factor,
                    getattr(m, "zbl_rc_inner_per_type", None),
                    getattr(m, "zbl_rc_outer_per_type", None), dtype, device)
                if Ei_zbl.requires_grad:
                    g_zbl = torch.autograd.grad(Ei_zbl.sum(), rz,
                                                allow_unused=True)[0]
                else:
                    g_zbl = None
            Ei = Ei + Ei_zbl.detach()
            if g_zbl is not None:
                empty_i = torch.zeros(0, dtype=torch.long, device=device)
                empty_r = torch.zeros(0, 3, dtype=dtype, device=device)
                zf, zv = ops.accumulate_forces_virial(
                    N, empty_i, empty_i, empty_r, empty_r,
                    pi_ang, pj_ang, rij_ang.detach(), g_zbl.detach(),
                    dtype, device)
                forces = forces + zf
                virial = virial + zv

        Etot = torch.zeros(num_structures, dtype=dtype, device=device)
        Etot.scatter_add_(0, struct_idx, Ei)
        result = {"Ei": Ei, "Etot": Etot, "forces": forces}
        if need_virial:
            result["virial"] = virial
        return result
