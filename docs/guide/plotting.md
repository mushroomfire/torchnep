# Plotting

`torchnep.plot.NEPPlotter` draws figures straight from the files a run writes — `loss.out`, `energy/force/virial/stress_train.out` and `*_test.out` — and from the output of `predict_dataset`. It needs matplotlib: `pip install torchnep[plot]`.

```python
from torchnep.plot import NEPPlotter

p = NEPPlotter()
p.dashboard("run", out="dashboard.png")      # loss curves + E/F/stress parity, train and validation
p.loss("run", out="loss.png")                 # the loss curves alone
p.parity("run", out="parity.png")             # E/F/stress parity plots
```

`"run"` is the output directory of a training run (or of `predict_dataset`).

<figure markdown>
![Training dashboard](../assets/plot_dashboard.png){ width="560" }
<figcaption><code>p.dashboard("run")</code></figcaption>
</figure>

## Parity plots

```python
p.parity("run", kind="density", margins=True, out="parity_density.png")
```

- `kind="density"` draws hexagonal density cells instead of points — readable for millions of force components. `bins` sets the cell size, `cell="square"` switches to square cells.
- `margins=True` adds a strip above each panel with the error (NEP − DFT) against the DFT value.
- `virial=True` shows the virial instead of the stress; `quantities` picks the panels (`E`, `F`, `V`, `S`).
- `exclude=` leaves listed training frames out, e.g. known outliers (needs `natoms=` or `xyz=`).

<figure markdown>
![Parity plots with error strips](../assets/plot_parity_margins_density.png)
<figcaption><code>p.parity("run", kind="density", margins=True)</code></figcaption>
</figure>

### Data from another DFT reference

When the reference energies come from different DFT settings, remove the per-element energy offset before plotting:

```python
p.parity("pred", shift_energy="element", xyz="test.xyz", out="pred.png")
```

## Errors per element

```python
p.errors("run", xyz="train.xyz", out="errors.png")
p.periodic_table(path="run", xyz="train.xyz", families=True, out="table.png")
```

`errors` draws the error distributions, the force error against the force magnitude, the force RMSE per element and the energy RMSE per `config_type`. `periodic_table` colours each element by its energy and force RMSE; `element_errors(path, xyz)` returns the numbers.

## Style

The constructor sets the style of every figure:

| Argument | Default | Meaning |
|---|---|---|
| `font` | `"Arial"` | Font family; matplotlib's default when not installed. |
| `font_dir` | `TORCHNEP_FONT_DIR` | Folder of `.ttf` / `.otf` files to register first. |
| `fontsize` | `7` | Base font size in points. |
| `dpi` | `300` | Figure and file resolution. |
| `colors` | built-in | Colours of the training / validation sets and of the E/F/V/S curves. |
| `cmaps`, `cmap_range`, `cmap_reverse` | Blues / Reds, `(0.1, 0.9)`, `True` | Density colormaps; reversed by default so sparse cells (the outliers) are dark. |
| `panel_labels`, `label_format` | `"abcd…"`, `"{}"` | Panel labels of multi-panel figures; `None` for none. |
| `frame` | `False` | Full box with ticks on four sides. |
| `rc` | `None` | Extra matplotlib rcParams. |
