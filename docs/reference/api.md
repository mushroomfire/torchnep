# Python API

The functions most scripts need are importable from the package itself:

```python
from torchnep import train_nep, train_nep_sharded, predict_dataset, predict_dataset_sharded, export_valid_split
```

## Training

::: torchnep.train.train_nep

::: torchnep.train_sharded.train_nep_sharded

::: torchnep.data.export_valid_split

::: torchnep.data.parse_nep_in

## Prediction

::: torchnep.predict.predict_dataset

::: torchnep.predict.predict_dataset_sharded

::: torchnep.nep.NEPCalculator
    options:
      members: [compute, get_descriptor, compute_tiled]

::: torchnep.ase_calculator.NEP
    options:
      members: [get_components, get_energy_components]

## Models

::: torchnep.model.slim_model

## Plotting

::: torchnep.plot.NEPPlotter
    options:
      members: [dashboard, loss, parity, errors, periodic_table]

::: torchnep.plot.element_errors

## Extrapolation grade

::: torchnep.extrapolation.build_active_set

::: torchnep.extrapolation.select_structures

::: torchnep.extrapolation.compute_gamma

::: torchnep.extrapolation.build_active_set_sharded

::: torchnep.extrapolation.select_structures_sharded

::: torchnep.extrapolation.compute_gamma_sharded

::: torchnep.extrapolation.ActiveSet
    options:
      members: [load, save, save_gpumd, gamma]
