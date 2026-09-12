# Contributing to TorchNEP

Pull requests are welcome. To keep reviews quick:

1. **Open an issue first** for anything larger than a bug fix, so the design can be discussed before code is written.
2. **One topic per PR.** A bug fix and a new feature go in separate PRs.
3. **Fill in the PR template** (description, what changed, results, notes). A PR without a description will be sent back.
4. **Tests must pass.** Run `pytest` locally before pushing; the `tests` workflow (Python 3.9 and 3.12, CPU) must be green before a PR is merged. Add a test for every new behaviour.
5. **Keep the public API stable.** New nep.in keys default to the old behaviour; do not change defaults without saying so in `releaseNotes.md`.
6. **Match the existing style**: numpy-style docstrings, no new dependencies without discussion.

Development setup:

```bash
git clone https://github.com/mushroomfire/torchnep && cd torchnep
pip install -e .[ase,dev]
pytest -q
```
