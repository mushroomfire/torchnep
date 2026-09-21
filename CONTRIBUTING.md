# Contributing to TorchNEP

Pull requests are welcome. To keep reviews quick:

1. **Open an issue first** for anything larger than a bug fix, so the design can be discussed before code is written.
2. **One topic per PR.** A bug fix and a new feature go in separate PRs.
3. **Fill in the PR template** (description, what changed, results, notes). A PR without a description will be sent back.
4. **Tests must pass.** Run `pytest` locally before pushing; the `tests` workflow (Python 3.9 and 3.12, CPU) must be green before a PR is merged. Add a test for every new behaviour.
5. **No lint warnings.** Run `ruff check .` locally before pushing; it must report no warnings, and the `ruff` job of the `tests` workflow checks it on every PR. The rules are in `pyproject.toml` (`[tool.ruff.lint]`); fix the code rather than adding exceptions.
6. **Keep the public API stable.** New nep.in keys default to the old behaviour; say so in the PR if a default changes.
7. **Update the documentation** (`docs/`) if your change is user-facing. Do not edit `docs/release-notes.md` or the version number (`torchnep/__init__.py`): the maintainers write the release notes and bump the version when a release is cut.
8. **Match the existing style**: numpy-style docstrings, no new dependencies without discussion.

Development setup:

```bash
git clone https://github.com/mushroomfire/torchnep && cd torchnep
pip install -e .[ase,dev]
pytest -q
ruff check .
```

Documentation preview (served at http://127.0.0.1:8000/torchnep/, reloads on every edit):

```bash
pip install -e .[docs]
mkdocs serve
```
