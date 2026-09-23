# Vendored from BLM

These files are copied from the BLM repository (`slm/external/cpx/`). Do not edit the
vendored parts by hand; change BLM and re-copy.

- BLM commit: `fb02e237d8d0b17eac92e6d770ad06ee9d913a10`
- BLM branch: `berk`
- BLM commit date: 2026-09-22

Files:

- `utils.py` <- `slm/external/cpx/utils.py` (whole file)
- `callbacks.py` <- `slm/external/cpx/callbacks.py` (`StatsCallback`, between the
  `# --- begin BLM StatsCallback (vendored; do not edit)` / `# --- end BLM StatsCallback` markers; imports hoisted to the top)

The copy is refreshed by the local, untracked helper `dev/vendor_blm.py` (maintainer-only;
needs a local BLM checkout).
