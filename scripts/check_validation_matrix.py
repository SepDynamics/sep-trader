#!/usr/bin/env python3
"""Validate that all evidence file paths in docs/proofs/validation_matrix.md exist."""

from pathlib import Path
import re
import sys

MATRIX_PATH = Path('docs/proofs/validation_matrix.md')

if not MATRIX_PATH.exists():
    print(f"Matrix file not found: {MATRIX_PATH}")
    sys.exit(1)

missing = []
for line in MATRIX_PATH.read_text().splitlines():
    line = line.strip()
    if not line.startswith('|') or line.startswith('| Pitch') or set(line) <= {'|', '-', ' '}:
        continue
    # Extract backticked paths from the evidence column
    parts = [p.strip() for p in line.strip('|').split('|')]
    if len(parts) < 2:
        continue
    evidence_cell = parts[1]
    paths = re.findall(r'`([^`]+)`', evidence_cell)
    for rel in paths:
        if not Path(rel).exists():
            missing.append(rel)

if missing:
    print('Missing evidence files:')
    for m in missing:
        print(f' - {m}')
    sys.exit(1)
else:
    print('All referenced evidence files exist.')
