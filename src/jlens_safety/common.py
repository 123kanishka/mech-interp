"""Durable state, explicit budget stops, and immutable resume identities."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from sae_jlens.run_io import atomic_json, append_jsonl, read_jsonl


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def source_hash(root):
    paths = sorted((root / 'src/jlens_safety').glob('*.py'))
    paths += [root / 'experiments/run_jlens_safety.py']
    paths += [root / 'src/sae_jlens/run_io.py']
    return digest({str(p.relative_to(root)): file_hash(p) for p in paths})


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Cumulative active process time, including failed/resumed stages."""
    def __init__(self, path, hours):
        self.path = Path(path)
        self.previous = json.loads(self.path.read_text())['seconds'] if self.path.exists() else 0
        self.start = time.monotonic()
        self.limit = float(hours) * 3600

    @property
    def seconds(self):
        return self.previous + time.monotonic() - self.start

    def save(self):
        atomic_json(self.path, {'seconds': self.seconds, 'limit_seconds': self.limit})

    def check(self):
        self.save()
        if self.seconds >= self.limit:
            raise BudgetExceeded('Active compute budget reached; outputs preserved. Not SUCCESS.')


def rows_by_id(path, key='id'):
    rows = read_jsonl(Path(path))
    result = {}
    for row in rows:
        if row[key] in result:
            raise ValueError(f'Duplicate durable key {row[key]} in {path}')
        result[row[key]] = row
    return result


def append_unique(path, row, known, key='id'):
    if row[key] in known:
        if known[row[key]] != row:
            raise ValueError('Conflicting record for durable key')
        return
    append_jsonl(Path(path), row)
    known[row[key]] = row


def atomic_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    with path.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(path.with_suffix('.tmp'), path)
