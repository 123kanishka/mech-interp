#!/usr/bin/env python3
"""Copy one safety run and its logs from Vast, then verify completed artifacts.

Does not delete local or remote files, stop the GPU, or copy credentials.
Safe to rerun while collection is in progress; complete verification requires
the remote experiment's SUCCESS marker and final checksum manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--identity', required=True, type=Path)
    parser.add_argument('--known-hosts', required=True, type=Path)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9.-]+', args.host) or not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_name):
        parser.error('Invalid host or run name')
    if not 0 < args.port < 65536:
        parser.error('Invalid SSH port')
    destination = ROOT / 'results/safety_runs' / args.run_name
    destination.mkdir(parents=True, exist_ok=True)
    ssh = shlex.join(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-o', 'StrictHostKeyChecking=yes', '-o', f'UserKnownHostsFile={args.known_hosts}',
        '-i', str(args.identity), '-p', str(args.port)])
    source = f'root@{args.host}:/workspace/mech-interp/results/'
    subprocess.run(['rsync', '-az', '--partial', '--safe-links', '-e', ssh,
        source + 'safety_runs/' + args.run_name + '/', str(destination) + '/'], check=True)
    logs = ROOT / 'results/logs' / args.run_name
    logs.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-az', '--partial', '--safe-links', '-e', ssh,
        source + 'logs/', str(logs) + '/'], check=True)
    if not (destination / 'SUCCESS').exists():
        print(f'Partial backup saved (experiment not complete): {destination}')
        return 0
    checksums = json.loads((destination / 'checksums.json').read_text())
    for name, expected in checksums.items():
        path = (destination / name).resolve()
        if destination.resolve() not in path.parents:
            raise ValueError('Unsafe checksum path')
        checksum = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                checksum.update(chunk)
        if checksum.hexdigest() != expected:
            raise ValueError(f'Backup checksum mismatch: {name}')
    print(f'Complete backup verified: {len(checksums)} artifact checksums match: {destination}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
