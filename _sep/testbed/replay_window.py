#!/usr/bin/env python3
"""Replay fused signal window for determinism checks."""
import argparse
import json
import hashlib


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay fused signal window")
    parser.add_argument("file", help="Path to fused signal JSON file")
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Direction: {data.get('direction')}")
    print(f"Confidence: {data.get('confidence')}")
    print(f"Input hash: {data.get('input_hash')}")
    print(f"Config version: {data.get('config_version')}")

    # Produce deterministic digest for replay validation
    replay_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
    print(f"Replay digest: {hashlib.sha256(replay_bytes).hexdigest()}")


if __name__ == "__main__":
    main()
