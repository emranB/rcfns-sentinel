#!/usr/bin/env python3
'''Command-line entry point for RCFNS Sentinel.'''

import argparse
from pathlib import Path

from src.backend import main


# Read the optional configuration path.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RCFNS Sentinel")
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Path to the configuration file (default: config.json beside main.py).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args().config_path)
