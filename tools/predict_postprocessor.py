"""Replay a verified ensemble-postprocessor release from its frozen saved state."""
import argparse
import json
import subprocess
from pathlib import Path
from postprocessor_replay import replay_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, help='Default: artifacts/replays/<release>.csv')
    args = parser.parse_args()
    root = args.root.resolve()
    from ml_workflow.contracts import identifier
    try:
        identifier(args.release)
        output = args.output or root / 'artifacts/replays' / (args.release + '.csv')
        print(json.dumps(replay_release(root, output, args.release), indent=2))
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
