"""Replay only a gate-approved V7.1 release from its frozen saved calibrator."""
import argparse
import json
import subprocess
from pathlib import Path
from postprocessor_replay import replay_release


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,help='Default: artifacts/replays/v7_1.csv')
    args=parser.parse_args();root=args.root.resolve()
    output=args.output or root/'artifacts/replays/v7_1.csv'
    try:
        print(json.dumps(replay_release(root,output),indent=2))
    except (ValueError,OSError,subprocess.TimeoutExpired) as error:parser.exit(1,str(error)+'\n')


if __name__=='__main__':main()
