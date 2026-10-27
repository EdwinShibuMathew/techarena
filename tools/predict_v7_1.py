"""Replay only a gate-approved V7.1 release from its frozen saved calibrator."""
import argparse
import json
from pathlib import Path
from ml_workflow import postprocessing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--token',help=argparse.SUPPRESS)
    args=parser.parse_args();root=args.root.resolve()
    output=args.output or root/'artifacts/submissions/v7_1.csv'
    try:
        if args.worker:raise SystemExit(postprocessing.replay_worker(root,output,args.token))
        print(json.dumps(postprocessing.replay_release(root,output),indent=2))
    except (ValueError,OSError) as error:parser.exit(1,str(error)+'\n')


if __name__=='__main__':main()
