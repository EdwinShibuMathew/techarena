"""Run one hash-bound experiment stage; no silent resume or gate bypass."""
import argparse
import json
from pathlib import Path
import sys

from ml_workflow import engine


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","develop","screen","stress","finalize","verify"])
    parser.add_argument("--root",type=Path,default=engine.ROOT)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--spec",type=Path)
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args();root=args.root.resolve()
    try:
        if args.action=="prepare":
            if args.worker or args.spec is None:parser.error("prepare requires --spec")
            result=engine.prepare(root,args.spec,args.run_id)
        elif args.action=="verify":
            result=engine.verify(root,args.run_id)
        elif args.worker:
            raise SystemExit(engine.worker(root,engine.locate(root,args.run_id),args.action))
        else:
            raise SystemExit(engine.run(root,args.run_id,args.action))
        print(json.dumps(result,indent=2,sort_keys=True))
    except (ValueError,OSError,TimeoutError) as error:
        parser.exit(1,str(error)+"\n")


if __name__=="__main__":main()
