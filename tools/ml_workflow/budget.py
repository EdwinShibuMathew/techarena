"""Exclusive phase ownership, persistent compute accounting and hard child timeout."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from consolidate import write_json
from .storage import read
from .contracts import require


class Budget:
    def __init__(self,path,owner,limit,total,carry=0.,carry_identity=None):
        self.path,self.owner,self.requested,self.total=Path(path),owner,limit,total
        self.carry,self.carry_identity=carry,carry_identity

    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.lock=self.path.with_suffix(".lock").open("a")
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.state=read(self.path) if self.path.exists() else {"used_seconds":self.carry,"carry_identity":self.carry_identity,"total_seconds":self.total,"attempts":[]}
            require(self.state["total_seconds"]==self.total and self.state["carry_identity"]==self.carry_identity,"Budget lineage changed")
            require(not any(a["owner"]==self.owner for a in self.state["attempts"]),"Phase already attempted; retries prohibited")
            self.limit=min(self.requested,self.total-self.state["used_seconds"])
            require(self.limit>0,"Cumulative compute exhausted")
            self.start=time.monotonic()
            # Reserve before launching: a killed controller cannot reset compute.
            self.state["used_seconds"]+=self.limit
            self.attempt={"owner":self.owner,"status":"running","reserved_seconds":self.limit,"controller_pid":os.getpid()}
            self.state["attempts"].append(self.attempt);write_json(self.path,self.state)
            return self
        except BaseException:
            self.lock.close();raise

    def run(self,command,cwd):
        child=subprocess.Popen(command,cwd=cwd,start_new_session=True)
        try:
            return child.wait(timeout=max(.001,self.limit-(time.monotonic()-self.start)))
        except BaseException:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait(timeout=10)
            raise
        finally:
            # A completed/failed worker must not leave fitting subprocesses behind.
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass

    def __exit__(self,kind,error,tb):
        elapsed=min(self.limit,time.monotonic()-self.start)
        self.state["used_seconds"]-=self.limit-elapsed
        self.attempt.update(status="complete" if error is None else "failed",seconds=elapsed,error=None if error is None else str(error))
        try:write_json(self.path,self.state)
        finally:self.lock.close()
        return False
