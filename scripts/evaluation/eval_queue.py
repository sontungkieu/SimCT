#!/usr/bin/env python3
"""Two independent GPU workers; tier barriers; bounded resumable internal eval."""
import argparse
import concurrent.futures as cf
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import threading
import urllib.request
import contract_eval as E
import queue_data as D
from context_check import CONTEXT_LENGTH


class Deadline(Exception): pass


def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_bytes(E.encoded(value)); os.replace(tmp,path)


@contextlib.contextmanager
def locked(path,blocking=True):
    with path.open("a+") as f:
        try: fcntl.flock(f,fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield None; return
        try: yield f
        finally: fcntl.flock(f,fcntl.LOCK_UN)


def journal(path):
    rows={}
    if not path.exists(): return rows
    end=0
    with path.open("rb") as f:
        while line:=f.readline():
            try: row=json.loads(line)
            except ValueError:
                if f.read(1): raise ValueError("corrupt interior journal line")
                if line.endswith(b"\n"): raise ValueError("corrupt complete journal line")
                # Preserve only an interrupted final write, never silently lose a full row.
                tail=path.with_name(path.name+".interrupted-tail")
                if tail.exists(): raise ValueError("inspect previous interrupted tail first")
                tail.write_bytes(line)
                with path.open("r+b") as repair: repair.truncate(end)
                break
            if row["id"] in rows: raise ValueError("duplicate journal ID")
            rows[row["id"]]=row; end=f.tell()
    if rows and path.stat().st_size:
        with path.open("rb") as f:
            f.seek(-1,2); missing_newline=f.read(1)!=b"\n"
        if missing_newline:
            with path.open("ab") as f: f.write(b"\n"); f.flush(); os.fsync(f.fileno())
    return rows


def append(path,row):
    with path.open("ab") as f:
        f.write(E.encoded(row)+b"\n"); f.flush(); os.fsync(f.fileno())


def stop_group(process):
    # Only the process group created by this queue; never pkill/ray stop.
    try: os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=3)
    except subprocess.TimeoutExpired: pass
    try: os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError: pass
    try: process.wait(timeout=3)
    except subprocess.TimeoutExpired: pass


def score_job(python,job,deadline):
    remaining=deadline-time.time()
    if remaining<=0: raise Deadline()
    timeout=min(120.,remaining)
    env={"PATH":"/usr/bin:/bin","OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1",
         "PYTHONDONTWRITEBYTECODE":"1","PYTHONUNBUFFERED":"1","LANG":"C.UTF-8"}
    with tempfile.TemporaryDirectory(prefix="simct-score-") as work, tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        env["HOME"]=work; env["TMPDIR"]=work
        p=subprocess.Popen([python,str(E.HERE/"internal_worker.py")],stdin=subprocess.PIPE,
                           stdout=out,stderr=err,cwd=work,env=env,start_new_session=True)
        try:
            try: p.communicate(E.encoded(job),timeout=timeout)
            except subprocess.TimeoutExpired:
                if timeout<120: raise Deadline()
                return {"passed":False,"timeout":True}
            out.seek(0); raw=out.read(1024**2)
            if p.returncode:
                raise RuntimeError("scorer runtime error: "+raw.decode(errors="replace")[:1000])
            result=json.loads(raw)
            if type(result.get("passed")) is not bool: raise ValueError("invalid scorer result")
            return result
        finally: stop_group(p)


def preflight(python):
    cases=[("gsm8k",{"gold":"#### 2"},"#### 2","#### 3"),
           ("math500",{"gold":"2"},r"\boxed{2}",r"\boxed{3}"),
           ("mbpp",{"tests":["assert add(2,3)==5"],"setup":""},"def add(a,b): return a+b","def add(a,b): return a-b"),
           ("live-code-bench-v6",{"tests":[{"input":"2 3\n","output":"5\n"}],"fn_name":None},"a,b=map(int,input().split());print(a+b)","print(0)"),
           ("live-code-bench-v6",{"tests":[{"input":"2\n3","output":"5"}],"fn_name":"add"},"class Solution:\n def add(self,a,b): return a+b","class Solution:\n def add(self,a,b): return a-b")]
    evidence=[]
    for benchmark,item,good,bad in cases:
        for correct,text in ((True,good),(False,bad)):
            result=score_job(python,{"profile":D.PROFILE,"benchmark":benchmark,"item":item,"text":text},time.time()+150)
            if result["passed"]!=correct or result.get("timeout"): raise RuntimeError("scorer qualification failed")
            evidence.append({"benchmark":benchmark,"functional":bool(item.get("fn_name")),"correct":correct,**result})
    return evidence


def state_path(root): return root/"state.json"


def read_state(root,plan_hash,plan):
    path=state_path(root)
    if path.exists():
        state=E.read_json(path)
        if state["plan_sha256"]!=plan_hash: raise ValueError("queue plan changed")
        return state
    start=time.time()
    state={"plan_sha256":plan_hash,"started":start,"deadline":start+plan["hours"]*3600,
           "admit_until":start+plan["admit_hours"]*3600,"jobs":{},"durations":[]}
    atomic_json(path,state);return state


def next_jobs(plan,state,now):
    pending=[j for j in plan["jobs"] if state["jobs"].get(j["id"],{}).get("status")!="completed"]
    if not pending or now>=state["admit_until"]: return []
    if any(x.get("status")=="failed" for x in state["jobs"].values()):
        raise RuntimeError("queue has a failed job; inspect error and use retry after fixing it")
    tier=min(j["tier"] for j in pending)
    estimate=max(state["durations"],default=0.)
    if now+estimate>state["deadline"]: return []
    return [j for j in pending if j["tier"]==tier]


def cell_contract(plan_hash,job,data,benchmark,seed,server):
    return {"plan_sha256":plan_hash,"checkpoint_sha256":job["checkpoint"]["sha256"],
            "data_sha256":data[benchmark]["sha256"],"benchmark":benchmark,"seed":seed,
            "profile":D.PROFILE,"server":server}


def generate_one(base,item,benchmark,seed,deadline):
    if time.time()>=deadline: raise Deadline()
    payload=E.generation_payload("eval-gemma",item,benchmark,seed)
    request=urllib.request.Request(base+"/v1/chat/completions",data=E.encoded(payload),headers={"Content-Type":"application/json"})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request,timeout=min(600.,max(.1,deadline-time.time()))) as response:
        result=json.load(response)
    E.validate_response(result)
    return {"id":item["id"],"seed":seed,"request_sha256":E.digest(E.encoded(payload)),"response":result}


def run_cell(root,plan,plan_hash,job,benchmark,seed,base,server,args,deadline):
    cell=root/"cells"/job["id"]/benchmark/str(seed);cell.mkdir(parents=True,exist_ok=True)
    contract=cell_contract(plan_hash,job,plan["data"],benchmark,seed,server)
    manifest=cell/"contract.json"
    if manifest.exists():
        if E.read_json(manifest)!=contract: raise ValueError("cell resume contract mismatch")
    else: E.write_new(manifest,contract)
    dataset=E.read_json(plan["data"][benchmark]["path"])
    items={x["id"]:x for x in dataset["items"]}
    if (cell/"metrics.json").exists():
        complete=E.read_json(cell/"metrics.json")
        for name in ("responses","scores"):
            if E.file_hash(cell/(name+".jsonl"))!=complete[name+"_sha256"]:
                raise ValueError("completed journal changed; no repair permitted")
    responses=journal(cell/"responses.jsonl"); scores=journal(cell/"scores.jsonl")
    if not set(scores)<=set(responses)<=set(items): raise ValueError("orphan/unknown results")
    for key,row in responses.items():
        expected=E.generation_payload("eval-gemma",items[key],benchmark,seed)
        if row["seed"]!=seed or row["request_sha256"]!=E.digest(E.encoded(expected)): raise ValueError("response request mismatch")
        E.validate_response(row["response"])
    for key,row in scores.items():
        if row["response_sha256"]!=E.digest(E.encoded(responses[key])) or type(row.get("passed")) is not bool:
            raise ValueError("score does not match response")
    if (cell/"metrics.json").exists():
        m=E.read_json(cell/"metrics.json")
        if len(scores)!=len(items) or m["responses_sha256"]!=E.file_hash(cell/"responses.jsonl") or m["scores_sha256"]!=E.file_hash(cell/"scores.jsonl"):
            raise ValueError("completed cell integrity mismatch")
        return m
    started=time.time()
    previous_seconds=E.read_json(cell/"timing.json").get("seconds",0.) if (cell/"timing.json").exists() else 0.
    pending=iter([x for x in items.values() if x["id"] not in scores])
    futures={}
    def process_item(item):
        # Data decoding remains in parent, one problem per scoring process.
        row=responses.get(item["id"])
        if row is None: row=generate_one(base,item,benchmark,seed,deadline)
        return row
    try:
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as generation, cf.ThreadPoolExecutor(max_workers=args.score_workers) as grading:
            exhausted=False
            while futures or not exhausted:
                if time.time()>=deadline: raise Deadline()
                # Bounded outstanding work: prevents retaining decoded LCB tests for all tasks.
                while not exhausted and len(futures)<args.concurrency+args.score_workers:
                    try: item=next(pending)
                    except StopIteration: exhausted=True;break
                    futures[generation.submit(process_item,item)]=("generation",item)
                if not futures: break
                done,_=cf.wait(futures,timeout=min(1.,max(.1,deadline-time.time())),return_when=cf.FIRST_COMPLETED)
                for future in done:
                    kind,item=futures.pop(future);value=future.result()
                    if kind=="generation":
                        if item["id"] not in responses:
                            append(cell/"responses.jsonl",value);responses[item["id"]]=value
                        score_input=D.score_item(benchmark,item)
                        payload={"profile":D.PROFILE,"benchmark":benchmark,"item":score_input,
                                 "text":value["response"]["choices"][0]["message"]["content"]}
                        futures[grading.submit(score_job,args.score_python,payload,deadline)]=("score",item)
                    else:
                        value.update(id=item["id"],response_sha256=E.digest(E.encoded(responses[item["id"]])))
                        append(cell/"scores.jsonl",value);scores[item["id"]]=value
                        if len(scores)%25==0: print(f"PROGRESS {job['id']} {benchmark} seed={seed} {len(scores)}/{len(items)}",flush=True)
    finally:
        atomic_json(cell/"timing.json",{"seconds":previous_seconds+time.time()-started})
    if len(scores)!=len(items): raise ValueError("incomplete cell")
    metrics={"status":"completed","contract":contract,"count":len(items),
             "score":sum(x["passed"] for x in scores.values())/len(items),
             "truncated_fraction":sum(x["response"]["choices"][0]["finish_reason"]=="length" for x in responses.values())/len(items),
             "seconds":previous_seconds+time.time()-started,
             "responses_sha256":E.file_hash(cell/"responses.jsonl"),"scores_sha256":E.file_hash(cell/"scores.jsonl")}
    E.write_new(cell/"metrics.json",metrics)
    print("CELL_COMPLETE",job["id"],benchmark,seed,metrics["score"],flush=True)
    return metrics


def run_checkpoint(root,plan,plan_hash,job,args,deadline):
    actual=E.checkpoint_identity(job["checkpoint"]["path"])
    if time.time()>=deadline: raise Deadline("wall budget during checkpoint verification")
    if actual!=job["checkpoint"]: raise ValueError("checkpoint changed after plan")
    port=31000+1000*args.gpu;base=f"http://127.0.0.1:{port}"
    import socket
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1",port))==0: raise RuntimeError("server port already occupied")
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
             HF_DATASETS_OFFLINE="1",OMP_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false",PYTHONUNBUFFERED="1",
             PYTHONPATH=str(E.HERE.parents[1]/"experiments/modal/vendor")+":"+str(E.HERE.parents[1]))
    for name in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy","WANDB_API_KEY","HF_TOKEN"):
        env.pop(name,None)
    env.pop("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN",None)
    subprocess.run(["bash",str(E.HERE.parents[1]/"experiments/runai/python-b200-host.sh"),
                    str(E.HERE/"context_check.py"),"--plan",str(args.plan.resolve()),
                    "--checkpoint",actual["path"]],env=dict(env,CUDA_VISIBLE_DEVICES=""),
                   check=True,timeout=max(.1,min(300.,deadline-time.time())))
    if time.time()>=deadline: raise Deadline("wall budget during context qualification")
    log=root/(job["id"]+f"-gpu{args.gpu}-server.log")
    command=["bash",str(E.HERE.parents[1]/"experiments/runai/python-b200-host.sh"),"-m","sglang.launch_server",
             "--model-path",actual["path"],"--served-model-name","eval-gemma","--host","127.0.0.1","--port",str(port),
             "--tp-size","1","--mem-fraction-static","0.8","--context-length",str(CONTEXT_LENGTH),
             "--attention-backend","triton","--disable-cuda-graph","--random-seed","42"]
    with log.open("ab") as output:
        process=subprocess.Popen(command,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        watchdog=threading.Timer(max(0.,deadline-time.time()),stop_group,args=(process,))
        watchdog.daemon=True; watchdog.start()
        try:
            startup=min(deadline,time.time()+900)
            while True:
                if time.time()>=deadline: raise Deadline("wall budget")
                if time.time()>=startup: raise RuntimeError("server readiness exceeded 900 seconds; see "+str(log))
                if process.poll() is not None: raise RuntimeError("SGLang exited; see "+str(log))
                try:
                    # Short health timeout; identity verification follows readiness.
                    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    with opener.open(base+"/health",timeout=2): pass
                    break
                except (OSError,ValueError): time.sleep(2)
            server=E.verify_server(base,actual["path"],"eval-gemma")
            for seed in plan["seeds"]:
                for benchmark in E.CAPS:
                    run_cell(root,plan,plan_hash,job,benchmark,seed,base,server,args,deadline)
            if E.verify_server(base,actual["path"],"eval-gemma")!=server: raise ValueError("server identity drift")
        finally:
            watchdog.cancel(); stop_group(process)


def worker(args):
    root=args.plan.resolve().parent;plan=E.read_json(args.plan);plan_hash=E.file_hash(args.plan)
    if plan["profile"]!=D.PROFILE or plan["source"]!=D.script_hashes(): raise ValueError("queue source/profile changed")
    for data in plan["data"].values():
        if E.file_hash(data["path"])!=data["sha256"]: raise ValueError("prepared data changed")
    qualification=preflight(args.score_python)
    with locked(root/"state.lock"):
        state=read_state(root,plan_hash,plan)
        if "qualification" in state and state["qualification"]!=qualification: raise ValueError("grader version drift")
        state["qualification"]=qualification;atomic_json(state_path(root),state)
    gpu_lock=Path(tempfile.gettempdir())/f"simct-eval-gpu{args.gpu}.lock"
    with locked(gpu_lock,blocking=False) as own:
        if own is None: raise RuntimeError("another eval worker owns this GPU")
        while True:
            with locked(root/"state.lock"):
                state=read_state(root,plan_hash,plan)
                candidates=next_jobs(plan,state,time.time())
                if not candidates:
                    print("QUEUE_STOP admission/deadline/complete",flush=True);return
            usage=subprocess.check_output(["nvidia-smi","-i",str(args.gpu),"--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip()
            if int(usage)>=1024:
                print("WAIT_GPU",args.gpu,usage,flush=True);time.sleep(15);continue
            claimed=False
            for job in candidates:
                with locked(root/(job["id"]+".lock"),blocking=False) as lock:
                    if lock is None: continue
                    with locked(root/"state.lock"):
                        state=read_state(root,plan_hash,plan)
                        if job["id"] not in {j["id"] for j in next_jobs(plan,state,time.time())}: continue
                        state["jobs"][job["id"]]={"status":"running","gpu":args.gpu,"started":time.time()}
                        atomic_json(state_path(root),state)
                    started=time.time();claimed=True
                    try:
                        run_checkpoint(root,plan,plan_hash,job,args,state["deadline"])
                        result={"status":"completed","seconds":time.time()-started}
                    except Deadline as exc:
                        result={"status":"partial","reason":str(exc) or "wall budget"}
                    except Exception as exc:
                        result=({"status":"partial","reason":"wall budget"} if time.time()>=state["deadline"] else
                                {"status":"failed","error":type(exc).__name__+": "+str(exc)})
                    with locked(root/"state.lock"):
                        state=read_state(root,plan_hash,plan);state["jobs"][job["id"]]=result
                        if result["status"]=="completed": state["durations"].append(result["seconds"])
                        atomic_json(state_path(root),state)
                    print("JOB",job["id"],json.dumps(result),flush=True)
                    if result["status"]=="failed": raise RuntimeError(result["error"])
                    if result["status"]!="completed": return
                    break
            if not claimed: time.sleep(5)


def summarize(args):
    root=args.plan.resolve().parent;plan=E.read_json(args.plan);ph=E.file_hash(args.plan)
    output={"profile":plan["profile"],"seeds":plan["seeds"],"checkpoints":{},"scope":plan["protocol"]["scope"]}
    for job in plan["jobs"]:
        benchmarks={}
        for benchmark in E.CAPS:
            paths=[root/"cells"/job["id"]/benchmark/str(seed)/"metrics.json" for seed in plan["seeds"]]
            if not all(p.exists() for p in paths): continue
            rows=[E.read_json(p) for p in paths]
            for seed,row,path in zip(plan["seeds"],rows,paths):
                c=row["contract"]
                if c["data_sha256"]!=plan["data"][benchmark]["sha256"] or c["benchmark"]!=benchmark or c["profile"]!=D.PROFILE or row["count"]!=plan["data"][benchmark]["count"]:
                    raise ValueError("summary data contract mismatch")
                for name in ("responses","scores"):
                    if E.file_hash(path.parent/(name+".jsonl"))!=row[name+"_sha256"]:
                        raise ValueError("summary journal integrity mismatch")
                if c["plan_sha256"]!=ph or c["seed"]!=seed or c["checkpoint_sha256"]!=job["checkpoint"]["sha256"] or row["status"]!="completed":
                    raise ValueError("summary cell contract mismatch")
            values=[x["score"] for x in rows]
            benchmarks[benchmark]={"mean":statistics.mean(values),"sample_std":statistics.stdev(values),"scores":values}
        result={"benchmarks":benchmarks,"complete":len(benchmarks)==4}
        if result["complete"]: result["average"]=statistics.mean(x["mean"] for x in benchmarks.values())
        output["checkpoints"][job["id"]]=result
    atomic_json(root/"summary-3seeds.json",output);print(json.dumps(output,indent=2))


def recover_startup(args):
    """Fork only a zero-result failed-startup queue; retain its original clock."""
    old=args.from_plan.resolve();oldroot=old.parent
    with locked(oldroot/"state.lock"), contextlib.ExitStack() as locks:
        plan=E.read_json(old);state=E.read_json(oldroot/"state.json")
        if state["plan_sha256"]!=E.file_hash(old) or plan["profile"]!=D.PROFILE:
            raise ValueError("old plan identity mismatch")
        if state["durations"] or not state["jobs"] or any(x["status"]!="failed" for x in state["jobs"].values()):
            raise ValueError("recovery requires exclusively failed startup jobs")
        if list(oldroot.glob("cells/**/*.json*")):
            raise ValueError("results exist; use an audited migration instead")
        for job in plan["jobs"]:
            if locks.enter_context(locked(oldroot/(job["id"]+".lock"),blocking=False)) is None:
                raise ValueError("old worker still owns a checkpoint")
        for data in plan["data"].values():
            if E.file_hash(data["path"])!=data["sha256"]: raise ValueError("old data changed")
        root=args.out.resolve();root.mkdir(parents=True,exist_ok=False)
        plan["source"]=D.script_hashes();plan["protocol"]["context_length"]=CONTEXT_LENGTH
        plan["recovery"]={"from_plan":str(old),"sha256":E.file_hash(old),"reason":"Gemma2 native context startup fix; original clock retained"}
        E.write_new(root/"plan.json",plan)
        state.update(plan_sha256=E.file_hash(root/"plan.json"),jobs={},durations=[])
        E.write_new(root/"state.json",state)
        print("RECOVERED_PLAN="+str(root/"plan.json"))
        print("REMAINING_HOURS="+str(max(0.,state["deadline"]-time.time())/3600))
        print("Original deadline retained; no GPU started.")


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="cmd",required=True)
    q=sub.add_parser("prepare")
    q.add_argument("--out",type=Path,required=True)
    q.add_argument("--shared",type=Path,default=Path("/workspace/storage-shared/nlp/tungks"))
    q.add_argument("--company-data",type=Path,default=Path("/workspace/storage-shared/nlp/hadv17/lm-eval/benchmarks"))
    q.add_argument("--proxy",default="http://10.30.154.118:80")
    q.add_argument("--hours",type=float,default=20.)
    q.set_defaults(func=D.prepare)
    q=sub.add_parser("preflight");q.add_argument("--score-python",default="/usr/bin/python3.12");q.set_defaults(func=lambda a:print(json.dumps(preflight(a.score_python),indent=2)))
    q=sub.add_parser("worker")
    q.add_argument("--plan",type=Path,required=True);q.add_argument("--gpu",type=int,choices=range(8),required=True)
    q.add_argument("--score-python",default="/usr/bin/python3.12")
    q.add_argument("--concurrency",type=int,default=8);q.add_argument("--score-workers",type=int,default=2)
    q.add_argument("--internal-code-execution",action="store_true",required=True,
                   help="Explicit company-internal profile; resource limits are not an OS sandbox")
    q.set_defaults(func=worker)
    q=sub.add_parser("summarize");q.add_argument("--plan",type=Path,required=True);q.set_defaults(func=summarize)
    q=sub.add_parser("recover-startup");q.add_argument("--from-plan",type=Path,required=True);q.add_argument("--out",type=Path,required=True);q.set_defaults(func=recover_startup)
    q=sub.add_parser("retry");q.add_argument("--plan",type=Path,required=True)
    def retry(a):
        root=a.plan.resolve().parent
        with locked(root/"state.lock"):
            s=E.read_json(state_path(root))
            for key,value in list(s["jobs"].items()):
                if value["status"]=="failed": del s["jobs"][key]
            atomic_json(state_path(root),s)
    q.set_defaults(func=retry)
    a=p.parse_args()
    if a.cmd=="prepare" and not 1<a.hours<=20: p.error("budget must be >1 and <=20 hours")
    if a.cmd=="worker" and not (1<=a.concurrency<=32 and 1<=a.score_workers<=8): p.error("invalid concurrency")
    a.func(a)

if __name__=="__main__": main()
