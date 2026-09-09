"""Offline queue regression tests; synthetic code only, no GPU/network."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/evaluation"))
import eval_queue as Q
import queue_data as D
E=Q.E

class QueueTests(unittest.TestCase):
    def test_trajectory(self):
        rows=[x for tier in D.tiers() for x in tier]
        self.assertEqual(len(rows),17)
        self.assertEqual(len(set(rows)),17)
        self.assertEqual([x for x in rows if x[0]=="sft"],[("sft",0)])
        for mode in ("atomic","fixed"):
            self.assertEqual(sorted(x[1] for x in rows if x[0]==mode),[40,80,120,156,200,240,280,312])
        self.assertEqual(D.SEEDS,(42,43,44))

    def test_journal_repairs_only_interrupted_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"rows"
            p.write_bytes(b'{"id":"a"}')
            self.assertEqual(set(Q.journal(p)),{"a"})
            Q.append(p,{"id":"b"})
            self.assertEqual(set(Q.journal(p)),{"a","b"})
            with p.open("ab") as f: f.write(b'{"id":')
            self.assertEqual(len(Q.journal(p)),2)
            self.assertTrue(Path(str(p)+".interrupted-tail").exists())
            p.write_bytes(b'broken\n')
            with self.assertRaises(ValueError): Q.journal(p)

    def test_tier_deadline_failure_and_resume_clock(self):
        plan={"jobs":[{"id":"a","tier":0},{"id":"b","tier":1}],"hours":20,"admit_hours":18}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            state=Q.read_state(root,"hash",plan)
            self.assertEqual(Q.read_state(root,"hash",plan)["deadline"],state["deadline"])
            now=state["started"]
            self.assertEqual([x["id"] for x in Q.next_jobs(plan,state,now)],["a"])
            state["jobs"]["a"]={"status":"completed"}
            self.assertEqual([x["id"] for x in Q.next_jobs(plan,state,now)],["b"])
            self.assertEqual(Q.next_jobs(plan,state,state["admit_until"]),[])
            state["jobs"]["b"]={"status":"failed"}
            with self.assertRaises(RuntimeError): Q.next_jobs(plan,state,now)

    def test_internal_qualification(self):
        self.assertEqual(len(Q.preflight(sys.executable)),10)
        with self.assertRaises(Q.Deadline): Q.score_job(sys.executable,{},time.time()-1)
        with self.assertRaises(RuntimeError):
            Q.score_job(sys.executable,{"profile":"author-code"},time.time()+10)

    def test_functional_prompt_and_source_row_hash(self):
        prompt=D.prompt("live-code-bench-v6",{"fn_name":"add","prompt":"sum","starter_code":"class Solution:"})
        self.assertIn("Solution",prompt[0]["content"])
        self.assertIn("add",prompt[0]["content"])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"data";p.write_bytes(b'{}')
            with self.assertRaises(ValueError):
                D.score_item("live-code-bench-v6",{"raw":{"path":str(p),"offset":0,"length":2,"sha256":"wrong"}})

    def test_cell_resume_no_repeated_generation_or_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"data.json"
            items=[{"id":str(i),"messages":[{"role":"user","content":"sum"}],"gold":"#### 2"} for i in range(3)]
            E.write_new(data,{"items":items})
            plan={"data":{"gsm8k":{"path":str(data),"sha256":E.file_hash(data),"count":3}}}
            job={"id":"atomic-40","checkpoint":{"sha256":"cp"}}
            args=argparse.Namespace(concurrency=1,score_workers=1,score_python=sys.executable)
            def generate(base,item,benchmark,seed,deadline):
                return {"id":item["id"],"seed":seed,"request_sha256":E.digest(E.encoded(E.generation_payload("eval-gemma",item,benchmark,seed))),
                        "response":{"choices":[{"finish_reason":"stop","message":{"content":"#### 2"}}]}}
            cell=root/"cells/atomic-40/gsm8k/42";cell.mkdir(parents=True)
            E.write_new(cell/"contract.json",Q.cell_contract("plan",job,plan["data"],"gsm8k",42,{}))
            first=generate("",items[0],"gsm8k",42,0);Q.append(cell/"responses.jsonl",first)
            with patch.object(Q,"generate_one",side_effect=generate) as g:
                m=Q.run_cell(root,plan,"plan",job,"gsm8k",42,"",{},args,time.time()+30)
                self.assertEqual(g.call_count,2);self.assertEqual(m["score"],1.)
            with patch.object(Q,"generate_one",side_effect=AssertionError("repeat")), patch.object(Q,"score_job",side_effect=AssertionError("repeat")):
                Q.run_cell(root,plan,"plan",job,"gsm8k",42,"",{},args,time.time()+30)
            with (cell/"scores.jsonl").open("ab") as f: f.write(b" ")
            with self.assertRaises(ValueError):
                Q.run_cell(root,plan,"plan",job,"gsm8k",42,"",{},args,time.time()+30)

if __name__=="__main__": unittest.main()
