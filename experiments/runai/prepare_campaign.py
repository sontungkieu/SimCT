#!/usr/bin/env python3
"""Freeze a 20-hour campaign. Preparation does not start a GPU."""
import argparse,json,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main(a):
    w=a.out.resolve();w.mkdir(parents=True,exist_ok=False)
    base=Path('/workspace/storage-shared/nlp/tungks')
    cfg=dict(dataset=str(base/'SimCT/data/qwen-author/data/prompts.parquet'),student=str(base/'SimCT/runs/qwen-gemma-sft-paper-20260908-045828/checkpoint'),teacher='/workspace/storage-shared/models/Qwen2.5-7B-Instruct',reference_input=str(a.references or base/'SimCT/data/qwen-author/data/selected.parquet'),reference_key=a.reference_key,exclude=a.exclude,eval_template=str(a.eval_template.resolve()),scope='external guide; provenance and near-duplicate audit still required' if a.references else 'SEEN-SFT mechanics only; not an unseen generalization gate')
    for key in ('dataset','student','teacher','reference_input','eval_template'):
        if not Path(cfg[key]).exists():raise ValueError('missing '+key+': '+cfg[key])
    (w/'config.json').write_text(json.dumps(cfg,indent=2))
    jobs=[]
    def add(key,action,hours,gpu=None,after=(),name='',lr='.1',self_locks=False):
        jobs.append(dict(id=key,gpu=gpu,after=list(after),budget_seconds=int(hours*3600),self_locks=self_locks,argv=['/usr/bin/python3.12',str(ROOT/'experiments/runai/campaign_job.py'),action,'--work',str(w),'--name',name,'--gpu',str(gpu or 0),'--lr',lr]))
    add('prepare-guide','prepare-probe',.25)
    # One lane confirms fixed; the other probes before confirming atomic.
    add('canary-fixed','canary',.5,0,name='fixed')
    add('fixed43','train',6.5,0,('canary-fixed',),name='fixed')
    add('canary-atomic','canary',.5,1,name='atomic')
    for i,lr in enumerate(('.01','.1','1.0')):add('probe-'+str(i),'probe',.75,1,('prepare-guide','canary-atomic'),name='probe-'+str(i),lr=lr)
    add('probe-decision','decision',.05,after=('probe-0','probe-1','probe-2'));jobs[-1]['after_terminal']=True
    add('adaptive-probe','adaptive',1.5,1,('probe-decision','canary-atomic'))
    # Baseline lane remains useful even if a probe fails. No automatic retry of partial training.
    add('atomic43','train',6.5,1,('adaptive-probe',),name='atomic');jobs[-1]['after_terminal']=True;jobs[-1]['require_success']=['canary-atomic']
    for mode,gpu in (('fixed',0),('atomic',1)):
        add('plan-'+mode,'eval-plan',.25,after=(mode+'43',),name=mode)
        add('gen-'+mode,'generate',2,gpu,('plan-'+mode,),name=mode,self_locks=True)
        add('score-'+mode,'score',4,after=('plan-'+mode,),name=mode)
    plan=dict(config_sha256=hashlib.sha256((w/'config.json').read_bytes()).hexdigest(),schema='campaign-v1',hours=20,source=str(ROOT),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),jobs=jobs,data_contract=cfg)
    (w/'plan.json').write_text(json.dumps(plan,indent=2));print('PLAN='+str(w/'plan.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--eval-template',type=Path,required=True);p.add_argument('--references',type=Path);p.add_argument('--reference-key',default='label');p.add_argument('--exclude',action='append',default=[]);main(p.parse_args())
