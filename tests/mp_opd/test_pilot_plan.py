import hashlib,json,time
from pathlib import Path
from experiments.runai.learned_partition_pilot import prepare

def test_pilot_gpu1_only_dependencies_and_original_clock(tmp_path):
    prior=tmp_path/'prior';prior.mkdir();original=tmp_path/'original';original.mkdir()
    data=prior/'data.json';data.write_text('{"groups": []}')
    started=time.time()-100
    argv=['bash','old-host','old-oracle','run','--data',str(data),'--output','old-out','--weighting-steps','1']
    (prior/'plan.json').write_text(json.dumps(dict(hours=20,jobs=[dict(id='probe-1',argv=argv,gpu=0)])))
    (prior/'campaign-state.json').write_text(json.dumps(dict(started=started,jobs={'probe-1':{'status':'completed'}})))
    (prior/'preflight.json').write_text(json.dumps(dict(data_sha256=hashlib.sha256(data.read_bytes()).hexdigest())))
    (original/'config.json').write_text(json.dumps(dict(student='s',teacher='t',dataset='d')))
    out=tmp_path/'out';prepare(prior,original,out)
    plan=json.loads((out/'plan.json').read_text())
    assert [j['gpu'] for j in plan['jobs']]==[1,None,1,1]
    assert plan['jobs'][2]['require_success']==['qualify']
    assert plan['jobs'][3]['require_success']==['soft-canary']
    assert json.loads((out/'campaign-state.json').read_text())['started']==started
    assert '--learn-partition' in plan['jobs'][0]['argv']
    assert plan['jobs'][3]['argv'][-2:]==['soft','50']
