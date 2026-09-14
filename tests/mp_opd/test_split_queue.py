import importlib.util
import json
from pathlib import Path
import sys
import pytest

path=Path(__file__).parents[2]/'experiments/runai/queue_split_alternating.py'
spec=importlib.util.spec_from_file_location('split_queue',path)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


@pytest.mark.parametrize('host,gpus',[(q.OWNER,['GPU-owner']),(q.EXTRA,q.EXTRA_GPUS)])
def test_plan_manager_contract_and_dependencies(tmp_path,host,gpus):
    jobs=q.specs(tmp_path,host,gpus)
    assert jobs==q.specs(tmp_path,host,gpus)
    seen=set()
    for j in jobs:
        assert set(j['dependencies'])<=seen
        seen.add(j['id'])
    train=[j for j in jobs if j['argv'][4]=='train']
    qualify=next(j for j in jobs if j['argv'][4]=='qualify')
    assert len(train)==3
    assert train[0]['dependencies']==[qualify['id']]
    if host==q.OWNER:
        assert train[1]['dependencies']==[train[0]['id']]
        assert train[2]['dependencies']==[train[1]['id']]
        assert all(j['gpus']==gpus for j in train)
    else:
        assert all(j['dependencies']==[qualify['id']] for j in train)
        assert [j['gpus'] for j in train]==[[g] for g in gpus[:3]]
        assert len({j['env']['KDFLOW_ROLLOUT_PORT_BASE'] for j in train})==3
        gen=[j for j in jobs if j['argv'][4]=='generate']
        assert len(gen)==5
        assert [j['gpus'] for j in gen[-2:]]==[[g] for g in gpus[3:]]
    manager=Path('/mnt/d/dev/codex/job-manager')
    if manager.exists():
        sys.path.insert(0,str(manager))
        from job_manager.store import initialize,connect,submit,rows
        state=tmp_path/'state';initialize(state,dict(gpus=gpus,cpu_slots=16,deadline=None))
        db=connect(state)
        try:
            submit(db,jobs)
            assert len(rows(db))==len(jobs)
        finally:db.close()


def test_export_not_visible_before_commit_marker_and_detects_change(tmp_path):
    from kdflow.export_ready import marker,publish,validate
    folder=tmp_path/'step40';folder.mkdir();(folder/'config.json').write_text('{}')
    (folder/'model.safetensors').write_bytes(b'model')
    assert not marker(tmp_path,40).exists()
    publish(tmp_path,40)
    assert validate(tmp_path,40)==folder
    (folder/'model.safetensors').write_bytes(b'truncated')
    with pytest.raises(ValueError,match='mismatch'):validate(tmp_path,40)


@pytest.mark.parametrize('fail',[False,True])
def test_worker_ignores_uncommitted_and_preserves_failure(tmp_path,monkeypatch,fail):
    from kdflow.export_ready import publish
    monkeypatch.setattr(q.F,'configurations',lambda:[{'id':'test'}])
    monkeypatch.setattr(q.F,'STEPS',(40,80))
    root=tmp_path/'train/test/checkpoint';(root/'step40').mkdir(parents=True)
    (root/'step40/config.json').write_text('{}');publish(root,40)
    (root/'step80').mkdir() # Producer is still writing. Never consume this.
    calls=[]
    def evaluate(case,run,step,phase):
        calls.append((run,step,phase))
        if fail:raise RuntimeError('retained failure')
    monkeypatch.setattr(q.F,'evaluate',evaluate)
    monkeypatch.setattr(q,'producers_done',lambda case:True)
    monkeypatch.setattr(q.time,'sleep',lambda seconds:None)
    for _ in range(2):
        if fail:
            with pytest.raises(RuntimeError,match='retained'):q.worker(tmp_path,'generate')
        else:q.worker(tmp_path,'generate')
    assert calls==[('test',40,'generate')]
    assert (tmp_path/'dispatch/test-step40'/('generate.error.json' if fail else 'generate.done.json')).exists()


def test_producers_require_both_host_status_and_all_trains_terminal(tmp_path):
    for host,gpus in [(q.OWNER,['GPU-owner']),(q.EXTRA,q.EXTRA_GPUS)]:
        jobs=q.specs(tmp_path,host,gpus);dest=tmp_path/'nodes'/host;dest.mkdir(parents=True)
        (dest/'receipt.json').write_text(json.dumps({'jobs':jobs}))
        (dest/'status.json').write_text(json.dumps({'jobs':{j['id']:'completed' for j in jobs}}))
        if host==q.OWNER:assert not q.producers_done(tmp_path)
    assert q.producers_done(tmp_path)
    state=json.loads((dest/'status.json').read_text())
    job=next(j for j in jobs if j['argv'][4]=='train');state['jobs'][job['id']]='running'
    (dest/'status.json').write_text(json.dumps(state))
    assert not q.producers_done(tmp_path)
