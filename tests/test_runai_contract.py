import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def test_preflight_configuration_and_input_hashes(tmp_path):
    for name in ('student','teacher'):
        (tmp_path/name).mkdir();(tmp_path/name/'config.json').write_text('{}')
    data=tmp_path/'prompts.parquet'; data.write_bytes(b'fixture')
    env=dict(os.environ,MP_PREFLIGHT_ONLY='1',MP_STUDENT_PATH=str(tmp_path/'student'),
             MP_TEACHER_PATH=str(tmp_path/'teacher'),MP_DATASET_PATH=str(data),CUDA_VISIBLE_DEVICES='0',
             MP_SEED='17',MP_MAX_SPAN_LENGTH='4',MP_FIXED_SPAN_LENGTH='4',MP_ALGORITHM='mp_opd')
    result=subprocess.run([sys.executable,str(ROOT/'experiments/runai/run_single_gpu.py'),'fixed','50',str(tmp_path/'out')],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    config=json.loads((tmp_path/'out/launch-config.json').read_text())
    assert config['options']['seed']==17
    assert config['options']['micro_train_batch_size']==4
    assert config['options']['mp_opd_fixed_span_length']==4
    assert config['contract']['scheduler_horizon']==312
    assert config['contract']['execution_updates']==50
    assert len(config['dataset_sha256'])==64
    assert config['models_sha256']['student']['config.json']
    # Check manifest changes on data drift; no Ray or model execution.
    data.write_bytes(b'changed')
    result=subprocess.run([sys.executable,str(ROOT/'experiments/runai/run_single_gpu.py'),'random','5',str(tmp_path/'out2')],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    config2=json.loads((tmp_path/'out2/launch-config.json').read_text())
    assert config2['dataset_sha256']!=config['dataset_sha256']
    env['MP_ALGORITHM']='span_ctkd'
    result=subprocess.run([sys.executable,str(ROOT/'experiments/runai/run_single_gpu.py'),'atomic','5',str(tmp_path/'out3')],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    config3=json.loads((tmp_path/'out3/launch-config.json').read_text())
    assert config3['contract']['parity'] is None
