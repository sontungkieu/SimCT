"""Bounded H2 stages on the existing SimCT runtime and model cache."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/opt/h2')
CONFIG_PATH = ROOT / 'experiments/diagnostics/h2_pilot_config.json'
CONFIG = json.loads(CONFIG_PATH.read_text()) if modal.is_local() else {}
RUN = os.environ.get('H2_RUN_ID', CONFIG.get('run_id'))
IMAGE = CONFIG.get('runtime_image', os.environ.get('H2_RUNTIME_IMAGE'))
app = modal.App(RUN)
assets = modal.Volume.from_name('sb-opd-assets-v1')
results = modal.Volume.from_name('sb-opd-results-v1')
volumes = {'/assets': assets, '/results': results}
image = modal.Image.from_registry(IMAGE).entrypoint([]).env({'H2_RUN_ID': RUN, 'H2_RUNTIME_IMAGE': IMAGE})
SOURCES = {
    'experiments/diagnostics/h2_pilot.py': 'h2_pilot.py',
    'experiments/diagnostics/h2_measurements.py': 'h2_measurements.py',
    'experiments/diagnostics/h2_e3.py': 'h2_e3.py',
    'experiments/diagnostics/h2_report.py': 'h2_report.py',
    'experiments/diagnostics/h2_sitecustomize.py': 'sitecustomize.py',
    'experiments/diagnostics/h2_pilot_config.json': 'h2_pilot_config.json',
    'experiments/modal/mp_opd_parity_worker.py': 'mp_opd_parity_worker.py',
    'experiments/mp_opd/hf_frozen_sampler.py': 'hf_frozen_sampler.py',
    'scripts/sft/answer_verifier.py': 'answer_verifier.py',
}
if modal.is_local():
    for local, remote in SOURCES.items():
        image = image.add_local_file(str(ROOT / local), '/opt/h2/' + remote)
PY = '/opt/venvs/simct-b200/bin/python'


def execute(stage, arm, seconds, attempt):
    assets.reload(); results.reload()
    root = Path('/results') / RUN
    root.mkdir(exist_ok=True)
    label = stage + '-' + arm + '-a' + str(attempt)
    log = root / (label + '.log')
    if log.exists():
        raise RuntimeError('Attempt exists; inspect evidence, then use a new run ID for repairs')
    env = dict(os.environ, HF_HOME='/assets/hf-cache', PYTHONUNBUFFERED='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', PYTHONPATH='/opt/h2')
    libs = ['/usr/local/cuda/lib64', '/usr/local/nvidia/lib64']
    libs += [str(p) for p in Path('/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia').glob('*/lib')]
    env['LD_LIBRARY_PATH'] = ':'.join(libs)
    if stage != 'prepare':
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        env['H2_CANONICAL_GEMMA_SOFTCAP'] = '1'
    else:
        env.pop('HF_HUB_OFFLINE', None); env.pop('TRANSFORMERS_OFFLINE', None)
    env['H2_STAGE_ATTEMPT'] = str(attempt)
    start = time.monotonic(); position = 0; process = None; error = None
    try:
        with log.open('x') as output:
            process = subprocess.Popen([PY, '-u', '/opt/h2/h2_pilot.py', stage, arm],
                env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                if time.monotonic() - start > seconds:
                    raise TimeoutError('H2 stage runtime cap reached')
                with log.open() as reader:
                    reader.seek(position); chunk = reader.read(); position = reader.tell()
                if chunk: print(chunk, flush=True)
                assets.commit(); results.commit(); time.sleep(4)
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    finally:
        if process and process.poll() is None:
            # Only the process group created by this stage is controlled.
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        receipt = {'run_id': RUN, 'stage': stage, 'arm': arm,
            'exit_code': process.returncode if process else None, 'error': error,
            'elapsed_seconds': time.monotonic() - start, 'runtime_image': IMAGE,
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in Path('/opt/h2').glob('*.py')},
            'config_sha256': hashlib.sha256(Path('/opt/h2/h2_pilot_config.json').read_bytes()).hexdigest(),
            'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()}
        (root / (label + '-receipt.json')).write_text(json.dumps(receipt, indent=2))
        assets.commit(); results.commit()
    with log.open() as reader:
        reader.seek(position); print(reader.read(), flush=True)
    print('H2_RECEIPT_JSON=' + json.dumps(receipt), flush=True)
    if receipt['exit_code'] != 0 or receipt['error']:
        raise RuntimeError('H2 stage failed; persisted log/receipt are authoritative')
    return receipt


@app.function(image=image, cpu=2, memory=8192, timeout=1860, retries=0,
              max_containers=1, scaledown_window=2, volumes=volumes,
              secrets=[modal.Secret.from_name('huggingface-secret')])
def prepare(arm, attempt): return execute('prepare', arm, 1800, attempt)


@app.function(image=image, gpu='L4', cpu=4, memory=24576, timeout=1260,
              retries=0, max_containers=1, scaledown_window=2, volumes=volumes)
def l4(stage, arm, attempt): return execute(stage, arm, 1200, attempt)


@app.function(image=image, gpu='L40S', cpu=4, memory=32768, timeout=1860,
              retries=0, max_containers=1, scaledown_window=2, volumes=volumes)
def l40s(stage, arm, attempt): return execute(stage, arm, 1800, attempt)


@app.function(image=image, cpu=2, memory=16384, timeout=1260, retries=0,
              max_containers=1, scaledown_window=2, volumes=volumes)
def analysis(stage, arm, attempt): return execute(stage, arm, 1200, attempt)


@app.local_entrypoint()
def main(stage: str = 'prepare', arm: str = 'S', gpu: str = 'L4', attempt: int = 1):
    if stage not in {'prepare', 'parity', 'rollout', 'score', 'events', 'quality', 'report',
                     'e3-candidates', 'e3-merge', 'e3-score'}:
        raise ValueError('Unknown H2 stage')
    if arm not in {'S', 'G9', 'Q', 'L', 'ALL'}: raise ValueError(arm)
    if gpu not in {'L4', 'L40S'}: raise ValueError(gpu)
    result = prepare.remote(arm, attempt) if stage == 'prepare' else analysis.remote(stage,arm,attempt) if stage in {'e3-merge','report'} else (
        l4.remote(stage, arm, attempt) if gpu == 'L4' else l40s.remote(stage, arm, attempt))
    print('H2_LOCAL_RECEIPT_JSON=' + json.dumps(result))
