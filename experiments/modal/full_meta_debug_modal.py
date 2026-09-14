"""Bounded B200 autograd diagnostic; no model downloads or company data."""
import json
import os
import subprocess
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[2]
IMAGE = 'docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f'
app = modal.App('simct-full-meta-autograd-debug')
image = (modal.Image.from_registry(IMAGE).entrypoint([])
    .add_local_dir(str(ROOT/'kdflow'), '/opt/overlay/kdflow', ignore=['**/__pycache__/**','**/*.pyc'])
    .add_local_file(str(ROOT/'experiments/modal/full_meta_debug_worker.py'), '/opt/overlay/probe.py'))

@app.function(image=image, gpu=os.environ.get('META_DEBUG_GPU', 'B200'), cpu=2, memory=8192, timeout=180, retries=0, max_containers=1)
def probe():
    import os
    env = dict(os.environ, PYTHONPATH='/opt/overlay', PYTHONUNBUFFERED='1')
    p = subprocess.run(['/opt/venvs/simct-b200/bin/python','/opt/overlay/probe.py'],
        env=env, capture_output=True, text=True, timeout=150)
    print(p.stdout, flush=True)
    print(p.stderr, flush=True)
    return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

@app.local_entrypoint()
def main(run_id: str):
    result = probe.remote()
    dest = ROOT/'remote_artifacts'/run_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'result.json').write_text(json.dumps(result, indent=2))
    if result['returncode']:
        raise RuntimeError('Diagnostic failed; inspect result.json')
