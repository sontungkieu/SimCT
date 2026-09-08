"""CPU-only inspection/export of pinned B200 runtime."""
import modal,subprocess,json
IMAGE='docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f'
app=modal.App('simct-b200-portable-inspect')
image=modal.Image.from_registry(IMAGE).entrypoint([])
@app.function(image=image,cpu=1,memory=2048,timeout=180,retries=0)
def inspect():
    code="import sys,json;print(json.dumps(dict(executable=sys.executable,base_prefix=sys.base_prefix,prefix=sys.prefix)))"
    for cmd in [['/opt/venvs/simct-b200/bin/python','-c',code],['cat','/etc/os-release'],['readlink','-f','/opt/venvs/simct-b200/bin/python'],['du','-sh','/opt/venvs/simct-b200','/usr/local/cuda'],['ls','-ld','/usr/local/cuda'],['ldd','/opt/venvs/simct-b200/bin/python']]:
        r=subprocess.run(cmd,text=True,capture_output=True);print(cmd,r.stdout,r.stderr,flush=True)
@app.local_entrypoint()
def main():inspect.remote()
