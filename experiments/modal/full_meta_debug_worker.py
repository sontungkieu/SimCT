"""Tiny GPU diagnostic of the production full-meta function under FSDP2."""
import json
import os
import importlib.util
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
spec = importlib.util.spec_from_file_location('full_meta', '/opt/overlay/kdflow/algorithms/_mp_opd_full_meta.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
full_meta_step = module.full_meta_step


def run(sharded):
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 2, bias=False).cuda()
    if sharded:
        fully_shard(model)
    energy = torch.nn.Linear(4, 1, bias=False).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    eo = torch.optim.AdamW(energy.parameters(), lr=1e-3)
    x = torch.randn(8, 4, device='cuda')
    y = torch.randn(8, 2, device='cuda')
    def inner():
        return (energy(x).sigmoid() * (model(x)-y).square()).mean()
    def outer():
        return (model(x + .1) - y).square().mean()
    params = tuple(model.parameters())
    loss = inner()
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    report = {'sharded': sharded, 'unused': [g is None for g in grads]}
    try:
        report['metrics'] = full_meta_step(params, opt, energy, eo,
            [inner], [outer], max_norm=1.,
            refresh_parameters=model.reshard if sharded else lambda: None)
    except Exception as exc:
        report['error'] = type(exc).__name__ + ': ' + str(exc)
    opt.zero_grad(set_to_none=True)
    inner().backward()
    report['backward_has_gradient'] = [p.grad is not None for p in model.parameters()]
    print('META_DEBUG ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:29671', rank=0, world_size=1)
    torch.cuda.set_device(0)
    print('ENV ' + json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0)}), flush=True)
    try:
        run(False)
        run(True)
    finally:
        dist.destroy_process_group()
