"""Synthetic CUDA preprocessing benchmark; not full-model throughput."""
import importlib.util
import json
import os
import statistics
import time
import torch

os.environ['KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT'] = '1'
spec = importlib.util.spec_from_file_location('references', '/opt/overlay/tests/mp_opd/test_vectorized_features.py')
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)


def timed(fn):
    for _ in range(2): fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = time.perf_counter(); fn(); torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000)
    return statistics.median(samples)


torch.manual_seed(42)
report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cases=[])
for n in (400, 1000):
    atoms, c = ref.fixture(n, 'cuda')
    for name, old, new in (
        ('features', lambda: ref.scalar_features(atoms, c), lambda: ref.atom_features(atoms, c)),
        ('spans', lambda: ref.scalar_spans(c.base_credit, c.weight, 2),
                  lambda: ref.span_tables(c.base_credit, c.weight, 2))):
        a, b = old(), new()
        aa, bb = (a,), (b,)
        if isinstance(a, tuple): aa, bb = a, b
        assert all(torch.equal(x, y) for x, y in zip(aa, bb)), name
        if name == 'features':
            energy = torch.nn.Linear(10, 2, device='cuda')
            ga = torch.autograd.grad(energy(a).square().sum(), tuple(energy.parameters()))
            gb = torch.autograd.grad(energy(b).square().sum(), tuple(energy.parameters()))
            assert all(torch.equal(x, y) for x, y in zip(ga, gb))
        baseline = timed(old); optimized = timed(new)
        report['cases'].append(dict(atoms=n, operation=name, baseline_ms=baseline,
            optimized_ms=optimized, speedup=baseline/optimized, bitwise_equal=True))
print(json.dumps(report, indent=2), flush=True)
