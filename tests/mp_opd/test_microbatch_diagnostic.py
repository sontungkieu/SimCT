"""Execute the launcher guards without importing GPU/model dependencies."""
import ast
from pathlib import Path
import pytest

SOURCE = Path(__file__).parents[2]/'experiments/runai/run_single_gpu.py'


def validate(micro, meta, limit):
    tree = ast.parse(SOURCE.read_text())
    guards = [node for node in tree.body if isinstance(node, ast.If)
              and any(isinstance(n, ast.Constant) and isinstance(n.value, str)
                      and n.value.startswith(('Student microbatch must', 'Meta microbatch must',
                                             'Microbatch overrides are'))
                      for n in ast.walk(node))]
    assert len(guards) == 3
    exec(compile(ast.Module(body=guards, type_ignores=[]), str(SOURCE), 'exec'),
         dict(opts={'micro_train_batch_size':micro,'mp_opd_meta_microbatch_size':meta},limit=limit))


@pytest.mark.parametrize('micro,meta,limit', [(4,4,312),(64,16,10),(8,8,30),(1,1,2)])
def test_valid_recipe(micro, meta, limit):
    validate(micro,meta,limit)


@pytest.mark.parametrize('micro,meta,limit', [(3,4,10),(4,3,10),(8,4,312),(4,8,0)])
def test_invalid_or_unqualified_production_override(micro, meta, limit):
    with pytest.raises(ValueError): validate(micro,meta,limit)
