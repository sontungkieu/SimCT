import random
import pytest
from kdflow.meta_data import MetaSampler, prompt_id


def test_disjoint_replay_and_shared_schedule_without_global_rng():
    prompts=[f'question {i}' for i in range(100)]
    rows=[dict(prompt=p,reference='reference '+p) for p in prompts]
    a=MetaSampler(rows,prompts,42,16)
    before=random.getstate()
    selected=a.batch(4,prompts[:64])
    assert len(selected)==16 and len({r['id'] for r in selected})==16
    assert not {r['id'] for r in selected} & {prompt_id(p) for p in prompts[:64]}
    assert MetaSampler(rows,prompts,42,16).batch(4,prompts[:64])==selected
    assert a.batch(8,prompts[:64])!=selected
    assert before==random.getstate()


def test_reference_membership_and_conflict():
    with pytest.raises(ValueError,match='outside'):
        MetaSampler([dict(prompt='test',reference='answer')],['train'],42,1)
    with pytest.raises(ValueError,match='Conflicting'):
        MetaSampler([dict(prompt='train',reference='a'),dict(prompt=' TRAIN ',reference='b')],['train'],42,1)
