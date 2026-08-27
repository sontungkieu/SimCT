from __future__ import annotations

import inspect

from vdt_tunix.trainer import PaperSimpleOPDTrainer


def test_simple_opd_update_skips_full_vocabulary_logits():
    source = inspect.getsource(PaperSimpleOPDTrainer)

    assert "forward_hidden_fn" in source
    assert "paper_simple_opd_aligned_batch_loss_from_hidden_projection" in source
    assert "full_logits" not in source
