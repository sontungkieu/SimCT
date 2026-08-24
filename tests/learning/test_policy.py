from __future__ import annotations

import unittest

from vdt_span.policy import adaptive_max_span_width
from vdt_span.types import PolicyContext, SpanPolicyConfig, TrainingState


class AdaptiveSpanPolicyTests(unittest.TestCase):
    def test_exact_auditable_policy_fixture(self) -> None:
        width = adaptive_max_span_width(
            PolicyContext(
                boundary_confidence=0.8,
                teacher_disagreement=0.25,
                context_risk=0.5,
            ),
            TrainingState(step=50, warmup_steps=100),
            SpanPolicyConfig(min_width=1, max_width=9),
        )
        self.assertEqual(width, 2)

    def test_training_progress_can_expand_span_budget(self) -> None:
        context = PolicyContext(0.9, 0.0, 0.0)
        config = SpanPolicyConfig(1, 9)
        early = adaptive_max_span_width(context, TrainingState(0, 100), config)
        late = adaptive_max_span_width(context, TrainingState(100, 100), config)
        self.assertEqual(early, 1)
        self.assertEqual(late, 8)
        self.assertGreater(late, early)

    def test_context_risk_and_disagreement_shrink_budget(self) -> None:
        training = TrainingState(100, 100)
        config = SpanPolicyConfig(1, 9)
        safe = adaptive_max_span_width(PolicyContext(1.0, 0.0, 0.0), training, config)
        risky = adaptive_max_span_width(PolicyContext(1.0, 0.5, 0.5), training, config)
        self.assertEqual(safe, 9)
        self.assertEqual(risky, 3)
        self.assertGreater(safe, risky)

    def test_invalid_state_or_context_is_rejected(self) -> None:
        invalid_calls = (
            (PolicyContext(1.1, 0.0, 0.0), TrainingState(1, 10), SpanPolicyConfig()),
            (PolicyContext(1.0, 0.0, 0.0), TrainingState(-1, 10), SpanPolicyConfig()),
            (PolicyContext(1.0, 0.0, 0.0), TrainingState(1, 0), SpanPolicyConfig()),
            (PolicyContext(1.0, 0.0, 0.0), TrainingState(1, 10), SpanPolicyConfig(4, 2)),
        )
        for context, training, config in invalid_calls:
            with self.subTest(context=context, training=training, config=config):
                with self.assertRaises(ValueError):
                    adaptive_max_span_width(context, training, config)


if __name__ == "__main__":
    unittest.main()
