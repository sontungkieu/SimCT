"""Study-scoped Gemma softcap arithmetic; does not edit installed dependencies."""
import os

H2_PATCH_APPLIED = False
if os.environ.get('H2_CANONICAL_GEMMA_SOFTCAP') == '1':
    import torch
    from sglang.srt.layers import logits_processor
    original_softcap = logits_processor.fused_softcap

    def canonical_gemma_softcap(logits, final_logit_softcapping):
        if final_logit_softcapping != 30.0:
            return original_softcap(logits, final_logit_softcapping)
        # _compute_lm_head returns BF16; the original implementation upcasts
        # before fused_softcap. Recover those exact BF16 values, perform the
        # three canonical BF16 operations, then return FP32 for log_softmax.
        capped = logits.to(torch.bfloat16) / final_logit_softcapping
        capped = torch.tanh(capped)
        capped = capped * final_logit_softcapping
        logits.copy_(capped.float())
        return logits

    logits_processor.fused_softcap = canonical_gemma_softcap
    H2_PATCH_APPLIED = True
