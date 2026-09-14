"""Streamed full-parameter one-step AdamW hypergradient.

The optimizer state is held constant (truncated one-step objective). A virtual
student is installed temporarily, always rolled back, and never checkpointed.
Only one rollout microbatch's higher-order graph is retained at a time.
"""
import torch
from kdflow.training_checkpoint import capture_rng, restore_rng


def adam_value(p, g, state, group):
    if group.get("amsgrad") or group.get("maximize"):
        raise ValueError("Full meta supports standard, non-AMSGrad AdamW only")
    beta1, beta2 = group["betas"]
    step = float(state.get("step", 0)) + 1
    m = state.get("exp_avg", torch.zeros_like(p)).detach()
    v = state.get("exp_avg_sq", torch.zeros_like(p)).detach()
    m1 = beta1*m + (1-beta1)*g
    v1 = beta2*v + (1-beta2)*g.square()
    denom = (v1 / (1-beta2**step)).sqrt() + group["eps"]
    return p.detach()*(1-group["lr"]*group["weight_decay"]) - group["lr"]*m1/(1-beta1**step)/denom


def clipped(grads, max_norm):
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach(), 2) for g in grads]), 2)
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite virtual gradient")
    scale = (max_norm/(norm+1e-6)).clamp(max=1) if max_norm > 0 else norm.new_tensor(1.)
    return [g*scale.to(g.dtype) for g in grads], norm, scale


def full_meta_step(parameters, optimizer, energy, energy_optimizer, inner_losses, meta_losses, *, max_norm, refresh_parameters=lambda:None):
    """Each callback returns its already normalized contribution to ONE batch.

    Inner callbacks keep credits/features detached, but expose energy marginals
    to autograd. Meta callbacks use independent reference data. No real student
    update occurs here: caller recomputes detached rates and uses its optimizer.
    """
    params = tuple(parameters)
    groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    if set(groups) != {id(p) for p in params}:
        raise ValueError("Virtual and real optimizer parameter sets differ")
    if any(p.grad is not None for p in params):
        raise ValueError("Meta update requires an optimizer boundary")
    phi = tuple(energy.parameters())
    initial_rng = capture_rng()
    g = [torch.zeros_like(p) for p in params]
    # Detached accumulation avoids retaining B=64 activation graphs.
    for loss_fn in inner_losses:
        loss = loss_fn()
        part = torch.autograd.grad(loss, params, allow_unused=True)
        if all(value is None for value in part):
            raise RuntimeError('Meta inner loss is disconnected from optimizer parameters; FSDP parameter views require an explicit autograd bridge')
        for target, value in zip(g, part):
            if value is not None: target.add_(value.detach())
    cg, norm, scale = clipped(g, max_norm)
    refresh_parameters()
    originals = [p.detach().cpu().clone() for p in params]
    outer = [torch.zeros_like(p) for p in params]
    meta_value = 0.
    try:
        with torch.no_grad():
            for p, grad in zip(params, cg):
                p.copy_(adam_value(p, grad, optimizer.state.get(p, {}), groups[id(p)]))
        for loss_fn in meta_losses:
            loss = loss_fn()
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite meta NLL")
            meta_value += float(loss.detach())
            part = torch.autograd.grad(loss, params, allow_unused=True)
            if all(value is None for value in part):
                raise RuntimeError('Meta outer loss is disconnected from optimizer parameters')
            for target, value in zip(outer, part):
                if value is not None: target.add_(value.detach())
    finally:
        refresh_parameters()
        with torch.no_grad():
            for p, old in zip(params, originals): p.copy_(old.to(p.device))
    del originals
    # VJP of the optimizer, holding pre-step moments fixed. Work parameter by
    # parameter so optimizer-expression graphs do not scale with the full model.
    u = []
    for p, grad, vector in zip(params, cg, outer):
        group=groups[id(p)]; state=optimizer.state.get(p,{})
        b1,b2=group['betas']; step=float(state.get('step',0))+1
        m1=b1*state.get('exp_avg',torch.zeros_like(p))+(1-b1)*grad
        v1=b2*state.get('exp_avg_sq',torch.zeros_like(p))+(1-b2)*grad.square()
        root=(v1/(1-b2**step)).sqrt(); denom=root+group['eps']
        # At v=g=0, the composite Adam map g/(|g|+eps) has derivative
        # 1/eps; naïve autograd through sqrt produces 0*inf -> NaN.
        root_safe=torch.where(root>0,root,torch.ones_like(root))
        correction=torch.where(root>0,m1*(1-b2)*grad/((1-b2**step)*root_safe*denom.square()),0.)
        tangent=(-group['lr']/(1-b1**step)*((1-b1)/denom-correction)*vector).detach()
        if not torch.isfinite(tangent).all():
            raise FloatingPointError("Nonfinite AdamW VJP; inspect zero-variance/numerical edge")
        u.append(tangent)
    del cg, outer
    # VJP through GLOBAL gradient clipping, including its cross-parameter term.
    if max_norm > 0 and float(scale) < 1:
        dot = sum((a*b).sum() for a,b in zip(u,g))
        correction = scale*dot/(norm*(norm+1e-6))
        u = [scale.to(a.dtype)*a - correction.to(b.dtype)*b for a,b in zip(u,g)]
    del g
    restore_rng(initial_rng)
    hyper = [torch.zeros_like(p) for p in phi]
    for loss_fn in inner_losses:
        loss = loss_fn()
        part = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        active = [(grad, vector) for grad, vector in zip(part,u) if grad is not None and grad.requires_grad]
        if active:
            hg = torch.autograd.grad(tuple(x for x,_ in active), phi,
                                     grad_outputs=tuple(v for _,v in active), allow_unused=True)
            for target, value in zip(hyper,hg):
                if value is not None: target.add_(value.detach())
    if not all(torch.isfinite(x).all() for x in hyper):
        raise FloatingPointError("Nonfinite full energy hypergradient")
    energy_optimizer.zero_grad(set_to_none=True)
    old_energy=[p.detach().clone() for p in phi]
    for p, value in zip(phi,hyper): p.grad = value
    energy_optimizer.step(); energy_optimizer.zero_grad(set_to_none=True)
    # Real student pass sees exactly the original inner-forward RNG sequence.
    restore_rng(initial_rng)
    return {"mp_opd_meta_nll": meta_value,
            "mp_opd_energy_parameter_delta": float(sum((p.detach()-old).square().sum() for p,old in zip(phi,old_energy)).sqrt()),
            "mp_opd_energy_gradient_norm": float(sum(x.square().sum() for x in hyper).sqrt()),
            "mp_opd_virtual_gradient_norm": float(norm)}
