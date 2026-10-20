"""Read the GRPO loss without loading a model or running a training job.

Use the SAME PyTorch 2.8.0 environment/pins as train_grpo.py. For an actual
training pipeline, use that TRL starter; this file only teaches loss arithmetic.
After installing PyTorch, `python tiny_grpo_loss.py` runs small CPU checks.

Two intentionally different modes:
* clip_epsilon=None: raw on-policy policy gradient, -log_prob * advantage,
  as in the simple nano-aha-moment implementation. This is not PPO clipping.
* clip_epsilon=0.2: PPO-style clipped ratios; saved old_logps are REQUIRED.

Every log-probability tensor is [B, T], for GENERATED tokens only. Entries
must already be aligned: the distribution at position t predicts token t+1.
Do that causal shift/gather when preparing inputs, not inside this loss.
response_mask is 1 for generated tokens to train on, 0 for padding/ignored
positions; include EOS if it is part of the sampled answer being optimized.

Rows must preserve group order: prompt 0 answer 0, prompt 0 answer 1, ...,
prompt 1 answer 0, ... . rewards is [n_prompts, G], so B = n_prompts * G.
Rewards, advantages, old log probabilities, and reference values are constants.

Reduction: sum over valid response TOKENS / total valid response tokens.
Longer answers receive more total weight. Original GRPO's mean of per-answer
token means differs; do not silently interchange the two normalizations.

Sources (the example below is independently written):
https://github.com/McGill-NLP/nano-aha-moment/blob/main/nano_r1_script.py
https://docs.pytorch.org/docs/2.8/generated/torch.std.html
https://huggingface.co/docs/trl/v0.23.1/en/grpo_trainer
"""

import math

import torch


def group_advantages(rewards: torch.Tensor, stabilizer: float = 1e-4) -> torch.Tensor:
    """Normalize each SAME-PROMPT group; return a detached [n_prompts, G].

    Population std matches NumPy's default in the referenced simple code.
    PyTorch defaults to sample std, so explicitly request unbiased=False.
    TRL 0.23.1 uses sample std for its group scaling; exact values differ.
    An all-equal group returns zeros. A stabilizer cannot create a signal.
    """
    if rewards.ndim != 2 or rewards.shape[0] < 1 or rewards.shape[1] < 2:
        raise ValueError("rewards must be [n_prompts, G] with G >= 2")
    if stabilizer <= 0:
        raise ValueError("stabilizer must be positive")
    fixed_rewards = rewards.detach().float()
    if not torch.isfinite(fixed_rewards).all():
        raise ValueError("rewards must be finite")
    mean = fixed_rewards.mean(dim=1, keepdim=True)
    std = fixed_rewards.std(dim=1, keepdim=True, unbiased=False)
    return ((fixed_rewards - mean) / (std + stabilizer)).detach()


def tiny_grpo_loss(
    new_logps: torch.Tensor,
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    old_logps: torch.Tensor | None = None,
    clip_epsilon: float | None = None,
    ref_logps: torch.Tensor | None = None,
    beta: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Return differentiable loss plus detached policy_loss and kl metrics.

    new_logps comes from a forward pass WITH gradients for sampled tokens.
    In clipped mode, old_logps is the frozen behavior-policy snapshot that
    generated this batch. Keep it unchanged for all updates on the batch.
    ref_logps is a separate reference anchor, not a replacement for old_logps.
    beta=0 omits the KL term. beta>0 requires fixed ref_logps.

    The optional KL expression is a sampled estimator, not the exact sum over
    the vocabulary: exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1.
    Its usual KL interpretation assumes samples from the current policy;
    finite samples/old-policy reuse and sampling transformations matter.
    """
    if new_logps.ndim != 2 or response_mask.shape != new_logps.shape:
        raise ValueError("new_logps and response_mask must have the same [B, T] shape")
    if beta < 0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and nonnegative")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("response_mask must contain only 0 and 1")
    valid = response_mask.to(device=new_logps.device, dtype=torch.bool)
    if not valid.any():
        raise ValueError("the batch needs at least one valid generated token")
    if not torch.isfinite(new_logps[valid]).all():
        raise ValueError("valid generated-token log probabilities must be finite")
    advantages = group_advantages(rewards).to(new_logps.device).reshape(-1, 1)
    if advantages.shape[0] != new_logps.shape[0]:
        raise ValueError("B must equal n_prompts * G, in matching group order")

    # Replace ignored positions before exp/multiplication, so padding such as
    # -inf cannot create NaN from 0 * inf. Compute sensitive arithmetic in fp32.
    current = new_logps.float().masked_fill(~valid, 0.0)

    def fixed_logps(values, name):
        if values is None or values.shape != new_logps.shape:
            raise ValueError(f"{name} must have the same [B, T] shape as new_logps")
        frozen = values.detach().to(device=new_logps.device, dtype=torch.float32)
        if not torch.isfinite(frozen[valid]).all():
            raise ValueError(f"valid {name} entries must be finite")
        return frozen.masked_fill(~valid, 0.0)

    if clip_epsilon is None:
        # For a fresh on-policy batch. Do not treat this as importance-corrected
        # optimization of arbitrarily stale answers over repeated updates.
        token_score = current * advantages
    else:
        if not math.isfinite(clip_epsilon) or not 0 < clip_epsilon < 1:
            raise ValueError("clip_epsilon must be between 0 and 1")
        old = fixed_logps(old_logps, "old_logps")
        ratio = (current - old).exp()
        clipped_ratio = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
        token_score = torch.minimum(ratio * advantages, clipped_ratio * advantages)

    policy_loss = -token_score[valid].mean()
    if beta:
        reference = fixed_logps(ref_logps, "ref_logps")
        log_ratio = reference - current
        token_kl = log_ratio.exp() - log_ratio - 1
        kl = token_kl[valid].mean()
    else:
        kl = current.new_zeros(())
    loss = policy_loss + beta * kl
    return {"loss": loss, "policy_loss": policy_loss.detach(), "kl": kl.detach()}


def self_test():
    """Small CPU-only mathematical checks; no model/tokenizer/network involved."""
    rewards = torch.tensor([[0.0, 1.0], [5.0, 5.0]], requires_grad=True)
    normalized = group_advantages(rewards)
    assert not normalized.requires_grad
    assert torch.allclose(normalized[0], torch.tensor([-1.0, 1.0]) / 1.0002)
    assert torch.equal(normalized[1], torch.zeros(2))

    # Raw policy gradient: positive advantage pushes log probability up,
    # negative advantage pushes it down under gradient descent. Padding has 0 grad.
    current = torch.tensor([[-1.0, -2.0], [-1.2, -0.9]], requires_grad=True)
    mask = torch.tensor([[1, 1], [1, 0]])
    result = tiny_grpo_loss(current, torch.tensor([[0.0, 1.0]]), mask)
    result["loss"].backward()
    assert current.grad[0, 0] > 0 and current.grad[1, 0] < 0
    assert current.grad[1, 1] == 0

    # With r=1.5, +A is past its upper incentive cap. -A is a harmful
    # probability increase, and its gradient must remain active.
    old = torch.full((2, 2), -2.0, requires_grad=True)
    current = (old.detach() + math.log(1.5)).requires_grad_()
    result = tiny_grpo_loss(current, torch.tensor([[0.0, 1.0]]), torch.ones(2, 2), old_logps=old, clip_epsilon=0.2)
    result["loss"].backward()
    assert current.grad[0, 0] > 0 and torch.equal(current.grad[1], torch.zeros(2))
    assert old.grad is None

    # Matching reference gives zero KL; all-equal rewards give no policy signal.
    current = torch.full((2, 2), -2.0, requires_grad=True)
    result = tiny_grpo_loss(current, torch.ones(1, 2), torch.ones(2, 2), ref_logps=current.detach(), beta=0.1)
    assert result["kl"].item() == 0 and result["loss"].item() == 0
    try:
        tiny_grpo_loss(current, torch.ones(1, 2), torch.zeros(2, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("an empty response mask should be rejected")
    print("PASS: group normalization, detachment, masks, gradient signs, clipping, and zero KL")


if __name__ == "__main__":
    self_test()
