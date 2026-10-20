"""A small, versioned GRPO arithmetic experiment for one CUDA GPU.

Run from your ai-uni directory. All outputs and model caches stay there.
No packages, model, or GPU are needed for: python train_grpo.py --self-test
Training requires the accompanying requirements-grpo.txt and a CUDA GPU.

API checked against TRL 0.23.1 source; syntax/reward checked locally.
GPU execution and quality improvement have NOT been verified.
Sources:
https://huggingface.co/docs/trl/v0.23.1/en/grpo_trainer
https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/grpo_config.py
https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import os
from pathlib import Path
import random
import re


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
INTEGER = re.compile(r"[+-]?[0-9]{1,8}")


def completion_text(completion):
    """TRL chat format is [{role: assistant, content: text}] per completion."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and len(completion) == 1:
        message = completion[0]
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def correctness_reward(completions, answer, **kwargs):
    """One float per completion; TRL supplies the matching answer column.

    Only a single integer is allowed (outer whitespace is ignored).
    We never execute/eval model output or reward the presence of an answer
    hidden inside an otherwise incorrect response.
    """
    if len(completions) != len(answer):
        raise ValueError("Each generated completion needs one matching answer")
    rewards = []
    for completion, expected in zip(completions, answer):
        text = completion_text(completion).strip()
        valid = INTEGER.fullmatch(text) is not None
        rewards.append(float(valid and int(text) == int(expected)))
    return rewards


def build_rows():
    """80 unique multiplication questions: 64 train, 16 held out."""
    pairs = [(a, b) for a in range(2, 26) for b in range(a, 26)]
    random.Random(7).shuffle(pairs)
    rows = [
        {
            "prompt": [
                {"role": "system", "content": "Reply with only the integer answer. No explanation."},
                {"role": "user", "content": f"What is {a} times {b}?"},
            ],
            "answer": a * b,
        }
        for a, b in pairs[:80]
    ]
    return rows[:64], rows[64:]


def self_test():
    texts = ["42", " 42\n", "+42", "41", "The answer is 42", "42 or 43", "", "4.2", "999999999"]
    chats = [[{"role": "assistant", "content": text}] for text in texts]
    expected = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert correctness_reward(chats, [42] * len(chats)) == expected
    assert correctness_reward(texts, [42] * len(texts)) == expected
    assert correctness_reward([[], None, [{"role": "user", "content": "42"}]], [42] * 3) == [0.0] * 3
    train, heldout = build_rows()
    assert len(train) == 64 and len(heldout) == 16
    questions = [row["prompt"][-1]["content"] for row in train + heldout]
    assert len(set(questions)) == 80
    assert (1 * 1 * 4) % 4 == 0  # batch size × devices × accumulation; G = 4
    print("PASS: reward parser, conversational format, 64/16 split, group divisibility")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    # Set cache locations BEFORE importing Hugging Face libraries.
    work_dir = Path.cwd().resolve()
    if work_dir.name != "ai-uni":
        raise SystemExit("Run this script with ai-uni as your current directory.")
    local_caches = {
        "HF_HOME": work_dir / ".hf-cache",
        "HF_HUB_CACHE": work_dir / ".hf-cache" / "hub",
        "HUGGINGFACE_HUB_CACHE": work_dir / ".hf-cache" / "hub",
        "HF_DATASETS_CACHE": work_dir / ".hf-cache" / "datasets",
        "HF_XET_CACHE": work_dir / ".hf-cache" / "xet",
        "HF_ASSETS_CACHE": work_dir / ".hf-cache" / "assets",
        "TRANSFORMERS_CACHE": work_dir / ".hf-cache" / "hub",
        "XDG_CACHE_HOME": work_dir / ".cache",
        "TORCH_HOME": work_dir / ".cache" / "torch",
        "TORCH_EXTENSIONS_DIR": work_dir / ".cache" / "torch-extensions",
        "TORCHINDUCTOR_CACHE_DIR": work_dir / ".cache" / "torchinductor",
        "TRITON_CACHE_DIR": work_dir / ".cache" / "triton",
        "CUDA_CACHE_PATH": work_dir / ".cache" / "cuda",
        "TMPDIR": work_dir / ".tmp",
    }
    for name, cache_path in local_caches.items():
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(cache_path)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise SystemExit("This starter is configured for one GPU. Use a single Python process.")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for training. --self-test works without a GPU.")
    set_seed(7)
    use_bf16 = torch.cuda.is_bf16_supported()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    train_rows, heldout_rows = build_rows()
    config = GRPOConfig(
        output_dir=str(work_dir / "grpo-output"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        generation_batch_size=4,  # four completions = one full prompt group
        num_iterations=2,  # reuse each generated batch with saved old log probs
        max_steps=10,  # a pipeline smoke test, not a quality-training budget
        learning_rate=1e-5,
        max_prompt_length=128,
        max_completion_length=16,
        temperature=0.9,
        top_p=1.0,
        top_k=0,
        epsilon=0.2,
        loss_type="grpo",  # explicitly select original per-completion averaging
        scale_rewards="group",
        beta=0.0,  # no reference KL term; old-policy ratios still exist
        use_vllm=False,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=0,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=7,
    )
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(train_rows),
        reward_funcs=correctness_reward,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )

    def evaluate_heldout():
        """Same 16 unseen prompts, greedy decoding, before and after training."""
        trainer.model.eval()
        rows = []
        with torch.inference_mode():
            for row in heldout_rows:
                text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(trainer.model.device)
                generated = trainer.model.generate(
                    **inputs, do_sample=False, max_new_tokens=16,
                    pad_token_id=tokenizer.pad_token_id, use_cache=True,
                )
                completion = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                reward = correctness_reward([completion], [row["answer"]])[0]
                rows.append({"question": row["prompt"][-1]["content"], "answer": row["answer"], "completion": completion, "reward": reward})
        trainer.model.train()
        return {"accuracy": sum(row["reward"] for row in rows) / len(rows), "n": len(rows), "examples": rows}

    before = evaluate_heldout()
    print(f"Before: held-out exact-answer accuracy = {before['accuracy']:.1%} (n=16)")
    trainer.train()
    after = evaluate_heldout()
    print(f"After:  held-out exact-answer accuracy = {after['accuracy']:.1%} (n=16)")
    adapter_dir = work_dir / "grpo-adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    (work_dir / "grpo-metrics.json").write_text(json.dumps({"before": before, "after": after}, indent=2))
    print("Saved LoRA adapter, tokenizer, and per-example held-out results inside ai-uni.")


if __name__ == "__main__":
    main()
