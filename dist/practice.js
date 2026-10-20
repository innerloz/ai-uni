const PRACTICE_PARTS = [
  {
    label: 'Data & reward',
    title: 'You provide questions and a checker.',
    text: 'Each row has a conversational prompt and an answer column. The answer goes to the reward function, not into the model’s question. TRL generates four completions and repeats the matching answer for the checker. A response earns 1 only when its entire trimmed text is the correct integer; otherwise it earns 0.',
    code: `row = {
    "prompt": [{"role": "user", "content": "What is 7 times 8?"}],
    "answer": 56,
}

def correctness_reward(completions, answer, **kwargs):
    # Chat completion: [{"role": "assistant", "content": "56"}]
    # Return one float per completion; see the download for robust parsing.
    ...`,
    note: '“The answer is 56” gets 0 because this task explicitly requires only an integer. That is part of the task, not a general definition of answer quality. The script never evaluates model-generated Python.',
  },
  {
    label: 'Trainer config',
    title: 'Make the group fit the generation batch.',
    text: 'For this single-GPU run, the effective batch contains 1 × 4 = 4 completions: one question with four candidate answers. Two iterations reuse that generated batch with fixed old log probabilities and advantages. After those updates, the trainer generates fresh answers.',
    code: `config = GRPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    generation_batch_size=4,
    num_generations=4,
    num_iterations=2,
    loss_type="grpo", scale_rewards="group",
    beta=0.0, max_steps=10,
)
trainer = GRPOTrainer(..., args=config,
    reward_funcs=correctness_reward, peft_config=lora)
trainer.train()`,
        note: 'This version requires generation_batch_size to be divisible by num_generations. Here 4 ÷ 4 = 1 complete group. Gradient accumulation splits training into small microbatches; it does not guarantee that generation uses equally little memory. The browser lab uses population standard deviation + 1e−8 for teaching; TRL 0.23.1 uses sample standard deviation (G − 1 divisor) + 1e−4, so exact advantage magnitudes differ.',
  },
  {
    label: 'Check results',
    title: 'Measure answers, not just the loss.',
    text: 'The starter creates 64 training questions and 16 separate held-out questions. It checks the same held-out prompts with greedy decoding before and after training, then saves every prediction. Ten optimizer steps check that the pipeline runs; they do not establish a useful improvement.',
    code: `before = evaluate_heldout()   # exact-answer accuracy on 16 questions
trainer.train()               # 10 optimizer steps, fresh generations
after = evaluate_heldout()    # same questions and decoding settings

trainer.save_model("grpo-adapter")
# grpo-metrics.json stores before/after predictions and scores.
# The saved LoRA adapter still needs its original base model.`,
    note: 'Inspect failed and successful outputs. Training reward can rise while unseen accuracy stays flat. Sixteen questions are a smoke test, not a reliable benchmark; expand the held-out set and use several seeds before drawing conclusions.',
  },
];

const PRACTICE_DIAGNOSTICS = {
  equal: {
    title: 'Check whether every candidate gets the same reward.',
    body: 'Look at reward_std and frac_reward_zero_std, then inspect actual completions. If a group is all 0s or all 1s, its advantages are all zero: there is no relative policy-gradient signal from that group. With beta = 0, there is no reference-KL gradient either. All correct? Increase task difficulty. All incorrect? Start with easier questions, check the parser and format instructions, and ensure sampling produces useful variation. Adding a tiny denominator prevents division by zero; it cannot create signal.',
  },
  memory: {
    title: 'Count generated tokens as well as trainable weights.',
    body: 'This starter uses LoRA, checkpointing, short prompts, and 16-token completions. Generation still keeps the base model and token caches in memory, and a group contains four completions. Start by reducing completion length if your task permits it. If reducing the group to 2, keep the generation batch divisible by 2. More gradient accumulation alone does not shrink the configured generation batch. No critic saves one learning component; it does not remove actor, generation, or optional reference-model costs.',
  },
  reward: {
    title: 'Check the reward against the outcome you actually want.',
    body: 'Compare held-out exact-answer accuracy with training reward and open grpo-metrics.json. Test the checker with wrong answers, multiple numbers, malformed text, and empty output. A formatting bonus or reward for longer answers can be exploited without solving the task. This starter rewards only a correct integer, but that narrow checker says nothing about reasoning quality or other tasks.',
  },
  api: {
    title: 'Use the versioned environment and check full groups.',
    body: 'Use the downloaded core pins in a fresh Python 3.11 environment. This example intentionally targets TRL 0.23.1 rather than the latest release. For one GPU, generation_batch_size = 4 and num_generations = 4 are valid. If changing device count or generation settings, re-check both global batch and group divisibility in that version’s GRPOConfig. Do not mix this older starter with an unrestricted future Transformers release.',
  },
};

const practiceEscape = value => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');

export function initPractice() {
  const mount = document.querySelector('#practice-mount');
  if (!mount || mount.dataset.practiceInitialized) return;
  mount.dataset.practiceInitialized = 'true';
  mount.innerHTML = `
    <div class="practice-shell">
      <div class="practice-start">
        <div><p class="practice-kicker">A SMALL EXPERIMENT YOU CAN RUN</p><h3>Fine-tune a compact model on checked arithmetic.</h3><p>Use <a href="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct" target="_blank" rel="noopener noreferrer">Qwen2.5-0.5B-Instruct</a>, four answers per question, a Python correctness checker, and a small LoRA adapter. You supply the task and the reward; TRL handles sampling, group advantages, token ratios, and optimization.</p><p class="practice-validation"><strong>Validation:</strong> Python syntax and the reward self-test were checked. APIs and batch rules were checked against tagged source. Dependencies have not been installed here, and no GPU training or quality improvement has been verified.</p></div>
        <aside class="practice-downloads"><span class="practice-kicker">DOWNLOAD THE STARTER</span><a class="practice-download" href="examples/train_grpo.py" download>train_grpo.py <span>Python script ↓</span></a><a class="practice-download" href="examples/requirements-grpo.txt" download>requirements-grpo.txt <span>Version pins ↓</span></a><p>Deliberately pinned tutorial:<br><strong>TRL 0.23.1 · Transformers 4.56.2</strong><br>PyTorch 2.8.0 · PEFT 0.17.1<br>Accelerate 1.10.1 · Datasets 4.1.1</p><p>Linux · Python 3.11 · one CUDA GPU<br>LoRA adapter saved with its tokenizer.</p></aside>
      </div>

      <div class="practice-walkthrough">
        <div class="practice-tabs" role="tablist" aria-label="GRPO implementation walkthrough">${PRACTICE_PARTS.map((part, index) => `<button type="button" role="tab" id="practice-tab-${index}" aria-controls="practice-panel-${index}" aria-selected="${index === 0}" tabindex="${index === 0 ? '0' : '-1'}"><span>0${index + 1}</span>${part.label}</button>`).join('')}</div>
        ${PRACTICE_PARTS.map((part, index) => `<div class="practice-part" role="tabpanel" id="practice-panel-${index}" aria-labelledby="practice-tab-${index}" ${index === 0 ? '' : 'hidden'}><div><h3>${part.title}</h3><p>${part.text}</p><p class="practice-detail">${part.note}</p></div><pre><code>${practiceEscape(part.code)}</code></pre></div>`).join('')}
      </div>

      <details class="practice-details"><summary>Set up and run on a CUDA machine</summary><div class="practice-details-inner"><p>Save both downloads into an <strong>ai-uni</strong> directory on your GPU machine. Replace the path below with that directory. The environment, caches, temporary files, adapter, and results stay inside it. These commands install dependencies and download the base model when you choose to run them.</p><pre><code>cd /path/to/innersight/ai-uni
mkdir -p .tmp .cache
export TMPDIR="$PWD/.tmp"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export XDG_CACHE_HOME="$PWD/.cache"
export HF_HOME="$PWD/.hf-cache"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Example wheel for a driver compatible with CUDA 12.8:
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-grpo.txt
python train_grpo.py --self-test
CUDA_VISIBLE_DEVICES=0 python train_grpo.py</code></pre><p>The self-test needs only standard Python, so you can run it before installing ML packages. Training requires an available CUDA GPU and a driver compatible with the installed PyTorch wheel. Use the matching PyTorch 2.8.0 CUDA command from the <a href="https://pytorch.org/get-started/previous-versions/#v280" target="_blank" rel="noopener noreferrer">official installation matrix</a> if your setup differs. Memory and runtime depend on your GPU; this page does not claim a tested VRAM minimum.</p><p>Outputs: <code>grpo-adapter/</code>, <code>grpo-output/</code>, and <code>grpo-metrics.json</code>. The adapter is a small set of changes and needs the original Qwen base model when loaded later.</p></div></details>

      <details class="practice-details"><summary>The training loop, without library machinery</summary><div class="practice-details-inner"><pre><code>repeat:
    sample questions
    freeze the actor that generates this batch as “old”
    generate G answers per question; save old token log probabilities
    score each answer with the checker
    A = (reward - group_mean) / (group_std + small_number)
    for a few update passes on this fixed batch:
        ratio = exp(new_token_log_prob - old_token_log_prob)
        score = minimum(ratio * A, clipped_ratio * A)
        minimize(-average(score) + beta * reference_KL)
    generate a fresh batch with the updated actor</code></pre><p>The reward and advantage belong to a whole answer; this simple outcome-reward version applies that advantage to its generated tokens. The old actor is the snapshot that made the batch. A reference policy is a separate anchor used for the optional KL penalty. <strong>beta = 0</strong> in this starter removes that reference penalty, while keeping old-policy ratios for repeated updates.</p><p><code>loss_type="grpo"</code> explicitly selects per-answer token averaging to connect with the original method. <code>scale_rewards="group"</code> selects group standard deviation. These choices can introduce length or difficulty biases; newer variants use different normalizations, so this is an educational configuration rather than a universal recommendation. See the <a href="https://huggingface.co/docs/trl/v0.23.1/en/grpo_trainer#loss-types" target="_blank" rel="noopener noreferrer">versioned loss discussion</a>.</p></div></details>

      <div class="practice-diagnostics"><div><p class="practice-kicker">WHEN THE RUN DOES NOT TEACH YOU WHAT YOU EXPECT</p><h3>Debug the signal first.</h3><label for="practice-symptom">Choose what you observe</label><select id="practice-symptom"><option value="equal">Rewards are identical / learning is flat</option><option value="memory">CUDA runs out of memory</option><option value="reward">Training reward rises, unseen accuracy does not</option><option value="api">Configuration or version error</option></select></div><div class="practice-diagnosis" aria-live="polite"><h4 id="practice-diagnosis-title"></h4><p id="practice-diagnosis-body"></p></div></div>
      <p class="practice-sources">Checked against <a href="https://huggingface.co/docs/trl/v0.23.1/en/grpo_trainer" target="_blank" rel="noopener noreferrer">TRL 0.23.1 documentation</a>, <a href="https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/grpo_config.py" target="_blank" rel="noopener noreferrer">tagged configuration code</a>, <a href="https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/grpo_trainer.py" target="_blank" rel="noopener noreferrer">trainer implementation</a>, and the <a href="https://pypi.org/project/trl/0.23.1/" target="_blank" rel="noopener noreferrer">released package</a>. For newer APIs, consult the <a href="https://huggingface.co/docs/trl/en/grpo_trainer" target="_blank" rel="noopener noreferrer">current GRPO documentation</a>.</p>
    </div>`;

  const tabs = [...mount.querySelectorAll('[role="tab"]')];
  const activate = index => tabs.forEach((tab, tabIndex) => {
    const selected = index === tabIndex;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
    mount.querySelector(`#practice-panel-${tabIndex}`).hidden = !selected;
  });
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => activate(index));
    tab.addEventListener('keydown', event => {
      const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : null;
      if (next !== null) { event.preventDefault(); activate(next); tabs[next].focus(); }
    });
  });
  const select = mount.querySelector('#practice-symptom');
  const diagnose = () => {
    const item = PRACTICE_DIAGNOSTICS[select.value];
    mount.querySelector('#practice-diagnosis-title').textContent = item.title;
    mount.querySelector('#practice-diagnosis-body').textContent = item.body;
  };
  select.addEventListener('change', diagnose);
  diagnose();
}
