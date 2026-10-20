const NANO_RESPONSES = [
  { answer: '3', reward: 0, probabilities: [0.15, 0.6], reference: [0.2, 0.8] },
  { answer: '5', reward: 0, probabilities: [0.25, 0.7], reference: [0.3, 0.7] },
  { answer: '4', reward: 1, probabilities: [0.5, 0.8], reference: [0.25, 0.5] }
];
const nanoFormat = (value, digits = 4) => (Math.abs(value) < 1e-10 ? 0 : value).toFixed(digits);
const nanoSigned = value => `${value >= 0 ? '+' : '−'}${nanoFormat(Math.abs(value))}`;

export function nanoExampleStats(beta = 0.1) {
  const mean = 1 / 3;
  const std = Math.sqrt(2 / 9); // Population standard deviation of [0, 0, 1].
  const advantages = NANO_RESPONSES.map(response => (response.reward - mean) / (std + 1e-4));
  const tokenTerms = NANO_RESPONSES.map((response, responseIndex) => response.probabilities.map((probability, tokenIndex) => {
    const logp = Math.log(probability);
    const refLogp = Math.log(response.reference[tokenIndex]);
    const delta = refLogp - logp;
    const klSample = Math.expm1(delta) - delta;
    const policy = -logp * advantages[responseIndex];
    return { logp, refLogp, policy, klSample, total: policy + beta * klSample };
  }));
  const validTokens = tokenTerms.reduce((sum, tokens) => sum + tokens.length, 0);
  const contributions = tokenTerms.map(tokens => tokens.reduce((sum, token) => sum + token.total, 0) / validTokens);
  return { mean, std, advantages, tokenTerms, validTokens, contributions, batchLoss: contributions.reduce((sum, loss) => sum + loss, 0) };
}

export function initNano() {
  const mount = document.querySelector('#nano-mount');
  if (!mount || mount.dataset.nanoInitialized) return;
  mount.dataset.nanoInitialized = 'true';
  let selected = 2;
  let beta = 0.1;
  mount.innerHTML = `
    <div class="nano-shell">
      <div class="nano-intro"><p class="nano-kicker">Inside the two code slides</p><h3>Follow one answer from reward to token loss.</h3><p>The <a href="https://github.com/McGill-NLP/nano-aha-moment" target="_blank" rel="noopener noreferrer">nano-aha-moment code</a> makes the plumbing visible. The following is a small worked example of its reward normalization, token masking, and direct policy-gradient loss.</p></div>
      <ol class="nano-four-steps"><li><span>01</span><strong>Score each rollout</strong><p>A rollout is one generated answer. The repository adds format and equation rewards; our toy checker uses binary correctness.</p></li><li><span>02</span><strong>Compare its group</strong><p>For the same prompt, calculate Â = (reward − mean) / (std + 10⁻⁴).</p></li><li><span>03</span><strong>Broadcast to tokens</strong><p>Give every generated token in that answer the same Â. Mask prompt and padding positions.</p></li><li><span>04</span><strong>Reduce and update</strong><p>Add the reference penalty, average valid token contributions, backpropagate, and take an optimizer step.</p></li></ol>
      <div class="nano-scope-note"><strong>Read this implementation precisely.</strong> The shown policy term is <code>−log π × Â</code>. It has no old-policy ratio or clipping: this is the direct fresh-batch, REINFORCE-style form. The TRL starter above uses saved ratios and a clipped objective for repeated updates. The snippets illustrate related ideas; their losses are not identical.</div>

      <div class="nano-lab" aria-labelledby="nano-lab-title">
        <div class="nano-lab-heading"><div><p class="nano-kicker">A tiny same-prompt group</p><h4 id="nano-lab-title">Prompt: “Solve 2 + 2”</h4><p class="nano-help">Choose an answer to inspect. All three are two generated tokens: its number and an end token.</p></div><div class="nano-beta-control"><label for="nano-beta">Reference weight β <output id="nano-beta-value" for="nano-beta">0.10</output></label><input id="nano-beta" class="nano-range" type="range" min="0" max="1" step="0.01" value="0.1"></div></div>
        <div class="nano-answer-buttons" role="group" aria-label="Choose which response to trace">${NANO_RESPONSES.map((response, index) => `<button type="button" data-nano-answer="${index}" aria-pressed="${index === selected}"><span>Answer ${String.fromCharCode(65 + index)}: <strong>${response.answer}</strong></span><small>reward ${response.reward} · Â <span id="nano-answer-advantage-${index}"></span></small></button>`).join('')}</div>
        <div class="nano-group-stats"><span>Rewards <strong>[0, 0, 1]</strong></span><span>Mean <strong id="nano-mean"></strong></span><span>Population std <strong id="nano-std"></strong></span><span>Valid tokens in batch <strong>6</strong></span></div>
        <p class="nano-help">The std is a square root of variance, not variance itself. NumPy's default <code>std()</code> divides variance by the group size. The 10⁻⁴ prevents division by zero; equal rewards would still give zero advantages.</p>

        <div class="nano-token-heading"><h4>1. Build the sequence and loss mask</h4><span class="nano-legend"><i class="nano-legend-prompt"></i>prompt <i class="nano-legend-response"></i>response <i class="nano-legend-pad"></i>padding</span></div>
        <div class="nano-token-strip" id="nano-token-strip" aria-label="Prompt, response, and padding token sequence"></div>
        <div class="nano-table-wrap"><table class="nano-table"><caption>Before shifting: prompt and pad labels are ignored; only response labels count toward loss.</caption><thead><tr><th scope="col">Position</th><th scope="col">Token</th><th scope="col">Training label</th><th scope="col">Loss mask</th><th scope="col">Advantage</th></tr></thead><tbody id="nano-mask-rows"></tbody></table></div>
        <p class="nano-help">Token pieces and IDs are illustrative. The prompt still supplies context through attention; masking its <em>loss</em> does not hide it from the model.</p>

        <div class="nano-token-heading"><h4>2. Score the two generated tokens</h4></div>
        <p class="nano-help">A causal model predicts the <em>next</em> token. Logits at position 2 predict the first answer token at position 3; logits at position 3 predict the end token at position 4.</p>
        <div class="nano-table-wrap"><table class="nano-table"><caption id="nano-loss-caption"></caption><thead><tr><th scope="col">Target token</th><th scope="col">Current p</th><th scope="col">Ref p</th><th scope="col">log p</th><th scope="col">−log p × Â</th><th scope="col">k = u−log u−1</th><th scope="col">Policy + βk</th></tr></thead><tbody id="nano-loss-rows"></tbody></table></div>
        <div class="nano-results" aria-live="polite" aria-atomic="true"><div><span>This answer's contribution</span><strong id="nano-answer-contribution"></strong><small>sum of its 2 token losses / all 6 valid tokens</small></div><div><span>Whole-batch loss</span><strong id="nano-batch-loss"></strong><small>sum of all 3 answer contributions</small></div></div>
        <p class="nano-help">Move β to change the reference penalty while rewards and advantages stay fixed. These are frozen example probabilities, not a training simulation. Negative loss values are possible; the sign of loss is not an accuracy score.</p>
      </div>

      <div class="nano-detail-grid">
        <details class="nano-details"><summary>Why labels[..., 1:] and advantages[..., 1:]?</summary><div class="nano-details-body"><p><code>...</code> means “keep earlier dimensions,” such as batch. <code>1:</code> drops the first sequence position; <code>:-1</code> drops the last.</p><pre><code>next_logits = logits[..., :-1, :]
next_labels = labels[..., 1:]
next_mask   = loss_mask[..., 1:]
next_A      = advantages[..., 1:]</code></pre><p>These arrays now align each prediction with its next-token target, its mask, and its advantage. <code>−100</code> is an ignore marker for labels, not a vocabulary token. A helper replaces ignored labels with a safe gather index and then masks their results to zero.</p><p>Keep attention masking separate: attention excludes padding from context; loss masking excludes prompt and padding from the objective. Normalize by valid response-token count, not padded sequence length. The repository's current code carries an explicit <code>labels_mask</code>.</p></div></details>
        <details class="nano-details"><summary>Where do gradients flow?</summary><div class="nano-details-body"><p>Rewards and normalized advantages are fixed training weights. They arrive from the checker and NumPy; if calculated with tensors, detach them from autograd. Reference log probabilities are computed under <code>no_grad()</code>.</p><p>Current-policy log probabilities remain differentiable. For the direct term, ∂(−log p × Â)/∂log p = −Â: minimizing the loss favors a larger probability when Â is positive, and a smaller one when it is negative.</p><p>The reference values stay fixed, but the penalty still depends on the current log probabilities and can contribute a gradient. Then clear accumulated gradients when appropriate, run backward, and let the optimizer update the actor. Gradient accumulation may combine microbatches before a weight update.</p><p>Repeatedly optimizing this direct log-probability term on stale answers is not the clipped GRPO procedure. Use the versioned TRL implementation above for that workflow.</p></div></details>
        <details class="nano-details"><summary>Decode KL and the “entropy” diagnostic</summary><div class="nano-details-body"><p>Let d = log p<sub>ref</sub> − log p and u = exp(d). The sample penalty is <code>exp(d) − d − 1</code>. For this example, it is nonnegative and zero when the two token probabilities match.</p><p>It is a sample statistic. Its expectation gives current-to-reference KL under current-policy sampling with compatible support; one token's number is not a full distribution's KL.</p><p>The snippet's <code>−sum(logps) / N</code> diagnostic is average <strong>sampled surprisal</strong>. It estimates conditional entropy when tokens are sampled from the distribution whose log probabilities are measured. It is not exact vocabulary entropy, which sums <code>−p log p</code> over every possible next token.</p><p>Here N means valid response tokens and ignored log probabilities are already zeroed. Logging this statistic under <code>no_grad()</code> does not itself add an entropy bonus to the loss.</p></div></details>
      </div>
      <div class="nano-download-note"><a class="button secondary" href="examples/tiny_grpo_loss.py" download>Download the tiny PyTorch loss ↓</a><p class="nano-help">Optional reading: a small, independently written loss with raw and clipped modes, masks, and CPU checks. Use the same pinned environment as the TRL starter. Python syntax was checked here; the included tensor checks were not run because PyTorch is unavailable in this workspace.</p></div>
      <p class="nano-sources">Original implementation: <a href="https://github.com/McGill-NLP/nano-aha-moment/blob/main/nano_r1_script.py#L177" target="_blank" rel="noopener noreferrer">reward grouping and token advantages ↗</a> · <a href="https://github.com/McGill-NLP/nano-aha-moment/blob/main/nano_r1_script.py#L512" target="_blank" rel="noopener noreferrer">direct policy-gradient loss ↗</a> · <a href="https://github.com/McGill-NLP/nano-aha-moment/blob/main/utils.py#L129" target="_blank" rel="noopener noreferrer">causal token log probabilities ↗</a>. This walkthrough uses original toy numbers and simplified notation; repository code can evolve.</p>
    </div>`;

  const find = id => mount.querySelector(`#nano-${id}`);
  const update = () => {
    const stats = nanoExampleStats(beta);
    const response = NANO_RESPONSES[selected];
    const advantage = stats.advantages[selected];
    find('beta-value').textContent = nanoFormat(beta, 2);
    find('mean').textContent = nanoFormat(stats.mean);
    find('std').textContent = nanoFormat(stats.std);
    stats.advantages.forEach((value, index) => { find(`answer-advantage-${index}`).textContent = nanoSigned(value); });
    mount.querySelectorAll('[data-nano-answer]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.nanoAnswer) === selected)));
    const tokens = [
      { token: 'Solve', kind: 'prompt', label: '−100', mask: 0, advantage: 0 },
      { token: '2+2', kind: 'prompt', label: '−100', mask: 0, advantage: 0 },
      { token: '→', kind: 'prompt', label: '−100', mask: 0, advantage: 0 },
      { token: response.answer, kind: 'response', label: 101 + selected, mask: 1, advantage },
      { token: '<end>', kind: 'response', label: 2, mask: 1, advantage },
      { token: '<pad>', kind: 'pad', label: '−100', mask: 0, advantage: 0 },
      { token: '<pad>', kind: 'pad', label: '−100', mask: 0, advantage: 0 }
    ];
    const escapeToken = value => value.replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    find('token-strip').innerHTML = tokens.map((token, index) => `<div class="nano-token nano-token-${token.kind}"><small>pos ${index}</small><strong>${escapeToken(token.token)}</strong><span>mask ${token.mask}</span></div>`).join('');
    find('mask-rows').innerHTML = tokens.map((token, index) => `<tr class="${token.mask ? 'nano-valid-row' : 'nano-ignored-row'}"><th scope="row">${index}</th><td>${escapeToken(token.token)}</td><td>${token.label}</td><td>${token.mask}</td><td>${token.mask ? nanoSigned(advantage) : '0'}</td></tr>`).join('');
    find('loss-caption').textContent = `Answer ${String.fromCharCode(65 + selected)}: reward ${response.reward}, shared token advantage ${nanoSigned(advantage)}. All logs are natural logs.`;
    find('loss-rows').innerHTML = stats.tokenTerms[selected].map((term, index) => `<tr><th scope="row">${index === 0 ? response.answer : '&lt;end&gt;'}</th><td>${nanoFormat(response.probabilities[index], 2)}</td><td>${nanoFormat(response.reference[index], 2)}</td><td>${nanoSigned(term.logp)}</td><td>${nanoSigned(term.policy)}</td><td>${nanoFormat(term.klSample)}</td><td class="nano-total-cell">${nanoSigned(term.total)}</td></tr>`).join('');
    find('answer-contribution').textContent = nanoSigned(stats.contributions[selected]);
    find('batch-loss').textContent = nanoSigned(stats.batchLoss);
  };
  mount.querySelectorAll('[data-nano-answer]').forEach(button => button.addEventListener('click', () => { selected = Number(button.dataset.nanoAnswer); update(); }));
  find('beta').addEventListener('input', event => { beta = Number(event.target.value); update(); });
  update();
}
