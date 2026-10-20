const GRPO_STABILIZER = 1e-8;
const grpoNumber = (value, digits = 3) => (Math.abs(value) < 0.00000001 ? 0 : value).toFixed(digits);
const grpoSigned = value => `${value >= 0 ? '+' : '−'}${grpoNumber(Math.abs(value))}`;

export function grpoGroupStats(rewards) {
  if (!Array.isArray(rewards) || rewards.length === 0 || rewards.some(reward => !Number.isFinite(reward))) {
    throw new TypeError('Expected a nonempty array of finite reward numbers.');
  }
  const mean = rewards.reduce((sum, value) => sum + value, 0) / rewards.length;
  const variance = rewards.reduce((sum, value) => sum + (value - mean) ** 2, 0) / rewards.length;
  const std = Math.sqrt(variance);
  const allEqual = std < 1e-12;
  const advantages = rewards.map(reward => allEqual ? 0 : (reward - mean) / (std + GRPO_STABILIZER));
  return { mean, variance, std, allEqual, advantages };
}

export function initGRPO() {
  const mount = document.querySelector('#grpo-mount');
  if (!mount || mount.dataset.grpoInitialized) return;
  mount.dataset.grpoInitialized = 'true';
  let rewards = [2, 4, 6, 8];
  mount.innerHTML = `
    <div class="grpo-shell">
      <div class="grpo-opening"><p><strong>GRPO replaces the learned critic with comparisons inside a group.</strong> Generate several answers to the same prompt, score them, then favor answers that outperform their peers.</p><div class="grpo-correction"><span class="grpo-correction-icon" aria-hidden="true">↳</span><p><strong>One correction to the slide:</strong> the value network is removed, but <em>advantages are still calculated</em>. Their baseline comes from the group's rewards.</p></div></div>
      <div class="grpo-flow" aria-label="Same prompt, generate a group, score responses, compute relative advantages, apply clipped update"><span>One prompt</span><b aria-hidden="true">→</b><span>G responses</span><b aria-hidden="true">→</b><span>Rewards</span><b aria-hidden="true">→</b><span>Group advantages</span><b aria-hidden="true">→</b><span>Clipped update</span></div>

      <div class="grpo-lab" aria-labelledby="grpo-lab-title">
        <div class="grpo-lab-header"><div><p class="grpo-kicker">Try a group of four · G = 4</p><h3 id="grpo-lab-title">Change one reward. Watch every advantage move.</h3><p class="grpo-help">These four responses all answer the same prompt. Drag their illustrative reward scores from 0 to 10.</p></div><div class="grpo-presets" role="group" aria-label="Set example group rewards"><button type="button" data-grpo-preset="graded">Graded: 2, 4, 6, 8</button><button type="button" data-grpo-preset="binary">Binary: 0, 0, 0, 1</button><button type="button" data-grpo-preset="equal">All equal: 5, 5, 5, 5</button></div></div>
        <div class="grpo-response-grid">${['A', 'B', 'C', 'D'].map((name, index) => `<article class="grpo-response"><div class="grpo-response-top"><h4>Response ${name}</h4><span class="grpo-response-tag">same prompt</span></div><label class="grpo-reward-label" for="grpo-reward-${index}"><span>Reward R<sub>${index + 1}</sub></span><output id="grpo-reward-output-${index}" for="grpo-reward-${index}">${rewards[index]}</output></label><input class="grpo-range" id="grpo-reward-${index}" data-grpo-reward="${index}" type="range" min="0" max="10" step="0.1" value="${rewards[index]}" aria-describedby="grpo-slider-help"><div class="grpo-response-calculation" id="grpo-calculation-${index}"></div><div class="grpo-meter" aria-hidden="true"><span class="grpo-meter-zero"></span><span class="grpo-meter-fill" id="grpo-meter-${index}"></span></div><div class="grpo-advantage-label"><span>Advantage Â<sub>${index + 1}</sub></span><strong id="grpo-advantage-${index}"></strong></div><p class="grpo-direction" id="grpo-direction-${index}"></p></article>`).join('')}</div>
        <p class="grpo-sr-only" id="grpo-slider-help">Changing any reward recalculates the mean, population standard deviation, and all four advantages.</p>
        <div class="grpo-stat-grid" aria-live="polite" aria-atomic="true"><div><span>Group mean μ</span><strong id="grpo-mean"></strong><small>sum of rewards / 4</small></div><div><span>Group spread σ</span><strong id="grpo-std"></strong><small>population standard deviation</small></div><div><span>Mean advantage</span><strong id="grpo-average-advantage"></strong><small>centered near zero</small></div></div>
        <div class="grpo-state-note" id="grpo-state-note"></div>
        <details class="grpo-arithmetic"><summary>Show the mean, variance, and z-score arithmetic</summary><div class="grpo-arithmetic-body"><p class="grpo-math-line" id="grpo-mean-equation"></p><p class="grpo-math-line" id="grpo-std-equation"></p><p class="grpo-math-line">Âᵢ = (Rᵢ − μ) / (σ + 10⁻⁸)</p><p class="grpo-help">This demo uses <strong>population</strong> variance: divide by G = 4, then take the square root. A sample-standard-deviation implementation uses G−1 and gives slightly different magnitudes. The tiny 10⁻⁸ stabilizer protects the denominator; if all rewards match, this demo explicitly returns four zeros.</p></div></details>
      </div>

      <div class="grpo-explanation-grid">
        <article class="grpo-explanation-card"><p class="grpo-kicker">Read the signal</p><h3>Relative quality, within one prompt</h3><p>A positive Â means above this group's mean; a negative Â means below it. It does not directly mean “correct” or “incorrect.”</p><p>With [2, 4, 6, 8], a reward of 4 gets a negative advantage despite being positive. With binary [0, 0, 0, 1], the one successful answer stands out.</p><p>Equal rewards provide no within-group reward preference, whether every answer is great or every answer is poor. A separate KL term can still contribute an update.</p></article>
        <article class="grpo-explanation-card grpo-objective-card"><p class="grpo-kicker">The PPO connection</p><h3>Same clipped incentive, new advantage</h3><div class="grpo-objective-formula">maximize average [<br><span>min(ρÂ, clip(ρ, 1−ε, 1+ε)Â)</span><br><span>− β × KL-to-reference</span><br>]</div><p>ρ = probability under current actor / probability under old actor. ε controls clipping; β sets the reference-policy penalty strength.</p><p><strong>Old policy:</strong> the snapshot that generated this batch. <strong>Reference policy:</strong> a separate anchor for regularization. They play different roles.</p></article>
      </div>

      <div class="grpo-comparison-wrap"><table class="grpo-comparison"><caption>Common actor-critic PPO versus outcome-reward GRPO</caption><thead><tr><th scope="col">Question</th><th scope="col">PPO</th><th scope="col">GRPO</th></tr></thead><tbody><tr><th scope="row">Where is the baseline?</th><td>A learned value prediction V(s).</td><td>Mean reward of same-prompt responses.</td></tr><tr><th scope="row">How is advantage estimated?</th><td>Return minus value, often with GAE.</td><td>Group-centered, often standardized rewards.</td></tr><tr><th scope="row">What experience is needed?</th><td>Rollouts with critic estimates.</td><td>Several scored responses per prompt.</td></tr><tr><th scope="row">What controls policy change?</th><td>A clipped objective in PPO-Clip.</td><td>A clipped objective, with optional KL regularization.</td></tr></tbody></table></div>

      <div class="grpo-fit-grid"><article><h3>Why it can help</h3><p>Removing critic training can save memory and simplify the system, especially when a critic would be large. Groups fit tasks where answers can be scored reliably, such as verifiable math or code.</p></article><article><h3>Where the trade-off appears</h3><p>You must generate and score multiple answers. Larger groups cost more sampling work; identical rewards give no relative signal. Savings depend on model sizes and the rest of the training stack. GRPO is not universally preferable to PPO.</p></article></div>

      <div class="grpo-deep-grid">
        <details class="grpo-details"><summary>What does this look like for actual LLM tokens?</summary><div class="grpo-details-body"><p>A language model chooses the next token from a distribution conditioned on the prompt and earlier tokens. A typical implementation calculates</p><p class="grpo-math-line">ρᵢ,ₜ = exp(log πθ(tokenᵢ,ₜ | prefix) − log πold(tokenᵢ,ₜ | prefix)).</p><p>For outcome supervision, the sequence's group advantage is broadcast to its generated tokens. Completion masks exclude padding and other tokens that should not count. The objective then averages token contributions; length normalization varies by implementation.</p><p>The compact slide writes a response-level ratio. Do not directly replace the token calculation with the product of all token ratios: that changes the objective and can be numerically unstable.</p></div></details>
        <details class="grpo-details"><summary>Decode the slide's KL estimator</summary><div class="grpo-details-body"><p>For one sampled token at a fixed prefix, let u = πref(token) / πθ(token). The slide uses</p><p class="grpo-math-line">k(u) = u − log(u) − 1 ≥ 0.</p><p>This is a nonnegative <em>sample statistic</em>, not the exact KL divergence of a whole distribution or response. When sampling from πθ with compatible support, E[u] = 1 and E[k(u)] = DKL(πθ ∥ πref). Samples from an older policy require care about distribution mismatch.</p><p>The original GRPO objective includes a reference penalty. Some variants use β = 0 and omit it. This is an implementation choice; it is not the definition of every GRPO run.</p></div></details>
        <details class="grpo-details"><summary>If the average advantage is zero, why can learning happen?</summary><div class="grpo-details-body"><p>Before the first update, current and old policies match, so each ratio is 1. The score may average to zero, but its <strong>gradient need not be zero</strong>.</p><p>Hold the sampled rewards, group statistics, and old probabilities fixed. At this point, clipping is inactive and ∇θρᵢ,ₜ = ∇θ log πθ(tokenᵢ,ₜ | prefix). Ignoring KL, the update is a group-normalized REINFORCE-style gradient:</p><p class="grpo-math-line">average [Âᵢ × ∇θ log πθ(tokenᵢ,ₜ | prefix)].</p><p>Different responses have different log-probability gradients. Their weighted gradients do not cancel just because the advantages sum to zero. This describes an immediate on-policy first step, not every later reuse of the batch.</p></div></details>
      </div>
      <p class="grpo-sources">Primary sources: <a href="https://arxiv.org/pdf/2402.03300#page=13" target="_blank" rel="noopener noreferrer">DeepSeekMath §4.1 ↗</a> · <a href="https://arxiv.org/pdf/2501.12948" target="_blank" rel="noopener noreferrer">DeepSeek-R1 §2.1 ↗</a> · <a href="https://huggingface.co/docs/trl/grpo_trainer" target="_blank" rel="noopener noreferrer">Hugging Face TRL implementation notes ↗</a>. Rewards here are illustrative.</p>
    </div>`;

  const find = id => mount.querySelector(`#grpo-${id}`);
  const update = () => {
    const { mean, variance, std, allEqual, advantages } = grpoGroupStats(rewards);
    find('mean').textContent = grpoNumber(mean);
    find('std').textContent = grpoNumber(std);
    find('average-advantage').textContent = grpoNumber(advantages.reduce((sum, value) => sum + value, 0) / advantages.length);
    find('mean-equation').textContent = `μ = (${rewards.map(value => grpoNumber(value, 1)).join(' + ')}) / 4 = ${grpoNumber(mean)}`;
    find('std-equation').textContent = `σ = √[Σ(Rᵢ − ${grpoNumber(mean)})² / 4] = √${grpoNumber(variance)} = ${grpoNumber(std)}`;
    advantages.forEach((advantage, index) => {
      find(`reward-output-${index}`).textContent = grpoNumber(rewards[index], 1);
      find(`reward-${index}`).value = rewards[index];
      find(`calculation-${index}`).textContent = `(${grpoNumber(rewards[index], 1)} − ${grpoNumber(mean, 1)}) / (${grpoNumber(std)} + 10⁻⁸)`;
      find(`advantage-${index}`).textContent = grpoSigned(advantage);
      const direction = find(`direction-${index}`);
      direction.textContent = Math.abs(advantage) < 1e-12 ? 'No relative reward signal' : advantage > 0 ? 'Above this group’s average' : 'Below this group’s average';
      direction.className = `grpo-direction ${advantage > 1e-12 ? 'grpo-positive' : advantage < -1e-12 ? 'grpo-negative' : 'grpo-neutral'}`;
      const meter = find(`meter-${index}`);
      const width = Math.abs(advantage) * 25;
      meter.style.left = `${advantage >= 0 ? 50 : 50 - width}%`;
      meter.style.width = `${width}%`;
      meter.className = `grpo-meter-fill ${advantage >= 0 ? 'grpo-meter-positive' : 'grpo-meter-negative'}`;
    });
    find('state-note').classList.toggle('grpo-equal-note', allEqual);
    find('state-note').textContent = allEqual ? 'Every response has the same reward. All four advantages are 0: there is no within-group reward preference to learn from.' : 'The rewards stay on their original scale. The advantages express distance from this group’s mean in standard-deviation units.';
  };
  mount.querySelectorAll('[data-grpo-reward]').forEach(input => input.addEventListener('input', event => {
    rewards[Number(event.target.dataset.grpoReward)] = Number(event.target.value);
    update();
  }));
  mount.querySelectorAll('[data-grpo-preset]').forEach(button => button.addEventListener('click', () => {
    rewards = button.dataset.grpoPreset === 'binary' ? [0, 0, 0, 1] : button.dataset.grpoPreset === 'equal' ? [5, 5, 5, 5] : [2, 4, 6, 8];
    update();
  }));
  update();
}
