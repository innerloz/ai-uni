const LOOP_STEPS = [
  { name: 'Initialize', code: 'Input: actor θ₀, critic ϕ₀', title: 'Start with two jobs, usually two networks.', body: '<p>The <strong>actor</strong> π<sub>θ</sub>(a | s) chooses an action distribution. The <strong>critic</strong> V<sub>ϕ</sub>(s) predicts the return from a state. θ and ϕ are their trainable weights; the subscript 0 means “before training.”</p><p>The loop counter k labels a collection-and-update round. It is different from t, which labels a time step inside a trajectory.</p>' },
  { name: 'Collect', code: 'Collect trajectories Dₖ using πₖ = π(θₖ)', title: 'Let the current actor gather experience.', body: '<p>A trajectory τ is a sequence of states, actions, and rewards. D<sub>k</sub> is a batch of these trajectories. Our tiny episode takes three actions and receives rewards <strong>[1, 0, 2]</strong>, then really terminates.</p><p>Save the probability of each sampled action under this actor. During the upcoming policy updates, this snapshot becomes π<sub>old</sub>: the denominator of every probability ratio stays fixed.</p>' },
  { name: 'Returns', code: 'Compute rewards-to-go R̂ₜ', title: 'A reward is immediate. A return includes what follows.', body: '<p>The reward r<sub>t</sub> belongs to one step. The return-to-go R̂<sub>t</sub> adds that reward and the later rewards, discounting each future step by γ. γ = 1 gives the future full weight; a smaller γ gives it less weight.</p><p>Move the discount slider and watch all three return targets change. No reward is being changed.</p>' },
  { name: 'Advantages', code: 'Estimate Âₜ using critic Vϕₖ', title: 'Was the outcome better than the critic expected?', body: '<p>A simple Monte Carlo estimate is <strong>Â<sub>t</sub> = R̂<sub>t</sub> − V<sub>ϕₖ</sub>(s<sub>t</sub>)</strong>. Positive means better than expected; negative means worse than expected. A positive return can still have a negative advantage.</p><p>The table uses this simple estimate to make the arithmetic visible. The algorithm permits other estimators; practical PPO commonly uses generalized advantage estimation (GAE).</p>' },
  { name: 'Update actor', code: 'θₖ₊₁ ≈ argmaxθ mean[min(rₜ(θ)Âₜ, g(ε, Âₜ))]', title: 'Improve the actor using the clipped batch score.', body: '<p>For each recorded action, compute its new probability divided by its saved old probability. Multiply by its fixed advantage, apply the clipped objective, then average over the batch.</p><p><strong>argmax</strong> means “find the weights giving the largest score.” We approximately do that with several minibatch gradient steps, often using Adam. Code usually minimizes the negative of this score.</p><p>The saved old probabilities and advantage estimates stay fixed across those passes. Clipping removes some incentives for large changes; it does not enforce a hard boundary.</p>' },
  { name: 'Fit critic', code: 'ϕₖ₊₁ ≈ argminϕ mean[(Vϕ(sₜ) − R̂ₜ)²]', title: 'Train the critic to make better return predictions.', body: '<p>Compare each value prediction with its return target. Square each error and average: this is <strong>mean squared error (MSE)</strong>. <strong>argmin</strong> means “find the weights giving the smallest error.”</p><p>In training, gradient descent changes ϕ. Our slider directly moves the three predictions toward their targets so you can see the effect. This is a teaching illustration, not a simulation of neural-network training.</p>' },
  { name: 'Repeat', code: 'k ← k + 1; collect a fresh batch with the updated actor', title: 'Use the improved actor to collect fresh experience.', body: '<p>The updated actor now becomes the behavior policy for the next round. Collect a fresh batch, save its old probabilities, estimate new advantages, and repeat.</p><p>PPO is on-policy: it makes several optimization passes on a freshly collected batch, then moves on. This is different from repeatedly training on an unrestricted replay buffer of older experience.</p>' }
];

const loopFormat = (value, digits = 2) => (Math.abs(value) < 0.0000001 ? 0 : value).toFixed(digits);
const loopSigned = value => `${value >= 0 ? '+' : '−'}${loopFormat(Math.abs(value))}`;

export function initLoop() {
  const mount = document.querySelector('#training-loop-mount');
  if (!mount || mount.dataset.loopInitialized) return;
  mount.dataset.loopInitialized = 'true';
  const values = [2.8, 1.2, 1.5];
  const oldProbabilities = [0.2, 0.5, 0.4];
  let active = 0;
  let gamma = 0.9;
  let fit = 0;
  let pass = 0;
  let epsilon = 0.2;

  mount.innerHTML = `
    <div class="loop-shell">
      <p class="loop-intro">The second slide is the training recipe. Follow one tiny episode through it: <strong>collect → calculate → update → repeat.</strong></p>
      <nav aria-label="PPO training steps" class="loop-navigation"><ol class="loop-steps">${LOOP_STEPS.map((s, i) => `<li><button class="loop-step" data-loop-step="${i}" type="button" aria-controls="loop-step-content"><span class="loop-step-number">${i + 1}</span><span>${s.name}</span></button></li>`).join('')}</ol></nav>
      <div class="loop-main-grid">
        <article class="loop-explanation" id="loop-step-content" aria-labelledby="loop-step-title">
          <p class="loop-kicker" id="loop-step-counter"></p>
          <div class="loop-code" id="loop-step-code"></div>
          <h3 id="loop-step-title"></h3>
          <div class="loop-step-body" id="loop-step-body"></div>
          <div class="loop-pager"><button type="button" id="loop-prev">← Previous</button><span id="loop-page-label" aria-live="polite"></span><button type="button" id="loop-next">Next step →</button></div>
        </article>
        <aside class="loop-episode" aria-labelledby="loop-episode-title">
          <p class="loop-kicker">One terminated toy episode</p><h3 id="loop-episode-title">Three rewards. Three return targets.</h3>
          <div class="loop-reward-track" aria-label="Immediate rewards are 1, 0, and 2"><div><span>t = 0</span><strong>+1</strong></div><span class="loop-arrow" aria-hidden="true">→</span><div><span>t = 1</span><strong>0</strong></div><span class="loop-arrow" aria-hidden="true">→</span><div><span>t = 2</span><strong>+2</strong></div><span class="loop-terminal">end</span></div>
          <label class="loop-control-label" for="loop-gamma"><span>Discount factor γ</span><output id="loop-gamma-value" for="loop-gamma">0.90</output></label>
          <input class="loop-range" id="loop-gamma" type="range" min="0" max="1" step="0.01" value="0.9" aria-describedby="loop-gamma-help">
          <p class="loop-help" id="loop-gamma-help">Future rewards get multiplied by γ once per step into the future.</p>
          <div class="loop-return-formulas" id="loop-return-formulas"></div>
          <div class="loop-table-wrap"><table class="loop-data-table"><caption>Returns and advantages, using the saved critic</caption><thead><tr><th scope="col">t</th><th scope="col">Reward<br>rₜ</th><th scope="col">Return<br>R̂ₜ</th><th scope="col">Value<br>Vₖ</th><th scope="col">Advantage<br>Âₜ</th></tr></thead><tbody id="loop-trajectory-rows"></tbody></table></div>
          <p class="loop-help">Value predictions [2.80, 1.20, 1.50] are made-up estimates. The return targets are calculated from this episode.</p>
        </aside>
      </div>

      <div class="loop-live-panel" id="loop-policy-panel" hidden>
        <div class="loop-panel-heading"><div><p class="loop-kicker">Actor calculation</p><h3>One fixed batch, several candidate updates</h3></div><div class="loop-controls-pair"><div><label class="loop-control-label" for="loop-pass">Candidate pass <output id="loop-pass-value" for="loop-pass">0</output></label><input class="loop-range" id="loop-pass" type="range" min="0" max="4" step="1" value="0"></div><div><label class="loop-control-label" for="loop-epsilon">Clip ε <output id="loop-epsilon-value" for="loop-epsilon">0.20</output></label><input class="loop-range" id="loop-epsilon" type="range" min="0.05" max="0.4" step="0.05" value="0.2"></div></div></div>
        <p class="loop-help">These hypothetical candidates move action probabilities in the direction favored by each saved advantage. The slider illustrates objective arithmetic; it does not run an optimizer.</p>
        <div class="loop-table-wrap"><table class="loop-data-table"><caption>Each row represents one recorded state-action pair</caption><thead><tr><th scope="col">t</th><th scope="col">Old p</th><th scope="col">New p</th><th scope="col">Ratio</th><th scope="col">Fixed Â</th><th scope="col">r × Â</th><th scope="col">g(ε, Â)</th><th scope="col">min score</th></tr></thead><tbody id="loop-policy-rows"></tbody></table></div>
        <div class="loop-objective-summary"><span>Batch mean = (score₀ + score₁ + score₂) / 3</span><strong id="loop-batch-mean"></strong></div>
        <div class="loop-g-definition"><strong>The shortcut g(ε, A)</strong><p>If A ≥ 0: <code>g = (1 + ε) A</code><br>If A &lt; 0: <code>g = (1 − ε) A</code></p><p>The full expression <code>min(rA, clip(r, 1−ε, 1+ε)A)</code> simplifies to <code>min(rA, g)</code>. A = 0 contributes zero either way.</p></div>
      </div>

      <div class="loop-live-panel" id="loop-value-panel" hidden>
        <div class="loop-panel-heading"><div><p class="loop-kicker">Critic calculation</p><h3>Watch squared prediction errors shrink</h3></div><div class="loop-fit-control"><label class="loop-control-label" for="loop-fit">Move predictions toward targets <output id="loop-fit-value" for="loop-fit">0%</output></label><input class="loop-range" id="loop-fit" type="range" min="0" max="100" step="1" value="0"></div></div>
        <div class="loop-table-wrap"><table class="loop-data-table"><caption>Controlled interpolation: V = initial V + fraction × (target − initial V)</caption><thead><tr><th scope="col">t</th><th scope="col">Target R̂ₜ</th><th scope="col">Current V</th><th scope="col">Error V − R̂</th><th scope="col">Squared error</th></tr></thead><tbody id="loop-value-rows"></tbody></table></div>
        <div class="loop-objective-summary"><span>Mean squared error = sum of three squared errors / 3</span><strong id="loop-mse"></strong></div>
        <p class="loop-help">At 100%, three independently controlled predictions exactly match these three targets. A real critic shares parameters across states and may not reach zero error. This illustration does not change the frozen advantages used by the actor above.</p>
      </div>

      <div class="loop-details-grid">
        <details class="loop-details"><summary>Decode the notation on the algorithm slide</summary><dl class="loop-symbols"><div><dt>θ / ϕ</dt><dd>Actor weights / critic weights. Two different learning jobs.</dd></div><div><dt>k / t</dt><dd>Training-round index / time-step index within experience.</dd></div><div><dt>sₜ / aₜ / rₜ</dt><dd>State, chosen action, and immediate reward at step t.</dd></div><div><dt>τ / Dₖ</dt><dd>One trajectory / the batch collected in round k.</dd></div><div><dt>πₖ = π(θₖ)</dt><dd>The policy produced by actor weights θₖ.</dd></div><div><dt>R̂ₜ / Âₜ</dt><dd>Estimated return target / estimated advantage. The hat means an estimate.</dd></div><div><dt>Vϕ(s)</dt><dd>The critic's predicted return from state s.</dd></div><div><dt>𝔼̂ / batch mean</dt><dd>A sample average over recorded experience.</dd></div><div><dt>argmax / argmin</dt><dd>Seek parameters with the largest score / smallest loss. Optimizers approximate this search.</dd></div><div><dt>|Dₖ|T</dt><dd>For |Dₖ| trajectories of equal length T, this counts all recorded time steps. With unequal lengths, average over the actual sample count.</dd></div><div><dt>θₖ₊₁ / ϕₖ₊₁</dt><dd>The updated weights for the next collection round.</dd></div></dl></details>
        <details class="loop-details"><summary>What changes with GAE or an unfinished rollout?</summary><div class="loop-details-body"><p><strong>GAE:</strong> instead of the simple return-minus-value estimate, combine prediction surprises across steps:</p><p class="loop-formula">δₜ = rₜ + γV(sₜ₊₁) − V(sₜ)<br>Âₜ = δₜ + γλδₜ₊₁ + (γλ)²δₜ₊₂ + …</p><p>λ controls how far those surprises are mixed. With an exactly terminated episode and λ = 1, this telescopes to the return-minus-value estimate used in our table.</p><p><strong>True terminal state:</strong> there is no future return, so use a final value of zero. That is the assumption for [1, 0, 2] here.</p><p><strong>Unfinished segment or external time-limit truncation:</strong> the task may continue beyond the data. Bootstrap with the critic's value at the last observed state, rather than automatically using zero. Whether a time limit is terminal depends on the task definition.</p><p>For an unfinished segment ending after step T−1, a bootstrapped target is <span class="loop-inline-formula">R̂ₜ = rₜ + γrₜ₊₁ + … + γ<sup>T−t−1</sup>r<sub>T−1</sub> + γ<sup>T−t</sup>V(s<sub>T</sub>)</span>.</p><p>The slide is a high-level recipe: advantage estimators and value-target details can differ across PPO implementations.</p></div></details>
      </div>
      <p class="loop-source">Algorithm and notation: <a href="https://spinningup.openai.com/en/latest/algorithms/ppo.html#pseudocode" target="_blank" rel="noopener noreferrer">OpenAI Spinning Up: PPO pseudocode ↗</a>. The numbers and controlled demonstrations here are teaching examples.</p>
    </div>`;

  const find = id => mount.querySelector(`#loop-${id}`);
  const getReturns = () => [1 + gamma * gamma * 2, gamma * 2, 2];
  const updateNumbers = () => {
    const returns = getReturns();
    const advantages = returns.map((target, index) => target - values[index]);
    find('gamma-value').textContent = loopFormat(gamma);
    find('return-formulas').innerHTML = `<p><span>R̂₀</span> = 1 + ${loopFormat(gamma)} × 0 + ${loopFormat(gamma)}² × 2 <strong>= ${loopFormat(returns[0])}</strong></p><p><span>R̂₁</span> = 0 + ${loopFormat(gamma)} × 2 <strong>= ${loopFormat(returns[1])}</strong></p><p><span>R̂₂</span> = 2 <strong>= 2.00</strong></p>`;
    find('trajectory-rows').innerHTML = returns.map((target, index) => `<tr><th scope="row">${index}</th><td>${[1, 0, 2][index]}</td><td class="loop-return-cell">${loopFormat(target)}</td><td>${loopFormat(values[index])}</td><td class="${advantages[index] >= 0 ? 'loop-positive' : 'loop-negative'}">${loopSigned(advantages[index])}</td></tr>`).join('');

    find('pass-value').textContent = pass;
    find('epsilon-value').textContent = loopFormat(epsilon);
    let totalScore = 0;
    find('policy-rows').innerHTML = advantages.map((advantage, index) => {
      const oldP = oldProbabilities[index];
      const newP = oldP * (1 + Math.sign(advantage) * pass * 0.1);
      const ratio = newP / oldP;
      const raw = ratio * advantage;
      const bound = (advantage >= 0 ? 1 + epsilon : 1 - epsilon) * advantage;
      const score = Math.min(raw, bound);
      const clipped = raw > bound + 1e-10;
      totalScore += score;
      return `<tr><th scope="row">${index}</th><td>${loopFormat(oldP)}</td><td>${loopFormat(newP)}</td><td>${loopFormat(ratio)}</td><td>${loopSigned(advantage)}</td><td>${loopSigned(raw)}</td><td>${loopSigned(bound)}</td><td class="${clipped ? 'loop-clipped-cell' : 'loop-return-cell'}">${loopSigned(score)}${clipped ? '<span class="loop-cell-note">capped</span>' : ''}</td></tr>`;
    }).join('');
    find('batch-mean').textContent = loopSigned(totalScore / 3);

    let squaredErrorSum = 0;
    find('fit-value').textContent = `${fit}%`;
    find('value-rows').innerHTML = returns.map((target, index) => {
      const prediction = values[index] + fit / 100 * (target - values[index]);
      const error = prediction - target;
      squaredErrorSum += error * error;
      return `<tr><th scope="row">${index}</th><td>${loopFormat(target)}</td><td>${loopFormat(prediction)}</td><td>${loopSigned(error)}</td><td>${loopFormat(error * error, 3)}</td></tr>`;
    }).join('');
    find('mse').textContent = loopFormat(squaredErrorSum / 3, 3);
  };
  const showStep = index => {
    active = Math.max(0, Math.min(LOOP_STEPS.length - 1, index));
    const step = LOOP_STEPS[active];
    find('step-counter').textContent = `Step ${active + 1} of ${LOOP_STEPS.length}`;
    find('step-code').textContent = step.code;
    find('step-title').textContent = step.title;
    find('step-body').innerHTML = step.body;
    find('page-label').textContent = `${active + 1} / ${LOOP_STEPS.length}`;
    find('prev').disabled = active === 0;
    find('next').textContent = active === LOOP_STEPS.length - 1 ? 'Back to start ↻' : 'Next step →';
    find('policy-panel').hidden = active !== 4;
    find('value-panel').hidden = active !== 5;
    mount.querySelectorAll('[data-loop-step]').forEach(button => {
      const current = Number(button.dataset.loopStep) === active;
      button.classList.toggle('loop-current', current);
      if (current) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
  };
  mount.querySelectorAll('[data-loop-step]').forEach(button => button.addEventListener('click', () => showStep(Number(button.dataset.loopStep))));
  find('prev').addEventListener('click', () => showStep(active - 1));
  find('next').addEventListener('click', () => showStep((active + 1) % LOOP_STEPS.length));
  find('gamma').addEventListener('input', event => { gamma = Number(event.target.value); updateNumbers(); });
  find('pass').addEventListener('input', event => { pass = Number(event.target.value); updateNumbers(); });
  find('epsilon').addEventListener('input', event => { epsilon = Number(event.target.value); updateNumbers(); });
  find('fit').addEventListener('input', event => { fit = Number(event.target.value); updateNumbers(); });
  updateNumbers();
  showStep(0);
}
