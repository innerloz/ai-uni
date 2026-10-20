import { ppoScore, bernoulliKL, sampleGradient } from './math.mjs';
import { initLoop } from './loop.js';
import { initGRPO } from './grpo.js';
import { initPractice } from './practice.js';
import { initNano } from './nano.js';

const $ = (id) => document.getElementById(id);
const pct = (x) => `${Math.round(x * 100)}%`;
const num = (x, digits = 2) => (Math.abs(x) < 1e-10 ? 0 : x).toFixed(digits);
const signed = (x, digits = 2) => `${x > 0 ? '+' : x < 0 ? '−' : ''}${num(Math.abs(x), digits)}`;

function updateBasics() {
  const p = Number($('basic-p').value) / 100;
  const expected = 8 * p + 2 * (1 - p);
  $('basic-p-value').textContent = pct(p);
  $('right-prob').textContent = pct(p);
  $('left-prob').textContent = pct(1 - p);
  $('right-bar').style.width = pct(p);
  $('left-bar').style.width = pct(1 - p);
  $('expected-reward').textContent = num(expected);
  $('expectation-working').textContent = `With p = ${num(p)}, that is ${num(p)} × 8 + ${num(1 - p)} × 2 = ${num(expected)}. Across many visits, the average approaches this value.`;
  $('advantage-working').textContent = `At your current policy, V(s) = ${num(expected)}, so A(right) = ${signed(8 - expected)} and A(left) = ${signed(2 - expected)}.`;
}
$('basic-p').addEventListener('input', updateBasics);
updateBasics();

const tabs = [...document.querySelectorAll('.foundation-tabs [role=tab]')];
function activateTab(tab) {
  tabs.forEach((other) => {
    const selected = other === tab;
    other.setAttribute('aria-selected', String(selected));
    other.tabIndex = selected ? 0 : -1;
    $(other.getAttribute('aria-controls')).hidden = !selected;
  });
}
tabs.forEach((tab, i) => {
  tab.addEventListener('click', () => activateTab(tab));
  tab.addEventListener('keydown', (event) => {
    let next;
    if (event.key === 'ArrowRight') next = (i + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (i + tabs.length - 1) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next !== undefined) { event.preventDefault(); activateTab(tabs[next]); tabs[next].focus(); }
  });
});

let samples = [true, false, true, true, false, false, true, false, true, true, false, true];
function renderGradients() {
  const useBaseline = $('pg-baseline').checked;
  const grads = samples.map((action) => sampleGradient(action, useBaseline ? 9.5 : 0));
  $('gradient-samples').innerHTML = grads.map((g, i) => `<div class="gradient-chip ${g < 0 ? 'negative' : ''}"><small>Action ${samples[i] ? 'A' : 'B'}</small><b>${signed(g)}</b></div>`).join('');
  $('pg-mean').textContent = signed(grads.reduce((a, b) => a + b, 0) / grads.length);
  $('pg-note').textContent = useBaseline ? 'Same sampled actions, much steadier estimates. This deliberately simple example has zero estimator variance after subtracting the baseline.' : 'A small batch can point in the wrong direction by chance. Resample a few times and watch the batch mean move; the true expected gradient stays +0.25.';
}
$('pg-baseline').addEventListener('change', renderGradients);
$('resample').addEventListener('click', () => { samples = Array.from({ length: 12 }, () => Math.random() < 0.5); renderGradients(); });
renderGradients();

function updateTRPO() {
  const q = Number($('trpo-q').value) / 100;
  const delta = Number($('trpo-delta').value);
  const kl = bernoulliKL(0.5, q);
  $('trpo-q-value').textContent = pct(q);
  $('trpo-delta-value').textContent = num(delta, 3);
  $('trpo-a').style.width = pct(q);
  $('trpo-b').style.width = pct(1 - q);
  $('trpo-kl').textContent = num(kl, 5);
  $('trpo-score').textContent = signed(2 * q - 1);
  const within = kl <= delta;
  $('trpo-status').classList.toggle('over', !within);
  $('trpo-status').textContent = within ? `✓ Within the KL budget${q < 0.5 ? ', but this proposal lowers the score.' : q === 0.5 ? ' — the policy is unchanged.' : ' — this proposal increases the score.'}` : '↗ Over the KL budget — TRPO would need a smaller step.';
}
['trpo-q', 'trpo-delta'].forEach((id) => $(id).addEventListener('input', updateTRPO));
updateTRPO();

let advantage = 2;
function drawChart(ratio, epsilon) {
  const width = 660, height = 330;
  const left = 53, right = 21, top = 38, bottom = 51;
  const xmax = Math.max(2, Math.ceil((ratio + 0.15) * 2) / 2);
  const maxY = Math.max(4, Math.ceil(2 * xmax));
  const ymin = advantage > 0 ? -0.3 : -maxY;
  const ymax = advantage > 0 ? maxY : 0.3;
  const x = (v) => left + v / xmax * (width - left - right);
  const y = (v) => top + (ymax - v) / (ymax - ymin) * (height - top - bottom);
  const low = 1 - epsilon, high = 1 + epsilon;
  let s = `<title id="ppo-chart-title">PPO score for advantage ${advantage > 0 ? '+2' : '−2'}</title><desc id="ppo-chart-desc">Ratio ${num(ratio)}. The solid green score is flat ${advantage > 0 ? `above ${num(high)}` : `below ${num(low)}`}. Changes in the harmful direction still lower the score.</desc>`;
  s += `<rect x="${x(low)}" y="${top}" width="${x(high) - x(low)}" height="${height - top - bottom}" fill="#edf3e4"/>`;
  const yticks = advantage > 0 ? [0, maxY / 2, maxY] : [-maxY, -maxY / 2, 0];
  yticks.forEach((t) => { s += `<line x1="${left}" y1="${y(t)}" x2="${width - right}" y2="${y(t)}" stroke="#e5e8df"/><text x="${left - 12}" y="${y(t) + 4}" text-anchor="end" font-size="10" fill="#7b877b">${num(t, Number.isInteger(t) ? 0 : 1)}</text>`; });
  s += `<line x1="${x(1)}" y1="${top}" x2="${x(1)}" y2="${height - bottom}" stroke="#bbc4b1" stroke-dasharray="4 5"/>`;
  s += `<text x="${left}" y="18" font-size="11" fill="#657563">Score L · higher is better ↑</text>`;
  s += `<line x1="${x(0)}" y1="${y(0)}" x2="${x(xmax)}" y2="${y(xmax * advantage)}" stroke="#a195bf" stroke-width="2" stroke-dasharray="6 5"/>`;
  const knee = advantage > 0 ? high : low;
  const points = [0, knee, xmax].map((r) => `${x(r)},${y(ppoScore(r, advantage, epsilon).score)}`).join(' ');
  s += `<polyline points="${points}" fill="none" stroke="#147d6b" stroke-width="3.5" stroke-linejoin="round"/>`;
  s += `<line x1="${x(low)}" y1="${top}" x2="${x(low)}" y2="${height - bottom}" stroke="#bdcdb0" stroke-dasharray="3 5"/><line x1="${x(high)}" y1="${top}" x2="${x(high)}" y2="${height - bottom}" stroke="#bdcdb0" stroke-dasharray="3 5"/>`;
  const tickValues = [0, 0.5, 1, 1.5, 2].filter((t) => t <= xmax);
  if (xmax > 2) tickValues.push(xmax);
  tickValues.forEach((t) => { s += `<text x="${x(t)}" y="${height - bottom + 22}" text-anchor="middle" font-size="10" fill="#7b877b">${num(t, 1)}</text>`; });
  s += `<text x="${(left + width - right) / 2}" y="${height - 6}" text-anchor="middle" font-size="11" fill="#657563">Probability ratio r = new ÷ old</text>`;
  s += `<text x="${x(low) - 4}" y="${top - 10}" text-anchor="end" font-size="10" fill="#658353">1 − ε</text><text x="${x(high) + 4}" y="${top - 10}" text-anchor="start" font-size="10" fill="#658353">1 + ε</text>`;
  const score = ppoScore(ratio, advantage, epsilon).score;
  s += `<line x1="${x(ratio)}" y1="${y(score)}" x2="${x(ratio)}" y2="${height - bottom}" stroke="#147d6b" stroke-width="1.2" stroke-dasharray="2 4"/><circle cx="${x(ratio)}" cy="${y(score)}" r="6" fill="#d1ee9e" stroke="#173d36" stroke-width="2"/>`;
  $('ppo-chart').innerHTML = s;
}

function updatePPO() {
  const oldP = Number($('ppo-old').value) / 100;
  const newP = Number($('ppo-new').value) / 100;
  const eps = Number($('ppo-epsilon').value);
  const result = ppoScore(newP / oldP, advantage, eps);
  $('ppo-old-value').textContent = pct(oldP);
  $('ppo-new-value').textContent = pct(newP);
  $('ppo-epsilon-value').textContent = num(eps);
  $('positive-a').classList.toggle('selected', advantage > 0);
  $('negative-a').classList.toggle('selected', advantage < 0);
  $('positive-a').setAttribute('aria-pressed', String(advantage > 0));
  $('negative-a').setAttribute('aria-pressed', String(advantage < 0));
  $('ppo-ratio').textContent = num(result.ratio);
  $('ppo-raw').textContent = signed(result.raw);
  $('ppo-capped').textContent = signed(result.capped);
  $('ppo-final').textContent = signed(result.score);
  const flat = advantage > 0 ? result.ratio > 1 + eps + 1e-10 : result.ratio < 1 - eps - 1e-10;
  const harmful = advantage > 0 ? result.ratio < 1 - 1e-10 : result.ratio > 1 + 1e-10;
  $('ppo-state').textContent = flat ? 'Incentive capped' : harmful ? 'Harmful direction' : 'Incentive active';
  $('ppo-state').classList.toggle('harmful', harmful);
  let explanation;
  if (flat && advantage > 0) explanation = `This action was better than expected. Increasing its probability helps, but past r = ${num(1 + eps)} this sample’s PPO score stays at ${signed(result.score)}. There is no extra incentive here to push it further.`;
  else if (flat) explanation = `This action was worse than expected. Decreasing its probability helps, but below r = ${num(1 - eps)} this sample’s PPO score stays at ${signed(result.score)}. The negative score becoming less negative counts as improvement.`;
  else if (harmful) explanation = `This moves a ${advantage > 0 ? 'good action’s probability down' : 'bad action’s probability up'}. The minimum keeps the original score, ${signed(result.raw)}, so clipping does not hide this harmful change. Maximizing the score pushes back.`;
  else explanation = `The original score wins (or ties): min(${signed(result.raw)}, ${signed(result.capped)}) = ${signed(result.score)}. ${advantage > 0 ? 'Increasing' : 'Decreasing'} this action’s probability can still improve the objective until the incentive cap.`;
  $('ppo-explanation').textContent = explanation;
  const lowP = oldP * (1 - eps) * 100, highP = oldP * (1 + eps) * 100;
  $('ppo-percentage-note').innerHTML = `<b>${Math.round(eps * 100)}% relative change, not percentage points.</b> With an old probability of ${pct(oldP)}, the ratio band [${num(1 - eps)}, ${num(1 + eps)}] corresponds to action probabilities ${num(lowP, 1)}%–${num(highP, 1)}%. This interval illustrates the objective’s clipping, not a hard limit on the new probability.`;
  drawChart(result.ratio, eps);
}
['ppo-old', 'ppo-new', 'ppo-epsilon'].forEach((id) => $(id).addEventListener('input', updatePPO));
$('positive-a').addEventListener('click', () => { advantage = 2; updatePPO(); });
$('negative-a').addEventListener('click', () => { advantage = -2; updatePPO(); });
function preset(type) {
  $('ppo-old').value = 20; $('ppo-epsilon').value = 0.2;
  $('ppo-new').value = type === 'bad-down' ? 10 : 30;
  advantage = type === 'good-up' ? 2 : -2;
  updatePPO();
}
document.querySelectorAll('[data-preset]').forEach((button) => button.addEventListener('click', () => preset(button.dataset.preset)));
$('ppo-reset').addEventListener('click', () => preset('good-up'));
updatePPO();

initLoop();
initGRPO();
initPractice();
initNano();

const questions = [
  { q: 'A good action has A = +2. Its ratio rises from 1.2 to 1.5 with ε = 0.2. What happens to its PPO score?', a: ['It rises from 2.4 to 3.0.', 'It stays at 2.4.', 'The update becomes impossible.'], correct: 1, why: 'min(1.5 × 2, 1.2 × 2) = 2.4. The incentive is flat beyond 1.2; the ratio itself can still exceed that value.' },
  { q: 'A bad action has A = −2 and r = 1.5. With ε = 0.2, which score does PPO keep?', a: ['−3.0, the original score.', '−2.4, the capped comparison.', '+3.0, because it is a penalty.'], correct: 0, why: 'min(−3.0, −2.4) = −3.0. Making a bad action more likely is harmful, so PPO does not soften that penalty.' },
  { q: 'During several policy-update passes on one batch, which quantities stay fixed?', a: ['The new action probabilities.', 'The old log probabilities and advantage estimates.', 'All model weights.'], correct: 1, why: 'Freeze the batch reference while optimizing θ. Recalculate new log probabilities as the policy changes, then collect a fresh batch on the next outer iteration.' },
  { q: 'GRPO samples four answers to one prompt. All receive the same reward. What group-relative learning signal remains?', a: ['The best answer gets a positive advantage.', 'Their relative advantages are zero.', 'The critic supplies the missing signal.'], correct: 1, why: 'There is no reward variation within the group. With a numerical guard for zero standard deviation, all relative advantages are zero. A separate KL term may still contribute if it is enabled; GRPO has no learned critic to fill this gap.' }
];
$('quiz-mount').innerHTML = questions.map((question, i) => `<article class="quiz-card"><span class="mini-label">CHECK ${String(i + 1).padStart(2, '0')}</span><h3 id="quiz-question-${i}">${question.q}</h3><div role="group" aria-labelledby="quiz-question-${i}">${question.a.map((answer, j) => `<button data-question="${i}" data-answer="${j}" aria-pressed="false">${answer}</button>`).join('')}</div><div class="quiz-feedback" id="quiz-feedback-${i}" aria-live="polite">Choose an answer to reveal the explanation.</div></article>`).join('');
document.querySelectorAll('[data-question]').forEach((button) => button.addEventListener('click', () => {
  const i = Number(button.dataset.question), j = Number(button.dataset.answer), q = questions[i];
  document.querySelectorAll(`[data-question="${i}"]`).forEach((other) => { other.setAttribute('aria-pressed', String(other === button)); other.classList.remove('correct', 'incorrect'); });
  button.classList.add(j === q.correct ? 'correct' : 'incorrect');
  $(`quiz-feedback-${i}`).textContent = `${j === q.correct ? '✓ Exactly. ' : 'Try again. '}${q.why}`;
}));

const chapterLinks = [...document.querySelectorAll('.chapter-inner a')];
const chapters = chapterLinks.map((link) => document.querySelector(link.getAttribute('href')));
let scheduled = false;
function updateScroll() {
  const scrollable = document.documentElement.scrollHeight - window.innerHeight;
  $('reading-progress').style.width = `${scrollable > 0 ? window.scrollY / scrollable * 100 : 0}%`;
  let active = 0;
  chapters.forEach((section, i) => { if (section.getBoundingClientRect().top < 160) active = i; });
  chapterLinks.forEach((link, i) => { link.classList.toggle('active', i === active); if (i === active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current'); });
  scheduled = false;
}
window.addEventListener('scroll', () => { if (!scheduled) { scheduled = true; requestAnimationFrame(updateScroll); } }, { passive: true });
updateScroll();
