import test from 'node:test';
import assert from 'node:assert/strict';
import { ppoScore, bernoulliKL, sampleGradient } from '../dist/math.mjs';
import { grpoGroupStats } from '../dist/grpo.js';
import { nanoExampleStats } from '../dist/nano.js';

const close = (actual, expected, tolerance = 1e-10) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} should be ${expected}`);

test('PPO worked examples preserve the advantage sign and the smaller branch', () => {
  close(ppoScore(1.5, 2, 0.2).score, 2.4);
  close(ppoScore(1.5, -2, 0.2).score, -3);
  close(ppoScore(0.5, 2, 0.2).score, 1);
  close(ppoScore(0.5, -2, 0.2).score, -1.6);
  close(ppoScore(1, 0, 0.2).score, 0);
});

test('PPO plateaus only in the favorable direction; harmful changes remain visible', () => {
  for (const eps of [0.05, 0.1, 0.2, 0.4]) {
    for (const advantage of [-5, -2, 0.25, 2, 5]) {
      for (let i = 0; i <= 100; i++) {
        const ratio = i / 25;
        const expected = advantage > 0 ? Math.min(ratio, 1 + eps) * advantage : Math.max(ratio, 1 - eps) * advantage;
        close(ppoScore(ratio, advantage, eps).score, expected);
      }
    }
  }
});

test('TRPO uses old-to-new KL and distinguishes the two budget examples', () => {
  close(bernoulliKL(0.5, 0.5), 0);
  close(bernoulliKL(0.5, 0.55), 0.005025167926750673);
  close(bernoulliKL(0.5, 0.6), 0.020410997260127586);
  assert.ok(bernoulliKL(0.5, 0.55) < 0.01);
  assert.ok(bernoulliKL(0.5, 0.6) > 0.01);
  assert.notEqual(bernoulliKL(0.2, 0.4), bernoulliKL(0.4, 0.2));
  assert.throws(() => bernoulliKL(0.5, 1), RangeError);
});

test('the baseline preserves the expected gradient and eliminates toy estimator variance', () => {
  close(sampleGradient(true), 5);
  close(sampleGradient(false), -4.5);
  close((sampleGradient(true) + sampleGradient(false)) / 2, 0.25);
  close(sampleGradient(true, 9.5), 0.25);
  close(sampleGradient(false, 9.5), 0.25);
});

test('GRPO graded and binary rewards give the displayed population z-scores', () => {
  const graded = grpoGroupStats([2, 4, 6, 8]);
  close(graded.mean, 5);
  close(graded.variance, 5);
  close(graded.std, Math.sqrt(5));
  [-3, -1, 1, 3].forEach((value, i) => close(graded.advantages[i], value / Math.sqrt(5), 1e-7));
  const binary = grpoGroupStats([0, 0, 0, 1]);
  close(binary.mean, 0.25);
  close(binary.std, Math.sqrt(0.1875));
  close(binary.advantages[0], -1 / Math.sqrt(3), 1e-7);
  close(binary.advantages[3], Math.sqrt(3), 1e-7);
});

test('GRPO equal rewards are finite and zero; shifting reward origin preserves advantages', () => {
  const equal = grpoGroupStats([5, 5, 5, 5]);
  assert.ok(equal.allEqual);
  assert.deepEqual(equal.advantages, [0, 0, 0, 0]);
  const a = grpoGroupStats([2, 4, 6, 8]);
  const b = grpoGroupStats([12, 14, 16, 18]);
  a.advantages.forEach((value, i) => close(value, b.advantages[i]));
  close(a.advantages.reduce((sum, value) => sum + value, 0), 0);
});

test('the tiny loss uses six valid tokens and a nonnegative, linearly weighted reference penalty', () => {
  const zero = nanoExampleStats(0);
  const one = nanoExampleStats(1);
  close(zero.mean, 1 / 3);
  close(zero.std, Math.sqrt(2 / 9));
  assert.equal(zero.validTokens, 6);
  close(zero.advantages.reduce((sum, a) => sum + a, 0), 0);
  close(zero.batchLoss, -0.2731602712, 1e-9);
  const klMean = one.tokenTerms.flat().reduce((sum, term) => { assert.ok(term.klSample >= 0); return sum + term.klSample; }, 0) / 6;
  close(one.batchLoss - zero.batchLoss, klMean);
  close(nanoExampleStats(0.5).batchLoss, (zero.batchLoss + one.batchLoss) / 2);
});
