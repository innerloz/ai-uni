export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

export function ppoScore(ratio, advantage, epsilon) {
  const clippedRatio = clamp(ratio, 1 - epsilon, 1 + epsilon);
  const raw = ratio * advantage;
  const capped = clippedRatio * advantage;
  return { ratio, clippedRatio, raw, capped, score: Math.min(raw, capped) };
}

export function bernoulliKL(oldProbability, newProbability) {
  if (!(oldProbability > 0 && oldProbability < 1 && newProbability > 0 && newProbability < 1)) throw new RangeError('Probabilities must be strictly between 0 and 1.');
  return oldProbability * Math.log(oldProbability / newProbability) + (1 - oldProbability) * Math.log((1 - oldProbability) / (1 - newProbability));
}

export function sampleGradient(actionA, baseline = 0) {
  return ((actionA ? 10 : 9) - baseline) * (actionA ? 0.5 : -0.5);
}
