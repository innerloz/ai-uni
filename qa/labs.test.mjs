// Tests event/state wiring with a small DOM test double. This is not visual QA.
import test from 'node:test';
import assert from 'node:assert/strict';
import { initLoop } from '../dist/loop.js';
import { initGRPO } from '../dist/grpo.js';

class Element {
  constructor(attrs = {}) {
    this.attrs = attrs;
    this.dataset = {};
    for (const [key, value] of Object.entries(attrs)) if (key.startsWith('data-')) this.dataset[key.slice(5).replace(/-([a-z])/g, (_, char) => char.toUpperCase())] = value;
    this.children = [];
    this.textContent = '';
    this.value = attrs.value ?? '';
    this.hidden = 'hidden' in attrs;
    this.style = {};
    this.listeners = {};
    const classes = new Set((attrs.class ?? '').split(' '));
    this.classList = { toggle: (name, state) => state ? classes.add(name) : classes.delete(name) };
  }
  set innerHTML(html) {
    this.html = html;
    this.children = [];
    for (const tag of html.matchAll(/<[a-z][a-z\d]*\b((?:[^"'>]|"[^"]*"|'[^']*')*)>/gi)) {
      const attrs = {};
      for (const attr of tag[1].matchAll(/([^\s=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s]+)))?/g)) attrs[attr[1]] = attr[2] ?? attr[3] ?? attr[4] ?? '';
      this.children.push(new Element(attrs));
    }
  }
  get innerHTML() { return this.html; }
  querySelector(selector) { return this.children.find((child) => child.attrs.id === selector.slice(1)); }
  querySelectorAll(selector) { return this.children.filter((child) => selector.startsWith('[') && selector.slice(1, -1) in child.attrs); }
  addEventListener(type, handler) { this.listeners[type] = handler; }
  setAttribute(key, value) { this.attrs[key] = value; }
  removeAttribute(key) { delete this.attrs[key]; }
  emit(type, value) { if (value !== undefined) this.value = value; this.listeners[type]({ target: this }); }
}

test('training loop responds to gamma, policy passes, critic fit, and step navigation', () => {
  const mount = new Element();
  globalThis.document = { querySelector: () => mount };
  initLoop();
  const get = (suffix) => mount.querySelector(`#loop-${suffix}`);
  assert.ok(get('return-formulas').innerHTML.includes('2.62'));
  assert.equal(get('mse').textContent, '0.214');
  get('gamma').emit('input', '0');
  assert.ok(get('return-formulas').innerHTML.includes('= 1.00'));
  get('gamma').emit('input', '1');
  assert.ok(get('return-formulas').innerHTML.includes('= 3.00'));
  get('gamma').emit('input', '0.9');
  const actor = mount.querySelectorAll('[data-loop-step]').find((node) => node.dataset.loopStep === '4');
  actor.emit('click');
  assert.equal(get('policy-panel').hidden, false);
  assert.equal(get('value-panel').hidden, true);
  get('pass').emit('input', '4');
  assert.equal(get('batch-mean').textContent, '+0.39');
  const fixedAdvantages = get('trajectory-rows').innerHTML;
  get('next').emit('click');
  assert.equal(get('value-panel').hidden, false);
  get('fit').emit('input', '100');
  assert.equal(get('mse').textContent, '0.000');
  assert.equal(get('trajectory-rows').innerHTML, fixedAdvantages);
});

test('GRPO reward inputs and equal/binary presets update all relative scores', () => {
  const mount = new Element();
  globalThis.document = { querySelector: () => mount };
  initGRPO();
  const get = (suffix) => mount.querySelector(`#grpo-${suffix}`);
  assert.equal(get('mean').textContent, '5.000');
  assert.equal(get('std').textContent, '2.236');
  const presets = mount.querySelectorAll('[data-grpo-preset]');
  presets.find((node) => node.dataset.grpoPreset === 'equal').emit('click');
  assert.equal(get('std').textContent, '0.000');
  assert.equal(get('advantage-0').textContent, '+0.000');
  assert.ok(get('state-note').textContent.includes('no within-group'));
  presets.find((node) => node.dataset.grpoPreset === 'binary').emit('click');
  assert.equal(get('mean').textContent, '0.250');
  assert.equal(get('advantage-3').textContent, '+1.732');
  get('reward-0').emit('input', '2');
  assert.equal(get('mean').textContent, '0.750');
});
