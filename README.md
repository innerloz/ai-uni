# AI Uni: from PPO to GRPO

An interactive lesson based on three supplied lecture slides, with maths foundations, policy-gradient sampling, TRPO's KL budget, PPO clipping, a complete training walkthrough, GRPO group-relative advantages, and a downloadable training starter.

All project source, copied slides, checks, deployment configuration, and local temporary files live in this folder.

## Run the lesson locally

This is a buildless static site with no JavaScript dependencies:

```bash
cd /home/lorenz/innersight/ai-uni
python3 -m http.server 4173 --directory dist
```

Open `http://localhost:4173`. Serve the folder over HTTP so the browser can load JavaScript modules.

## Checks

```bash
node --test qa/math.test.mjs qa/labs.test.mjs
python3 qa/check_assets.py
python3 dist/examples/train_grpo.py --self-test
```

The installed Node executable in this environment is `/home/lorenz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node`.

Calculators are teaching examples. The downloadable training script requires its documented ML environment and a compatible GPU; no model training was run as part of authoring this lesson. Browser inspection depends on browser tooling being available.

## Hosting

The Sites project identity is recorded in `.openai/hosting.json`. Publish this same project on updates rather than creating another one. Its configured site origin is https://ai-uni-ppo-lab.lorenzuberger.chatgpt.site . Deployment success and access must be confirmed by the hosting tools before claiming publication.

## Sources

- OpenAI Spinning Up: policy gradients and PPO pseudocode.
- Schulman et al., *Trust Region Policy Optimization* (2015).
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017).
- Shao et al., *DeepSeekMath* (2024), introducing GRPO.
- DeepSeek-AI, *DeepSeek-R1* (2025).
- Hugging Face TRL versioned documentation for the practical starter.

The lesson links directly to primary sources, distinguishes old-policy snapshots from reference policies, and explains the limits of each simplified demonstration.
