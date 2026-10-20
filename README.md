# AI Uni: from PPO to GRPO

An interactive lesson based on five supplied lecture slides, with maths foundations, policy-gradient sampling, TRPO's KL budget, PPO clipping, a complete training walkthrough, GRPO group-relative advantages, a token-mask/loss walkthrough, and downloadable training and PyTorch-loss starters.

**Interactive lesson:** [Open on GitHub Pages](https://innerloz.github.io/ai-uni/) · [Existing Sites publication](https://ai-uni-ppo-lab.lorenzuberger.chatgpt.site/)

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

### GitHub Pages

The `.github/workflows/pages.yml` workflow publishes the existing `dist/` directory on every push to `main`. It does not install ML libraries or run model training. The lesson uses relative asset paths, so it also works under a GitHub Pages project path.

The repository is [innerloz/ai-uni](https://github.com/innerloz/ai-uni). After a successful deployment, the interactive lesson is available through GitHub Pages; its URL is shown under **Settings → Pages** and in the workflow's `github-pages` environment. GitHub Pages is configured to deploy from **GitHub Actions**.

### Existing Sites publication

The Sites project identity is recorded in `.openai/hosting.json`. Publish this same project on updates rather than creating another one. Its configured site origin is https://ai-uni-ppo-lab.lorenzuberger.chatgpt.site . Deployment success and access must be confirmed by the hosting tools before claiming publication.

## Sources

- OpenAI Spinning Up: policy gradients and PPO pseudocode.
- Schulman et al., *Trust Region Policy Optimization* (2015).
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017).
- Shao et al., *DeepSeekMath* (2024), introducing GRPO.
- DeepSeek-AI, *DeepSeek-R1* (2025).
- Hugging Face TRL versioned documentation for the practical starter.
- McGill-NLP's nano-aha-moment source for the two implementation slides. Its direct policy-gradient loss is distinguished from the clipped multi-pass objective.

The lesson links directly to primary sources, distinguishes old-policy snapshots from reference policies, and explains the limits of each simplified demonstration.
