# SO-101 Embodied Agent

A safety-first, extensible scaffold for a single SO-101 robot. The browser architecture page is in `docs/so101_embodied_agent.html`.

## Layers

- Brain: provider interface for GPT-6 Astra or another planner. The brain selects registered skills; it does not emit joint targets.
- In-context learning: retrieve a small number of approved, task-matched demonstrations. Demonstrations for policy training must pair video with timestamp-aligned action and state.
- Skills and VLA router: versioned skills use one compatible policy adapter per action segment; routing checks embodiment, modalities, action schema and readiness.
- Robot backend: SO-101 transport is injected through `SO101Transport`; actual LeRobot port, cameras and calibration remain machine-specific.
- Safety: reject stale or mismatched observations/actions, calibrated joint-limit violations and oversized chunks. Initial adapter permits one action step only.
- RSI: compare candidates on the same fixed seeds and verifier; reject hard safety events, require non-regression and human approval before explicit artifact activation.

## Local preview

From this directory:

```bash
python3 -m http.server 8050 --bind 127.0.0.1
```

Open `http://127.0.0.1:8050/docs/so101_embodied_agent.html`.

## Current scope

This repository contains interface contracts, a single-step runtime, policy routing, a skill allow-list, a safety validator, append-only episode events and an RSI promotion gate. It does not include a GPT provider implementation, a demo indexer, a configured LeRobot serial transport, compatible SO-101 VLA weights, or a simulator evaluator. The safety profile template intentionally has empty joint limits and therefore cannot authorize motion.

The existing HoloBrain weights in the source project target other embodiments and must not be assumed compatible with SO-101. Populate calibration and action units from the actual robot driver before implementing the transport.
