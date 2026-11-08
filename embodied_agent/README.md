# SO-101 Embodied Agent Scaffold

This is the extension seam for the proposed single-SO-101 agent. It keeps the
high-level brain, demonstration memory, versioned skills, VLA policies, robot
backend, safety checks, and offline RSI gates behind separate contracts.

## Proposed integration

- `Brain`: use a provider adapter for GPT-6 Astra to choose only registered
  skills. It returns structured `SkillCall` data; it never emits motor values.
- `DemoMemory`: retrieve operator-approved episodes for in-context learning.
  A demonstration should include video plus timestamp-aligned joint state and
  action data. Video without actions is context, not a supervised control label.
- `Skill`: a versioned behavior with stated preconditions, postconditions,
  embodiment, action schema, time budget, and approval requirements.
- `SkillRegistry`: stores verified versions separately from the active version;
  activation is an explicit deployment operation, so RSI never hot-swaps code.
- `PolicyRouter`: choose only a ready VLA adapter whose declared embodiment,
  sensors, action schema, and skill match. Add SO-101-specific models here.
- `RobotBackend`: wrap the installed LeRobot SO-101 transport or simulator.
  This repo does not yet contain a tested SO-101 hardware adapter.
- `SafetySupervisor`: reject stale observations, joint-order/shape mismatches,
  non-finite values, calibrated-limit violations, and excessive action changes.
  Rejection means no write to the motor bus and a backend safe stop.
- `EpisodeStore`: append-only metadata/events; raw MP4/MCAP and aligned action
  data are referenced by URI instead of overwritten.
- `rsi.assess_promotion`: require identical holdout seeds, enough episodes, no
  hard safety events, non-regressing success/latency, and human approval.
- `AgentRuntime.step`: one fresh observation leads to one bounded action chunk;
  an uncertain execution receipt is stopped and escalated, never replayed.

## Hardware boundary

No model or robot movement is enabled by importing this scaffold. Before SO-101
deployment, implement and validate a LeRobot adapter, read the actual joint
names/units/calibration, and populate a `SafetyProfile` from that calibration.
The current HoloBrain checkpoint is for Panda Droid data; no SO-101-compatible
checkpoint/adapter was found in the existing project. Do not route SO-101
commands to it until the embodiment and action schema have been trained and
validated.

## Suggested package growth

```text
embodied_agent/
  contracts.py          shared observations, actions, skills and records
  interfaces.py         provider, memory, policy, skill and robot contracts
  policy_router.py      VLA registry and capability-based selection
  safety.py             fail-closed command checks
  episode_store.py      immutable episode metadata and event log
  rsi.py                offline candidate comparison and promotion gate
  adapters/so101.py     calibrated single-step LeRobot transport boundary
  skills/               verified SO-101 skills and bounded runner
  brain/                GPT-6 Astra and alternate provider adapters
  memory/               demo indexing and ICL retrieval
  adapters/              simulator and existing HoloBrain adapters
  evaluation/           fixed-seed suite, verifier and reports
```

The reference implementation is a safety-first scaffold, not a production robot
driver. GPT provider calls, demo indexing, the simulator evaluator and the
LeRobot transport are injected interfaces. Add the physical adapter only after
the SO-101 serial device, cameras, calibration, joint units, and stop mechanism
are known.
