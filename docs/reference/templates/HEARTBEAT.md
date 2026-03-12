---
title: "HEARTBEAT.md Template"
summary: "Workspace template for HEARTBEAT.md"
read_when:
  - Bootstrapping a workspace manually
---

# HEARTBEAT.md

Always-on control loop. Keep this short and deterministic.

## State

- Track mode in `memory/always-on-state.json`.
- Allowed modes: `sleeping`, `working`, `researching`.
- Each heartbeat: read state, choose mode intentionally, then update state.

## Working Mode Contract

When mode is `working`, each cycle must do exactly one branch:

1. `trail`: follow the active trail in `memory/trails/` and complete the next concrete step.
2. `priorities`: check `memory/priorities.md`, pick highest-priority item, then create/refresh active trail.

Do not run a working heartbeat without choosing `trail` or `priorities`.

## Mode Guidance

- `sleeping`: only urgent checks/actions. If nothing urgent, reply `HEARTBEAT_OK`.
- `researching`: collect info and write findings; switch to `working` when enough context exists.
- `working`: execute one branch from Working Mode Contract and record progress.
