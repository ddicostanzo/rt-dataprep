# Findings: --workers auto (#175)

Spec: docs/specs/2026-11-08-auto-workers.md. Plan: docs/plans/auto-workers/plan.json (5 tasks; groups 01 → 02 → 03+04 → 05).
Append under `## <task-id>`; never rewrite earlier sections.

## orchestrator
- task-01 and task-02 run in one implementer on feat/auto-workers (both touch pool.py, strictly sequential); one commit per task.
- Vault guardrails: [[measure-a-phase-peak-rss-by-resetting-vmhwm]] (ru_maxrss is lifetime HWM; overestimate by design), [[dataloader-cpu-bound-on-hyperthreaded-cores]] (ITK thread oversubscription: deferred).
- OSC enforces cgroup v2 memory.max (job 8403365).

## task-01
- available_ram(root='/') walks the cgroup path and all ancestors (incl. the cgroup root); a v1 memory line wins over a `0::` line (hybrid hosts). An unreadable level (missing memory.max/current/stat) is no limit for that level; a missing inactive key counts as 0 (conservative).
- size_workers uses integer math (4*available // (5*peak)) so the 0.8 floor has no float rounding at exact boundaries.
- Private helpers (_read, _cgroup_free) are exercised only through the public readers in tests, so the commit hook's private-access override was not needed.
- tests/test_pool.py alone takes ~2.5 min (spawned pools), full suite ~3m45s.
