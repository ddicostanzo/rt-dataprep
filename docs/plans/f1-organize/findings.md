# F1 organize: shared findings

Spec: `docs/plans/2026-11-03-rt-dataprep/phase1.md` (section F1) and `proposal.md`. Plan: `plan.json` here.
Design was settled interactively in the planning session (2026-11-03); no separate spec file.

- Source functions to port live in `/home/dominic/Repositories/rnt_transformer` and `/home/dominic/Repositories/DICOM_Transport` (read-only; never edit them).
- Vault trap: pydicom validates lazily on first tag access; quiet the whole access (see project-profile.yml review rules).
- PHI: tests use only synthetic data generated in tmp_path.

## Decisions (orchestrator)
- No separate fresh-context plan-review: phase1.md already had two advisor passes; plan.json had one more.
- Tasks run sequentially in the session worktree (task-02 depends on task-01), not per-task worktrees.
- DicomRTTool, SimpleITK and the mclahe extra are deferred to F2: mclahe pins numpy 1.26 in a universal uv lock.
- Implementer never sets CLAUDE_ALLOW_PRIVATE_API; any third-party private access is reported to the orchestrator.
