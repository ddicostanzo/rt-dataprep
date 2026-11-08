# Contributing to rt-dataprep

Contributions should keep image geometry and patient/study association correct, preserve source files, and keep patient
information out of code, diagnostics, and review artifacts. Start with the [README](README.md), [security policy](SECURITY.md),
and the relevant specifications under [docs/specs/](docs/specs/).

## Set up a checkout

Use Python 3.12 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/ddicostanzo/rt-dataprep.git
cd rt-dataprep
uv sync --locked --group dev
git switch -c fix/short-description
```

Install the PHI guard once per checkout on systems supporting Git hook symlinks:

```bash
ln -s ../../.githooks/pre-commit .git/hooks/pre-commit
```

Inspect an existing hook before changing it. If your system cannot create symlinks, install an executable copy of
`.githooks/pre-commit` at `.git/hooks/pre-commit` and keep it in sync. The guard checks staged file types and artifact paths;
it also runs the optional local content guard if present. It is an additional safeguard, not a substitute for reviewing
every staged change.

## Data and diagnostic rules

- Generate synthetic test data at runtime in `tmp_path` or an explicit disposable directory. Use `SYNTHETIC_` identifiers
  and freshly generated or deliberately synthetic UIDs. Reuse `tests/synthetic_dicom.py` where practical.
- Never commit clinical DICOM, converted arrays/images, per-case CSVs, manifests, splits, reports, logs, credentials,
  screenshots containing patient information, or notebooks with patient-bearing outputs.
- Never log patient IDs, names, dates, paths, UIDs, ROI labels, descriptions, or raw exception messages. Use fixed reason
  constants and, for unexpected failures, only safe error class/module/line details.
- Keep source trees read-only in behavior: copy when organizing, and write conversion/preprocessing to separate roots.
- Quiet third-party reads through their complete value-access lifetime. pydicom may validate lazily on first tag access;
  SimpleITK/GDCM can write to native file descriptors. See `dicom_io.py` before modifying these paths.
- Do not turn unexpected programming errors into ordinary case refusals. Preserve the existing abort/report behavior,
  cleanup, and privacy-safe exception handling.

Use [private reporting](SECURITY.md#reporting-a-vulnerability) for vulnerabilities. Ordinary bugs should include a synthetic
reproducer, expected result, affected commit, and safe failure details; do not attach patient-bearing artifacts.

## Make a focused change

1. Branch from current `main` and inspect the relevant tests and specification.
2. For a behavior fix, add a meaningful regression that fails before the fix. Cover the failure boundary, accounting,
   valid sibling behavior, and output/diagnostic privacy where applicable.
3. Keep valid conversion output stable unless the change explicitly requires otherwise. Geometry, voxels, mask labels,
   dose units, metadata rows, and deterministic ordering all matter.
4. Update the README and examples when commands or output contracts change. Keep generated artifacts out of the diff.
5. Avoid unrelated formatting, dependency churn, or changes that weaken characterization tests.

The [synthetic workflow](examples/synthetic_workflow.py) is a small end-to-end check. The
`tests/test_plan_regression.py` characterization tests pin valid plan behavior; investigate differences rather than
rewriting expectations merely to make a change pass.

## Verify

Run from the repository root:

```bash
uv run --group dev pytest -q --tb=short
uv run --group dev ruff check .
uv run --group dev python -m compileall -q src examples
git diff --check
```

Use `pytest -rs` to inspect skips. TorchIO is an optional comparison dependency used by selected tests; it is not required
by the runtime pipeline. Report skipped checks and environment limitations instead of claiming they passed. If changing
dependencies, update `pyproject.toml` and `uv.lock` together and explain why.

For documentation/example changes, also execute the documented workflow in a fresh synthetic workspace:

```bash
uv run python examples/synthetic_workflow.py /tmp/rt-dataprep-contribution-demo
uv run python examples/custom_preprocess.py /tmp/rt-dataprep-contribution-demo/converted /tmp/rt-dataprep-custom-demo --shape 16 16 8
```

Choose new directories for each run. Do not remove an existing clinical or analysis output just to reuse an example path.

## Open a pull request

Before committing, inspect `git diff`, `git diff --cached`, and `git status --short`. Include only the intended source,
tests, documentation, or synthetic generator code. A `.gitignore` entry does not make a file safe to commit.

Describe the problem, resulting behavior, relevant issue, compatibility implications, and exact validation results.
Explain any skipped checks. Use synthetic examples in the PR description. Keep implementation and documentation changes
separate when that makes review easier. Repository merges normally use squash commits.

## Cut a release

Users install from a Git tag, so a release is a version bump plus a tag on `main`:

1. Merge the release's changes, then bump `version` in `pyproject.toml` in its own PR (semantic versioning: a
   breaking CLI, API, or output-layout change bumps the minor version while the major version is 0).
2. On the updated `main`, run `uv build` and install the wheel into a fresh Python 3.12 environment. Check that
   `rt-dataprep --help` runs and that `examples/synthetic_workflow.py` finishes with no refusals.
3. Tag and push: `git tag -a vX.Y.Z -m "rt-dataprep X.Y.Z" && git push origin vX.Y.Z`.
4. Update the install tag in the README if it names an older release.

Share the wheel, not the source archive, if someone cannot use Git. The source archive is built from the working tree, so
check it with `tar tzf dist/*.tar.gz` before it leaves the machine. `docs/handoffs/` is excluded from it.

## License

By submitting a contribution, you agree to license your contribution under this project's GNU Affero General Public
License version 3 only (`AGPL-3.0-only`), as provided in [LICENSE](LICENSE). Preserve existing attribution and identify the
origin and license of any third-party code you propose adding. The project license grants no rights to clinical datasets.
