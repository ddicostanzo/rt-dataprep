# Security policy

rt-dataprep processes medical images and radiotherapy exports. Its security priorities are preventing data disclosure,
cross-patient or cross-study association, unsafe file writes, and silent corruption of research outputs.

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/ddicostanzo/rt-dataprep/security/advisories/new) if that
option is available for your repository access. If GitHub does not offer it, contact [@ddicostanzo](https://github.com/ddicostanzo)
through an established private channel to arrange a secure reporting destination. Do not place sensitive details in a
public issue, pull request, commit, or discussion.

Include the affected commit/version, dependency versions, impact, affected operation, and a minimal **synthetic**
reproducer. Describe expected and observed behavior. Share only sanitized error class/module/line details. No response
deadline or supported-release service level is promised; fixes target the current `main` branch. Older snapshots should
be evaluated against current source before assuming they contain a fix.

Do not submit patient DICOM, NIfTI, screenshots, real identifiers, per-case CSVs, manifests, splits, or unrestricted logs.
If patient data may have been exposed, follow your institution's incident process and preserve evidence in its approved
systems. A software issue report is not a substitute for that process.

## Data classification and boundaries

| Artifact | Handling requirement |
| --- | --- |
| Source and organized DICOM trees | Treat as original patient data; organization copies files without de-identifying them. |
| Converted/preprocessed volumes and applicator coordinates | Treat as sensitive derived medical data, including possible identifying image content. |
| Patient folders, reports, `series.json`, `manifest.jsonl`, `splits.json` and dated copies | Treat as sensitive metadata: identifiers, dates, labels, descriptions, or linkage can remain. |
| Logs and exception text | Intended to contain counts, fixed reasons, and safe class/site details; inspect before sharing. |
| Synthetic examples and tests | Generate at runtime; keep the synthetic naming convention and do not replace them with clinical files. |

Sanitizing PatientID for a filename does not anonymize it. UID values, dates, ROI names, and descriptions are retained in
per-case metadata. The tool neither performs DICOM de-identification nor certifies compliance with a privacy standard.
It provides no authentication, authorization, encryption, retention policy, or access audit service.

## Operating safely

- Run in approved, access-controlled storage. Apply institution-managed filesystem permissions, encryption, backups,
  retention, and deletion procedures to inputs, outputs, scratch space, and reports.
- Use a least-privilege account and a dedicated output root outside its input tree. Source exports are read or copied;
  converting or preprocessing must not overwrite the preceding stage.
- Keep output and scratch paths private. Intermediate mask generation can create temporary DICOM links/copies under a
  patient's partial output. Restrict access to the entire tree, including hidden files.
- Use one writer per output root. Atomic file writes and partial-folder promotion do not make concurrent pipeline runs
  against the same output tree safe.
- Treat DICOM and NIfTI as untrusted parser input. Use reviewed dependency versions, appropriate resource limits, and
  isolated execution for unfamiliar datasets. This software does not sandbox third-party parsers.
- Pin the reviewed commit and lockfile; assess dependency/security updates before reprocessing established cohorts.
- Review refusals, skips, image geometry, dose units, masks, and split membership before accepting an analysis dataset.
  A completed process can contain refused cases. Clinical use requires separate local validation and governance.

## Development rules that protect data

No real patient data, identifiers, report files, notebooks with outputs, or credentials belong in this repository.
`.gitignore` and the [pre-commit guard](.githooks/pre-commit) provide additional checks, not a guarantee. Review the staged
diff and paths before committing. See [CONTRIBUTING.md](CONTRIBUTING.md).

When changing diagnostic paths, preserve fixed refusal reasons and suppress third-party output for both reading and
lazy value access. Never add raw exception messages, Dataset dumps, paths, dates, UIDs, or patient-bearing metadata to
logs. Add synthetic regression coverage for the relevant output channels and fail-closed behavior.
