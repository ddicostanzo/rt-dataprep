# AHE, image support, and allocation-failure hardening

Implements #147 and the AHE correctness/configuration work in #154; completes allocation-failure handling in #53.
This extends the earlier AHE and aligned-series FOV specifications. Source DICOM/converted trees remain read-only.

## AHE

- Check voxel finiteness before entering AHE compute. Planning CT failures refuse the plan with a fixed reason; study
  failures refuse the individual series. Invalid aligned series are skipped and counted. No path, UID or voxel value is logged.
- A constant input becomes zero. When CLAHE turns a varying input into a constant (tiny tiles or two-level images), retain
  normalized input instead. Backend non-finite output is an internal failure, never a successful image write.
- Preserve default 128 bins, 0.05 clip limit and the backend's per-axis voxel window. Opt-in physical kernel widths use
  x/y/z mm, converted with ceil(width/spacing), minimum one voxel, then reversed for the z/y/x array. Configurable bins
  and clip limit are validated before creating outputs. Defaults preserve non-degenerate AHE results.
- Load the exposure backend when needed. Handle float32 subtraction overflow for otherwise finite extreme inputs.
- The `rescale_` prefix is processing history, not a claim that equalized output retains HU. Initial-resample edge fill
  remains part of the histogram; FOV masks do not implement masked CLAHE. Use AHE-only when no geometric steps are wanted.
- Real-data AHE peak-memory/throughput measurement remains in #47, including the planned OSC run. Do not infer a
  production memory requirement from tiny synthetic acceptance tests.

## Primary-image FOV masks

- Extend the existing opt-in `series_fov_mask` switch to planning CT and study CT/MR/PT. Off preserves output files,
  reports and manifests. On writes `fovmask_<output image>` immediately after its image in the report's file list.
- Begin with a native-grid uint8 all-ones image. Propagate through each enabled resample with nearest-neighbor and zero
  fill, then the same centered crop/pad with zero padding. Ignore intensity rescale/AHE. Grid, dtype and contents are
  tested on both resamples, odd crop/pad, anisotropic/rotated geometry, and genuine zero-valued data.
- Keep these auxiliary files out of `series.csv`. Aligned-series masks continue in the manifest's `series` JSON.
- Planning CT and study masks are discovered by header/grid checks, without voxel reads or following symlinks. A usable
  primary mask introduces a trailing `fov_mask` manifest column; other rows are blank. No usable primary mask means the
  original fourteen-column manifest is retained. Broken optional masks never remove otherwise usable cases.

## Allocation failures

- `MemoryError` and `OSError(ENOMEM)` become bare `MemoryError` outside the quiet block. Recognized case-insensitive
  ITK/C++ RuntimeError forms include allocation-failed, out-of-memory and `std::bad_alloc` wording. Ordinary input errors
  remain ordinary refusals. An unrecognized future backend message is not presumed to be an allocation error.
- All original messages and exception chains remain suppressed. Integration tests cover CT and dose reads, run abortion,
  report flushing, cleanup and diagnostic privacy. Existing failing-`str()` handling stays covered.
