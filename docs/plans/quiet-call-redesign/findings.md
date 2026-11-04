# quiet_call redesign: findings

Spec: docs/specs/2026-11-04-quiet-call-redesign.md. Plan: docs/plans/quiet-call-redesign/plan.json.
Baseline at 5977e72: 483 passed, 1 skipped.

## task-01

- Red before the change (tests/test_dicom_io.py, 36 collected, 5 failed): A4 failed for KeyboardInterrupt('SYNTHETIC_x'), SystemExit('SYNTHETIC_x'), the needs-an-argument BaseException subclass and GeneratorExit (original object leaked through); A5 failed (extra fd 12 left open after the second os.dup fails). A6 and the SystemExit(3) / SystemExit() / SystemExit(None) cases passed pre-change (pins).
- Decisions: the except BaseException sits after the existing except Exception inside fn's try; it records only (type, code), and the bare object is built and raised after the with/finally blocks. SystemExit subclasses map to SystemExit; SystemExit(None) and SystemExit() both come out as SystemExit() (None code). Construction failure of type(exc)() falls back to KeyboardInterrupt().
- _fds_quiet: dups are appended one at a time inside the try; the finally restores fds 1/2 only when both dups exist, then closes whatever was made.
- Gotchas: ruff PLW (stacklevel) wants stacklevel on the thread warn in the A6 test. The scan flags dotted private refs, so the A5 test reaches _fds_quiet via getattr(dicom_io, '_fds_quiet')  # noqa: B009.
- Result: 492 passed, 1 skipped (baseline 483 + 9 new cases); ruff and compileall clean.

## task-02

- Red before the change (tests/test_convert.py, 8 new cases, 5 failed): A1 failed (DID NOT RAISE ConvertAborted: the TypeError read as REASON_DOSE_UNREADABLE); A10 failed for all four cases (NaN frame offset, and NaN in ImagePositionPatient, ImageOrientationPatient and PixelSpacing: each converted without refusing). A2 passed pre-change (pin), for the 5-value IOP, the 1-value PixelSpacing and a 2-value IPP.
- Decisions: _read_dose is extract (quiet) / arithmetic (bare) / build (quiet). Extraction checks counts 6 / 3 / 2 and np.isfinite on the plain floats, raising a bare ValueError (refused by the wrapper). offsets is used by len() alone inside extraction; indexing and subtraction start in the arithmetic step. The offsets finiteness check sits in _check_dose_header after np.diff and before the grid test, so a NaN offset gives REASON_DOSE_GRID.
- Equivalence: reorder, origin, spacing, direction and the float64 scaling are the old statements moved unchanged, so output for a well-formed RTDOSE is the same; every existing dose test passes unmodified.
- Gotcha: pydicom reads a one-value DS back as a scalar (DSfloat), not a list; the old and new code both refuse it through the TypeError from iterating it. The A2 fixture check uses np.ravel for that reason.
- Result: 500 passed, 1 skipped (492 + 8 new cases); ruff and compileall clean.

## task-03

- Red before the change (tests/test_convert.py, 6 new cases, 6 failed): own-frame AttributeError, own-dict KeyError and the misspelled keyword (A9 a, c, d) were all refused instead of propagating; the two missing-tag cases (attribute and item) refused but wrote no INFO record; the end-to-end case with masks.classify patched to raise AttributeError did not abort. The two switched tests (KeyError to ValueError, test_convert.py refuse_after_the_plan_folder_exists and test_series.py test_a_tag_error_while_building_the_series_rows_refuses_the_plan_and_removes_the_folder) pass before and after.
- Decision (deviation from the plan's note): the propagated exception is re-raised after the except block, not by a bare raise inside it, because the existing guard test_no_case_refused_is_raised_inside_an_except_block forbids any raise inside an except handler in convert.py and may not change. The handler keeps the object, and the raise after it re-raises the same object (same class, args and traceback; innermost frame unchanged, so the abort path's crash site is the same).
- Typo rule: tag_for_keyword(getattr(exc, 'name', None)); a pydicom AttributeError with no name (None) also propagates, since None is no keyword (tag_for_keyword(None) is None, checked).
- Audit, attribute or method access on a value pydicom may return as None (empty numeric tags read as None; empty text tags as ''; empty sequences as an empty Sequence), in _assemble, _check_volume, _check_contour_images, _dose_candidates, _check_dose_header, masks.classify, _series_rows and the lambda at the FrameOfReferenceUID read:
  - Every tag value is first passed to str(), int(), float() or iterated: str(None) is text, int(None) / float(None) / iterating None raise TypeError, and TypeError is refused as REASON_MISSING_TAG (unchanged). Covered: ImageOrientationPatient, PixelSpacing, ImagePositionPatient, Rows, Columns, NumberOfFrames, GridFrameOffsetVector, ReferencedROINumber, ROINumber, ContourData.
  - The methods called after that (strip, lower, isalnum, reshape, sort, pop) are on a str, a numpy array or our own list, never on a tag value that could be None.
  - get(name, []) on a Dataset yields a Sequence (possibly empty) for sequence tags, and the optional-text reader _tag maps None to ''; the empty-sequence index reads (ReferencedStructureSetSequence[0]) raise IndexError, refused as before.
  - Dict reads in our code are guarded (the contours lookup by `in`, the roi_names lookup by the file filter).
  - Verdict: no attribute or method access on a possibly-None pydicom value found in the wrapped functions. Accepted behaviour stands for any future one: it raises AttributeError from our frame and aborts the run.
- Result: 506 passed, 1 skipped (500 + 6 new); ruff and compileall clean.

## task-04

- Scratch run (SimpleITK reading a 4-slice synthetic CT series with slice 2 deleted, outside the repo): the raw ITK message is `ITK ERROR: GDCMImageIO(...): Could not open file: <tmp dir>/ct_001.dcm for reading. Reason: No such file or directory`, so it names the full path. Deleting the slice is therefore the right setup; replacing or truncating it names no path.
- Test: test_a2_10_a_slice_lost_after_the_header_read_refuses_without_the_path_ITK_names (tests/test_convert.py, after the A2.10 broken-RTSTRUCT guard). It wraps read_folder_headers on the convert module so that, after the real call for plan_1, one planning-CT slice is deleted (setup only; the code under test deletes nothing). It asserts REASON_CT_UNREADABLE for plan_1 and scans rows, log records (DEBUG), warnings and fds 1/2 for the tmp path, the slice file name, the patient id and every UID. The rows are scanned without the patient column, which is the organized folder name by design.
- Passes on the unchanged code (the first run failed only because the scan included the patient column, a test bug, not a leak).
- Mutation, run once by hand and not committed: convert's quiet_call name patched with a variant that runs fn() and raises CaseRefused(str(exc)) on any Exception. Result: the test fails. The plan_1 row reason is the full ITK message (with the tmp path and ct_001.dcm) instead of REASON_CT_UNREADABLE, and the plan-refused log line carries it too. Test file restored afterwards.
- Result: 507 passed, 1 skipped (506 + 1 new); ruff clean.

## review-group-1
No BLOCKING. WARNING: a failing restore dup2 in the fd-quiet finally leaves fd 2 redirected (deferred). NOTES: the inf-offset RuntimeWarning from np.diff before the finiteness check; a third-party BaseException class's init runs outside the quiet block (deferred).

## review-group-2
No BLOCKING. WARNING: the A9 site regex can't tell the right site from the header-call fallback line (deferred). NOTES: a wrong-VR sequence read as text aborts instead of refusing (accepted in B5); a misspelled keyword in item form still refuses (outside criteria).

## review-loop-pass-1
Bug review and spec-axis review (Opus): no plan bug. Both flagged a non-finite DoseGridScaling not being refused (a NaN dose image would be written). Labelled BLOCKING (clinical output, same class as B6) and fixed in `_read_dose`'s extract step, refused as REASON_DOSE_UNREADABLE; tests for NaN and inf were red first. Deferred: task-02's spacing and origin tuples are built inside the quiet build closure (trivial); a frame without a module name in the header-call origin check; CT-side non-finite header values are unchecked (not verified to reach output).
