"""masks (task-05): RTSTRUCT -> {0,1} masks through DicomRTTool 6.2, and applicator text files. Every tree is synthetic."""
import ast
import gzip
import inspect
import os

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
import synthetic_dicom as sd
from DicomRTTool import DicomReaderWriter

from rt_dataprep import convert as conv
from rt_dataprep import masks
from rt_dataprep.dicom_io import CaseRefused

PID = 'SYNTHETICPTA'
SHAPE = (6, 8, 10)  # (z, y, x)
SPACING = (1.0, 1.5, 2.0)  # (x, y, z): anisotropic, so an axis swap cannot pass
ORIGIN = (-3.0, 5.0, 7.0)


def box(number, name, z, y, x, **kwargs):
    return dict(number=number, name=name, kind='box', box=(z, y, x), **kwargs)


def expected(z, y, x):
    out = np.zeros(SHAPE, np.uint8)
    out[z[0]:z[1] + 1, y[0]:y[1] + 1, x[0]:x[1] + 1] = 1
    return out


def make_plan(tmp_path, rois, channels=None):
    folder = tmp_path / 'org' / PID / 'plan_1'
    folder.mkdir(parents=True)
    ct = sd.write_ct_series(folder, np.zeros(SHAPE, int), spacing=SPACING, origin=ORIGIN, patient_id=PID)
    study = dict(patient_id=PID, study_uid=ct['study_uid'])
    struct = sd.write_rtstruct_rois(folder / 'struct.dcm', ct, rois, **study)
    sd.write_rtplan(folder / 'plan.dcm', struct, np.zeros((2, 3)), channels=channels, **study)
    (tmp_path / 'out').mkdir()
    return folder


def run(tmp_path, folder):
    """(PlanResult, case folder) of convert_plan; the case folder sits in tmp_path/out, the stand-in for a partial folder."""
    case = tmp_path / 'out' / 'plan_1'
    return conv.convert_plan(folder, case), case


def leftovers(tmp_path):
    return [p.name for p in (tmp_path / 'out').iterdir() if p.name.startswith('.')]


def refused(tmp_path, folder):
    case = tmp_path / 'out' / 'plan_1'
    with pytest.raises(CaseRefused) as caught:
        conv.convert_plan(folder, case)
    assert not case.exists() and leftovers(tmp_path) == []
    return str(caught.value)


def read(path):
    return sitk.ReadImage(str(path))


def arr(path):
    return sitk.GetArrayFromImage(read(path))


def edit(path, change):
    ds = pydicom.dcmread(path)
    change(ds)
    pydicom.dcmwrite(path, ds)


def spy(monkeypatch, after_walk=None, after_mask=None, run_fails=False):
    """Replace masks.DicomReaderWriter with a recording subclass; returns the call log."""
    calls = []

    class Spy(DicomReaderWriter):
        def __init__(self, *args, **kwargs):
            calls.append(('init', kwargs))
            super().__init__(*args, **kwargs)

        def walk_through_folders(self, input_path, *args, **kwargs):
            calls.append(('walk', str(input_path), sorted(os.listdir(input_path))))
            if run_fails:
                raise RuntimeError('boom')
            super().walk_through_folders(input_path, *args, **kwargs)
            if after_walk:
                after_walk(self)

        def set_contour_names_and_associations(self, *args, **kwargs):
            if kwargs.get('contour_names') is not None:  # the constructor calls this with None first
                calls.append(('names', kwargs))
            super().set_contour_names_and_associations(*args, **kwargs)

        def set_index(self, index):
            calls.append(('index', index, self.series_instances_dictionary[index].SeriesInstanceUID))
            super().set_index(index)

        def get_images_and_mask(self):
            calls.append(('mask',))
            super().get_images_and_mask()
            if after_mask:
                after_mask(self)
    monkeypatch.setattr(masks, 'DicomReaderWriter', Spy)
    return calls


class Boom:
    def __init__(self, *args, **kwargs):
        raise AssertionError('DicomRTTool must not run')


# ---------------------------------------------------------------- A2.2: masks

def test_a2_2_cube_gives_a_binary_mask_on_the_ct_grid(tmp_path):
    z, y, x = (1, 3), (2, 5), (3, 7)
    folder = make_plan(tmp_path, [box(1, 'PTV 1-a', z, y, x)])
    result, case = run(tmp_path, folder)
    assert result.files == ['planning_ct.nii.gz', 'ptv1a.nii.gz']
    mask, ct = read(case / 'ptv1a.nii.gz'), read(case / 'planning_ct.nii.gz')
    values = sitk.GetArrayFromImage(mask)
    assert values.dtype == np.uint8 and set(np.unique(values)) == {0, 1}
    assert values.sum() == 3 * 4 * 5  # the cube volume
    np.testing.assert_array_equal(values, expected(z, y, x))  # and in the right place on every axis
    assert (mask.GetSize(), mask.GetOrigin(), mask.GetSpacing(), mask.GetDirection()) == (
        ct.GetSize(), ct.GetOrigin(), ct.GetSpacing(), ct.GetDirection())
    assert sorted(p.name for p in case.iterdir()) == ['planning_ct.nii.gz', 'ptv1a.nii.gz', 'series.csv']


def test_a2_2_through_convert_writes_the_mask_into_the_plan_folder(tmp_path):
    make_plan(tmp_path, [box(1, 'Liver', (0, 2), (0, 2), (0, 2))])
    rows = conv.convert(tmp_path / 'org', tmp_path / 'out2').rows
    assert [(r.outcome, r.files) for r in rows] == [('converted', ['planning_ct.nii.gz', 'liver.nii.gz', 'series.csv'])]
    assert arr(tmp_path / 'out2' / PID / 'plan_1' / 'liver.nii.gz').sum() == 27


def test_a2_2b_overlapping_rois_keep_their_full_voxel_count(tmp_path):
    a, b = ((1, 3), (1, 4), (1, 5)), ((2, 4), (3, 6), (4, 8))
    folder = make_plan(tmp_path, [box(1, 'A', *a), box(2, 'B', *b)])
    _, case = run(tmp_path, folder)
    assert arr(case / 'a.nii.gz').sum() == 3 * 4 * 5 and arr(case / 'b.nii.gz').sum() == 3 * 4 * 5
    np.testing.assert_array_equal(arr(case / 'a.nii.gz'), expected(*a))
    np.testing.assert_array_equal(arr(case / 'b.nii.gz'), expected(*b))
    assert (arr(case / 'a.nii.gz') & arr(case / 'b.nii.gz')).sum() > 0  # they do overlap


def test_a2_2c_a_kept_roi_with_an_empty_mask_refuses_the_plan(tmp_path):
    folder = make_plan(tmp_path, [box(1, 'Far', (1, 3), (2, 5), (3, 7)), box(2, 'Near', (1, 3), (2, 5), (3, 7))])

    def shift_far(ds):  # far outside the CT in x: it rasterizes to nothing
        for item in ds.ROIContourSequence[0].ContourSequence:
            data = [float(v) for v in item.ContourData]
            item.ContourData = [v + 10000.0 if k % 3 == 0 else v for k, v in enumerate(data)]
    edit(folder / 'struct.dcm', shift_far)
    assert refused(tmp_path, folder) == masks.REASON_MASK_EMPTY


def test_an_empty_mask_from_dicomrttool_refuses_the_plan(tmp_path, monkeypatch):
    folder = make_plan(tmp_path, [box(1, 'A', (1, 3), (2, 5), (3, 7))])

    def blank(reader):
        reader.mask_dictionary['a'] = sitk.Image(reader.mask_dictionary['a'].GetSize(), sitk.sitkUInt8)
    spy(monkeypatch, after_mask=blank)
    assert refused(tmp_path, folder) == masks.REASON_MASK_EMPTY


# ---------------------------------------------------------------- which ROIs are kept

def test_point_empty_and_dose_rois_produce_no_file_and_the_plan_converts(tmp_path):
    rois = [box(1, 'Kept', (1, 2), (1, 2), (1, 2)), dict(number=2, name='Marker', kind='point', point=(0.0, 5.0, 8.0)),
            dict(number=3, name='Nothing', kind='empty'), box(4, 'Dose_Max', (1, 2), (1, 2), (1, 2)),
            box(5, 'DOSE 50', (1, 2), (1, 2), (1, 2)), box(6, 'Overdosed', (1, 2), (1, 2), (1, 2))]
    folder = make_plan(tmp_path, rois)
    result, case = run(tmp_path, folder)
    assert result.files == ['planning_ct.nii.gz', 'kept.nii.gz']
    assert sorted(p.name for p in case.iterdir()) == ['kept.nii.gz', 'planning_ct.nii.gz', 'series.csv']


def test_dicomrttool_gets_exactly_the_kept_names(tmp_path, monkeypatch):
    rois = [box(1, 'Kept One', (1, 2), (1, 2), (1, 2)), dict(number=2, name='Marker', kind='point', point=(0.0, 5.0, 8.0)),
            dict(number=3, name='Ring', kind='channel', points=np.arange(12.0).reshape(4, 3)),
            box(4, 'Kept Two', (3, 4), (3, 4), (3, 4))]
    calls = spy(monkeypatch)
    run(tmp_path, make_plan(tmp_path, rois))
    assert [c[1] for c in calls if c[0] == 'names'] == [dict(contour_names=['Kept One', 'Kept Two'])]


def test_a_plan_with_no_kept_roi_does_not_run_dicomrttool_and_converts(tmp_path, monkeypatch):
    monkeypatch.setattr(masks, 'DicomReaderWriter', Boom)
    rois = [dict(number=1, name='Marker', kind='point', point=(0.0, 5.0, 8.0)), dict(number=2, name='Nothing', kind='empty'),
            box(3, 'Dose 1', (1, 2), (1, 2), (1, 2)), dict(number=4, name='Tandem', kind='channel', points=np.zeros((3, 3)))]
    result, case = run(tmp_path, make_plan(tmp_path, rois))
    assert result.files == ['planning_ct.nii.gz', 'Tandem.nii.gz']
    assert leftovers(tmp_path) == []


@pytest.mark.parametrize('names', [('PTV 1', 'ptv1'), ('Brain-Stem', 'brain stem'), ('--', 'Other'), ('A', 'a')])
def test_empty_or_clashing_format_names_refuse_the_plan(tmp_path, monkeypatch, names):
    monkeypatch.setattr(masks, 'DicomReaderWriter', Boom)
    rois = [box(k + 1, name, (1, 2), (1, 2), (1, 2)) for k, name in enumerate(names)]
    assert refused(tmp_path, make_plan(tmp_path, rois)) == masks.REASON_ROI_NAMES


def test_an_empty_name_refuses_even_with_one_roi(tmp_path):
    assert refused(tmp_path, make_plan(tmp_path, [box(1, '- -', (1, 2), (1, 2), (1, 2))])) == masks.REASON_ROI_NAMES


# ---------------------------------------------------------------- DicomRTTool call and the folder it reads

def test_dicomrttool_runs_once_on_a_scratch_folder_of_the_ct_and_the_rtstruct_only(tmp_path, monkeypatch):
    folder = make_plan(tmp_path, [box(1, 'A', (1, 2), (1, 2), (1, 2))])
    sd.write_ct_series(folder, np.zeros(SHAPE, int), spacing=SPACING, origin=ORIGIN, patient_id=PID, prefix='other')
    seen = {}
    calls = spy(monkeypatch, after_walk=lambda reader: seen.update(
        series=[e.SeriesInstanceUID for e in reader.series_instances_dictionary.values()]))
    result, case = run(tmp_path, folder)
    assert [c[0] for c in calls] == ['init', 'walk', 'names', 'index', 'mask']  # once per plan, in this order
    assert calls[0][1] == dict(arg_max=False, verbose=False)
    ct_files = sorted(p.name for p in folder.glob('ct_*.dcm'))
    assert calls[1][2] == sorted([*ct_files, 'struct.dcm'])  # the planning CT slices and the RTSTRUCT, not the other series
    assert os.path.dirname(calls[1][1]) == str(tmp_path / 'out')  # inside the patient's partial folder (here: the case's parent)
    assert calls[3][2] == seen['series'][0] and len(seen['series']) == 1  # set_index took the planning CT's index
    assert not os.path.exists(calls[1][1]) and leftovers(tmp_path) == []  # removed after the run
    assert result.files == ['planning_ct.nii.gz', 'a.nii.gz']


def test_scratch_folder_is_removed_when_the_run_fails(tmp_path, monkeypatch):
    folder = make_plan(tmp_path, [box(1, 'A', (1, 2), (1, 2), (1, 2))])
    calls = spy(monkeypatch, run_fails=True)
    assert refused(tmp_path, folder) == masks.REASON_MASK_RUN
    assert calls[1][0] == 'walk' and not os.path.exists(calls[1][1])


def test_scratch_folder_is_removed_when_a_check_refuses(tmp_path, monkeypatch):
    folder = make_plan(tmp_path, [box(1, 'A', (1, 2), (1, 2), (1, 2))])
    calls = spy(monkeypatch, after_mask=lambda reader: reader.mask_dictionary.clear())
    assert refused(tmp_path, folder) == masks.REASON_MASK_MISSING
    assert not os.path.exists(calls[1][1])


def test_scratch_folder_falls_back_to_copies_when_linking_fails(tmp_path, monkeypatch):
    folder = make_plan(tmp_path, [box(1, 'A', (1, 2), (1, 2), (1, 2))])

    def no_link(*args, **kwargs):
        raise OSError('cross-device')
    monkeypatch.setattr(masks.os, 'link', no_link)
    result, case = run(tmp_path, folder)
    assert result.files == ['planning_ct.nii.gz', 'a.nii.gz'] and arr(case / 'a.nii.gz').sum() == 8


def test_scratch_files_are_hard_links_when_the_file_system_allows_it(tmp_path, monkeypatch):
    folder = make_plan(tmp_path, [box(1, 'A', (1, 2), (1, 2), (1, 2))])
    links = []
    original = os.link
    monkeypatch.setattr(masks.os, 'link', lambda a, b: (links.append(b), original(a, b))[1])
    run(tmp_path, folder)
    assert len(links) == SHAPE[0] + 1


def test_the_rtstruct_not_ending_on_the_planning_ct_is_not_linked_and_refuses(tmp_path):
    """DicomRTTool links an RTSTRUCT to the LAST series it references (hardening #18): refuse, never mask the wrong series."""
    folder = make_plan(tmp_path, [box(1, 'A', (1, 2), (1, 2), (1, 2))])

    def reference_another_series(ds):
        series = pydicom.Dataset()
        series.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence.append(series)
    edit(folder / 'struct.dcm', reference_another_series)
    assert refused(tmp_path, folder) == masks.REASON_MASK_LINK


def test_two_indexes_carrying_the_planning_ct_uid_refuse(tmp_path, monkeypatch):
    folder = make_plan(tmp_path, [box(1, 'A', (1, 2), (1, 2), (1, 2))])

    def duplicate(reader):
        entries = reader.series_instances_dictionary
        entries[max(entries) + 1] = entries[0]
    spy(monkeypatch, after_walk=duplicate)
    assert refused(tmp_path, folder) == masks.REASON_MASK_LINK


def test_a_roi_dicomrttool_resolves_to_another_roi_refuses(tmp_path):
    """Two ROIs equal ignoring case: DicomRTTool keeps the first, which here is the POINT ROI, not the kept one."""
    rois = [dict(number=1, name='ptv', kind='point', point=(0.0, 5.0, 8.0)), box(2, 'PTV', (1, 2), (1, 2), (1, 2))]
    assert refused(tmp_path, make_plan(tmp_path, rois)) == masks.REASON_MASK_MISSING


def test_a_kept_roi_missing_from_the_mask_dictionary_refuses(tmp_path, monkeypatch):
    folder = make_plan(tmp_path, [box(1, 'A', (1, 2), (1, 2), (1, 2)), box(2, 'B', (3, 4), (3, 4), (3, 4))])
    spy(monkeypatch, after_mask=lambda reader: reader.mask_dictionary.pop('b'))
    assert refused(tmp_path, folder) == masks.REASON_MASK_MISSING


def shifted(reader):
    image = reader.mask_dictionary['a']
    image.SetOrigin((image.GetOrigin()[0] + 0.01, *image.GetOrigin()[1:]))


def respaced(reader):
    image = reader.mask_dictionary['a']
    image.SetSpacing((image.GetSpacing()[0] + 0.01, *image.GetSpacing()[1:]))


def turned(reader):
    image = reader.mask_dictionary['a']
    image.SetDirection((0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))


def resized(reader):
    reader.mask_dictionary['a'] = sitk.Cast(reader.mask_dictionary['a'][:, :, :-1], sitk.sitkUInt8)


@pytest.mark.parametrize('damage', [shifted, respaced, turned, resized])
def test_a_mask_off_the_ct_grid_refuses(tmp_path, monkeypatch, damage):
    folder = make_plan(tmp_path, [box(1, 'A', (1, 2), (1, 2), (1, 2))])
    spy(monkeypatch, after_mask=damage)
    assert refused(tmp_path, folder) == masks.REASON_MASK_GRID


def test_mask_values_are_written_as_zero_and_one(tmp_path, monkeypatch):
    folder = make_plan(tmp_path, [box(1, 'A', (1, 2), (1, 2), (1, 2))])

    def loud(reader):
        image = reader.mask_dictionary['a']
        reader.mask_dictionary['a'] = sitk.Cast(image, sitk.sitkUInt8) * 200
    spy(monkeypatch, after_mask=loud)
    _, case = run(tmp_path, folder)
    assert set(np.unique(arr(case / 'a.nii.gz'))) == {0, 1} and arr(case / 'a.nii.gz').dtype == np.uint8


def test_geometry_within_tolerance_passes_and_the_mask_takes_the_ct_geometry_exactly(tmp_path, monkeypatch):
    folder = make_plan(tmp_path, [box(1, 'A', (1, 2), (1, 2), (1, 2))])

    def nudge(reader):
        image = reader.mask_dictionary['a']
        image.SetOrigin((image.GetOrigin()[0] + conv.ORIGIN_ATOL_MM / 2, *image.GetOrigin()[1:]))
        image.SetDirection((1.0, conv.DIRECTION_ATOL / 2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    spy(monkeypatch, after_mask=nudge)
    _, case = run(tmp_path, folder)
    mask, ct = read(case / 'a.nii.gz'), read(case / 'planning_ct.nii.gz')
    assert (mask.GetSize(), mask.GetOrigin(), mask.GetSpacing(), mask.GetDirection()) == (
        ct.GetSize(), ct.GetOrigin(), ct.GetSpacing(), ct.GetDirection())


# ---------------------------------------------------------------- A2.3: applicators

def channel_points(count=25):
    return np.random.default_rng(3).uniform(-40, 40, size=(count, 3))  # 25 points: three ContourSequence items


def test_a2_3_brachy_channel_is_written_as_gzip_text_named_from_the_plan_channel(tmp_path):
    points = channel_points()
    other = channel_points(7) + 1.0
    rois = [dict(number=1, name='ROI A', kind='channel', points=points, interpreted='BRACHY_CHANNEL'),
            dict(number=2, name='Tandem 2', kind='channel', points=other, interpreted='BRACHY_CHANNEL')]
    channels = [dict(applicator_id='Ring-1', dwells_mm=np.zeros((2, 3)), roi=1)]  # nothing names ROI 2
    result, case = run(tmp_path, make_plan(tmp_path, rois, channels))
    assert result.files == ['planning_ct.nii.gz', 'Ring1.nii.gz', 'Tandem2.nii.gz']
    for name, expected_points in (('Ring1.nii.gz', points), ('Tandem2.nii.gz', other)):
        with gzip.open(case / name, 'rt') as handle:
            assert handle.readline().split() == [f'{v:.4f}' for v in expected_points[0]]  # '%.4f' text
        np.testing.assert_allclose(np.loadtxt(case / name), expected_points, rtol=0, atol=1e-4)  # in contour order
    assert sorted(p.name for p in case.iterdir()) == ['Ring1.nii.gz', 'Tandem2.nii.gz', 'planning_ct.nii.gz', 'series.csv']


@pytest.mark.parametrize('kind,interpreted', [('channel', 'ORGAN'), ('channel', None), ('channel', 'BRACHY_CHANNEL')])
def test_open_nonplanar_roi_is_an_applicator_whatever_its_interpreted_type(tmp_path, kind, interpreted):
    rois = [dict(number=1, name='Needle 1', kind=kind, points=channel_points(4), interpreted=interpreted)]
    result, case = run(tmp_path, make_plan(tmp_path, rois))
    assert result.files == ['planning_ct.nii.gz', 'Needle1.nii.gz']


def test_brachy_channel_type_makes_a_closed_roi_an_applicator_not_a_mask(tmp_path, monkeypatch):
    monkeypatch.setattr(masks, 'DicomReaderWriter', Boom)
    rois = [box(1, 'Odd', (1, 2), (1, 2), (1, 2), interpreted='BRACHY_CHANNEL')]
    result, case = run(tmp_path, make_plan(tmp_path, rois))
    assert result.files == ['planning_ct.nii.gz', 'Odd.nii.gz'] and np.loadtxt(case / 'Odd.nii.gz').shape == (8, 3)


def test_mask_and_applicator_files_are_listed_masks_first_in_rtstruct_order(tmp_path):
    rois = [dict(number=1, name='Ring', kind='channel', points=channel_points(3)), box(2, 'Zed', (1, 2), (1, 2), (1, 2)),
            box(3, 'Alpha', (3, 4), (3, 4), (3, 4)), dict(number=4, name='Needle', kind='channel', points=channel_points(3))]
    result, _ = run(tmp_path, make_plan(tmp_path, rois))
    assert result.files == ['planning_ct.nii.gz', 'zed.nii.gz', 'alpha.nii.gz', 'Ring.nii.gz', 'Needle.nii.gz']


@pytest.mark.parametrize('rois,channels', [
    ([dict(number=1, name='Ring', kind='channel', points=np.zeros((3, 3))),
      dict(number=2, name='RING', kind='channel', points=np.zeros((3, 3)))], None),  # two applicators, case differs
    ([dict(number=1, name='Ring', kind='channel', points=np.zeros((3, 3))),
      box(2, 'ring', (1, 2), (1, 2), (1, 2))], None),  # applicator and mask
    ([dict(number=1, name='x', kind='channel', points=np.zeros((3, 3))),
      dict(number=2, name='y', kind='channel', points=np.zeros((3, 3)))],
     [dict(applicator_id='Same', dwells_mm=np.zeros((2, 3)), roi=1), dict(applicator_id='same', dwells_mm=np.zeros((2, 3)), roi=2)]),
    ([dict(number=1, name='x', kind='channel', points=np.zeros((3, 3)))],
     [dict(applicator_id='--', dwells_mm=np.zeros((2, 3)), roi=1)]),  # the channel id reduces to nothing
    ([dict(number=1, name='x', kind='channel', points=np.zeros((3, 3)))],
     [dict(applicator_id='Dose', dwells_mm=np.zeros((2, 3)), roi=1)]),  # would overwrite dose.nii.gz
])
def test_clashing_or_empty_applicator_names_refuse_the_plan(tmp_path, rois, channels):
    assert refused(tmp_path, make_plan(tmp_path, rois, channels)) == masks.REASON_ROI_NAMES


def test_applicator_text_goes_through_a_temp_name_that_ends_nii_gz(tmp_path, monkeypatch):
    names = []
    original = np.savetxt

    def record(fname, points, **kwargs):
        names.append((os.path.basename(fname), kwargs))
        return original(fname, points, **kwargs)
    monkeypatch.setattr(masks.np, 'savetxt', record)
    rois = [dict(number=1, name='Ring', kind='channel', points=channel_points(3))]
    run(tmp_path, make_plan(tmp_path, rois))
    assert len(names) == 1 and names[0][0].endswith('.nii.gz') and names[0][0] != 'Ring.nii.gz'
    assert names[0][1] == dict(fmt='%.4f')


# ---------------------------------------------------------------- guardrails on the source

def test_no_case_refused_is_raised_inside_an_except_block_in_masks():
    tree = ast.parse(inspect.getsource(masks))
    inside = [node for handler in ast.walk(tree) if isinstance(handler, ast.ExceptHandler)
              for node in ast.walk(handler) if isinstance(node, ast.Raise)]
    assert inside == []


def test_no_array_view_of_a_temporary_image_in_masks():
    assert 'GetArrayViewFromImage' not in inspect.getsource(masks)
