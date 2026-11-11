"""
DICOM metadata for series.json: the catalog (docs/specs/2026-11-09-dicom-metadata.md), the value rules and the aggregation
over a row's headers.

A key is present only when its value is. Values are the DICOM values as written (strings; a list for a multi-valued
element); patient_age (a number) and slices (a count) are the only derived ones. Each key is computed under a guard, so a
tag that is missing or odd omits that key and never refuses a file or aborts a run. Nothing here logs, and no value goes
into an exception message: the values are patient data.

Headers come from dicom_io.read_header, which has already converted every element, so reading them here is silent.
"""
import datetime
import re

from pydicom.multival import MultiValue

ALL, IMAGES, RTPLAN, RTSTRUCT, RTDOSE, REG = 'all', 'images', 'rtplan', 'rtstruct', 'rtdose', 'reg'

NUMERIC_VRS = frozenset(('DS', 'IS', 'FL', 'FD', 'US', 'UL', 'SS', 'SL'))
TEXT_ORDER_VRS = frozenset(('DA', 'TM', 'DT'))
_GUARDED = (AttributeError, KeyError, IndexError, TypeError, ValueError)  # as series._or_none
_PET = 'RadiopharmaceuticalInformationSequence[0].'


def _patient_age(heads, roi):
    ages = [age for age in (_age(ds) for ds in heads) if age is not None]
    ages = list(dict.fromkeys(ages))  # 63 and 63.0 are one value, first seen kept
    if len(ages) > 1:
        return {'min': min(ages), 'max': max(ages)}
    return ages[0] if ages else None


def _roi_lookup(sequence, number_keyword, keyword):
    """The `keyword` of the `sequence` item whose `number_keyword` is the row's ROI number (never by name)."""
    def lookup(heads, roi):
        if roi is None:
            return None
        items = [item for ds in heads if sequence in ds for item in ds[sequence].value
                 if number_keyword in item and keyword in item and _int(item[number_keyword].value) == int(roi)]
        return _aggregate([sample for item in items if (sample := _render(item[keyword]))], False)
    return lookup


# One table, in the spec's order. A source is a path ('Seq[0].Keyword', 'Seq[*].Keyword' for every item), a tuple of
# paths (the first that has a value on any header is used for all), or a function (heads, roi_number) -> value.
_CATALOG = {
    ALL: {
        'patient_age': _patient_age, 'patient_sex': 'PatientSex', 'patient_size': 'PatientSize',
        'patient_weight': 'PatientWeight', 'study_date': 'StudyDate', 'study_time': 'StudyTime',
        'study_description': 'StudyDescription', 'accession_number': 'AccessionNumber', 'study_id': 'StudyID',
        'series_number': 'SeriesNumber', 'series_time': 'SeriesTime', 'acquisition_date': 'AcquisitionDate',
        'acquisition_time': 'AcquisitionTime', 'content_date': 'ContentDate', 'content_time': 'ContentTime',
        'manufacturer': 'Manufacturer', 'model': 'ManufacturerModelName', 'software_versions': 'SoftwareVersions',
        'institution_name': 'InstitutionName', 'station_name': 'StationName', 'device_serial_number': 'DeviceSerialNumber',
        'body_part_examined': 'BodyPartExamined', 'patient_position': 'PatientPosition', 'protocol_name': 'ProtocolName',
    },
    IMAGES: {
        'image_type': 'ImageType', 'rows': 'Rows', 'columns': 'Columns', 'slices': lambda heads, roi: len(heads),
        'pixel_spacing': 'PixelSpacing', 'slice_thickness': 'SliceThickness',
        'spacing_between_slices': 'SpacingBetweenSlices', 'contrast_agent': 'ContrastBolusAgent',
        # CT
        'kvp': 'KVP', 'tube_current': 'XRayTubeCurrent', 'exposure': 'Exposure', 'exposure_time': 'ExposureTime',
        'convolution_kernel': 'ConvolutionKernel', 'recon_diameter': 'ReconstructionDiameter', 'ctdi_vol': 'CTDIvol',
        # MR
        'field_strength': 'MagneticFieldStrength', 'scanning_sequence': 'ScanningSequence',
        'sequence_variant': 'SequenceVariant', 'sequence_name': 'SequenceName', 'acquisition_type': 'MRAcquisitionType',
        'repetition_time': 'RepetitionTime', 'echo_time': 'EchoTime', 'inversion_time': 'InversionTime',
        'flip_angle': 'FlipAngle', 'echo_train_length': 'EchoTrainLength', 'b_value': 'DiffusionBValue',
        'receive_coil': 'ReceiveCoilName',
        # PT
        'radiopharmaceutical': _PET + 'Radiopharmaceutical',
        'radiopharmaceutical_code': _PET + 'RadiopharmaceuticalCodeSequence[0].CodeMeaning',
        'radionuclide': _PET + 'RadionuclideCodeSequence[0].CodeMeaning',
        'injected_dose': _PET + 'RadionuclideTotalDose',
        'injection_time': (_PET + 'RadiopharmaceuticalStartDateTime', _PET + 'RadiopharmaceuticalStartTime'),
        'half_life': _PET + 'RadionuclideHalfLife', 'units': 'Units', 'decay_correction': 'DecayCorrection',
        'corrected_image': 'CorrectedImage', 'recon_method': 'ReconstructionMethod',
    },
    RTPLAN: {
        'plan_time': 'RTPlanTime', 'plan_intent': 'PlanIntent', 'approval_status': 'ApprovalStatus',
        'plan_description': 'RTPlanDescription',
        'prescription_dose': 'DoseReferenceSequence[*].TargetPrescriptionDose',
        'radiation_type': ('BeamSequence[*].RadiationType', 'IonBeamSequence[*].RadiationType'),
        'beam_type': ('BeamSequence[*].BeamType', 'IonBeamSequence[*].BeamType'),
        'treatment_machine': ('BeamSequence[*].TreatmentMachineName', 'IonBeamSequence[*].TreatmentMachineName',
                              'TreatmentMachineSequence[*].TreatmentMachineName'),
        'source_isotope': 'SourceSequence[*].SourceIsotopeName',
        'source_strength': 'SourceSequence[*].ReferenceAirKermaRate',
        'source_strength_date': 'SourceSequence[*].SourceStrengthReferenceDate',
        'source_strength_time': 'SourceSequence[*].SourceStrengthReferenceTime',
        'number_of_beams': 'FractionGroupSequence[0].NumberOfBeams', 'brachy_technique': 'BrachyTreatmentTechnique',
        'brachy_type': 'BrachyTreatmentType',
    },
    RTSTRUCT: {
        'structure_set_label': 'StructureSetLabel', 'structure_set_date': 'StructureSetDate',
        'structure_set_time': 'StructureSetTime',
        'roi_interpreted_type': _roi_lookup('RTROIObservationsSequence', 'ReferencedROINumber', 'RTROIInterpretedType'),
        'roi_generation_algorithm': _roi_lookup('StructureSetROISequence', 'ROINumber', 'ROIGenerationAlgorithm'),
    },
    RTDOSE: {'dose_units': 'DoseUnits', 'dose_type': 'DoseType'},
    REG: {
        'reg_type': 'RegistrationSequence[*].MatrixRegistrationSequence[*].MatrixSequence[*].'
                    'FrameOfReferenceTransformationMatrixType',
        'reg_date': 'ContentDate', 'reg_time': 'ContentTime', 'reg_manufacturer': 'Manufacturer',
    },
}


def keys(group):
    """The catalog key names of `group`, in table order."""
    return tuple(_CATALOG[group])


def collect(heads, groups, *, roi_number=None):
    """
    The keys of `groups` that are present, aggregated over `heads` (a list of already-read headers). `roi_number` selects
    the RTSTRUCT's ROI for roi_interpreted_type and roi_generation_algorithm; None leaves those two out.
    """
    found = {}
    if not heads:
        return found
    for group in groups:
        for key, source in _CATALOG[group].items():
            value = _guarded(_compute, heads, source, roi_number)
            if value is not None:
                found[key] = value
    return found


def reg_keys(reg):
    """The REG group keys of one REG header."""
    return collect([reg], (REG,))


def _guarded(fn, *args):
    """fn(*args), or None when a tag is missing or odd. Not quoted: the message could hold a value."""
    failed, result = False, None
    try:
        result = fn(*args)
    except _GUARDED:
        failed = True
    return None if failed else result


def _compute(heads, source, roi_number):
    if callable(source):
        return source(heads, roi_number)
    for path in (source,) if isinstance(source, str) else source:
        samples = [sample for ds in heads for sample in _samples(ds, path)]
        if samples:
            return _aggregate(samples, '[*]' not in path)  # sequence items are distinct values, not a range
    return None


def _samples(ds, path):
    return [sample for element in _walk(ds, path) if (sample := _render(element))]


def _walk(ds, path):
    """The data elements a path like 'Seq[0].Inner[*].Keyword' reaches in `ds`; a tag that is not there is skipped."""
    *steps, last = path.split('.')
    items = [ds]
    for step in steps:
        name, index = step[:-1].split('[')
        reached = []
        for item in items:
            if name in item:
                children = item[name].value
                reached += list(children) if index == '*' else list(children[int(index):int(index) + 1])  # empty: skipped like a missing tag
        items = reached
    return [item[last] for item in items if last in item]


def _render(element):
    """(text or list of texts, VR) for one element, or None when it holds nothing."""
    value = element.value
    if isinstance(value, MultiValue):
        texts = [str(v).strip() for v in value]  # never str() of the MultiValue, which is a Python list repr
        return (texts, element.VR) if texts else None
    if value is None:
        return None
    text = str(value).strip()
    return (text, element.VR) if text else None


def _text(ds, keyword):
    sample = _render(ds[keyword]) if keyword in ds else None
    return sample[0] if sample and isinstance(sample[0], str) else None


def _int(value):
    """int(value), or None when it holds no integer (one odd item must not hide the others)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _floats(texts):
    try:
        return [float(t) for t in texts]
    except ValueError:
        return None


def _aggregate(samples, minmax):
    """
    One value from the (value, VR) samples of all headers, in first-seen order: none -> None; one distinct value -> it;
    several -> {'min', 'max'} (when `minmax`, for a VM 1 numeric element that parses or a DA/TM/DT) else {'distinct': [...]}.
    """
    if not samples:
        return None
    vrs = {vr for _, vr in samples}
    scalar = all(isinstance(value, str) for value, _ in samples)
    numbers = _floats([value for value, _ in samples]) if scalar and vrs <= NUMERIC_VRS else None
    distinct = {}  # number (or text) -> first-seen value
    for i, (value, _) in enumerate(samples):
        distinct.setdefault(numbers[i] if numbers else value if isinstance(value, str) else tuple(value), value)  # '12' != ['1', '2']
    values = list(distinct.values())
    if len(values) == 1:
        return values[0]
    if minmax and numbers:
        return {'min': distinct[min(distinct)], 'max': distinct[max(distinct)]}
    if minmax and scalar and len(vrs) == 1 and vrs <= TEXT_ORDER_VRS:
        return {'min': min(values), 'max': max(values)}  # standard (colon-free) TM is fixed width from the left: string order is time order
    return {'distinct': values}


def _age(ds):
    """One header's patient age as a number, or None."""
    text = _text(ds, 'PatientAge')
    match = re.fullmatch(r'([0-9]{3})([DWMY])', text or '')
    if match:
        n = int(match[1])
        return {'Y': n, 'M': round(n / 12, 2), 'W': round(n * 7 / 365.25, 2), 'D': round(n / 365.25, 2)}[match[2]]
    born = _text(ds, 'PatientBirthDate')
    seen = next((t for t in (_text(ds, k) for k in ('StudyDate', 'SeriesDate', 'AcquisitionDate')) if t), None)
    if not (born and seen):
        return None
    try:
        b, s = (datetime.date.fromisoformat(f'{d[:4]}-{d[4:6]}-{d[6:]}') for d in (born, seen))  # 20230229 is a ValueError
    except ValueError:
        return None
    age = s.year - b.year - ((s.month, s.day) < (b.month, b.day))
    return age if age >= 0 else None
