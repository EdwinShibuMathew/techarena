"""Explicit upload-trial policy; omitted policy retains original V6 promotion."""
import numpy as np
from .contracts import require

V7_SHA = '9aebf08df53b9b1d0661479151c37118859831b3b4c9d371d533c3bf203b141a'


def is_trial(spec):
    return spec.get('release_policy', 'v6_four_gate') == 'v7_upload_trial'


def comparison(spec, row):
    return row['v7_comparison'] if is_trial(spec) else row


def validate_anchor(root, spec):
    if not is_trial(spec):
        return
    from .contracts import inside
    from .storage import read
    config = read(inside(root, spec['inputs']['v7_config']['path']))
    require(config['submission_sha256'] == V7_SHA == spec['anchor_sha256']
            and config['sharpening_gamma'] == spec['anchor_gamma'] == 1.12
            and config['validation_report_sha256'] == spec['inputs']['v7_validation']['sha256'],
            'V7 anchor configuration/evidence changed')


def validate_metric(row, confidence=False):
    require(isinstance(row,dict) and {'baseline_auc','candidate_auc','auc_diff'}<=set(row),'Missing comparison metrics')
    if confidence:require({'resamples','ci_lower','ci_upper'}<=set(row),'Missing confidence evidence')
    require(all(np.isfinite(row[k]) for k in ('baseline_auc', 'candidate_auc', 'auc_diff'))
            and 0 <= row['baseline_auc'] <= 1 and 0 <= row['candidate_auc'] <= 1
            and abs(row['candidate_auc'] - row['baseline_auc'] - row['auc_diff']) < 1e-12,
            'Inconsistent comparison metrics')
    if confidence:
        require(type(row['resamples']) is int and 990 <= row['resamples'] <= 1000
                and np.isfinite(row['ci_lower']) and np.isfinite(row['ci_upper'])
                and -1 <= row['ci_lower'] <= row['ci_upper'] <= 1, 'Invalid comparison confidence evidence')


def stage_passed(spec, stage, results):
    if stage == 'screen':
        primary = comparison(spec, results['primary'])
        secondary = comparison(spec, results['secondary'])
        return primary['ci_lower'] > 0 and secondary['auc_diff'] >= 0
    require(stage == 'stress', 'Unknown policy stage')
    five = comparison(spec, results['five_day'])
    return five['ci_lower'] > 0 and (is_trial(spec) or five['late']['auc_diff'] >= 0)


def v6_gates(screen, stress):
    return {'primary': screen['results']['primary']['ci_lower'] > 0,
            'secondary': screen['results']['secondary']['auc_diff'] >= 0,
            'five_day': stress['results']['five_day']['ci_lower'] > 0,
            'late': stress['results']['five_day']['late']['auc_diff'] >= 0}


def release_flags(spec):
    if is_trial(spec):
        return {'release_policy': 'v7_upload_trial', 'upload_ready': True,
                'submission_ready': False, 'promotion_supported': False}
    return {'submission_ready': True}


def receipt_ready(spec, receipt):
    if spec.get("release_policy","v6_four_gate") not in {"v6_four_gate","v7_upload_trial"}:return False
    return all(receipt.get(k) == v and type(receipt.get(k)) is type(v)
               for k, v in release_flags(spec).items())
