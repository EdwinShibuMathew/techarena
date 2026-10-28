"""Budgeted post-hoc analysis of closed V7.5 artifacts; never fits or exports."""
import argparse
from dataclasses import asdict
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'legacy'),
                str(ROOT / 'experiments/v7.5-history-age-v1'), str(ROOT / 'experiments/v7.4-rebuild-v1')]
from ml_workflow.contracts import canonical, compose, digest, first_indices, inside, require
from ml_workflow.storage import read, write_new, verify_snapshot
from ml_workflow import engine
from experiment_runner import PairedBootstrapEvaluator
from predict_v7 import calibrate_v7_probabilities
import campaign
import pipeline
import protocol
import run as rebuild
import data as dataset
import audit_metrics as metrics
import successor
import numpy as np

RELATIVE = 'experiments/v7.5-audit'
WORK = 'artifacts/audits/v75-v1'


def unchanged(root, inventory):
    for name, expected in inventory.items():
        require(digest(inside(root, name)) == expected, 'Protected audit input changed: ' + name)


def original_data_identity(root):
    parent = root / 'artifacts/rebuild'
    manifest = read(root / 'artifacts/v7_5/manifest.json')
    require(digest(parent / 'data/metadata.json') == manifest['data_sha256'], 'Original dataset metadata changed')
    require(digest(parent / 'genesis.json') == manifest['parent_genesis_sha256'], 'Original reconstruction genesis changed')
    for name, expected in read(parent / 'genesis.json')['raw_sha256'].items():
        require(digest(inside(root, name)) == expected, 'Original raw input changed: ' + name)


def inventory(root):
    """Explicit task inventories only; never traverse local/ or inspect raw rows."""
    original_data_identity(root)
    work = root / 'artifacts/v7_5'; files = set()
    def add(path): files.add(path.relative_to(root).as_posix())
    for path in sorted((work / 'tasks').glob('*.json')):
        require(campaign.completed(work, path.stem), 'Unaccounted V7.5 task')
        add(path); token = read(path)['token']
        for suffix in ('.claim.json', '.validated.json'): add(work / 'attempts' / (token + suffix))
        for relative in read(work / 'attempts' / (token + '.validated.json'))['files']: add(inside(work, relative))
    for pointer in ('baseline-source.json', 'scientific-source.json'):
        path = work / pointer; add(path); record = read(path)
        folder = work / record['snapshot_relative']; verify_snapshot(folder, record['source'])
        add(folder / 'identity.json'); add(folder / 'source.tar.gz')
    for name in ('manifest.json', 'baseline-delivery.json'): add(work / name)
    m = read(work / 'manifest.json')
    for relative, expected in m['inherited_models'].items():
        require(digest(inside(root, relative)) == expected, 'Inherited checkpoint changed'); add(inside(root, relative))
    parent = root / 'artifacts/rebuild'; add(parent / 'genesis.json')
    add(parent / 'accounting_snapshots' / (m['parent_ledger_sha256'] + '.json'))
    metadata = dataset.verify(parent / 'data')
    for name in ('metadata.json', 'vocabulary.json', *metadata['arrays']): add(parent / 'data' / name)
    for name in read(parent / 'genesis.json')['raw_sha256']: add(inside(root, name))
    delivery = read(work / 'baseline-delivery.json')
    csv = inside(root, delivery['submission'])
    for path in (csv, csv.with_suffix('.csv.sha256'), csv.with_suffix('.csv.receipt.json'),
                 inside(work, delivery['accounting_record'])): add(path)
    for name in ('results.json', 'validation.json', 'delivery.json', 'README.md'):
        add(root / 'docs/results/v7.5-history-age-v1' / name)
    # Includes the original, sealed terminal ledger without pinning the mutable live ledger.
    seal = read(root / 'docs/results/v7.5-history-age-v1/validation.json')['terminal_guard_checks']['ledger_sha256_after']
    add(work / 'accounting_snapshots' / (seal + '.json'))
    return {name: digest(inside(root, name)) for name in sorted(files)}


def compute(root, source):
    work = root / 'artifacts/v7_5'; inputs = inventory(root)
    verified = pipeline.verify(root, work); development = pipeline.checked(work, 'develop')
    require(not development['passed'] and pipeline.select_winner(development['results']) is None,
            'The closed V7.5 result differs')
    require(not any((work / 'stages' / (s + '.json')).exists() for s in ('screen', 'stress'))
            and not (root / 'configs/v7_5.json').exists(), 'Unexpected later V7.5 promotion evidence')
    raw = dataset.load(root / 'artifacts/rebuild/data', 'train')
    report = {'schema_version': 1, 'diagnostic_only': True, 'promotion_supported': False,
              'new_model_training_performed': False, 'new_submission_exported': False,
              'validation_status': 'post-hoc reused development dates; not independent holdout evidence',
              'audit_source': source, 'protected_inputs': inputs,
              'status': {'development': 'rejected', 'eight_promotion_gates': 'not_run',
                         'source_engineering': read(root / 'docs/results/v7.5-history-age-v1/validation.json')['source_ci'],
                         'delivery': verified, 'fallback_model_role': 'rebuilt_baseline'}, 'cases': {}}
    from catboost import CatBoostClassifier
    for case in ('jan22_long', 'jan24_forward'):
        meta, target, end = protocol.CASES[case]
        ix = first_indices(raw['u'], raw['t'], target, end); y = raw['y'][ix]
        masks = metrics.subgroup_masks(raw, ix, target)
        masks['aged_48_72'] = (raw['t'][ix] >= target + 2 * protocol.DAY)
        saved = {}
        for candidate in ('A', 'B', 'C'):
            with np.load(work / 'predictions' / (candidate + '-' + case + '.npz'), allow_pickle=False) as values:
                require(np.array_equal(values['indices'], ix) and np.array_equal(values['labels'], y)
                        and np.array_equal(values['rowids'], raw['rowids'][ix]), 'Saved cohort differs')
                saved[candidate] = {k: values[k] for k in ('baseline', 'v7', 'candidate')}
        base, v7 = saved['A']['baseline'], saved['A']['v7']
        rows = {'events': len(ix), 'models': {}, 'subgroups': {}, 'member_diagnostics': {},
                'feature_ranges': {}, 'feature_importance': {}, 'replicas': {}}
        predictions = {'V6': base, 'V7': v7, **{c: saved[c]['candidate'] for c in saved}}
        for name, p in predictions.items(): rows['models'][name] = metrics.describe(y, p)
        for c in ('A', 'B', 'C'):
            original = development['results'][c + '/' + case]
            for anchor, reference in (('V6', base), ('V7', v7)):
                recorded = original if anchor == 'V6' else original['v7_comparison']
                delta = metrics.auc(y, saved[c]['candidate']) - metrics.auc(y, reference)
                require(abs(delta - recorded['auc_diff']) < 1e-12, 'Development delta does not reproduce')
        for group, mask in masks.items():
            if mask.any():
                rows['subgroups'][group] = {'events': int(mask.sum()), 'models': {
                    name: metrics.describe(y[mask], p[mask]) for name, p in predictions.items()}}
        rows['posthoc_A_vs_V7_bootstrap'] = {**asdict(PairedBootstrapEvaluator().evaluate(saved['A']['candidate'], v7, y, raw['u'][ix])),
             'diagnostic_only': True, 'promotion_supported': False, 'seed': 2026, 'unit': 'whole_user'}
        rows['component_diagnostics'] = metrics.component_report(y, saved['A']['candidate'], v7, base,
                                                                  {'all': masks['all'], 'aged': masks['aged_48_72']})
        members = {family: {} for family in ('original', 'current', 'pooled')}
        importance = {family: [] for family in ('current', 'pooled')}
        for member in protocol.MEMBERS:
            target_data = pipeline.matrix47(work, target, target, end, member)
            names = target_data['feature_names'].tolist(); X = target_data['X']
            require(np.array_equal(target_data['indices'], ix), 'Target feature order differs')
            for family in members:
                path = (work / 'baseline-combiners' / str(meta) / (member + '.cbm') if family == 'original'
                        else pipeline.candidate_model(work, meta, member, family))
                width = 37 if family == 'original' else 47
                require(read(path.with_suffix('.receipt.json'))['identity']['parameters'] == protocol.COMBINER,
                        'Frozen combiner parameter receipt differs')
                members[family][member] = pipeline.predict_model(path, X[:, :width], names[:width], width)
                model = CatBoostClassifier(); model.load_model(str(path))
                require(model.get_all_params()['iterations'] == protocol.COMBINER['iterations'], 'Frozen combiner parameters differ')
                if family != 'original': importance[family].append(model.get_feature_importance(type='PredictionValuesChange', thread_count=4))
            rows['feature_ranges'][member] = {}
            for family in ('current', 'pooled'):
                prefixes = protocol.fitting_prefixes(meta, family == 'pooled')
                minima, maxima = [], []
                for cutoff in prefixes:
                    fit = pipeline.matrix47(work, cutoff, meta, meta + protocol.DAY, member)
                    minima.append(fit['X'].min(0)); maxima.append(fit['X'].max(0))
                rows['feature_ranges'][member][family] = metrics.range_coverage(np.min(minima, axis=0), np.max(maxima, axis=0), X, names)
        for family, expected in (('original', base), ('current', saved['A']['candidate']), ('pooled', saved['B']['candidate'])):
            p = compose(members[family])
            if family == 'current': p = calibrate_v7_probabilities(p, 1.12)
            require(np.allclose(p, expected, rtol=0, atol=1e-15), 'Saved combiner predictions differ: ' + family)
        require(np.allclose(calibrate_v7_probabilities(base, 1.12), v7, rtol=0, atol=1e-15)
                and np.allclose(calibrate_v7_probabilities(compose(members['pooled']), 1.12), saved['C']['candidate'], rtol=0, atol=1e-15),
                'Sharpened saved predictions differ')
        rows['member_diagnostics'] = metrics.member_ablation(y, members['original'], members['current'], v7,
                                                           {'all': masks['all'], 'aged': masks['aged_48_72']})
        rows['feature_importance'] = {f: metrics.group_importance(importance[f], names) for f in importance}
        for family in ('current', 'pooled'):
            prefixes = protocol.fitting_prefixes(meta, family == 'pooled')
            rows['replicas'][family] = {'prefixes': prefixes, 'weight_per_replica': 1 / len(prefixes),
                  'total_weight_per_original_row': 1., 'independent_extra_labels': 0,
                  'prefix_event_counts': [int((raw['t'] < c).sum()) for c in prefixes],
                  'target_prefix_event_count': int((raw['t'] < target).sum())}
        rows['identifiability_limits'] = ['current-prefix age equals time since the metadata-day midnight; staleness is not independently varied',
            'pooled prefixes jointly change neural quality, graph history, age and prefix volume',
            'no pooled37, raw-current47 or feature-group training ablations exist; importance is descriptive']
        report['cases'][case] = rows
        print('audited', case, len(ix), flush=True)
    unchanged(root, inputs)
    return report


def worker(root, token):
    work = root / WORK; parent = root / 'artifacts/rebuild'
    claim = read(work / 'attempts' / (token + '.claim.json'))
    genesis = read(parent / 'genesis.json'); latest = read(parent / 'compute-ledger.json')['attempts'][-1]
    require(latest['owner'] == genesis['generation'] + '/audit-v75-' + token and latest['status'] == 'running'
            and latest['controller_pid'] == os.getppid(), 'Audit worker requires its owning budget controller')
    folder = inside(work, claim['snapshot_relative']); verify_snapshot(folder, claim['source'])
    require(ROOT == folder / 'source', 'Audit must execute frozen source')
    require(genesis['runtime'] == {n: importlib.metadata.version(n) for n in genesis['runtime']}
            and genesis['python'] == sys.version, 'Frozen runtime changed')
    engine._start_watchdog(latest['controller_pid'])
    result = work / 'attempts' / (token + '.result.json')
    write_new(result, compute(root, claim['source']))
    write_new(work / 'attempts' / (token + '.validated.json'), {'report': result.relative_to(work).as_posix(),
              'sha256': digest(result), 'source_sha256': claim['source']['sha256']})


def verify(root):
    work = root / WORK; receipt = read(work / 'receipt.json')
    result = inside(work, receipt['report']); require(digest(result) == receipt['report_sha256'], 'Audit report changed')
    report = read(result); unchanged(root, report['protected_inputs']); original_data_identity(root)
    ledger = inside(work, receipt['accounting_record']); require(digest(ledger) == receipt['accounting_sha256'], 'Audit accounting seal changed')
    state = read(ledger); prior = receipt['prior_attempts']
    require(canonical(state['attempts'][:prior]) == receipt['prior_attempts_sha256']
            and state['attempts'][-1]['owner'] == receipt['owner'] and state['attempts'][-1]['status'] == 'complete'
            and not any(a['status'] == 'running' for a in state['attempts'])
            and abs(sum(a['seconds'] for a in state['attempts']) - state['used_seconds']) < 1e-6, 'Audit accounting does not reconcile')
    snapshot = work / 'snapshots' / report['audit_source']['commit']; verify_snapshot(snapshot, report['audit_source'])
    successor.validate_design(read(work / 'successor_spec.json'))
    require(report['diagnostic_only'] is True and report['promotion_supported'] is False
            and report['new_model_training_performed'] is False, 'Audit cannot claim promotion')
    output = root / 'docs/results/v7.5-audit'
    for name, expected in (('audit.json', report), ('receipt.json', receipt), ('accounting.json', state),
                           ('successor_spec.json', successor.design())):
        if (output / name).exists(): require(read(output / name) == expected, 'Published audit evidence changed: ' + name)
    return report, receipt


def publish(root):
    report, receipt = verify(root); work = root / WORK
    output = root / 'docs/results/v7.5-audit'
    for name, value in (('audit.json', report), ('accounting.json', read(inside(work, receipt['accounting_record']))),
                        ('receipt.json', receipt), ('successor_spec.json', successor.design())):
        pipeline.write_same(output / name, value)


def run(root):
    work = root / WORK; work.mkdir(parents=True, exist_ok=True)
    with (work / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (work / 'receipt.json').exists(): publish(root); return
        parent = root / 'artifacts/rebuild'; genesis = read(parent / 'genesis.json')
        token = str(time.time_ns()); owner = genesis['generation'] + '/audit-v75-' + token
        before = read(parent / 'compute-ledger.json')
        require(not any(a['status'] == 'running' for a in before['attempts']), 'Settle prior accounting before the audit')
        require(43200 - before['used_seconds'] >= 1200 + 3600, 'Protect finalization reserve')
        with rebuild.ReconstructionBudget(parent / 'compute-ledger.json', owner, 1200, 43200,
                                         carry_identity=digest(parent / 'genesis.json')) as budget:
            folder, source = rebuild.freeze(root, work)
            write_new(work / 'attempts' / (token + '.claim.json'), {'source': source, 'snapshot_relative': folder.relative_to(work).as_posix()})
            command = [sys.executable, str(folder / 'source' / RELATIVE / 'audit.py'), 'run', '--root', str(root), '--worker', '--token', token]
            require(budget.run(command, folder / 'source') == 0, 'Audit failed; preserve charged attempt')
            validation = read(work / 'attempts' / (token + '.validated.json'))
            require(validation['source_sha256'] == source['sha256'] and digest(inside(work, validation['report'])) == validation['sha256'], 'Audit output changed')
        # Publication follows the durable final accounting write; never inside the lease.
        settled = read(parent / 'compute-ledger.json'); seal = work / 'accounting' / (digest(parent / 'compute-ledger.json') + '.json')
        write_new(seal, settled); write_new(work / 'successor_spec.json', successor.design())
        write_new(work / 'receipt.json', {'report': validation['report'], 'report_sha256': validation['sha256'], 'owner': owner,
                  'accounting_record': seal.relative_to(work).as_posix(), 'accounting_sha256': digest(seal),
                  'prior_attempts': len(before['attempts']), 'prior_attempts_sha256': canonical(before['attempts']),
                  'diagnostic_only': True, 'promotion_supported': False})
        publish(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'verify'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--token', help=argparse.SUPPRESS)
    args = parser.parse_args(); root = args.root.resolve()
    if args.worker:
        require(args.action == 'run' and args.token, 'Missing audit worker claim'); worker(root, args.token)
    elif args.action == 'run': run(root)
    else:
        report, _ = verify(root)
        print(json.dumps({'verified': True, 'diagnostic_only': True, 'v7_5_promoted': False, 'cases': list(report['cases'])}))


if __name__ == '__main__': main()
