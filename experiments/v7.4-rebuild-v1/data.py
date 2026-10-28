"""Two-pass streaming conversion with deterministic opaque-ID dictionaries."""
import csv
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from ml_workflow.contracts import digest, first_indices, require, validate_events
from ml_workflow.storage import read, write_new, save_arrays
from recipe import DAY, JAN20, FULL


def rows(archive, name):
    with zipfile.ZipFile(archive) as z:
        with z.open(name + '.csv') as raw:
            yield from csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig', newline=''))


def prepare(root, destination, expected=None):
    """No target labels enter vocabulary construction or any fitting statistic."""
    expected = expected or {'train': 686824, 'evaluation': 52939}
    require(not destination.exists(), 'Preserve existing reconstructed data')
    vocab = {'users': set(), 'items': set()}
    widths, counts = {}, {}
    raw_hashes = {}
    for name in ('train', 'evaluation'):
        path = root / 'data/raw' / (name + '.zip')
        raw_hashes[path.relative_to(root).as_posix()] = digest(path)
        seen_rows, seen_events = set(), set()
        last = None; width = 1; count = 0
        for row in rows(path, name):
            require(row['rowid'] not in seen_rows, 'Duplicate row ID')
            seen_rows.add(row['rowid']); width = max(width, len(row['rowid']))
            vocab['users'].add(row['user_id']); vocab['items'].add(row['item_id'])
            if row['event_id'] != last:
                require(row['event_id'] not in seen_events, 'Noncontiguous repeated event')
                seen_events.add(row['event_id']); last = row['event_id']
                vocab['items'].update(json.loads(row['history_item_ids']))
            count += 1
            if count % 1000000 == 0: print('vocabulary', name, count, flush=True)
        require(count == expected[name] * 10 and len(seen_events) == expected[name], 'Official event/row count differs')
        widths[name], counts[name] = width, len(seen_events)
        del seen_rows, seen_events
    values = {name: sorted(items) for name, items in vocab.items()}
    maps = {name: {key: i for i, key in enumerate(items)} for name, items in values.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.data-', dir=destination.parent) as temporary:
        stage = Path(temporary) / 'data'; stage.mkdir()
        write_new(stage / 'vocabulary.json', values)
        shapes = {}
        for name in ('train', 'evaluation'):
            n = counts[name]
            fields = {'u': (np.int32, (n,)), 'i': (np.int32, (n, 10)),
                      't': (np.int64, (n,)), 'tb': (np.int8, (n,)), 'dt': (np.int8, (n,)),
                      'h': (np.int32, (n, 20)), 'ht': (np.int64, (n, 20)),
                      'rowids': ('U' + str(widths[name]), (n, 10))}
            if name == 'train': fields['y'] = (np.int8, (n, 10))
            arrays = {key: np.lib.format.open_memmap(stage / (name + '-' + key + '.npy'),
                      mode='w+', dtype=dtype, shape=shape) for key, (dtype, shape) in fields.items()}
            arrays['h'][:] = -1; arrays['ht'][:] = 0
            group = []; index = 0; last = None
            def store(group, index):
                require(len(group) == 10, 'Events must contain ten candidates')
                a = group[0]
                shared = ('event_id', 'user_id', 'timestamp', 'time_bucket', 'day_type', 'history_item_ids', 'history_timestamps')
                require(all(all(row[k] == a[k] for k in shared) for row in group), 'Event context differs across rows')
                arrays['u'][index] = maps['users'][a['user_id']]
                arrays['i'][index] = [maps['items'][row['item_id']] for row in group]
                arrays['rowids'][index] = [row['rowid'] for row in group]
                for key, column in (('t', 'timestamp'), ('tb', 'time_bucket'), ('dt', 'day_type')):
                    arrays[key][index] = int(a[column])
                h, ht = json.loads(a['history_item_ids']), json.loads(a['history_timestamps'])
                require(len(h) == len(ht) <= 20 and ht == sorted(ht), 'Invalid history shape/order')
                require(all(t < int(a['timestamp']) for t in ht), 'History crosses event timestamp')
                arrays['h'][index, :len(h)] = [maps['items'][item] for item in h]
                arrays['ht'][index, :len(ht)] = ht
                if name == 'train': arrays['y'][index] = [int(row['clicked']) for row in group]
                else: require(all('clicked' not in row for row in group), 'Evaluation labels forbidden')
            for row in rows(root / 'data/raw' / (name + '.zip'), name):
                if group and row['event_id'] != last:
                    store(group, index); index += 1; group = []
                    if index % 100000 == 0: print('cache', name, index, flush=True)
                group.append(row); last = row['event_id']
            if group: store(group, index); index += 1
            require(index == n, 'Event count changed between passes')
            # Validate bounded batches to avoid a multi-million-ID temporary set.
            for start in range(0, n, 4096):
                validate_events({k: v[start:start + 4096] for k, v in arrays.items()}, name == 'train')
            require((arrays['tb'] == (arrays['t'] // 21600) % 4).all()
                    and (arrays['dt'] == ((arrays['t'] // DAY + 3) % 7 >= 5)).all(), 'UTC context differs')
            if name == 'train':
                require(((arrays['t'] >= JAN20) & (arrays['t'] < FULL)).all(), 'Training timeline differs')
                if n == 686824: require(int(arrays['y'].sum()) == 432023, 'Official click count differs')
            else:
                require((arrays['t'] >= FULL).all() and len(np.unique(arrays['u'])) == n, 'Evaluation users/timeline differ')
            for key, array in arrays.items():
                array.flush(); shapes[name + '-' + key] = {'shape': list(array.shape), 'dtype': array.dtype.str}
            del arrays
        for relative, expected_hash in raw_hashes.items():
            require(digest(root / relative) == expected_hash, 'Raw input changed during reconstruction')
        metadata = {'schema_version': 5, 'raw_sha256': raw_hashes,
                    'vocabulary_sha256': digest(stage / 'vocabulary.json'),
                    'n_users': len(values['users']), 'n_items': len(values['items']),
                    'arrays': {p.name: {'sha256': digest(p), **shapes[p.stem]} for p in sorted(stage.glob('*.npy'))}}
        write_new(stage / 'metadata.json', metadata)
        stage.rename(destination)
    return metadata


def load(folder, name):
    return {p.stem[len(name) + 1:]: np.load(p, mmap_mode='r', allow_pickle=False)
            for p in sorted(folder.glob(name + '-*.npy'))}


def prefix_examples(raw, cutoff, n_users, n_items):
    fitting = np.flatnonzero(raw['t'] < cutoff)
    require(len(fitting), 'Empty fitting prefix')
    first = first_indices(raw['u'], raw['t'], JAN20, cutoff)
    h, ht = raw['h'][first], raw['ht'][first]
    present = h >= 0
    event, column = np.nonzero(raw['y'][fitting])
    clicked = fitting[event]
    result = {'users': np.concatenate([np.broadcast_to(raw['u'][first, None], h.shape)[present], raw['u'][clicked]]).astype(np.int64),
              'items': np.concatenate([h[present], raw['i'][clicked, column]]).astype(np.int64),
              'timestamps': np.concatenate([ht[present], raw['t'][clicked]]).astype(np.int64),
              'n_users': np.asarray(n_users), 'n_items': np.asarray(n_items), 'cutoff': np.asarray(cutoff)}
    require((result['timestamps'] < cutoff).all() and np.isin(result['users'], raw['u'][fitting]).all(), 'Future preference examples')
    return result


def verify(folder, keys=None):
    metadata = read(folder / 'metadata.json')
    require(digest(folder / 'vocabulary.json') == metadata['vocabulary_sha256'], 'Vocabulary changed')
    for name, info in metadata['arrays'].items():
        if keys is None or name in keys:
            require(digest(folder / name) == info['sha256'], 'Reconstructed input changed: ' + name)
    return metadata
