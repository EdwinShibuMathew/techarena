"""From-raw provenance, chronology and exact technical-resume contracts."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'experiments/v7.4-rebuild-v1'
sys.path[:0] = [str(DIRECTORY), str(ROOT / 'tools'), str(ROOT / 'legacy')]
import recipe
import data as dataset
from ml_workflow.contracts import digest


def raw_fixture(root):
    directory = root / 'data/raw'; directory.mkdir(parents=True)
    for name, times in [('train', [recipe.JAN20, recipe.JAN20 + 1, recipe.JAN20 + recipe.DAY]),
                        ('evaluation', [recipe.FULL])]:
        records = []
        for e, timestamp in enumerate(times):
            for j in range(10):
                row = dict(rowid=f'{name}-{e}-{j}', event_id=f'{e}', user_id='u' + str(e // 2),
                           item_id=f'item-{j:02}', timestamp=timestamp,
                           time_bucket=timestamp // 21600 % 4, day_type=int((timestamp // 86400 + 3) % 7 >= 5),
                           history_item_ids=json.dumps(['item-01', 'item-01']),
                           history_timestamps=json.dumps([timestamp - 60, timestamp - 30]))
                if name == 'train': row['clicked'] = int(j == 2)
                records.append(row)
        stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
        with zipfile.ZipFile(directory / (name + '.zip'), 'w') as z:
            z.writestr(name + '.csv', stream.getvalue())
    return {'train': 3, 'evaluation': 1}


def test_streaming_cache_and_causal_multiset(tmp_path):
    destination = tmp_path / 'cache'
    metadata = dataset.prepare(tmp_path, destination, raw_fixture(tmp_path))
    assert metadata['n_users'] == 2 and metadata['n_items'] == 10
    raw = dataset.load(destination, 'train')
    assert raw['rowids'][2, 9] == 'train-2-9'
    examples = dataset.prefix_examples(raw, recipe.JAN20 + recipe.DAY, 2, 10)
    assert examples['items'].tolist() == [1, 1, 2, 2]
    assert examples['users'].tolist() == [0, 0, 0, 0]
    assert dataset.verify(destination) == metadata
    with (destination / 'train-i.npy').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='input changed'): dataset.verify(destination)


def test_checkpoint_resume_restores_optimizer_and_all_randomness(tmp_path):
    torch = pytest.importorskip('torch')
    import train
    torch.manual_seed(42)
    model = train.PreferenceModel(4, 12, 8, torch.ones(4, dtype=torch.bool))
    opt = torch.optim.Adam(model.parameters(), lr=.012)
    users, items = torch.arange(4), torch.arange(4)
    train.preference_epoch(model, opt, users, items, 12)
    train.save(tmp_path / 'last.pt', model, opt, {'source': 'a'}, 'preference', 1)
    train.preference_epoch(model, opt, users, items, 12)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    restored = train.PreferenceModel(4, 12, 8, torch.ones(4, dtype=torch.bool))
    other_opt = torch.optim.Adam(restored.parameters(), lr=.99)
    train.restore(tmp_path / 'last.pt', restored, other_opt, {'source': 'a'})
    train.preference_epoch(restored, other_opt, users, items, 12)
    for key, value in restored.state_dict().items(): assert torch.equal(value, expected[key])
    with pytest.raises(ValueError, match='provenance'):
        train.restore(tmp_path / 'last.pt', restored, other_opt, {'source': 'changed'})


def test_synthetic_training_completion_is_immutable(tmp_path):
    torch = pytest.importorskip('torch')
    import train
    cache = tmp_path / 'cache'; folder = tmp_path / 'models'
    dataset.prepare(tmp_path, cache, raw_fixture(tmp_path))
    train.fit(folder, cache, recipe.JAN20 + recipe.DAY, 'dim8_seed42', 'test', 'cpu', synthetic=True)
    before = digest(folder / 'preference.pt'), digest(folder / 'event.pt')
    train.fit(folder, cache, recipe.JAN20 + recipe.DAY, 'dim8_seed42', 'test', 'cpu', synthetic=True)
    assert before == (digest(folder / 'preference.pt'), digest(folder / 'event.pt'))
    saved = torch.load(folder / 'event.pt', weights_only=True)
    assert saved['purpose'] == 'synthetic_validation'
    assert saved['preference_source_sha256'] == before[0]
    with pytest.raises(ValueError, match='Completed checkpoint differs'):
        train.fit(folder, cache, recipe.JAN20 + recipe.DAY, 'dim8_seed42', 'changed', 'cpu', synthetic=True)
