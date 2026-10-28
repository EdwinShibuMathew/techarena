"""Portable fixed-epoch reconstruction with exact epoch-boundary resume."""
from pathlib import Path
import os
import time

import numpy as np
import torch
from torch import nn

from ml_workflow.contracts import canonical, digest, require
from consolidate import write_json
from recipe import MEMBERS, RECIPE
import data as dataset


class PreferenceModel(nn.Module):
    def __init__(self, n_users, n_items, dimension, active_users):
        super().__init__()
        self.user = nn.Embedding(n_users, dimension)
        self.item = nn.Embedding(n_items, dimension)
        self.item_bias = nn.Embedding(n_items, 1)
        self.register_buffer('active_users', active_users)
        nn.init.normal_(self.user.weight, std=.1)
        nn.init.normal_(self.item.weight, std=.1)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users, items):
        u = self.user(users) * self.active_users[users, None]
        return (u[:, None, :] * self.item(items)).sum(-1) + self.item_bias(items).squeeze(-1)


class EventModel(nn.Module):
    def __init__(self, preference, n_users, n_items):
        super().__init__()
        self.preference = preference
        self.user_bias = nn.Embedding(n_users, 1)
        self.user_context = nn.Embedding(n_users, 8)
        self.item_context = nn.Embedding(n_items, 8)
        self.global_bias = nn.Parameter(torch.tensor(0.))
        for layer in (self.user_bias, self.user_context, self.item_context): nn.init.zeros_(layer.weight)

    def forward(self, users, items, context):
        return (self.preference(users, items) + self.user_bias(users) + self.global_bias
                + self.item_context(items).gather(2, context[:, None, None].expand(-1, items.shape[1], 1)).squeeze(-1)
                + self.user_context(users).gather(1, context[:, None]))


def save(path, model, optimizer, metadata, stage, epoch, pair_generator=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    saved = {**metadata, 'stage': stage, 'epoch': epoch,
             'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
             'cpu_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None,
             'pair_rng': pair_generator.get_state() if pair_generator is not None else None}
    temporary = path.with_suffix(path.suffix + '.staging')
    torch.save(saved, temporary); os.replace(temporary, path)


def restore(path, model, optimizer, metadata, pair_generator=None):
    saved = torch.load(path, map_location=next(model.parameters()).device, weights_only=True)
    require(all(saved.get(k) == value for k, value in metadata.items()), 'Checkpoint provenance differs; no silent resume')
    model.load_state_dict(saved['model_state']); optimizer.load_state_dict(saved['optimizer_state'])
    torch.set_rng_state(saved['cpu_rng'].cpu())
    if saved['cuda_rng'] is not None: torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    if pair_generator is not None and saved['pair_rng'] is not None: pair_generator.set_state(saved['pair_rng'].cpu())
    return saved


def preference_epoch(model, optimizer, users, items, n_items, batch_size=4096):
    model.train(); order = torch.randperm(len(users), device=users.device)
    total = torch.zeros((), device=users.device)
    for start in range(0, len(users), batch_size):
        q = order[start:start + batch_size]; u = users[q]
        negative = torch.randint(n_items, (len(q), 31), device=users.device)
        candidates = torch.cat([items[q, None], negative], 1)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(u, candidates), torch.zeros(len(q), dtype=torch.long, device=users.device))
        loss = loss + .02 * (model.user(u).square().sum(-1).mean() + model.item(candidates).square().sum(-1).mean())
        loss.backward(); optimizer.step(); total += loss.detach() * len(q)
    return float((total / len(users)).item())


def event_epoch(model, optimizer, inputs, target, anchors, objective, pair_generator, batch_size=2048):
    from row_auc_objective import mixed_row_auc_loss
    model.train(); order = torch.randperm(len(target), device=target.device)
    total = torch.zeros((), device=target.device)
    for start in range(0, len(target), batch_size):
        q = order[start:start + batch_size]; u, i, c = [a[q] for a in inputs]
        optimizer.zero_grad(); scores = model(u, i, c)
        if objective == 'mixed': loss = mixed_row_auc_loss(scores, target[q], pair_generator)
        else: loss = nn.functional.cross_entropy(torch.cat([torch.zeros_like(scores[:, :1]), scores], 1), target[q])
        loss = loss + .1 * ((model.preference.user(u) - anchors[0][u]).square().sum(-1).mean()
                           + (model.preference.item(i) - anchors[1][i]).square().sum(-1).mean())
        loss = loss + .05 * (model.user_bias(u).square().mean() + model.user_context(u).square().sum(-1).mean()
                            + model.item_context(i).square().sum(-1).mean())
        loss.backward(); optimizer.step(); total += loss.detach() * len(q)
    return float((total / len(target)).item())


def fit(folder, cache, cutoff, member, source_identity, device='cuda', synthetic=False):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    dimension, seed, final_epoch, objective = MEMBERS[member]
    initial_epoch, event_epochs = (2, 2) if synthetic else (196, 8)
    if synthetic: final_epoch = 3 if dimension == 8 else 2
    metadata = dataset.verify(cache, ['train-' + k + '.npy' for k in ('u', 'i', 't', 'tb', 'dt', 'h', 'ht', 'y')])
    raw = dataset.load(cache, 'train')
    folder.mkdir(parents=True, exist_ok=True)
    positive = dataset.prefix_examples(raw, cutoff, metadata['n_users'], metadata['n_items'])
    positive_path = folder / 'positive.npz'
    if not positive_path.exists():
        from ml_workflow.storage import save_arrays
        save_arrays(positive_path, **positive)
    else:
        with np.load(positive_path, allow_pickle=False) as stored:
            require(all(np.array_equal(positive[k], stored[k]) for k in positive), 'Preference examples changed')
    expected = {'schema_version': 5, 'lineage': 'rebuilt_recipe', 'dimension': dimension, 'seed': seed,
                'n_users': metadata['n_users'], 'n_items': metadata['n_items'], 'cutoff': cutoff,
                'recipe_sha256': canonical(RECIPE), 'source_identity': source_identity,
                'data_sha256': digest(cache / 'metadata.json'), 'positive_sha256': digest(positive_path),
                'purpose': 'synthetic_validation' if synthetic else 'research', 'event_objective': objective}
    if (folder / 'event.pt').exists():
        saved = torch.load(folder / 'event.pt', map_location='cpu', weights_only=True)
        require(all(saved.get(k) == v for k, v in expected.items()) and saved['epoch'] == event_epochs
                and saved.get('preference_source_sha256') == digest(folder / 'preference.pt'), 'Completed checkpoint differs')
        return {'reused_completed': True, 'checkpoint': str(folder / 'event.pt')}
    torch.manual_seed(seed); dev = torch.device(device)
    active = torch.tensor(np.bincount(positive['users'], minlength=metadata['n_users']) > 0)
    pref = PreferenceModel(metadata['n_users'], metadata['n_items'], dimension, active).to(dev)
    optimizer = torch.optim.Adam(pref.parameters(), lr=.012)
    start_epoch = 0
    last = folder / 'preference.last.pt'
    if last.exists(): start_epoch = restore(last, pref, optimizer, expected)['epoch']
    users = torch.tensor(positive['users'], dtype=torch.long, device=dev)
    items = torch.tensor(positive['items'], dtype=torch.long, device=dev)
    started = time.monotonic()
    for epoch in range(start_epoch + 1, final_epoch + 1):
        if epoch == initial_epoch + 1:
            for group in optimizer.param_groups: group['lr'] = .003
            torch.manual_seed(seed + 1000)
        loss = preference_epoch(pref, optimizer, users, items, metadata['n_items'])
        require(np.isfinite(loss), 'Nonfinite preference training')
        save(last, pref, optimizer, expected, 'preference', epoch)
        if epoch == initial_epoch: save(folder / 'preference-initial.pt', pref, optimizer, expected, 'preference', epoch)
        if epoch % 10 == 0 or epoch in (1, initial_epoch, final_epoch):
            print('preference', cutoff, member, epoch, 'loss', loss, 'seconds', round(time.monotonic() - started, 1), flush=True)
        write_json(folder / 'progress.json', {'member': member, 'cutoff': cutoff, 'stage': 'preference', 'epoch': epoch,
                                             'loss': loss, 'seconds_this_attempt': time.monotonic() - started})
    final_preference = folder / 'preference.pt'
    if final_preference.exists():
        previous = torch.load(final_preference, map_location=dev, weights_only=True)
        require(previous['epoch'] == final_epoch and all(previous.get(k) == v for k, v in expected.items())
                and all(torch.equal(previous['model_state'][k], v) for k, v in pref.state_dict().items()),
                'Completed preference checkpoint differs')
    else:
        save(final_preference, pref, optimizer, expected, 'preference', final_epoch)
    del optimizer, pref, users, items
    torch.manual_seed(seed)
    pref = PreferenceModel(metadata['n_users'], metadata['n_items'], dimension, torch.ones(metadata['n_users'], dtype=torch.bool))
    saved_pref = torch.load(folder / 'preference.pt', map_location='cpu', weights_only=True)
    require(saved_pref['epoch'] == final_epoch, 'Preference schedule incomplete')
    pref.load_state_dict(saved_pref['model_state']); del saved_pref
    expected = {**expected, 'preference_source_sha256': digest(final_preference)}
    model = EventModel(pref, metadata['n_users'], metadata['n_items']).to(dev)
    anchors = (model.preference.user.weight.detach().clone(), model.preference.item.weight.detach().clone())
    ix = np.flatnonzero(raw['t'] < cutoff)
    inputs = [torch.tensor(raw['u'][ix], dtype=torch.long, device=dev),
              torch.tensor(raw['i'][ix], dtype=torch.long, device=dev),
              torch.tensor(raw['tb'][ix].astype(np.int64) + 4 * raw['dt'][ix], dtype=torch.long, device=dev)]
    labels = raw['y'][ix]
    target = torch.tensor(np.where(labels.sum(1) > 0, labels.argmax(1) + 1, 0), dtype=torch.long, device=dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0015)
    pairs = torch.Generator(device=dev).manual_seed(seed + 3000)
    last = folder / 'event.last.pt'; start_epoch = 0
    if last.exists(): start_epoch = restore(last, model, optimizer, expected, pairs)['epoch']
    else:
        with torch.inference_mode():
            totals = torch.cat([model(*[a[st:st + 4096] for a in inputs]).logsumexp(1) for st in range(0, len(target), 4096)])
            rate = (target > 0).float().mean(); low, high = -40., 40.
            for _ in range(30):
                middle = (low + high) / 2
                if torch.sigmoid(totals + middle).mean() > rate: high = middle
                else: low = middle
            model.global_bias.fill_((low + high) / 2)
    for epoch in range(start_epoch + 1, event_epochs + 1):
        loss = event_epoch(model, optimizer, inputs, target, anchors, objective, pairs)
        require(np.isfinite(loss), 'Nonfinite event training')
        save(last, model, optimizer, expected, 'event', epoch, pairs)
        write_json(folder / 'progress.json', {'member': member, 'cutoff': cutoff, 'stage': 'event', 'epoch': epoch,
                                             'loss': loss, 'seconds_this_attempt': time.monotonic() - started})
        print('event', cutoff, member, epoch, 'loss', loss, 'seconds', round(time.monotonic() - started, 1), flush=True)
    save(folder / 'event.pt', model, optimizer, expected, 'event', event_epochs, pairs)
    return {'member': member, 'cutoff': cutoff, 'preference_epoch': final_epoch, 'event_epoch': event_epochs,
            'event_sha256': digest(folder / 'event.pt'), 'seconds': time.monotonic() - started}
