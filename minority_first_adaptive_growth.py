#ccore implementation of adaptive class introduction and capacity learning.

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    #stores the limits and targets used during training
    lr: float = 0.01
    target_ba: float = 0.90
    recall_floor: float = 0.80
    minimum: int = 60
    check_every: int = 5
    persistence: int = 3
    patience: int = 40
    ba_delta: float = 0.002
    loss_delta: float = 0.005
    overfit_loss_rise: float = 0.05
    overfit_train_drop: float = 0.01
    overfit_ba_drop: float = 0.005
    probe_epochs: int = 100
    growth_gain: float = 0.005
    recall_drop: float = 0.02
    max_probes: int = 4
    max_width: int = 4
    stage_updates: int = 1200
    max_updates: int = 9000
    max_examples: int = 60_000_000
    max_seconds: float = 590

    def __post_init__(self):
        #stops the algorithm early when its settings are not sensible
        assert 0 <= self.recall_floor <= self.target_ba <= 1
        assert self.minimum >= self.check_every > 0 and self.patience > 0
        assert self.persistence > 0 and self.probe_epochs > 0
        assert min(self.stage_updates, self.max_updates, self.max_examples) > 0
        assert self.max_probes >= 0 and self.max_width >= 0 and self.max_seconds > 0


class DirectHiddenNet(nn.Module):
    def __init__(self, features, classes, width=0):
        super().__init__()
        #the direct layer keeps the original model active as capacity grows
        self.direct = nn.Linear(features, classes)
        self.hidden_in = nn.Linear(features, width) if width else None
        self.hidden_out = nn.Linear(width, classes, bias=False) if width else None

    @property
    def width(self):
        return self.hidden_in.out_features if self.hidden_in is not None else 0

    def forward(self, x):
        scores = self.direct(x)
        if self.width:
            scores = scores + self.hidden_out(torch.tanh(self.hidden_in(x)))
        return scores

    def grow_one(self):
        #adds one hidden unit while preserving the current model
        old = self.width
        incoming = nn.Linear(self.direct.in_features, old + 1)
        outgoing = nn.Linear(old + 1, self.direct.out_features, bias=False)
        with torch.no_grad():
            if old:
                incoming.weight[:old].copy_(self.hidden_in.weight)
                incoming.bias[:old].copy_(self.hidden_in.bias)
                outgoing.weight[:, :old].copy_(self.hidden_out.weight)
            outgoing.weight[:, old].zero_()
        self.hidden_in, self.hidden_out = incoming, outgoing

    def add_class(self):
        #adds an output class without changing earlier class weights
        old = self.direct.out_features
        output = nn.Linear(self.direct.in_features, old + 1)
        with torch.no_grad():
            output.weight[:old].copy_(self.direct.weight)
            output.bias[:old].copy_(self.direct.bias)
        self.direct = output
        if self.width:
            output = nn.Linear(self.width, old + 1, bias=False)
            with torch.no_grad():
                output.weight[:old].copy_(self.hidden_out.weight)
            self.hidden_out = output


@dataclass
class Data:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor

    def __post_init__(self):
        #checks that the training and validation tensors use the expected format
        for x, y in ((self.x_train, self.y_train), (self.x_val, self.y_val)):
            assert x.device.type == y.device.type == 'cpu'
            assert x.dtype == torch.float32 and y.dtype == torch.int64
            assert x.ndim == 2 and y.ndim == 1 and len(x) == len(y)
            assert torch.isfinite(x).all() and len(x) > 0
        assert self.x_train.shape[1] == self.x_val.shape[1]
        labels, counts = torch.unique(self.y_train, sorted=True, return_counts=True)
        assert len(labels) >= 2 and torch.equal(labels, torch.arange(len(labels)))
        assert torch.equal(labels, torch.unique(self.y_val, sorted=True))
        assert torch.all(counts[1:] >= counts[:-1]), 'Encode labels by training count.'

    @property
    def classes(self):
        return int(self.y_train.max()) + 1

    def stage(self, k):
        #returns only the classes that belong to the current learning stage
        train, val = self.y_train < k, self.y_val < k
        return self.x_train[train], self.y_train[train], self.x_val[val], self.y_val[val]


def metrics(model, x, y):
    """Equal-class BA/F1; the training loss itself remains unweighted."""
    #computes validation measures from a confusion matrix
    model.eval()
    with torch.no_grad():
        logits = model(x)
        k = logits.shape[1]
        cm = np.bincount((y*k + logits.argmax(1)).numpy(), minlength=k*k).reshape(k, k)
        recall = np.diag(cm) / cm.sum(1)
        denominator = cm.sum(0) + cm.sum(1)
        f1 = np.divide(2*np.diag(cm), denominator, out=np.zeros(k), where=denominator > 0)
        return dict(loss=nn.functional.cross_entropy(logits, y).item(),
                    ba=float(recall.mean()), f1=float(f1.mean()),
                    min_recall=float(recall.min()), recall=recall.tolist(),
                    class_f1=f1.tolist(), support=cm.sum(1).tolist())


def rank(score, cfg):
    return (score['min_recall'] >= cfg.recall_floor, score['ba'], -score['loss'])


def acceptable(score, cfg):
    return score['ba'] >= cfg.target_ba and score['min_recall'] >= cfg.recall_floor


class Monitor:
    """Recheck evidence every check_every updates; never equate a cap to a plateau."""
    def __init__(self, train, val, cfg):
        self.cfg = cfg
        self.ba_anchor = self.best_ba = val['ba']
        self.loss_anchor = self.best_loss = val['loss']
        self.train_at_best_loss = train['loss']
        self.ba_epoch = self.loss_epoch = 0
        self.ready_count = self.overfit_count = 0

    def observe(self, epoch, train, val, allow_ready=True):
        #tracks improvement, deterioration, and repeated acceptable scores
        c = self.cfg
        self.best_ba = max(self.best_ba, val['ba'])
        if val['loss'] < self.best_loss:
            self.best_loss, self.train_at_best_loss = val['loss'], train['loss']
        if val['ba'] > self.ba_anchor + c.ba_delta:
            self.ba_anchor, self.ba_epoch = val['ba'], epoch
        if val['loss'] < self.loss_anchor * (1-c.loss_delta):
            self.loss_anchor, self.loss_epoch = val['loss'], epoch
        if epoch < c.minimum or epoch % c.check_every:
            return None
        deteriorating = (
            val['loss'] >= self.best_loss * (1+c.overfit_loss_rise)
            and train['loss'] <= self.train_at_best_loss * (1-c.overfit_train_drop)
            and val['ba'] <= self.best_ba - c.overfit_ba_drop)
        self.overfit_count = self.overfit_count + 1 if deteriorating else 0
        self.ready_count = self.ready_count + 1 if acceptable(val, c) else 0
        if self.overfit_count >= c.persistence:
            return 'overfit'
        if self.overfit_count:
            return None  #confirm deterioration before considering a capacity probe.
        if allow_ready and self.ready_count >= c.persistence:
            return 'acceptable'
        if allow_ready and self.ready_count:
            return None  #let an adequate score establish persistence before probing.
        if epoch-self.ba_epoch >= c.patience and epoch-self.loss_epoch >= c.patience:
            return 'plateau'
        return None


class Budget:
    def __init__(self, cfg, deadline=None):
        self.cfg, self.updates, self.examples, self.stage_updates = cfg, 0, 0, 0
        self.started = perf_counter()
        self.deadline = min(self.started+cfg.max_seconds, deadline or float('inf'))

    def stop_reason(self, rows, updates=1):
        #checks every limit before another update is started
        if perf_counter() >= self.deadline:
            return 'runtime_limit'
        if self.updates + updates > self.cfg.max_updates:
            return 'update_limit'
        if self.examples + rows*updates > self.cfg.max_examples:
            return 'example_limit'
        if self.stage_updates + updates > self.cfg.stage_updates:
            return 'stage_limit'
        return None

    def charge(self, rows):
        self.updates += 1
        self.stage_updates += 1
        self.examples += rows


def train_segment(model, batch, cfg, budget, run_id, curves, *,
                  epochs=None, adaptive=True):
    """Return a selected checkpoint. Fixed-length probe segments disable decisions."""
    #keeps the best model found during this training segment
    xt, yt, xv, yv = batch
    train, val = metrics(model, xt, yt), metrics(model, xv, yv)
    best = dict(model=deepcopy(model), score=val, epoch=0, run_id=run_id)
    monitor = Monitor(train, val, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    began, epoch, reason = perf_counter(), 0, None

    def record():
        curves.append(dict(run_id=run_id, classes=model.direct.out_features,
                           width=model.width, epoch=epoch, updates=budget.updates,
                           examples=budget.examples, stage_updates=budget.stage_updates,
                           train_loss=train['loss'], validation_loss=val['loss'],
                           train_ba=train['ba'], validation_ba=val['ba'],
                           train_f1=train['f1'], validation_f1=val['f1'],
                           validation_min_recall=val['min_recall']))

    record()
    while epochs is None or epoch < epochs:
        reason = budget.stop_reason(len(xt))
        if reason:
            break
        model.train()
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(xt), yt).backward()
        optimizer.step()
        budget.charge(len(xt))
        epoch += 1
        train, val = metrics(model, xt, yt), metrics(model, xv, yv)
        record()
        if rank(val, cfg) > rank(best['score'], cfg):
            best = dict(model=deepcopy(model), score=val, epoch=epoch, run_id=run_id)
        if epochs is None:
            reason = monitor.observe(epoch, train, val, allow_ready=adaptive)
            if reason:
                break
    if reason is None:
        reason = 'probe_complete'
    return dict(best=best, reason=reason, epochs=epoch,
                seconds=perf_counter()-began, train=train, val=val)


def probe_choice(start, continuation, grown, counts, cfg):
    """A practical gain threshold; no statistical-significance claim."""
    #compares continuing with adding capacity using recall safeguards
    counts = np.asarray(counts)
    gain = max(cfg.growth_gain, 1/(len(counts)*counts.min()))
    allowed_drop = np.maximum(cfg.recall_drop, 1/counts)
    guards = all(np.all(np.asarray(grown['recall']) >= np.asarray(other['recall'])
                        - allowed_drop - 1e-12) for other in (start, continuation))
    if (grown['ba'] >= max(start['ba'], continuation['ba']) + gain - 1e-12
            and rank(grown, cfg) > rank(continuation, cfg)
            and rank(grown, cfg) > rank(start, cfg) and guards):
        return 'grow_supported_by_probe', gain
    if (rank(continuation, cfg) > rank(start, cfg)
            and (continuation['ba'] >= start['ba'] + gain - 1e-12
                 or continuation['loss'] <= start['loss'] * (1-cfg.loss_delta))):
        return 'continue_supported_by_probe', gain
    return 'unresolved_no_probe_gain', gain


def fit_adaptive(data, condition='B', cfg=Config(), seed=42, deadline=None):
    """B introduces classes; C starts with all. No test argument is accepted."""
    #condition B grows the class set, while condition C starts with every class
    if condition not in ('B', 'C'):
        raise ValueError('condition must be B or C')
    torch.manual_seed(seed)
    k = 2 if condition == 'B' else data.classes
    model = DirectHiddenNet(data.x_train.shape[1], k)
    budget, curves, decisions, stages = Budget(cfg, deadline), [], [], []
    seconds, completed = 0., False
    while True:
        budget.stage_updates, probes, segment = 0, 0, 0
        batch = data.stage(k)
        stage_best = dict(model=deepcopy(model), score=metrics(model, batch[2], batch[3]),
                          run_id=f'{condition}_k{k}_initial', epoch=0)
        while True:
            #train the current stage before deciding whether to probe or advance
            segment += 1
            result = train_segment(model, batch, cfg, budget,
                                   f'{condition}_k{k}_s{segment}', curves)
            seconds += result['seconds']
            if rank(result['best']['score'], cfg) > rank(stage_best['score'], cfg):
                stage_best = result['best']
            reason = result['reason']
            decisions.append(dict(condition=condition, classes=k, segment=segment,
                                  event=reason, width=result['best']['model'].width,
                                  updates=budget.updates, stage_updates=budget.stage_updates,
                                  train_ba=result['train']['ba'], val_ba=result['val']['ba'],
                                  training_inadequate=not acceptable(result['train'], cfg)))
            if reason != 'plateau':
                break
            if stage_best['model'].width >= cfg.max_width:
                reason = 'width_limit'; break
            if probes >= cfg.max_probes:
                reason = 'probe_limit'; break
            reason = budget.stop_reason(len(batch[0]), 2*cfg.probe_epochs)
            if reason:
                break
            probes += 1
            base = stage_best['model']
            continuation, grown = deepcopy(base), deepcopy(base)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 1000 + 100*k + probes)
                with torch.no_grad():
                    before = grown(batch[2][:16]).clone()
                grown.grow_one()
                with torch.no_grad():
                    assert torch.allclose(before, grown(batch[2][:16]), atol=1e-6)
            pair = []
            #give continuation and growth the same fixed probe budget
            for label, branch in [('continue', continuation), ('grow', grown)]:
                r = train_segment(branch, batch, cfg, budget,
                                  f'{condition}_k{k}_p{probes}_{label}', curves,
                                  epochs=cfg.probe_epochs)
                pair.append(r)
                seconds += r['seconds']
            if any(r['reason'] != 'probe_complete' for r in pair):
                reason = 'incomplete_probe'; break
            cont, grow = pair[0]['best'], pair[1]['best']
            action, required_gain = probe_choice(stage_best['score'], cont['score'],
                                                 grow['score'], stage_best['score']['support'], cfg)
            decisions.append(dict(condition=condition, classes=k, event=action,
                                  probe=probes, start_ba=stage_best['score']['ba'],
                                  continuation_ba=cont['score']['ba'], grown_ba=grow['score']['ba'],
                                  start_loss=stage_best['score']['loss'],
                                  continuation_loss=cont['score']['loss'], grown_loss=grow['score']['loss'],
                                  start_recall=stage_best['score']['recall'],
                                  continuation_recall=cont['score']['recall'], grown_recall=grow['score']['recall'],
                                  required_gain=required_gain,
                                  continuation_run=cont['run_id'], grown_run=grow['run_id'],
                                  continuation_best_epoch=cont['epoch'], grown_best_epoch=grow['epoch'],
                                  updates=budget.updates, stage_updates=budget.stage_updates))
            if action == 'unresolved_no_probe_gain':
                reason = action; break
            stage_best = grow if action == 'grow_supported_by_probe' else cont
            model = deepcopy(stage_best['model'])
        model = deepcopy(stage_best['model'])
        advance = reason in ('acceptable', 'overfit')
        stages.append(dict(condition=condition, classes=k, reason=reason,
                           advance=advance, width=model.width, selected_run=stage_best['run_id'],
                           selected_epoch=stage_best['epoch'], **stage_best['score']))
        if not advance or k == data.classes:
            completed = advance and k == data.classes
            break
        with torch.no_grad():
            old = model(batch[2][:16]).clone()
        model.add_class()
        with torch.no_grad():
            assert torch.allclose(old, model(batch[2][:16])[:, :k], atol=1e-6)
        k += 1
    return dict(model=model, summary=dict(condition=condition, stop_reason=reason,
                reached_all_classes=k == data.classes, completed_protocol=completed,
                target_met=acceptable(stage_best['score'], cfg), classes=k, width=model.width,
                parameters=sum(p.numel() for p in model.parameters()), **stage_best['score'],
                updates=budget.updates, examples=budget.examples, training_seconds=seconds,
                wall_seconds=perf_counter()-budget.started), curves=curves,
                decisions=decisions, stages=stages)


def fit_conventional(data, cfg=Config(), seed=42, deadline=None, widths=(0, 4, 8, 16)):
    """Finite search; every fitted candidate contributes to search cost."""
    #fits each requested width and keeps the highest-ranked candidate
    budget, curves, decisions = Budget(cfg, deadline), [], []
    best, seconds, interrupted = None, 0., False
    candidates = sorted(set(widths))
    if not candidates or min(candidates) < 0:
        raise ValueError('Provide nonnegative candidate widths.')
    for width in candidates:
        budget.stage_updates = 0
        if budget.stop_reason(len(data.y_train)):
            interrupted = True; break
        torch.manual_seed(seed+width)
        model = DirectHiddenNet(data.x_train.shape[1], data.classes, width)
        before_updates, before_examples = budget.updates, budget.examples
        result = train_segment(model, data.stage(data.classes), cfg, budget,
                               f'A_w{width}', curves, adaptive=False)
        seconds += result['seconds']
        entry = dict(width=width, reason=result['reason'], selected_epoch=result['best']['epoch'],
                     updates=budget.updates-before_updates, examples=budget.examples-before_examples,
                     seconds=result['seconds'], **result['best']['score'])
        decisions.append(entry)
        if best is None or rank(entry, cfg) > rank(best['entry'], cfg):
            best = dict(entry=entry, model=result['best']['model'])
        if result['reason'] in ('runtime_limit', 'update_limit', 'example_limit'):
            interrupted = True; break
        if width == 16 and best['entry']['width'] == 16 and 32 not in candidates:
            candidates.append(32)
    if best is None:
        raise RuntimeError('No conventional candidate could start within the budget.')
    model, entry = best['model'], best['entry']
    return dict(model=model, summary=dict(condition='A', **entry, classes=data.classes,
                reached_all_classes=True, completed_protocol=not interrupted,
                target_met=acceptable(entry, cfg), search_interrupted=interrupted,
                boundary_selected=entry['width'] == max(d['width'] for d in decisions),
                parameters=sum(p.numel() for p in model.parameters()),
                search_updates=budget.updates, search_examples=budget.examples,
                search_seconds=seconds, wall_seconds=perf_counter()-budget.started),
                curves=curves, decisions=decisions, stages=[])
