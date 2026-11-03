"""Checks tied to the structural-model table and prefix causality."""
import pytest
import torch
from tes_process.models import Process
from tes_process.event_memory import EventMemory


@pytest.mark.parametrize('family,parameters', [
    ('semimarkov_events_text', 10647),
    ('explicit_full_text', 11685),
    ('explicit_additive_text', 11181),
    ('explicit_reset_text', 11685),
    ('explicit_full', 2469),
    ('gru_text', 24623),
    ('neural_hawkes_observed_text', 42063),
])
def test_paper_parameter_counts(family, parameters):
    cls = EventMemory if family.startswith('explicit') else Process
    model = cls(family, 7, 6)
    assert sum(p.numel() for p in model.parameters()) == parameters


@pytest.mark.parametrize('family', ['explicit_full_text', 'gru_text', 'neural_hawkes_observed_text'])
def test_future_segments_cannot_change_prefix_predictions(family):
    torch.manual_seed(42)
    cls = EventMemory if family.startswith('explicit') else Process
    model = cls(family, 7, 6).eval()
    b = dict(base=torch.rand(1, 5, 28), memory=torch.rand(1, 5, 24),
             event=torch.zeros(1, 5, 6), known=torch.ones(1, 5, 6),
             text=torch.rand(1, 5, 32), text_ema=torch.rand(1, 5, 32),
             length=torch.ones(1, 5) * 20,
             reset=torch.zeros(1, 5, dtype=torch.bool),
             true_start=torch.zeros(1, 5, dtype=torch.bool),
             start=torch.zeros(1, 5, dtype=torch.bool))
    b['base'][..., :7] = 0
    b['base'][..., 0] = 1
    changed = {k: v.clone() for k, v in b.items()}
    changed['event'][:, 3:] = 1
    changed['text'][:, 3:] = -100
    with torch.no_grad():
        first, second = model(b), model(changed)
    for key in first:
        torch.testing.assert_close(first[key][:, :3], second[key][:, :3], rtol=0, atol=0)
