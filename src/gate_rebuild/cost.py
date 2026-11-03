"""Full cost of one attempt under a policy, in ONE declared unit, with its components kept apart.

Unit: seconds of one reference card of one class, one request at a time. Every component of a compared policy is measured on that
card; solver tokens are reported next to the seconds and are never added to them without the declared converter.

Components of one attempt (review finding GATE-V1-R1):
  common    what every policy pays alike: reading the prompt and decoding the reasoning up to the decision position (or, for an
            attempt that ended before it, the whole attempt). It is part of every total, ratio and ceiling of the whole population.
  looking   what a TRAINED policy pays on every state on which it is consulted, also when it then stops: segmentation of the prefix
            for its observer, the observer over every completed segment, and one evaluation of every fitted member of the gate.
            A trained policy without an observer still pays for its gate. A fixed action or a fixed mixture pays nothing here.
  action    the marginal cost of the chosen action: its sampled and injected tokens including the final answer, plus the
            re-reading of prompt and prefix when the key-value cache of the solver was not kept. The re-reading is the SAME for
            stop and for the continuations: every action needs the prefix as context (in the bank generator each branch starts
            with no cache, stop included).

kv_cache is a declared part of the deployment: 'kept' (the solver holds its cache of prompt and prefix while the gate decides; no
re-reading; the memory held is that cache) or 'dropped' (every action re-reads prompt and prefix).

The observer is run at the decision position over the completed segments of the prefix; an attempt that ended before the decision
position never reaches the gate and pays no looking cost.

Not part of the deployment cost, kept in its own ledger line: generating the alternative branches for training, the review of the
answers, the fitting of the heads. The numbers of a provisional model stay provisional: nothing is frozen on them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KV_CACHE = ('kept', 'dropped')


@dataclass(frozen=True)
class CostModel:
    unit: str                               # for example 'seconds on RTX A6000, one request at a time'
    seconds_per_solver_token: float         # decoding one sampled or injected solver token
    prefill_seconds_per_token: float        # reading one token of the prompt, or re-reading one token of the prefix
    kv_cache: str                           # 'kept' or 'dropped'
    observer_seconds_per_segment: dict      # row source -> seconds of all observer calls for one completed segment
    segmentation_seconds_per_prefix: dict   # row source -> seconds of segmenting one prefix for that observer
    gate_seconds_per_member: float          # one evaluation of the encoder and the heads of ONE fitted member

    def __post_init__(self):
        numbers = [self.seconds_per_solver_token, self.prefill_seconds_per_token, self.gate_seconds_per_member,
                   *self.observer_seconds_per_segment.values(), *self.segmentation_seconds_per_prefix.values()]
        if not self.unit or self.kv_cache not in KV_CACHE or any(isinstance(v, bool) or not np.isfinite(v) or v < 0 for v in numbers) \
                or self.seconds_per_solver_token <= 0 or set(self.observer_seconds_per_segment) != set(self.segmentation_seconds_per_prefix):
            raise ValueError('invalid cost model')

    def common(self, prompt_tokens, decoded_tokens):
        """Reading the prompt and decoding `decoded_tokens`: the reasoning before the decision, or a whole attempt that ended earlier."""
        return prompt_tokens * self.prefill_seconds_per_token + decoded_tokens * self.seconds_per_solver_token

    def action(self, additional_tokens, prompt_tokens, prefix_tokens):
        """Marginal cost of an action; the same rule for stop and for a continuation."""
        reread = 0.0 if self.kv_cache == 'kept' else (prompt_tokens + prefix_tokens) * self.prefill_seconds_per_token
        return additional_tokens * self.seconds_per_solver_token + reread

    def looking(self, source, segments, members):
        """What a trained policy pays for one consulted state. source None: a trained policy without an observer (its gate only).
        A source without a price is an error, never a free observer."""
        if isinstance(members, bool) or members < 1:
            raise ValueError('a trained policy has at least one fitted member')
        gate = members * self.gate_seconds_per_member
        if source is None:
            return gate
        return self.segmentation_seconds_per_prefix[source] + self.observer_seconds_per_segment[source] * segments + gate
