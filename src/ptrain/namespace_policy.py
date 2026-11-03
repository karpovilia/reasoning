"""Frozen namespace-level supervision table (not a runtime count rule).
Source: COORDINATOR_POSITIVE_ONLY_EVENT_MASK_DECISION_V2.json (sha256 2c7c834d…), which replaces V1 (2e3ab65e…) without
changing the four namespaces; defect PREDICTOR_POSITIVE_ONLY_EVENT_SUPERVISION_001. V2 also confirms that v4.6 failed_attempt and
quality-review event types are unknown (never turned into negatives); native known masks are used as delivered.
In these namespaces events are positive-only conversions without certified absence: their event loss is disabled for every
axis in TRAIN and DEV and their event heads are reported NOT_TRAINED_POSITIVE_ONLY. Native values and known masks are kept."""
DECISION = dict(path='COORDINATOR_POSITIVE_ONLY_EVENT_MASK_DECISION_V2.json',
                sha256='2c7c834d701d71690a2d018b2ee92a06e2739d35427f3836f2afb47a9bd45007',
                replaces=dict(path='COORDINATOR_POSITIVE_ONLY_EVENT_MASK_DECISION_V1.json',
                              sha256='2e3ab65e15dd9af711cc8841792df77d5c90e8c2062700bd9780cf086a1909bc'))
EVENT_LOSS_DISABLED = (
    'legacy_deepseek-chat|SYS_M',
    'gold_deepseek_dual|SYS_R',
    'event_local_quote_v3_single_stage_v1:T1:left15:prefix_only',
    'process_V6_unit',
)
NOT_TRAINED = 'NOT_TRAINED_POSITIVE_ONLY'


def event_supervision_eligible(namespace_id):
    return namespace_id not in EVENT_LOSS_DISABLED
