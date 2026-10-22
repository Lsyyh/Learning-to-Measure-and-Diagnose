"""Evaluation layer.

``stage1`` carries Stage-1 quality metrics, endpoint metrics, patient
primitives, leakage-safe OOF cross-fitting, and the deterministic-replay
comparator; ``metrics`` is the shared binary-metric core; ``panel`` carries
the TU1.0 dictionary-panel Stage-2 (reliability selection, additive
endpoint, exact ledger); ``concept_groups`` carries the dictionary concept
groups.
"""
