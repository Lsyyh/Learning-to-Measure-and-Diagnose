"""Training primitives.

``stage1`` owns the Stage-1 radiomics regression loop with its determinism,
scheduler, scope, and loss-weight plumbing; ``stage2_joint`` owns the
train–inference-consistent Stage-2 anchored joint refinement (the method of
record); ``direct`` and ``segmentation`` own the two comparator training
loops; ``target_transform`` owns the rank-Gauss target transform whose
module path is the persistence contract for every frozen run's
``target_transform.joblib``.
"""
