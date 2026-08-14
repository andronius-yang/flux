# EPLB algorithm, vendored verbatim from deepseek-ai/EPLB @ d52c72d
# (workspace checkout: workspace/changchen/andrewy/EPLB, MIT — see LICENSE).
#
# Unlike the moonep/ultraep oracles (reference implementations the semantic
# ports are tested bit-equal against), this vendored file is the production
# algorithm itself: the eplb arm calls rebalance_experts() directly to compute
# its one-shot static placement. The flux-side plan builder that maps its
# output onto the UltraEPPlan tensors lives in
# python/flux/testing/eplb_semantics.py.
from .eplb import rebalance_experts  # noqa: F401
