"""Optional integrations with external research code."""

from .nlf_smpl_provider import NLFSMPLProvider

__all__ = ["NLFSMPLProvider"]
from .smpl_temporal_refiner import SMPLTemporalRefinementAdapter

__all__ = ["SMPLTemporalRefinementAdapter"]
