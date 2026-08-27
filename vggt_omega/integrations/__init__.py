"""Optional integrations with external research code."""

from .nlf_smpl_provider import NLFSMPLProvider

__all__ = ["NLFSMPLProvider"]
# Keep optional post-processing adapters out of this package initializer.
# ``VGGTOmega`` imports NLF integrations while ``vggt_omega.models`` is still
# being initialized; importing the temporal adapter here would recurse back
# into ``vggt_omega.models``.  Deploy code should import it explicitly:
# ``from vggt_omega.integrations.smpl_temporal_refiner import SMPLTemporalRefinementAdapter``.
