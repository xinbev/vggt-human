from dataclasses import dataclass


@dataclass(frozen=True)
class AggregatorTokenLayout:
    camera_start: int
    camera_end: int
    register_start: int
    register_end: int
    smpl_start: int
    smpl_end: int
    patch_start: int

    @property
    def camera_register_end(self) -> int:
        return self.register_end

    @property
    def num_smpl_queries(self) -> int:
        return self.smpl_end - self.smpl_start
