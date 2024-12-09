# TUM - MIRMI - ATARI lab
# Victor DHEDIN, 2024

from typing import Any

import numpy as np


class DataRecorderAbstract(object):
    def __init__(self, record_dir: str = "") -> None:
        self.record_dir = record_dir

    def record(
        self,
        q: np.array,
        v: np.array,
        **kwargs,
    ) -> None:
        pass

    def reset(self) -> None:
        pass

    def save(self) -> None:
        pass
