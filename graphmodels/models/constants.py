"""Constants used for models"""

import enum
from collections.abc import Sequence
from typing import Final

from graphmodels.layers import constants as layer_constants

ALLOWED_POOLING: Final[Sequence[layer_constants.PoolingMethod]] = [
    layer_constants.PoolingMethod.MEAN,
    layer_constants.PoolingMethod.CONCAT,
    layer_constants.PoolingMethod.MAX,
]


class TaskType(enum.StrEnum):
    REGRESSION = enum.auto()
    CLASSIFICATION = enum.auto()


class OutputLevel(enum.StrEnum):
    GRAPH = enum.auto()
    NODE = enum.auto()
    EDGE = enum.auto()


ALLOWED_OUTPUT_LEVEL: Final[Sequence[OutputLevel]] = [
    OutputLevel.GRAPH,
    OutputLevel.NODE,
    OutputLevel.EDGE,
]

ALLOWED_TASK_TYPES: Final[Sequence[TaskType]] = [
    TaskType.REGRESSION,
    TaskType.CLASSIFICATION,
]
