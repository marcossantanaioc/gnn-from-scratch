import enum


class PoolingMethod(enum.StrEnum):
    """Defines pooling operations for graph-level output."""

    MEAN = enum.auto()
    CONCAT = enum.auto()
    MAX = enum.auto()
