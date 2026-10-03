"""slidemaster: compose ordered PNG images and WAV music into a lossless MOV master."""

from .timeline import Timeline, FramePlan, FrameKind
from .composer import compose

__all__ = ["Timeline", "FramePlan", "FrameKind", "compose"]
