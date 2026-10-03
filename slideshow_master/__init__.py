"""商品幻灯片母版合成库。

将多张 PNG 商品图与一段 WAV 音乐合成为无损 MOV 母版。
"""

from .composer import compose
from .errors import ComposeError

__all__ = ["compose", "ComposeError"]
