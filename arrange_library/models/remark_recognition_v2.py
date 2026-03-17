"""
备注识别结果数据模型（v2）
创建时间：2026-01-16 14:49:06
更新时间：2026-01-16 14:49:06
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CommandItem:
    """单条排机指令"""
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RemarkRecognitionResultV2:
    """备注识别结果（v2）"""
    library_id: str
    original_text: str
    is_need: str
    explain: str
    commands: List[CommandItem] = field(default_factory=list)
    confidence: float = 0.0
    route_type_id: Optional[str] = None
    route_source: Optional[str] = None
    error_message: Optional[str] = None
    is_recognized: bool = True
    raw_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "library_id": self.library_id,
            "original_text": self.original_text,
            "is_need": self.is_need,
            "explain": self.explain,
            "commands": [
                {"type": cmd.type, "params": cmd.params} for cmd in self.commands
            ],
            "confidence": self.confidence,
            "route_type_id": self.route_type_id,
            "route_source": self.route_source,
            "error_message": self.error_message,
            "is_recognized": self.is_recognized,
        }

    @classmethod
    def create_unrecognized(
        cls, library_id: str, original_text: str, reason: str
    ) -> "RemarkRecognitionResultV2":
        """创建未识别结果"""
        return cls(
            library_id=library_id,
            original_text=original_text,
            is_need="人工识别",
            explain=reason,
            commands=[],
            confidence=0.0,
            error_message=reason,
            is_recognized=False,
        )

    @classmethod
    def create_ignored(
        cls, library_id: str, original_text: str, reason: str
    ) -> "RemarkRecognitionResultV2":
        """创建忽略结果"""
        return cls(
            library_id=library_id,
            original_text=original_text,
            is_need="忽略",
            explain=reason,
            commands=[],
            confidence=1.0,
            error_message=None,
            is_recognized=True,
        )

