"""IR layer -- Module 5 (ir_builder)."""

from converter.ir.ir_builder import build_ir, detect_target_version, write_ir_json
from converter.ir.validator import validate_ir

__all__ = ["build_ir", "detect_target_version", "validate_ir", "write_ir_json"]
