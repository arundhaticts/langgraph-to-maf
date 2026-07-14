"""Output generators -- Modules 7/8/9.

- Module 7 (code_generator)   -- renders the output repo (plugins/context/orchestrator)
- Module 8 (readme_generator) -- renders the output README.md
- Module 9 (report_generator) -- builds + writes MIGRATION_REPORT.md

All consume the frozen contracts and are shared by every approach.
"""

from converter.generator.code_generator import (
    GenerationResult,
    generate,
    generate_from_paths,
)
from converter.generator.docs_generator import (
    build_architecture_md,
    build_install_md,
    write_docs,
)
from converter.generator.readme_generator import build_readme, write_readme
from converter.generator.report_generator import (
    build_report,
    render_report,
    write_report,
)

__all__ = [
    "GenerationResult",
    "generate",
    "generate_from_paths",
    "build_readme",
    "write_readme",
    "build_report",
    "render_report",
    "write_report",
    "build_install_md",
    "build_architecture_md",
    "write_docs",
]
