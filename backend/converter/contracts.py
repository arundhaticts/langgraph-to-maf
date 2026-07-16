"""Frozen dataclass contracts shared by every module.

Section 9 of the build plan: these interfaces are FROZEN. Every module consumes
and produces these types. Do not redefine or mutate them during implementation.
Changes require explicit approval because they affect every downstream module.

All three approaches (Deterministic / Full LLM / Hybrid) speak these contracts,
which is what lets them share the scanner, generators, and report logic.

The schema is locked; the dataclasses are left mutable so builder modules can
assemble them incrementally. "Frozen" here means "do not change the fields",
not `@dataclass(frozen=True)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Optional


def _serialise(value: Any) -> Any:
    """Recursively convert dataclasses/enums into JSON-ready primitives.

    Used by `IR.to_json_dict` for the `ir.json` debug checkpoint. Enums become
    their string value; nested dataclasses become plain dicts.
    """
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _serialise(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {key: _serialise(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FileType(str, Enum):
    """Classification a file receives during scanning."""
    PYTHON = "python"
    README = "readme"
    PROMPT = "prompt"          # prompt .md files -> copy through
    REQUIREMENTS = "requirements"
    OTHER = "other"


class FileAction(str, Enum):
    """What the generator does with a file (assigned by the extractor)."""
    REWRITE = "rewrite"        # regenerate from templates + IR
    ADAPT = "adapt"            # partial rewrite, some sections stitched
    COPY_THROUGH = "copy_through"  # copied verbatim to output


class NodeRole(str, Enum):
    """Role a graph node plays, classified by the IR builder."""
    ENTRY = "entry"
    TERMINAL = "terminal"
    LINEAR = "linear"
    BRANCH = "branch"
    LOOP = "loop"
    AUX = "aux"
    HITL = "hitl"


class OrchestrationPattern(str, Enum):
    """Overall workflow shape, classified by the IR builder."""
    LINEAR = "linear"
    BRANCH = "branch"
    LOOP = "loop"
    LOOP_WITH_EXIT = "loop_with_exit"
    AGENT_DRIVEN = "agent_driven"


class OrchestrationMode(str, Enum):
    """How the target should realise the orchestration (Phase 0 decision).

    Encoded in the IR so the generator picks the right target shape:
    - SINGLE_AGENT: one ReAct-style agent with tools, no branches/loops/HITL
      -> a single `ChatAgent(tools=[...])` on the target side.
    - GRAPH_WORKFLOW: multi-node graph with branches / loops / HITL
      -> a `WorkflowBuilder` + executors on the target side.
    """
    SINGLE_AGENT = "single_agent"
    GRAPH_WORKFLOW = "graph_workflow"


class Tier(str, Enum):
    """Which tier resolved a given conversion."""
    TIER1 = "tier1"            # deterministic rules
    TIER2 = "tier2"            # README keyword match
    TIER3 = "tier3"            # LLM (or future SLM)
    UNRESOLVED = "unresolved"  # flagged for manual conversion


# ---------------------------------------------------------------------------
# Scanner contract (Module 1)
# ---------------------------------------------------------------------------

@dataclass
class FileEntry:
    """One file in the input repo, tagged by type. Path is relative to input root."""
    relative_path: str
    file_type: FileType
    # file_action is assigned later by the extractor (Module 4); None until then.
    file_action: Optional[FileAction] = None


@dataclass
class RepoManifest:
    """Output of Module 1 (repo scanner)."""
    input_root: str
    files: list[FileEntry] = field(default_factory=list)
    detected_framework: Optional[str] = None  # e.g. "langgraph"
    readme_path: Optional[str] = None          # relative path to README.md

    def python_files(self) -> list[FileEntry]:
        return [f for f in self.files if f.file_type is FileType.PYTHON]


# ---------------------------------------------------------------------------
# README contract (Module 3)
# ---------------------------------------------------------------------------

@dataclass
class ReadmeToolEntry:
    name: str
    description: str


@dataclass
class ReadmeStateEntry:
    name: str
    type: str
    description: str


@dataclass
class ReadmeSections:
    """Output of Module 3 (README parser). Sections keyed by level-2 header.

    `workflow_description` is stored verbatim -- Tier 2 input, never pre-parsed.
    Missing sections are recorded in `missing_sections` (warning, not hard stop).
    """
    purpose: Optional[str] = None
    framework: Optional[str] = None
    tools: list[ReadmeToolEntry] = field(default_factory=list)
    workflow_description: Optional[str] = None
    state: list[ReadmeStateEntry] = field(default_factory=list)
    configuration: Optional[str] = None
    dependencies: Optional[str] = None
    raw_sections: dict[str, str] = field(default_factory=dict)  # header -> raw text
    missing_sections: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser / component contracts (Modules 2 & 4)
# ---------------------------------------------------------------------------

@dataclass
class ToolParam:
    name: str
    annotation: Optional[str] = None
    default: Optional[str] = None


@dataclass
class ToolSpec:
    """A source `@tool` function."""
    name: str
    params: list[ToolParam] = field(default_factory=list)
    docstring: Optional[str] = None
    returns: Optional[str] = None
    source_file: Optional[str] = None
    # Raw source of the function body (statements, docstring stripped). Carried
    # so the generator can port real tool logic instead of emitting a stub.
    body: Optional[str] = None
    # Original parameter list source (e.g. "path: str, limit: int = 10"), so the
    # plugin method preserves defaults / *args / **kwargs exactly.
    signature: Optional[str] = None


@dataclass
class FunctionSpec:
    """Any source top-level function definition with its body.

    Captured for logic porting: node functions, routers, and helpers are matched
    by name and their bodies are transformed (state -> ctx) or copied through.
    """
    name: str
    params: list[ToolParam] = field(default_factory=list)
    body: Optional[str] = None
    returns: Optional[str] = None
    docstring: Optional[str] = None
    is_async: bool = False
    decorators: list[str] = field(default_factory=list)
    source_file: Optional[str] = None
    # Full original function source (def + args + body) so porting can preserve
    # the exact signature (defaults, *args, **kwargs) via an AST transform.
    source: Optional[str] = None
    # Original parameter list source (e.g. "test_id, state").
    signature: Optional[str] = None

    @property
    def first_param(self) -> Optional[str]:
        return self.params[0].name if self.params else None


@dataclass
class StateField:
    """A field on the source state TypedDict."""
    name: str
    type: str
    description: Optional[str] = None
    is_append_only: bool = False   # Annotated[list, add] reducer detected
    default: Optional[str] = None  # source-level default, if any (verbatim src)


@dataclass
class GraphNode:
    name: str
    target_callable: Optional[str] = None  # function bound to the node
    role: Optional[NodeRole] = None         # assigned by IR builder
    # Data-flow captured by the IR builder (Phase 1). `reads`/`writes` are state
    # field names the node reads / writes; `calls_tools` are tool names invoked.
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    calls_tools: list[str] = field(default_factory=list)


@dataclass
class GraphEdge:
    source: str
    target: str


@dataclass
class ConditionalEdge:
    source: str
    router: Optional[str] = None            # routing function name
    outcomes: dict[str, str] = field(default_factory=dict)  # branch label -> target node


@dataclass
class FlatEdge:
    """A router/edge flattened to a single explicit triple at IR-build time.

    Every unconditional edge and every conditional-edge outcome becomes one
    `FlatEdge`, so the generator never has to flatten a router mapping itself.
    `condition_label` is None for unconditional edges; `router` names the routing
    function for conditional ones. `is_loop` marks a back-edge (Phase 1).
    """
    source: str
    target: str
    condition_label: Optional[str] = None
    router: Optional[str] = None
    is_loop: bool = False


@dataclass
class LoopGuard:
    """The termination guard for one loop back-edge (Phase 1 metadata).

    `loop_node` is the node that the back-edge returns to; `router` is the
    function that decides continue-vs-exit; `counter_const` is the config
    constant that caps the iterations (e.g. MAX_GEN_RETRIES) if one was found;
    `exit_labels` are the router outcomes that leave the loop.
    """
    loop_node: str
    router: Optional[str] = None
    counter_const: Optional[str] = None
    exit_labels: list[str] = field(default_factory=list)


@dataclass
class HitlPoint:
    """A human-in-the-loop pause point (Phase 1).

    `payload` is the verbatim source of the `interrupt(...)` argument (the shape
    handed to the human); `resume_contract` describes what the resumed value is.
    """
    node: str
    payload: Optional[str] = None
    resume_contract: Optional[str] = None


@dataclass
class GraphSpec:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    conditional_edges: list[ConditionalEdge] = field(default_factory=list)
    entry_point: Optional[str] = None


@dataclass
class ConfigSpec:
    """Extracted config. `temperature` is None if not found -- never invented."""
    llm_kwargs: dict[str, Any] = field(default_factory=dict)
    env_vars: list[str] = field(default_factory=list)
    constants: dict[str, Any] = field(default_factory=dict)
    temperature: Optional[float] = None
    # Source LLM provider constructor name (e.g. "ChatOpenAI"), if detected.
    llm_provider: Optional[str] = None
    # Source persistence/checkpointer construct (e.g. "MemorySaver"), if any.
    checkpointer: Optional[str] = None


@dataclass
class ComponentInventory:
    """Output of Module 4 (component extractor). Consolidated across all files."""
    tools: list[ToolSpec] = field(default_factory=list)
    graph: GraphSpec = field(default_factory=GraphSpec)
    state: list[StateField] = field(default_factory=list)
    config: ConfigSpec = field(default_factory=ConfigSpec)
    # All top-level source functions by name (for node/router/helper porting).
    functions: dict[str, FunctionSpec] = field(default_factory=dict)
    # Non-framework import lines, deduped, for carrying into generated files.
    imports: list[str] = field(default_factory=list)
    # Module-level setup lines (e.g. `llm = ChatOpenAI(...)`) minus graph wiring.
    preamble: list[str] = field(default_factory=list)
    # Names of the source state TypedDict class(es), for backward-compat aliases.
    state_class_names: list[str] = field(default_factory=list)
    # Source persistence/checkpointer construct (e.g. "MemorySaver"), if any.
    checkpointer: Optional[str] = None
    # Cross-reference warnings (AST is authoritative over README).
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# IR contract (Module 5) -- the single source of truth
# ---------------------------------------------------------------------------

@dataclass
class WorkflowSpec:
    """The classified orchestration for the IR."""
    pattern: OrchestrationPattern
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    conditional_edges: list[ConditionalEdge] = field(default_factory=list)
    entry_point: Optional[str] = None
    readme_description: Optional[str] = None  # verbatim workflow prose (Tier 2)
    # Phase 1 flattening + explicit control-flow metadata.
    flat_edges: list[FlatEdge] = field(default_factory=list)
    loop_guards: list[LoopGuard] = field(default_factory=list)
    hitl_points: list[HitlPoint] = field(default_factory=list)


@dataclass
class IRMetadata:
    description: Optional[str] = None          # from README Purpose
    source_framework: Optional[str] = None
    target_framework: Optional[str] = None
    # Phase 0 decisions recorded into the IR.
    target_framework_version: Optional[str] = None  # installed target SDK version
    orchestration_mode: Optional[OrchestrationMode] = None
    llm_provider: Optional[str] = None         # source LLM provider carried over
    checkpointer: Optional[str] = None         # source persistence construct
    entrypoint: Optional[str] = None           # "api" (FastAPI/Flask) or "cli"


@dataclass
class IR:
    """Framework-neutral Intermediate Representation. Single source of truth.

    No stage after this is built looks at the original source again; no stage
    before code generation knows the target framework's syntax.
    """
    metadata: IRMetadata = field(default_factory=IRMetadata)
    tools: list[ToolSpec] = field(default_factory=list)
    state: list[StateField] = field(default_factory=list)
    config: ConfigSpec = field(default_factory=ConfigSpec)
    workflow: Optional[WorkflowSpec] = None
    # Source functions (node/router/helper bodies) for logic porting.
    functions: dict[str, FunctionSpec] = field(default_factory=dict)
    # Non-framework imports and module-level setup carried into the output.
    imports: list[str] = field(default_factory=list)
    preamble: list[str] = field(default_factory=list)
    # Names of the source state TypedDict class(es), for backward-compat aliases.
    state_class_names: list[str] = field(default_factory=list)
    # Files carried from the manifest with their assigned action.
    files: list[FileEntry] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialise for the ir.json debug checkpoint (Module 5)."""
        return _serialise(self)


# ---------------------------------------------------------------------------
# Conversion contracts (Module 6)
# ---------------------------------------------------------------------------

@dataclass
class Tier3Result:
    """Structured output contract for the Tier 3 LLM/SLM call.

    This shape is FROZEN so the future SLM can be swapped in behind it
    (Section 15) with no pipeline changes.
    """
    pattern: str
    generated_code: str
    reasoning: str
    confidence: float


@dataclass
class ConversionUnit:
    """One resolved conversion decision, produced by the engine per construct."""
    rule_id: Optional[str]                 # e.g. "R-01", or None for Tier 3
    tier: Tier
    source_ref: str                        # what was converted (e.g. tool name)
    target_ref: Optional[str] = None       # what it became
    generated_code: Optional[str] = None   # for stitched sections
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    needs_review: bool = False
    manual_action: Optional[str] = None    # non-None => manual step required


@dataclass
class ConversionResult:
    """Full output of Module 6 (conversion engine)."""
    units: list[ConversionUnit] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Migration report contract (Module 9)
# ---------------------------------------------------------------------------

@dataclass
class ReportEntry:
    text: str
    detail: Optional[str] = None


@dataclass
class MigrationReport:
    """Output of Module 9. Three sections per Section 14 of the plan."""
    agent_name: str
    generated: str = ""                    # date string, passed in (no Date.now here)
    auto_converted: list[ReportEntry] = field(default_factory=list)
    needs_review: list[ReportEntry] = field(default_factory=list)
    manual_action_required: list[ReportEntry] = field(default_factory=list)
