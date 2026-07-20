# Framework Authoring Guide — Adding and Removing Frameworks

This guide explains the full add/delete lifecycle for conversion frameworks.
The tool discovers frameworks from two sources: **built-in adapters** (Python
classes checked into the repo) and **vocabulary packs** (folders under
`backend/converter/frameworks/<name>/`). Both can be used independently or
together.

---

## Part 1 — How frameworks are discovered

```
backend/converter/
├── adapters/
│   ├── __init__.py           ← TARGET_ADAPTERS + SOURCE_ADAPTERS registries
│   ├── source/               ← one *SourceAdapter per framework
│   └── target/               ← one *TargetAdapter per framework + DynamicTargetAdapter
├── generator/
│   └── targets/
│       ├── __init__.py       ← TARGET_GENERATORS registry
│       └── <fw>_generator.py ← one *TargetGenerator per framework
└── frameworks/
    └── <name>/               ← vocabulary pack folder (optional)
        ├── vocabulary.json   ← machine-readable term map
        ├── docs.md           ← Tier-3 LLM context doc
        └── examples/         ← example Python files (Tier-3 context)
```

At startup `list_frameworks_detailed()` (in `adapters/__init__.py`) merges:
- names from `SOURCE_ADAPTERS` dict → `source: true`
- names from `TARGET_ADAPTERS` dict → `target: true`
- any `frameworks/<name>/vocabulary.json` file found on disk → `target: true`
  (unless `"supports_target": false` in the vocabulary)

A vocabulary pack alone is enough to make a framework appear in the UI and
accept conversions. A built-in adapter gives richer, deterministic output but
is not required.

---

## Part 2 — Adding a new framework (three levels of depth)

### Level 0 — Upload-only (zero code, instant)

Drop a folder into `backend/converter/frameworks/<yourfw>/` containing a
`vocabulary.json`. A template lives at `frameworks/new_framework/vocabulary.json`.
Copy it, fill in every `YOUR_*` placeholder, and the framework appears in the
UI immediately. The converter uses `DynamicTargetAdapter` (reads the vocab) and
`MAFTargetGenerator` as a fallback generator, which produces working output
for simple linear graphs.

**When to use:** prototyping, custom internal frameworks, frameworks with
small vocabularies that fit the MAF code-shape.

---

### Level 1 — Built-in TargetAdapter (better naming & idioms)

Add a `TargetAdapter` subclass when the framework needs custom tool
decoration, context-class naming, or decorator style that `DynamicTargetAdapter`
can't infer from the vocab alone.

1. **Create** `backend/converter/adapters/target/<yourfw>_adapter.py`:

```python
from converter.adapters.base import TargetAdapter, to_pascal_case

class YourFwTargetAdapter(TargetAdapter):
    name = "yourfw"

    def plugin_class_name(self, tool_name: str) -> str:
        return f"{to_pascal_case(tool_name)}Tool"  # or whatever the fw expects

    def method_name(self, tool_name: str) -> str:
        return tool_name

    @property
    def context_class_name(self) -> str:
        return "YourContext"

    def tool_style(self) -> str:
        return "function"           # "function" | "plugin_class"

    def tool_decorator(self) -> str:
        return "your_tool_decorator"

    def tool_decorator_import(self) -> str:
        return "from your_framework import your_tool_decorator"

    def runtime_requirements(self) -> tuple[str, ...]:
        return ("your-framework-package",)
```

2. **Register** in `adapters/__init__.py`:
```python
from converter.adapters.target.yourfw_adapter import YourFwTargetAdapter

TARGET_ADAPTERS: dict[str, type[TargetAdapter]] = {
    ...
    "yourfw": YourFwTargetAdapter,     # ← add this line
}
```

3. **Export** in `adapters/__init__.py`'s `__all__` list.

4. **Add** `"yourfw"` to the `test_frameworks_endpoint_lists_capabilities` assertions in
   `converter/tests/test_webapp.py` if it should be both source+target.

---

### Level 2 — Built-in TargetGenerator (high-fidelity code emission)

Add a `TargetGenerator` subclass when the framework needs its own orchestration
code shape (not just idiom tweaks). This is the highest-quality tier.

1. **Create** `backend/converter/generator/targets/<yourfw>_generator.py`:

```python
from converter.generator.targets.base import TargetGenerator
from converter.adapters.base import TargetAdapter
from converter.contracts import IR

class YourFwTargetGenerator(TargetGenerator):
    name = "yourfw"

    def workflow_block(self, ir: IR, adapter: TargetAdapter) -> str:
        """Emit the framework-specific orchestration (builder calls, edges, etc.)."""
        # Build and return a Python source string.
        ...

    def sdk_stub_files(self) -> dict[str, str]:
        """Return offline SDK stubs keyed by relative path, or {} for public packages."""
        return {}

    def smoke_test(self, adapter: TargetAdapter, target: str) -> str:
        """Return a minimal offline-safe smoke test (imports + instantiation only)."""
        ...

    def entrypoint(self, ir: IR, adapter: TargetAdapter, target: str) -> str:
        """Return a runnable main.py that drives the converted agent."""
        ...
```

See `maf_generator.py` as the reference implementation.

2. **Register** in `generator/targets/__init__.py`:
```python
from converter.generator.targets.yourfw_generator import YourFwTargetGenerator

TARGET_GENERATORS: dict[str, type[TargetGenerator]] = {
    ...
    "yourfw": YourFwTargetGenerator,   # ← add this line
}
```

3. **Export** in `__all__`.

---

### Level 3 — SourceAdapter (make it a conversion source)

Add a `SourceAdapter` to let the tool read *from* the new framework.

1. **Create** `backend/converter/adapters/source/<yourfw>_adapter.py`:

```python
from converter.adapters.base import SourceAdapter
from converter.contracts import SourceVocabulary

class YourFwSourceAdapter(SourceAdapter):
    name = "yourfw"

    def import_signatures(self) -> tuple[str, ...]:
        return ("your_framework",)    # top-level import root(s)

    def source_packages(self) -> tuple[str, ...]:
        return ("your-framework-package",)   # pip packages stripped from output

    def vocabulary(self) -> SourceVocabulary:
        return SourceVocabulary(
            tool_decorators=frozenset({"your_tool_decorator"}),
            graph_methods=frozenset({"add_step", "add_node", "set_entry_point"}),
            llm_constructors=frozenset({"YourModelClient"}),
            llm_constructor_prefixes=("Chat",),
            checkpointer_constructors=frozenset({"YourCheckpointStorage"}),
            sentinels={"END": "END"},
            state_base_classes=frozenset({"BaseModel", "TypedDict"}),
            dropped_import_roots=frozenset({"your_framework"}),
            graph_tokens=("YourBuilder(", ".add_step(", ".build("),
        )
    # Override extract_graph() only if the framework uses a non-standard
    # graph-construction pattern (see CrewAISourceAdapter for an example).
```

2. **Register** in `adapters/__init__.py`:
```python
from converter.adapters.source.yourfw_adapter import YourFwSourceAdapter

SOURCE_ADAPTERS: dict[str, type[SourceAdapter]] = {
    ...
    "yourfw": YourFwSourceAdapter,   # ← add this line
}
```

3. **Set** `"supports_source": true` in the framework's `vocabulary.json`.

4. **Add** the new source framework's agent fixture to
   `converter/tests/test_all_frameworks_matrix.py` and extend the parametrize
   lists so the matrix stays complete.

---

### Vocabulary pack checklist

```
frameworks/<yourfw>/
├── vocabulary.json    required: _meta, display_name, supports_source/target,
│                       primitives, nodes, edges, tools, hitl, invocation,
│                       providers, reject_list, dependencies
├── docs.md            optional but strongly recommended for Tier-3 quality
└── examples/          optional; each .py is injected as Tier-3 LLM context
    ├── 01_basic_agent.py
    ├── 02_conditional.py
    ├── 03_hitl.py
    ├── 04_tools.py
    └── ...
```

Tier-3 (LLM) accuracy is directly proportional to the quality of `docs.md`
and `examples/`. If docs.md is sparse, the LLM falls back to training memory
which may be wrong. Treat docs.md as the authoritative spec.

---

## Part 3 — Removing a framework

### Remove a vocabulary pack (upload-only framework)

Delete the folder: `backend/converter/frameworks/<name>/`.  
The framework disappears from the UI on next restart. No code changes needed.

### Remove a built-in framework

1. Delete the adapter file(s): `adapters/source/<fw>_adapter.py`,
   `adapters/target/<fw>_adapter.py`, `generator/targets/<fw>_generator.py`.
2. Remove the import and registry entry from:
   - `adapters/__init__.py` (`SOURCE_ADAPTERS`, `TARGET_ADAPTERS`, imports, `__all__`)
   - `generator/targets/__init__.py` (`TARGET_GENERATORS`, imports, `__all__`)
3. Delete the vocabulary pack folder if present: `frameworks/<name>/`.
4. Remove the framework from test parametrize lists in:
   - `test_all_frameworks_matrix.py` (`_ALL_TARGETS`, parametrize marks)
   - `test_webapp.py` (`test_frameworks_endpoint_lists_capabilities`)
5. Run the test suite: `pytest backend/converter/tests/ -q`.

---

## Part 4 — Testing after adding/removing

```bash
# Full suite (from backend/ directory):
pytest converter/tests/ -q

# Matrix only (catches all cross-framework regressions):
pytest converter/tests/test_all_frameworks_matrix.py -v

# Single target smoke:
pytest converter/tests/test_all_frameworks_matrix.py -k "target_yourfw" -v
```

The regression matrix (`test_all_frameworks_matrix.py`) tests every
source×target combination. A new framework must be added to the matrix to
be covered. A removed framework must be removed from the matrix to stop
failures.

---

## Part 5 — The `new_framework` template

`backend/converter/frameworks/new_framework/` is a Claude-generated template
that shows every required and optional section of a vocabulary pack, modelled
on the MAF pack. Use it as a starting point:

```bash
# Copy and rename:
cp -r backend/converter/frameworks/new_framework \
      backend/converter/frameworks/<yourfw>

# Edit vocabulary.json — replace every YOUR_* placeholder.
# Edit docs.md — describe the real framework API.
# Add examples/ files that show real usage patterns.
```

The template is kept up to date with the vocabulary schema so that a copy-edit
is always sufficient to produce a valid pack.
