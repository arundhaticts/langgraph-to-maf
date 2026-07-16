"""Phase 2 -- IR validation gate.

Runs after the IR is built and BEFORE any target code is generated. It catches
source-level inconsistencies (dangling routers, unguarded loops, orphan tools,
under-specified HITL, undeclared state) so they surface as review items instead
of being translated into equally-broken target code.

`validate_ir` returns a list of human-readable issue strings (empty == clean).
It never raises and never mutates the IR -- it is a pure gate whose findings the
pipeline routes into the migration report's "needs review" section.
"""

from __future__ import annotations

from converter.contracts import IR, NodeRole

_SENTINELS = frozenset({"START", "END", "__start__", "__end__"})


def validate_ir(ir: IR) -> list[str]:
    """Return a list of IR-consistency issues (empty list == the IR is clean)."""
    issues: list[str] = []
    wf = ir.workflow
    if wf is None:
        return issues

    node_names = {n.name for n in wf.nodes}
    fn_names = set(ir.functions)
    state_names = {f.name for f in ir.state}

    # 1. Every router referenced by a conditional edge is a defined function.
    for router in sorted({c.router for c in wf.conditional_edges if c.router}):
        if router not in fn_names:
            issues.append(
                f"Router '{router}' is referenced by a conditional edge but is not "
                "defined in the source functions."
            )

    # 2. Every loop back-edge has a paired termination guard with an exit.
    guard_by_node = {g.loop_node: g for g in wf.loop_guards}
    for target in sorted({e.target for e in wf.flat_edges if e.is_loop}):
        guard = guard_by_node.get(target)
        if guard is None:
            issues.append(
                f"Loop back-edge into '{target}' has no termination guard in the IR."
            )
        elif guard.counter_const is None and not guard.exit_labels:
            issues.append(
                f"Loop '{target}' has neither an iteration-cap constant nor a router "
                "exit label -- it may not terminate."
            )

    # 3. Every tool is referenced by at least one node (no orphans).
    called: set[str] = set()
    for node in wf.nodes:
        called.update(node.calls_tools)
    for tool in ir.tools:
        if tool.name not in called:
            issues.append(
                f"Tool '{tool.name}' is defined but not referenced by any node."
            )

    # 4. Every HITL node has a payload shape and a resume contract.
    hitl_points = {h.node: h for h in wf.hitl_points}
    for node in wf.nodes:
        if node.role is not NodeRole.HITL:
            continue
        point = hitl_points.get(node.name)
        if point is None:
            issues.append(
                f"HITL node '{node.name}' is missing from the IR hitl_points."
            )
            continue
        if not point.payload:
            issues.append(
                f"HITL node '{node.name}' has no interrupt() payload shape captured."
            )
        if not point.resume_contract:
            issues.append(f"HITL node '{node.name}' has no resume contract.")

    # 5. State schema covers every field any node reads or writes.
    for node in wf.nodes:
        for field in sorted(set(node.reads) | set(node.writes)):
            if field not in state_names:
                issues.append(
                    f"Node '{node.name}' reads/writes state field '{field}' that is "
                    "not declared in the state schema."
                )

    # 6. Every edge endpoint is a defined node (or a sentinel).
    for edge in wf.flat_edges:
        for endpoint in (edge.source, edge.target):
            if endpoint not in node_names and endpoint not in _SENTINELS:
                issues.append(
                    f"Edge endpoint '{endpoint}' is not a defined node."
                )

    return issues
