"""
CrewAI Example 07 — Flow with a router (conditional edges + loop).

SOURCE PATTERN (LangGraph):
    graph.add_conditional_edges("validate", router_fn, {
        "approve": "approve",
        "retry":   "gap_gen",     # loop back
        "escalate": "escalate",
    })

TARGET PATTERN (CrewAI):
    Conditional branching and loops are NOT possible in a plain Crew. Use a Flow.
      @start()          -> entry method(s)
      @listen(trigger)  -> runs when trigger completes (a method or a label)
      @router(method)   -> returns a string LABEL; the matching @listen("label") fires
      or_/and_          -> a listener fires when ANY / ALL triggers complete
      self.state        -> the typed Pydantic state shared across the flow

    Every label a @router can return MUST have a matching @listen — a dangling
    router label is a conversion bug. Guard loops with a state counter so they
    terminate.

    A Flow typically drives one or more Crews; here we keep the crew inline for
    brevity (build a real Crew as in example 01).
"""
from __future__ import annotations

from pydantic import BaseModel
from crewai.flow.flow import Flow, start, listen, router


# ---------------------------------------------------------------------------
# Flow state — replaces a LangGraph graph-state TypedDict.
# ---------------------------------------------------------------------------
class SuiteState(BaseModel):
    suite_path: str = "./uploads/suite/"
    score: float = 0.0
    passed: bool = False
    retries: int = 0


MAX_RETRIES = 3


class OptimiseFlow(Flow[SuiteState]):

    # -----------------------------------------------------------------------
    # Entry point.
    # -----------------------------------------------------------------------
    @start()
    def intake(self):
        # In a real flow this would kick off an intake Crew.
        return "loaded"

    # -----------------------------------------------------------------------
    # Runs after intake completes. Would call an analysis Crew here.
    # -----------------------------------------------------------------------
    @listen(intake)
    def analyse(self, _):
        # result = AnalysisCrew().crew().kickoff(inputs={"path": self.state.suite_path})
        # self.state.score = result.pydantic.score
        self.state.score = _fake_score(self.state.retries)
        self.state.passed = self.state.score >= 0.8
        return self.state.score

    # -----------------------------------------------------------------------
    # Router: returns a LABEL that decides the next step (conditional edges).
    # -----------------------------------------------------------------------
    @router(analyse)
    def route_on_score(self):
        if self.state.passed:
            return "approve"
        if self.state.retries < MAX_RETRIES:
            self.state.retries += 1     # guard the loop with a counter
            return "retry"
        return "escalate"

    # -----------------------------------------------------------------------
    # Every router label below has a matching @listen.
    # -----------------------------------------------------------------------
    @listen("approve")
    def approve(self):
        return f"approved: score={self.state.score:.2f}"

    @listen("retry")
    def retry(self):
        # Loop back into analysis; the counter guarantees termination.
        return self.analyse(None)

    @listen("escalate")
    def escalate(self):
        return f"escalated to human after {self.state.retries} retries"


def main():
    flow = OptimiseFlow()
    final = flow.kickoff()
    print(final)


# ---------------------------------------------------------------------------
# Stub — pretend the score improves with each retry so the loop terminates.
# ---------------------------------------------------------------------------
def _fake_score(retries: int) -> float:
    return 0.5 + 0.2 * retries


if __name__ == "__main__":
    main()
