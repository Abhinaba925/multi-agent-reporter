from __future__ import annotations

import json
from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph

from .retrieval import RetrievedSource, format_sources
from .verification import verify_report, verification_metrics


class ReporterState(TypedDict, total=False):
    task: str
    sources: list[RetrievedSource]
    outline: str
    draft: str
    critique: str
    revision_number: int
    checks: list


class MultiAgentWorkflow:
    def __init__(self, model, max_revisions: int = 3):
        self.model = model
        self.max_revisions = max_revisions
        self.parser = StrOutputParser()

    def _invoke(self, system: str, template: str, values: dict) -> str:
        prompt = PromptTemplate.from_template(template)
        return (prompt | self.model | self.parser).invoke(values)

    def planner(self, state: ReporterState):
        outline = self._invoke(
            "planner",
            """You are a research planner. Return a concise numbered outline for the task.
Use the retrieved evidence to identify the claims that must be supported.\nTASK: {task}\nEVIDENCE:\n{evidence}""",
            {"task": state["task"], "evidence": format_sources(state.get("sources", []))},
        )
        return {"outline": outline}

    def writer(self, state: ReporterState):
        draft = self._invoke(
            "writer",
            """You write a rigorous technical Markdown report.
Use only the supplied evidence for factual claims. Cite every externally verifiable claim with [S#].
If evidence is insufficient, say so. Do not fabricate sources or citations.
TASK: {task}\nOUTLINE:\n{outline}\nEVIDENCE:\n{evidence}""",
            {"task": state["task"], "outline": state["outline"], "evidence": format_sources(state.get("sources", []))},
        )
        return {"draft": draft}

    def critic(self, state: ReporterState):
        checks = verify_report(state["draft"], state.get("sources", []))
        metrics = verification_metrics(checks)
        critique = self._invoke(
            "critic",
            """You are a strict report critic. Review the draft and verification results.
Return actionable numbered changes. Mention missing, invalid or weak citations. If all claims are supported and the report is complete, return APPROVED.
DRAFT:\n{draft}\nVERIFICATION JSON:\n{checks}\nMETRICS:\n{metrics}""",
            {"draft": state["draft"], "checks": json.dumps([vars(item) for item in checks]), "metrics": metrics},
        )
        return {"critique": critique, "checks": checks}

    def reviser(self, state: ReporterState):
        draft = self._invoke(
            "reviser",
            """Revise the report using the critique. Preserve supported content and citations.
Do not invent evidence. Return complete Markdown with [S#] citations.
DRAFT:\n{draft}\nCRITIQUE:\n{critique}\nEVIDENCE:\n{evidence}""",
            {"draft": state["draft"], "critique": state["critique"], "evidence": format_sources(state.get("sources", []))},
        )
        return {"draft": draft, "revision_number": state.get("revision_number", 0) + 1}

    def should_continue(self, state: ReporterState):
        if "APPROVED" in state.get("critique", "").upper() or state.get("revision_number", 0) >= self.max_revisions:
            return END
        return "reviser"

    def run(self, task: str, sources: list[RetrievedSource]) -> ReporterState:
        graph = StateGraph(ReporterState)
        graph.add_node("planner", self.planner)
        graph.add_node("writer", self.writer)
        graph.add_node("critic", self.critic)
        graph.add_node("reviser", self.reviser)
        graph.set_entry_point("planner")
        graph.add_edge("planner", "writer")
        graph.add_edge("writer", "critic")
        graph.add_conditional_edges("critic", self.should_continue, {"reviser": "reviser", END: END})
        graph.add_edge("reviser", "critic")
        return graph.compile().invoke({"task": task, "sources": sources, "revision_number": 0})

