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


def serialize_check(item):
    return {"claim": item.claim, "citation_ids": item.citation_ids, "status": item.status, "support_score": item.support_score, "reason": item.reason}


class MultiAgentWorkflow:
    def __init__(self, model, max_revisions: int = 3):
        self.model, self.max_revisions = model, max_revisions

    def invoke(self, template: str, values: dict) -> str:
        return (PromptTemplate.from_template(template) | self.model | StrOutputParser()).invoke(values)

    def planner(self, state):
        return {"outline": self.invoke("""Create a concise numbered outline for TASK using only the evidence. Identify claims that need citations.
TASK: {task}
EVIDENCE: {evidence}""", {"task": state["task"], "evidence": format_sources(state.get("sources", []))})}

    def writer(self, state):
        return {"draft": self.invoke("""Write a rigorous Markdown report. Use only EVIDENCE for factual claims and cite every claim using [S#]. State uncertainty; never fabricate sources.
TASK: {task}
OUTLINE: {outline}
EVIDENCE: {evidence}""", {"task": state["task"], "outline": state["outline"], "evidence": format_sources(state.get("sources", []))})}

    def critic(self, state):
        checks = verify_report(state["draft"], state.get("sources", []))
        metrics = verification_metrics(checks)
        critique = self.invoke("""Review the draft and verification JSON. Return numbered, actionable changes. Mention missing, invalid or weak citations. Return APPROVED only if complete and supported.
DRAFT: {draft}
CHECKS: {checks}
METRICS: {metrics}""", {"draft": state["draft"], "checks": json.dumps([serialize_check(item) for item in checks]), "metrics": metrics})
        return {"critique": critique, "checks": checks}

    def reviser(self, state):
        draft = self.invoke("""Revise the report conservatively using the critique. Preserve supported claims and citations, never invent evidence. Return complete Markdown.
DRAFT: {draft}
CRITIQUE: {critique}
EVIDENCE: {evidence}""", {"draft": state["draft"], "critique": state["critique"], "evidence": format_sources(state.get("sources", []))})
        return {"draft": draft, "revision_number": state.get("revision_number", 0) + 1}

    def route(self, state):
        return END if "APPROVED" in state.get("critique", "").upper() or state.get("revision_number", 0) >= self.max_revisions else "reviser"

    def run(self, task: str, sources: list[RetrievedSource]):
        graph = StateGraph(ReporterState)
        graph.add_node("planner", self.planner); graph.add_node("writer", self.writer); graph.add_node("critic", self.critic); graph.add_node("reviser", self.reviser)
        graph.set_entry_point("planner"); graph.add_edge("planner", "writer"); graph.add_edge("writer", "critic")
        graph.add_conditional_edges("critic", self.route, {"reviser": "reviser", END: END}); graph.add_edge("reviser", "critic")
        return graph.compile().invoke({"task": task, "sources": sources, "revision_number": 0})

