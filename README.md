# Auto-Report Team: A Multi-Agent System

This project demonstrates a sophisticated multi-agent system built with LangGraph. A team of AI agents collaborates to research, write, revise, and score a detailed report on a given topic, showcasing a powerful "self-correction" loop.



---

## Core Concept

Instead of relying on a single large language model (LLM), this system deploys a team of specialized agents, each with a distinct role. This "division of labor" results in a more robust, detailed, and accurate final product compared to a single LLM call.

The workflow is as follows:
1.  **🧑‍✈️ Planner:** Creates a detailed, step-by-step plan.
2.  **📚 Researcher:** Gathers information based on the plan.
3.  **✍️ Writer:** Composes an initial draft based on the research.
4.  **🧐 Critic:** Reviews the draft for errors and suggests specific, actionable revisions.
5.  **📝 Reviser:** Implements the critic's feedback on the draft.
6.  *Loop:* The revised draft goes back to the Critic until it's **APPROVED** or hits a revision limit.
7.  **⚖️ Scorer:** Finally, an impartial Scorer agent rates both the multi-agent report and a baseline single-agent report to quantitatively measure the quality difference.

---

## 🛠️ Tech Stack

* **Orchestration:** LangGraph
* **Core Library:** LangChain

---


