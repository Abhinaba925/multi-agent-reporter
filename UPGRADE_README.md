# Upgrade status

The improved implementation is now the primary `main.py` entry point:

```bash
pip install -r requirements.txt -r requirements_extra.txt
streamlit run main.py
```

Implemented modules:

- `multi_agent_reporter/retrieval.py`: external DuckDuckGo search and page hydration.
- `multi_agent_reporter/verification.py`: claim extraction, citation validation and support metrics.
- `multi_agent_reporter/evaluation.py`: token estimates and order-swapped anonymous judge.
- `multi_agent_reporter/workflow_fixed.py`: modular LangGraph planner/writer/critic/reviser workflow.

Both single-agent and multi-agent reports receive the same retrieved evidence. Reports must cite evidence as `[S1]`, `[S2]`, etc. The UI displays citation completeness, support precision, unsupported-claim rate, estimated tokens and judge position consistency.

The enhanced folder `Research_Multi_Agent` is not modified by this upgrade.
