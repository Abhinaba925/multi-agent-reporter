import os
import json
import operator
import statistics
import streamlit as st
from typing import TypedDict, Annotated, List
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from multi_agent_reporter.retrieval import ArxivRetriever, WebRetriever, format_sources, RetrievedSource
from multi_agent_reporter.verification import verify_report, verification_metrics

# --- PAGE CONFIG ---
st.set_page_config(page_title="Multi-Agent Reporter", layout="wide")
st.title("Agentic Workflows: Single Prompt vs. LangGraph Team")
st.markdown("Experience the difference in depth and reasoning when a standard LLM response is replaced by a multi-step, specialized agent architecture.")

# --- SIDEBAR: API KEY & SETTINGS ---
st.sidebar.markdown("### Configuration")
api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")
if not api_key and "GROQ_API_KEY" in os.environ:
    api_key = os.environ["GROQ_API_KEY"]

# Add Temperature Slider
user_temperature = st.sidebar.slider(
    "Model Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.1,
    help="Higher values (e.g., 0.8) make the output more creative/random. Lower values (e.g., 0.1) make it more focused and deterministic."
)

research_mode = st.sidebar.selectbox(
    "Research source",
    ["No external search", "Web search", "Research papers (arXiv)"],
    help="Choose whether the researcher should use the model only, web pages, or arXiv papers.",
)
source_count = st.sidebar.slider("Number of sources", min_value=2, max_value=8, value=5)

# Model selector
model_options = {
    "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
    "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
    "GPT-OSS 120B": "openai/gpt-oss-120b",
    "GPT-OSS 20B": "openai/gpt-oss-20b",
    "Qwen 3.6 27B": "qwen/qwen3.6-27b",
}
selected_model_name = st.sidebar.selectbox(
    "Language model",
    options=list(model_options.keys()),
    index=0,
    help="Choose the Groq-hosted model used by the single-agent and multi-agent workflows.",
)
selected_model = model_options[selected_model_name]
judge_model_name = st.sidebar.selectbox(
    "Judge model",
    options=list(model_options.keys()),
    index=0,
    help="Use a separate model for evaluation when possible to reduce self-preference bias.",
)
judge_rounds = st.sidebar.slider(
    "Judge repetitions",
    min_value=2,
    max_value=5,
    value=3,
    help="Each repetition evaluates both candidate orders. More repetitions improve stability but increase API usage.",
)

if not api_key:
    st.warning("Please enter your Groq API Key in the sidebar to continue. Get one for free at console.groq.com")
    st.stop()

# Initialize Groq Model
os.environ["GROQ_API_KEY"] = api_key
try:
    model = ChatGroq(model=selected_model, temperature=user_temperature)
    judge_model = ChatGroq(model=model_options[judge_model_name], temperature=0.0)
    parser = StrOutputParser()
except Exception as e:
    st.error(f"Error initializing the model: {e}")
    st.stop()

# --- 1. DEFINE THE AGENT STATE ---
class AgentState(TypedDict):
    task: str
    plan: str
    research: str
    draft: str
    critique: str
    revision_number: int
    sources: List[RetrievedSource]

# --- 2. DEFINE THE CORE AGENTS ---
def planner_agent(state: AgentState):
    prompt = PromptTemplate.from_template(
        "You are an expert technical planner. Create a detailed outline for a comprehensive technical article on this task: {task}. "
        "IMPORTANT: Your outline must use concise, professional section titles (e.g., 'Neural Network Architecture' instead of 'Description of the Neural Network...'). "
        "Do not use bureaucratic formats like 'Executive Summary'. Focus purely on educational value and logical flow. "
        "Use the supplied evidence to identify which claims need citations.\n"
        "Evidence:\n{evidence}"
    )
    runnable = prompt | model | parser
    plan = runnable.invoke({"task": state['task'], "evidence": format_sources(state.get('sources', []))})
    return {"plan": plan}

def researcher_agent(state: AgentState):
    prompt = PromptTemplate.from_template(
        "You are an expert research analyst. Synthesize dense, factual information and mathematical formulas from this outline and the retrieved evidence. "
        "Cite every externally verifiable claim using the source IDs exactly as [S1] or [P1]. Never invent citations. "
        "If the evidence is insufficient, explicitly say so.\nOutline: {plan}\nEvidence: {evidence}"
    )
    runnable = prompt | model | parser
    research = runnable.invoke({"plan": state['plan'], "evidence": format_sources(state.get('sources', []))})
    return {"research": research}

def writer_agent(state: AgentState):
    prompt = PromptTemplate.from_template(
        "You are an expert technical writer. Write a comprehensive, seamless technical article using this research: {research}. "
        "FORMATTING RULES: "
        "1. Use proper Markdown formatting (e.g., `###` for subheadings, `**bold**` for key terms). "
        "2. You MUST use LaTeX formatting (e.g., `$$ equation $$` or `$ equation $`) for ALL mathematical formulas and variables. "
        "3. Ensure smooth narrative transitions between sections. Do not just list the outline points. "
        "4. Write objectively. NEVER refer to 'this project', 'our team', or 'this report'. "
        "5. Preserve source citations such as [S1] and [P1] for factual claims. Do not create citations not present in the research. "
        "6. End with a `## References` section mapping every cited source ID to its source title and URL."
    )
    runnable = prompt | model | parser
    draft = runnable.invoke({"research": state['research']})
    return {"draft": draft}

def revision_agent(state: AgentState):
    prompt = PromptTemplate.from_template(
        "You are an expert technical editor. Revise this draft: {draft} based strictly on these critiques: {critique}. "
        "Ensure the final text is beautifully formatted using Markdown, uses LaTeX for all math, is dense with facts, and flows logically. "
        "Make sure the article is fully complete and does not cut off abruptly at the end."
    )
    runnable = prompt | model | parser
    revised_draft = runnable.invoke({"draft": state['draft'], "critique": state['critique']})
    return {"draft": revised_draft}

def critic_agent(state: AgentState):
    prompt = PromptTemplate.from_template(
        """You are an expert critic. Review the draft article. 
        Penalize any corporate jargon, long/awkward headings, or missing Markdown/LaTeX formatting.
        If the draft is beautifully formatted, mathematically sound, and well-written, say 'APPROVED'. 
        Otherwise, provide a numbered list of specific, actionable revisions.
        Draft: {draft}"""
    )
    runnable = prompt | model | parser
    critique = runnable.invoke({"draft": state['draft']})
    revision_number = state.get('revision_number', 0) + 1
    return {"critique": critique, "revision_number": revision_number}

# --- 3. DEFINE THE GRAPH AND ITS LOGIC ---
def should_continue(state: AgentState):
    if state['revision_number'] > 3:
        return "end"
    if "APPROVED" in state['critique'].upper():
        return "end"
    else:
        return "revise"

workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("critic", critic_agent)
workflow.add_node("reviser", revision_agent)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "critic")
workflow.add_conditional_edges("critic", should_continue, {"revise": "reviser", "end": END})
workflow.add_edge("reviser", "critic")
app = workflow.compile()

# --- 4. SCORING AND SINGLE AGENT ---
def run_single_agent(task_string: str, sources: List[RetrievedSource]):
    prompt = PromptTemplate.from_template(
        "You are an expert technical writer. Write a comprehensive explanation using only the supplied evidence. "
        "Cite factual claims with the provided source IDs and disclose uncertainty. "
        "End with a `## References` section mapping each cited source ID to its title and URL.\n"
        "Task: {task}\nEvidence: {evidence}"
    )
    runnable = prompt | model | parser
    return runnable.invoke({"task": task_string, "evidence": format_sources(sources)})


def retrieve_sources(task: str) -> List[RetrievedSource]:
    if research_mode == "Web search":
        return WebRetriever(max_results=source_count).search(task)
    if research_mode == "Research papers (arXiv)":
        return ArxivRetriever(max_results=source_count).search(task)
    return []


def add_reference_list(report: str, sources: List[RetrievedSource]) -> str:
    """Guarantee that the multi-agent report exposes human-readable references."""
    if not sources:
        return report
    if "\n## References" in report or "\n### References" in report or "\nReferences" in report:
        return report
    references = ["\n\n## References\n"]
    references.extend(
        f"- **[{source.source_id}] {source.title}** — {source.url}"
        for source in sources
    )
    return report.rstrip() + "\n" + "\n".join(references)

def scoring_agent(single_agent_report: str, multi_agent_report: str, task: str):
    prompt = PromptTemplate.from_template(
        """You are an impartial judge. Your task is to score two texts based on a set of criteria.
        The original task was: "{task}"
        
        **Scoring Criteria (Total 10 Points):**
        1. **Factual Density (out of 4):** Does it explain the core mechanics and facts, or does it use filler words/corporate fluff? Deduct points for meta-commentary (e.g., "This report explores...").
        2. **Clarity and Structure (out of 3):** Is the text logically organized and easy to follow?
        3. **Completeness (out of 3):** Does it fully address the original task without missing key context?
        
        You must evaluate two texts:
        
        **Report 1 (Single-Agent):**
        {single_report}
        
        **Report 2 (Multi-Agent):**
        {multi_report}
        
        Please provide a score for each text out of 10. Your response MUST be a valid JSON object with two keys: "single_agent_score" and "multi_agent_score". Do not include markdown formatting like ```json in the output.
        
        {{
          "single_agent_score": 6.5,
          "multi_agent_score": 9.0
        }}
        """
    )
    scorer_runnable = prompt | model | parser
    response = scorer_runnable.invoke({
        "single_report": single_agent_report,
        "multi_report": multi_agent_report,
        "task": task
    })
    
    try:
        json_part = response[response.find('{'):response.rfind('}')+1]
        scores = json.loads(json_part)
        return scores
    except (json.JSONDecodeError, IndexError):
        return {"single_agent_score": "N/A", "multi_agent_score": "N/A"}

# --- Robust LLM-as-a-Judge evaluation ---
def _parse_judge_json(response: str):
    try:
        start, end = response.find("{"), response.rfind("}")
        payload = json.loads(response[start:end + 1])
        return payload["candidate_a"], payload["candidate_b"]
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _judge_overall(payload):
    return float(payload["overall"])


def robust_scoring_agent(single_agent_report: str, multi_agent_report: str, task: str, sources: List[RetrievedSource]):
    """Anonymous, order-swapped, repeated judging plus deterministic evidence metrics."""
    deterministic = {
        "single": verification_metrics(verify_report(single_agent_report, sources)),
        "multi": verification_metrics(verify_report(multi_agent_report, sources)),
    }
    rubric = (
        "You are a blinded evaluator. Score both candidates independently. Do not reward length, "
        "writing style, candidate position, or system identity. Use this rubric: factuality (3), "
        "citation correctness (2), citation completeness (2), task coverage (1), clarity (1), "
        "mathematical correctness (1). Return ONLY JSON with candidate_a and candidate_b objects, "
        "each containing factuality, citation_correctness, citation_completeness, coverage, clarity, "
        "mathematical_correctness, and overall (0-10).\nTASK: " + task
    )
    single_scores, multi_scores, stable = [], [], []
    for _ in range(judge_rounds):
        first = judge_model.invoke(rubric + "\nCANDIDATE A:\n" + single_agent_report + "\nCANDIDATE B:\n" + multi_agent_report)
        second = judge_model.invoke(rubric + "\nCANDIDATE A:\n" + multi_agent_report + "\nCANDIDATE B:\n" + single_agent_report)
        first_pair = _parse_judge_json(getattr(first, "content", str(first)))
        second_pair = _parse_judge_json(getattr(second, "content", str(second)))
        if first_pair is None or second_pair is None:
            continue
        first_single, first_multi = _judge_overall(first_pair[0]), _judge_overall(first_pair[1])
        second_multi, second_single = _judge_overall(second_pair[0]), _judge_overall(second_pair[1])
        single_scores.append((first_single + second_single) / 2)
        multi_scores.append((first_multi + second_multi) / 2)
        first_preference = first_single - first_multi
        second_preference = second_single - second_multi
        stable.append(first_preference * second_preference >= 0 or abs(first_preference) < 0.25 or abs(second_preference) < 0.25)
    if not single_scores:
        return {"single_agent_score": "N/A", "multi_agent_score": "N/A", "judge_status": "parse_failed", "position_consistency": 0.0, "deterministic": deterministic}
    single_score, multi_score = statistics.fmean(single_scores), statistics.fmean(multi_scores)
    consistency = statistics.fmean(stable) if stable else 0.0
    return {
        "single_agent_score": round(single_score, 2),
        "multi_agent_score": round(multi_score, 2),
        "judge_status": "stable" if consistency >= 0.67 else "position_sensitive",
        "position_consistency": round(consistency, 2),
        "single_quality_per_1000_tokens": round(single_score / max(0.001, len(single_agent_report.split()) * 1.33 / 1000), 2),
        "multi_quality_per_1000_tokens": round(multi_score / max(0.001, len(multi_agent_report.split()) * 1.33 / 1000), 2),
        "deterministic": deterministic,
    }


# --- 5. STREAMLIT UI ---
task_input = st.text_area("What topic would you like the agents to write a report on?", height=100, placeholder="e.g., Explain Quantum Entanglement...")

if st.button("Generate Reports", type="primary"):
    if not task_input.strip():
        st.error("Please enter a task or topic.")
    else:
        # 1. Run Single Agent
        with st.spinner("Retrieving research evidence..."):
            try:
                sources = retrieve_sources(task_input)
            except Exception as exc:
                sources = []
                st.warning(f"External search failed: {exc}. Continuing without retrieved sources.")

        if sources:
            st.info(f"Retrieved {len(sources)} sources using {research_mode}.")
            with st.expander("View retrieved evidence"):
                for source in sources:
                    st.markdown(f"**[{source.source_id}] [{source.title}]({source.url})**")
                    st.caption(source.snippet[:800])

        with st.spinner("Running single-agent baseline..."):
            single_report = run_single_agent(task_input, sources)
            single_report = add_reference_list(single_report, sources)
            
        # 2. Run Multi-Agent
        with st.spinner("Running multi-agent team (planning, researching, drafting, and critiquing)... This may take a moment."):
            initial_state = {"task": task_input, "revision_number": 0, "sources": sources}
            multi_agent_state = app.invoke(initial_state)
            multi_report = add_reference_list(multi_agent_state['draft'], sources)
            revisions_done = multi_agent_state['revision_number']

        # 3. Score Reports
        with st.spinner("Judging and scoring the reports..."):
            scores = robust_scoring_agent(single_report, multi_report, task_input, sources)

        st.success(f"Done! Multi-Agent loop completed {revisions_done} revision cycle(s).")
        st.info(
            f"Judge status: {scores.get('judge_status', 'unknown')} | "
            f"Order consistency: {scores.get('position_consistency', 'N/A')} | "
            f"Quality/1K tokens — single: {scores.get('single_quality_per_1000_tokens', 'N/A')}, "
            f"multi: {scores.get('multi_quality_per_1000_tokens', 'N/A')}"
        )
        st.divider()

        # Display side-by-side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Single-Agent Response")
            single_score = scores.get("single_agent_score", "N/A")
            st.metric(label="AI Judge Score", value=f"{single_score} / 10")
            with st.container(border=True):
                st.markdown(single_report)
                
        with col2:
            st.subheader("Multi-Agent Team Response")
            multi_score = scores.get("multi_agent_score", "N/A")
            st.metric(label="AI Judge Score", value=f"{multi_score} / 10")
            with st.container(border=True):
                st.markdown(multi_report)

        with st.expander("Deterministic evidence metrics"):
            st.json(scores.get("deterministic", {}))
