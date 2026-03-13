import os
import json
import operator
import streamlit as st
from typing import TypedDict, Annotated, List
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# --- PAGE CONFIG ---
st.set_page_config(page_title="Multi-Agent Reporter", layout="wide")
st.title("⚡ Agentic Workflows: Single Prompt vs. LangGraph Team")
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

if not api_key:
    st.warning("Please enter your Groq API Key in the sidebar to continue. Get one for free at console.groq.com")
    st.stop()

# Initialize Groq Model
os.environ["GROQ_API_KEY"] = api_key
try:
    # Using Llama 3.1 8B Instant and passing the dynamic user_temperature
    model = ChatGroq(model="llama-3.1-8b-instant", temperature=user_temperature)
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

# --- 2. DEFINE THE CORE AGENTS ---
def planner_agent(state: AgentState):
    prompt = PromptTemplate.from_template(
        "You are an expert technical planner. Create a detailed outline for a comprehensive technical article on this task: {task}. "
        "IMPORTANT: Your outline must use concise, professional section titles (e.g., 'Neural Network Architecture' instead of 'Description of the Neural Network...'). "
        "Do not use bureaucratic formats like 'Executive Summary'. Focus purely on educational value and logical flow."
    )
    runnable = prompt | model | parser
    plan = runnable.invoke({"task": state['task']})
    return {"plan": plan}

def researcher_agent(state: AgentState):
    prompt = PromptTemplate.from_template("You are an expert researcher. Gather dense, factual information and mathematical formulas based on this outline: {plan}")
    runnable = prompt | model | parser
    research = runnable.invoke({"plan": state['plan']})
    return {"research": research}

def writer_agent(state: AgentState):
    prompt = PromptTemplate.from_template(
        "You are an expert technical writer. Write a comprehensive, seamless technical article using this research: {research}. "
        "FORMATTING RULES: "
        "1. Use proper Markdown formatting (e.g., `###` for subheadings, `**bold**` for key terms). "
        "2. You MUST use LaTeX formatting (e.g., `$$ equation $$` or `$ equation $`) for ALL mathematical formulas and variables. "
        "3. Ensure smooth narrative transitions between sections. Do not just list the outline points. "
        "4. Write objectively. NEVER refer to 'this project', 'our team', or 'this report'."
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
def run_single_agent(task_string: str):
    prompt = PromptTemplate.from_template("You are an expert. Write a comprehensive technical explanation of this topic: {task}")
    runnable = prompt | model | parser
    return runnable.invoke({"task": task_string})

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

# --- 5. STREAMLIT UI ---
task_input = st.text_area("What topic would you like the agents to write a report on?", height=100, placeholder="e.g., Explain Quantum Entanglement...")

if st.button("Generate Reports", type="primary"):
    if not task_input.strip():
        st.error("Please enter a task or topic.")
    else:
        # 1. Run Single Agent
        with st.spinner("🤖 Running Single-Agent baseline..."):
            single_report = run_single_agent(task_input)
            
        # 2. Run Multi-Agent
        with st.spinner("🕵️‍♂️ Running Multi-Agent team (Planning, Researching, Drafting, and Critiquing)... This may take a moment."):
            initial_state = {"task": task_input, "revision_number": 0}
            multi_agent_state = app.invoke(initial_state)
            multi_report = multi_agent_state['draft']
            revisions_done = multi_agent_state['revision_number']

        # 3. Score Reports
        with st.spinner("⚖️ Judging and scoring the reports..."):
            scores = scoring_agent(single_report, multi_report, task_input)

        st.success(f"Done! Multi-Agent loop completed {revisions_done} revision cycle(s).")
        st.divider()

        # Display side-by-side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Single-Agent Response")
            single_score = scores.get("single_agent_score", "N/A")
            st.metric(label="AI Judge Score", value=f"{single_score} / 10")
            with st.container(border=True):
                st.markdown(single_report)
                
        with col2:
            st.subheader("👥 Multi-Agent Team Response")
            multi_score = scores.get("multi_agent_score", "N/A")
            st.metric(label="AI Judge Score", value=f"{multi_score} / 10")
            with st.container(border=True):
                st.markdown(multi_report)
