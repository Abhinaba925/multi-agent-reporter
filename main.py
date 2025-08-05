import os
import json
import operator
from typing import TypedDict, Annotated, List
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI

# Set up the Gemini model
try:
    model = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
except Exception as e:
    print(f"Error initializing the model: {e}")
    print("Please ensure your GOOGLE_API_KEY is set correctly.")
    exit()

################################################################################
# 1. DEFINE THE AGENT STATE
################################################################################

class AgentState(TypedDict):
    task: str
    plan: str
    research: str
    draft: str
    critique: str
    report: Annotated[List[str], operator.add]
    revision_number: int

################################################################################
# 2. DEFINE THE CORE AGENTS
################################################################################

def planner_agent(state: AgentState):
    print("---AGENT: PLANNER---")
    prompt = PromptTemplate.from_template("You are an expert planner. Create a detailed plan for this task: {task}")
    runnable = prompt | model
    plan = runnable.invoke({"task": state['task']})
    return {"plan": plan}

def researcher_agent(state: AgentState):
    print("---AGENT: RESEARCHER---")
    prompt = PromptTemplate.from_template("You are an expert researcher. Gather information based on this plan: {plan}")
    runnable = prompt | model
    research = runnable.invoke({"plan": state['plan']})
    return {"research": research}

def writer_agent(state: AgentState):
    print("---AGENT: WRITER (DRAFTER)---")
    prompt = PromptTemplate.from_template("You are an expert writer. Write a draft report using this research: {research}")
    runnable = prompt | model
    draft = runnable.invoke({"research": state['research']})
    return {"draft": draft}

def revision_agent(state: AgentState):
    print("---AGENT: REVISER---")
    prompt = PromptTemplate.from_template(
        "You are an expert editor. Revise this draft: {draft} based on these critiques: {critique}"
    )
    runnable = prompt | model
    revised_draft = runnable.invoke({"draft": state['draft'], "critique": state['critique']})
    return {"draft": revised_draft}

def critic_agent(state: AgentState):
    print("---AGENT: CRITIC---")
    prompt = PromptTemplate.from_template(
        """You are an expert critic. Review the draft report. If it's good, say 'APPROVED'. 
        Otherwise, provide a numbered list of specific, actionable revisions.
        Draft: {draft}"""
    )
    runnable = prompt | model
    critique = runnable.invoke({"draft": state['draft']})
    revision_number = state.get('revision_number', 0) + 1
    return {"critique": critique, "revision_number": revision_number}

################################################################################
# 3. DEFINE THE GRAPH AND ITS LOGIC
################################################################################

def should_continue(state: AgentState):
    print("---CRITIC'S VERDICT---")
    if state['revision_number'] > 3:
        print("-> Reached revision limit.")
        return "end"
    if "APPROVED" in state['critique'].upper():
        print("-> Report is approved.")
        return "end"
    else:
        print("-> Report needs revision.")
        return "revise"

from langgraph.graph import StateGraph, END

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

################################################################################
# 4. DEFINE THE COMPARISON AND SCORING AGENTS
################################################################################

def run_single_agent(task_string: str):
    print("\n--- Running Single-Agent Comparison ---")
    prompt = PromptTemplate.from_template("You are an expert. Write a detailed report on this task: {task}")
    runnable = prompt | model
    return runnable.invoke({"task": task_string})

# --- NEW: A dedicated agent for scoring the final reports ---
def scoring_agent(single_agent_report: str, multi_agent_report: str, task: str):
    """
    Scores the two reports based on a predefined rubric.
    """
    print("\n--- AGENT: SCORER ---")
    
    prompt = PromptTemplate.from_template(
        """You are an impartial judge. Your task is to score two reports based on a set of criteria.
        The original task was: "{task}"
        
        **Scoring Criteria:**
        1.  **Clarity and Structure (out of 3):** Is the report well-organized, with clear headings and logical flow?
        2.  **Depth and Detail (out of 4):** Is the analysis comprehensive and detailed, or superficial?
        3.  **Completeness (out of 3):** Does the report fully address all aspects of the original task?
        
        You must evaluate two reports:
        
        **Report 1 (Single-Agent):**
        {single_report}
        
        **Report 2 (Multi-Agent):**
        {multi_report}
        
        Please provide a score for each report out of 10. Your response MUST be a valid JSON object with two keys: "single_agent_score" and "multi_agent_score".
        
        Example Response:
        {{
          "single_agent_score": 6.5,
          "multi_agent_score": 9.0
        }}
        """
    )
    
    scorer_runnable = prompt | model
    response = scorer_runnable.invoke({
        "single_report": single_agent_report,
        "multi_report": multi_agent_report,
        "task": task
    })
    
    try:
        # The response is often wrapped in markdown, so we find the JSON part
        json_part = response[response.find('{'):response.rfind('}')+1]
        scores = json.loads(json_part)
        return scores
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error parsing scores: {e}")
        return {"single_agent_score": "N/A", "multi_agent_score": "N/A"}

################################################################################
# 5. MAIN EXECUTION BLOCK
################################################################################

if __name__ == '__main__':
    task_string = "Analyze the impact of remote work on the tech industry's productivity and employee well-being. Write a detailed report."
    initial_task = {"task": task_string}
    
    # --- Run the Multi-Agent System ---
    print("🚀 Starting the Multi-Agent System...")
    multi_agent_final_state = app.invoke(initial_task)
    multi_agent_report = multi_agent_final_state['draft']

    # --- Run the Single-Agent System ---
    single_agent_report = run_single_agent(task_string)

    # --- Run the Scoring Agent ---
    scores = scoring_agent(single_agent_report, multi_agent_report, task_string)
    single_score = scores.get("single_agent_score", "N/A")
    multi_score = scores.get("multi_agent_score", "N/A")

    # --- Print the Final Comparison ---
    print("\n" + "="*80)
    print("                      COMPARISON OF FINAL REPORTS")
    print("="*80 + "\n")

    print(f"--- SINGLE-AGENT RESPONSE (SCORE: {single_score}/10) ---")
    print(single_agent_report)
    print("\n" + "-"*80 + "\n")

    print(f"--- MULTI-AGENT RESPONSE (SCORE: {multi_score}/10) ---")
    print(multi_agent_report)
    print("\n" + "="*80)
