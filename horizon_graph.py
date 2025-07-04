# horizon_graph.py

import os
import uuid
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict
import pandas as pd
from langgraph.graph import StateGraph, END

from crew_scraper_agent import ScraperAgent
from crew_cleaner_agent import CleanerAgent
from crew_html_extractor_agent import HTMLExtractorAgent
from crew_llm_extractor_agent import LLMExtractorAgent
from crew_exclusion_agent import ExclusionAgent

load_dotenv()
State = Dict[str, any]

def log(msg: str):
    print(msg)

def generate_run_id():
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

# --- LangGraph Nodes ---
async def scraper_node(state: State) -> State:
    log(f"🌐 [ScraperNode] Scraping from: {state['url']}")
    return await ScraperAgent().run(state)

def cleaner_node(state: State) -> State:
    log("🧼 [CleanerNode] Cleaning HTML...")
    return CleanerAgent().run(state)

def html_extractor_node(state: State) -> State:
    log("🔍 [HTMLExtractorNode] Extracting visible content and links...")
    return HTMLExtractorAgent().run({
        "url": state["url"],
        "cleaned_file": state["cleaned_file"]
    })

def llm_extractor_node(state: State) -> State:
    log("🤖 [LLMExtractorNode] Extracting updates with LLM...")
    return LLMExtractorAgent().run({
        "url": state["url"],
        "extracted_file": state["extracted_file"],
        "extracted_links": state["extracted_links"]
    })

def exclusion_node(state: State) -> State:
    log("🚫 [ExclusionNode] Filtering updates...")
    state = ExclusionAgent().run({
        "url": state["url"],
        "llm_output_file": state["llm_output_file"]
    })
    state["final_updates"] = state.get("filtered_dataframe", pd.DataFrame()).to_dict(orient="records")
    return state

def output_node(state: State) -> State:
    updates = state.get("final_updates", [])
    source_url = state.get("url", "unknown")
    run_id = state.get("run_id", "none")

    if "combined_updates" not in state:
        state["combined_updates"] = []

    enriched_updates = []
    for item in updates:
        item = dict(item)
        item.setdefault("source_url", source_url)
        item.setdefault("run_id", run_id)
        enriched_updates.append(item)

    state["combined_updates"].extend(enriched_updates)
    return state

# --- LangGraph Assembly ---
def build_graph():
    graph = StateGraph(State)
    graph.add_node("scraper", scraper_node)
    graph.add_node("cleaner", cleaner_node)
    graph.add_node("html_extractor", html_extractor_node)
    graph.add_node("llm_extractor", llm_extractor_node)
    graph.add_node("exclusion", exclusion_node)
    graph.add_node("output", output_node)

    graph.set_entry_point("scraper")
    graph.add_edge("scraper", "cleaner")
    graph.add_edge("cleaner", "html_extractor")
    graph.add_edge("html_extractor", "llm_extractor")
    graph.add_edge("llm_extractor", "exclusion")
    graph.add_edge("exclusion", "output")
    graph.add_edge("output", END)

    return graph.compile()

async def run_single_site(app, url: str) -> List[Dict]:
    run_id = generate_run_id()
    state = {"url": url.strip(), "run_id": run_id}
    log(f"🚀 Starting pipeline for: {url}")
    result = await app.ainvoke(state)
    return result.get("combined_updates", [])

async def run_horizon_scan(urls: List[str], return_updates: bool = False):
    app = build_graph()
    tasks = [run_single_site(app, url) for url in urls]
    all_results = await asyncio.gather(*tasks)
    all_updates = [item for sublist in all_results for item in sublist]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_path = f"regulatory_outputs/horizon_scan_combined_{timestamp}.csv"
    os.makedirs("regulatory_outputs", exist_ok=True)

    if all_updates:
        pd.DataFrame(all_updates).to_csv(merged_path, index=False)
        log(f"✅ Combined results saved to: {merged_path}")
    else:
        log("⚠️ No updates found.")

    if return_updates:
        return all_updates
