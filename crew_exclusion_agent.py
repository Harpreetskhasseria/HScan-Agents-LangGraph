# crew_exclusion_agent.py

import os
import json
import re
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urlparse
from datetime import datetime
from openai import OpenAI
from crewai import Agent

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Output folder
OUTPUT_DIR = Path("regulatory_outputs/site_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUSION_PROMPT = """
You are an AI exclusion agent for a large U.S.-based financial institution.

Your job is to review the following regulatory update based on the provided Topic and (if present) Additional Context. You must decide whether the update should be INCLUDED or EXCLUDED from downstream compliance monitoring.

Return your answer as a JSON object with:
- "recommendation": either "Include" or "Exclude"
- "reason": a short explanation (1–2 sentences) explaining your decision

Exclude items that are:
- Appointments, personnel changes, or events not impacting regulations
- General economic commentary not tied to a regulatory directive
- Updates only about non-U.S. markets unless they directly affect U.S. banks

Include items that:
- Involve regulations, policies, enforcement, or systemic financial implications
- Mention compliance, banking operations, risk management, or supervision
- Involve a firm being fined, penalized, or cited for violations

Review this update:

Topic: {topic}
Additional Context: {additional_context}
"""

class ExclusionAgent(Agent):
    def __init__(self):
        super().__init__(
            name="ExclusionAgent",
            role="Relevance Filter",
            goal="Decide whether each regulatory update should be included in monitoring.",
            backstory="You help compliance teams by excluding irrelevant updates like events, appointments, or non-regulatory items."
        )

    def _review_llm(self, topic: str, additional_context: str):
        additional_context = additional_context.strip() if additional_context else "None"
        prompt = EXCLUSION_PROMPT.format(topic=topic, additional_context=additional_context)

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a regulatory compliance assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            content = response.choices[0].message.content.strip()
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            json_text = content[json_start:json_end]
            return json.loads(json_text)

        except Exception as e:
            return {
                "recommendation": "Exclude",
                "reason": f"⚠️ LLM parsing error: {str(e)}"
            }

    def run(self, input_data: dict) -> dict:
        df = input_data.get("llm_dataframe")
        url = input_data.get("url")

        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Missing or invalid 'llm_dataframe' in input_data")

        print(f"🔍 Running exclusion filter on {len(df)} rows")

        df["Recommendation"] = ""
        df["Reason"] = ""

        for i, row in df.iterrows():
            result = self._review_llm(row["topic"], row.get("additional_context", ""))
            df.at[i, "Recommendation"] = result.get("recommendation", "Exclude")
            df.at[i, "Reason"] = result.get("reason", "⚠️ No reason provided")

        # Save output
        domain = urlparse(url).netloc.replace('.', '_') if url else "unknown"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = OUTPUT_DIR / f"{domain}_exclusion_checked_{timestamp}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ Exclusion results saved to: {output_path}")

        input_data["exclusion_file"] = str(output_path)
        input_data["filtered_dataframe"] = df
        return input_data
