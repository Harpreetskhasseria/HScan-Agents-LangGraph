# crew_llm_extractor_agent.py

import json
import re
import pandas as pd
import os
from urllib.parse import urlparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from difflib import get_close_matches
from crewai import Agent
from pydantic import BaseModel
from typing import List

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Output path
OUTPUT_DIR = Path("regulatory_outputs/site_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Pydantic I/O Schemas ---
class LLMExtractorInput(BaseModel):
    url: str
    extracted_file: str
    extracted_links: List[str]

class LLMExtractorOutput(BaseModel):
    url: str
    llm_output_file: str

# --- Agent Class ---
class LLMExtractorAgent(Agent):
    def __init__(self):
        super().__init__(
            name="LLMExtractorAgent",
            role="Regulatory Update Extractor",
            goal="Use LLM to extract structured updates from visible HTML text.",
            backstory="You read visible HTML content and links to extract structured regulatory updates."
        )

    def _classify_with_llm(self, full_text: str, links: List[str]) -> pd.DataFrame:
        escaped_text = full_text.replace('"', '\\"')
        prompt = f"""
You are a regulatory update extraction assistant.

From the following DOCUMENT CONTENT, extract each distinct regulatory update.
Return the output as a strict JSON array of objects, each with the following keys:
- "date": the date of the update in YYYY-MM-DD format (if available)
- "topic": short title or subject of the update
- "additional_context": supporting detail or summary text
- "link": full URL to the source (choose from known_links)
- "regulator": the issuing regulatory body

⚠️ Do not include any explanatory text or markdown. Output must start with [ and end with ].
⚠️ Ensure all string values are wrapped in double quotes. Escape any internal quotes.

Known links to choose from: {json.dumps(links)}

DOCUMENT CONTENT:
\"\"\"
{escaped_text}
\"\"\"
"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You extract structured regulatory updates from documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4096
            )
            llm_output = response.choices[0].message.content.strip()
            cleaned = re.sub(r"^```json|```$", "", llm_output.strip(), flags=re.MULTILINE).strip()
            parsed = json.loads(cleaned)
            for item in parsed:
                if "additional_context" not in item:
                    item["additional_context"] = ""
            return pd.DataFrame(parsed, columns=["date", "topic", "additional_context", "link", "regulator"])
        except Exception as e:
            print(f"⚠️ LLM extraction failed: {e}")
            return pd.DataFrame(columns=["date", "topic", "additional_context", "link", "regulator"])

    def _fix_links(self, df: pd.DataFrame, known_links: List[str]) -> pd.DataFrame:
        print("🔗 Running fuzzy link match.")
        for i, row in df.iterrows():
            topic = row["topic"]
            if not row["link"] or row["link"].strip() == "":
                matches = get_close_matches(topic.lower(), known_links, n=1, cutoff=0.3)
                if matches:
                    df.at[i, "link"] = matches[0]
        return df

    def run(self, input_data: dict) -> dict:
        input_obj = LLMExtractorInput(**input_data)
        txt_file = input_obj.extracted_file
        url = input_obj.url
        known_links = input_obj.extracted_links

        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Extracted file not found: {txt_file}")

        with open(txt_file, "r", encoding="utf-8") as f:
            full_text = f.read()

        df = self._classify_with_llm(full_text, known_links)
        df = self._fix_links(df, known_links)

        domain = urlparse(url).netloc.replace('.', '_') if url else "unknown"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = OUTPUT_DIR / f"{domain}_llm_output_{timestamp}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"✅ LLM-extracted data saved to: {output_path}")
        return LLMExtractorOutput(
            url=url,
            llm_output_file=str(output_path)
        ).dict()
