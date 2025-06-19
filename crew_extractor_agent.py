# crew_extractor_agent.py

import fitz  # PyMuPDF
import re
import json
import pandas as pd
import os
from urllib.parse import urlparse, urljoin
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from crewai import Agent
from difflib import get_close_matches

# Load OpenAI API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Output folder
OUTPUT_DIR = Path("regulatory_outputs/site_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class ExtractorAgent(Agent):
    def __init__(self):
        super().__init__(
            name="ExtractorAgent",
            role="Regulatory Update Extractor",
            goal="Use LLM to extract structured regulatory updates from PDF content.",
            backstory="You read regulatory documents and extract key details like topic, date, and regulator using a language model."
        )

    def _extract_full_text_and_links(self, pdf_path: str, base_url: str = ""):
        doc = fitz.open(pdf_path)
        full_text = ""
        all_links = []

        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            full_text += f"\n--- Page {page_num} ---\n{page_text.strip()}"

            for link in page.get_links():
                if "uri" in link and link["uri"]:
                    uri = link["uri"]
                    full_url = urljoin(base_url, uri) if base_url else uri
                    all_links.append(full_url)

        url_pattern = re.compile(r'https?://[^\s)>\]\"]+')
        visible_links = re.findall(url_pattern, full_text)
        combined_links = list(set(all_links + visible_links))
        return full_text.strip(), combined_links

    def _classify_with_llm(self, full_text, links):
        prompt = (
            "You are a regulatory update extraction assistant.\n\n"
            "From the following document content, extract each distinct regulatory update as a JSON object with:\n"
            "- date (as shown in content)\n"
            "- topic (main title or headline for the update)\n"
            "- additional_context (copy any sentence or paragraph that provides supporting explanation, rationale, or clarification of the topic — do NOT generate or summarize; only extract exact text following the topic if it adds detail)\n"
            "- link (extract from the text directly if visibly present near the topic, or pick the best matching one from the known links list below)\n"
            f"- known_links: {json.dumps(links)}\n"
            "- regulator (if mentioned, e.g., White House, BIS, FINRA, etc.)\n\n"
            "Respond with a JSON array of objects.\n\n"
            f"DOCUMENT CONTENT:\n\"\"\"\n{full_text}\n\"\"\""
        )

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

    def _is_link_suspicious(self, link):
        if pd.isna(link):
            return True
        link = str(link).strip().lower()
        return (
            link.endswith('/') or
            '#content__main' in link or
            'topics=' in link or
            'filters=' in link
        )

    def _fix_links_by_matching_titles(self, df, known_links):
        link_counts = df["link"].fillna("").value_counts()
        if len(link_counts) <= 3 or link_counts.iloc[0] > len(df) // 2:
            print("⚠️ Detected low link variety — applying fuzzy matching to fix links.")
            for i, row in df.iterrows():
                topic = row["topic"]
                if self._is_link_suspicious(row["link"]):
                    matches = get_close_matches(topic.lower(), known_links, n=1, cutoff=0.3)
                    if matches:
                        df.at[i, "link"] = matches[0]
        else:
            print("✅ Links appear unique — no fixing needed.")
        return df

    def _filter_relevant_links(self, links):
        return [
            link for link in links
            if not re.search(r'#|topics=|filters=|content__main', link)
            and re.search(r'/\d{4}/|cfpb-|finra-|whitehouse', link)
        ]

    def _fix_repeated_links(self, df, all_links):
        total = len(df)
        repeated = df["link"].fillna("").value_counts()
        if not repeated.empty:
            most_common_link = repeated.index[0]
            repetition_rate = repeated.iloc[0] / total

            if repetition_rate >= 0.2:
                print(f"⚠️ Detected repeated link ({most_common_link}) in {repetition_rate*100:.1f}% of entries — rerunning fuzzy match with cleaned links.")
                cleaned_links = self._filter_relevant_links(all_links)
                for i, row in df.iterrows():
                    topic = row["topic"]
                    if self._is_link_suspicious(row["link"]) or row["link"] == most_common_link:
                        matches = get_close_matches(topic.lower(), cleaned_links, n=1, cutoff=0.3)
                        if matches:
                            df.at[i, "link"] = matches[0]
            else:
                print("✅ No excessive link repetition detected.")
        return df

    def run(self, input_data: dict) -> dict:
        pdf_path = input_data.get("pdf_file")
        url = input_data.get("url", "")

        if not pdf_path:
            raise ValueError("Missing 'pdf_file' in input_data")

        full_text, links = self._extract_full_text_and_links(pdf_path, url)
        print(f"\n📄 Extracted full text and {len(links)} links.")

        df = self._classify_with_llm(full_text, links)
        df = self._fix_links_by_matching_titles(df, links)
        df = self._fix_repeated_links(df, links)

        domain = urlparse(url).netloc.replace('.', '_') if url else "unknown"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = OUTPUT_DIR / f"{domain}_llm_output_{timestamp}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ LLM-extracted data saved to: {output_path}")

        input_data["llm_dataframe"] = df
        input_data["llm_output_file"] = str(output_path)
        return input_data
