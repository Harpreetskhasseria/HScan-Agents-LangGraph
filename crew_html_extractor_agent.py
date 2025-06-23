# crew_html_extractor_agent.py

from crewai import Agent
from pydantic import BaseModel
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
from urllib.parse import urlparse, urljoin
from pathlib import Path
import os

# ✅ Updated absolute output directory
OUTPUT_DIR = Path(r"C:\Users\hp\Documents\Agent Store 1 - Copy\regulatory_outputs\site_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class HTMLExtractorInput(BaseModel):
    url: str
    cleaned_file: str

class HTMLExtractorOutput(BaseModel):
    url: str
    extracted_text: str
    extracted_links: list
    extracted_file: str

class HTMLExtractorAgent(Agent):
    def __init__(self):
        super().__init__(
            name="HTMLExtractorAgent",
            role="HTML Content Extractor",
            goal="Extract visible text and links from cleaned HTML for LLM processing.",
            backstory="You work after the HTML has been cleaned, extract human-visible content and links in a format suitable for LLM extraction."
        )

    def _extract_visible_text_and_links(self, html_path: str, base_url: str = "") -> tuple[str, list[str]]:
        with open(html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        for tag in soup(["script", "style", "noscript", "footer", "header", "nav", "aside"]):
            tag.decompose()

        result = []

        def traverse(node):
            if isinstance(node, NavigableString):
                text = node.strip()
                if text:
                    result.append(text)
            elif isinstance(node, Tag):
                if node.name == "a" and node.get("href"):
                    text = node.get_text(strip=True)
                    href = urljoin(base_url, node["href"])
                    if text:
                        result.append(f"{text} ({href})")
                else:
                    for child in node.children:
                        traverse(child)

        traverse(soup.body or soup)
        visible_text = " ".join(result)
        links = [part.split(" (")[-1].rstrip(")") for part in result if " (" in part]
        return visible_text, list(set(links))

    def run(self, input_data: dict) -> dict:
        input_obj = HTMLExtractorInput(**input_data)
        html_path = input_obj.cleaned_file
        url = input_obj.url

        print(f"🔍 Extracting from: {html_path}")
        visible_text, links = self._extract_visible_text_and_links(html_path, url)

        domain = urlparse(url).netloc.replace(".", "_")
        output_path = OUTPUT_DIR / f"{domain}_extracted.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(visible_text)

        print(f"✅ Saved extracted content to: {output_path}")

        output = HTMLExtractorOutput(
            url=url,
            extracted_text=visible_text,
            extracted_links=links,
            extracted_file=str(output_path)
        )
        return output.dict()
