# crew_cleaner_agent.py

from bs4 import BeautifulSoup
from pathlib import Path
from crewai import Agent
from urllib.parse import urlparse

# Output folder
OUTPUT_DIR = Path("regulatory_outputs/site_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tags to remove
TAGS_TO_REMOVE = ["script", "style", "noscript", "footer", "header", "nav", "aside"]

class CleanerAgent(Agent):
    def __init__(self):
        super().__init__(
            name="CleanerAgent",
            role="HTML Cleaner",
            goal="Clean the scraped HTML by removing unnecessary tags, styles, and scripts.",
            backstory="You ensure that downstream agents receive only the most relevant and readable HTML content."
        )


    def _clean_html_content(self, raw_html: str) -> str:
        soup = BeautifulSoup(raw_html, "html.parser")

        for tag in TAGS_TO_REMOVE:
            for element in soup.find_all(tag):
                element.decompose()

        # Remove empty or whitespace-only tags
        for tag in soup.find_all():
            if not tag.get_text(strip=True) and tag.name not in ["br", "hr"]:
                tag.decompose()

        return soup.prettify()

    def run(self, input_data: dict) -> dict:
        raw_html = input_data.get("scraped_html")
        url = input_data.get("url")

        if not raw_html or not url:
            raise ValueError("Missing 'scraped_html' or 'url' in input_data")

        cleaned_html = self._clean_html_content(raw_html)

        # Save cleaned output
        domain = urlparse(url).netloc.replace('.', '_')
        output_path = OUTPUT_DIR / f"{domain}_cleaned.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_html)
        print(f"✅ Cleaned HTML saved to: {output_path}")

        input_data["cleaned_html"] = cleaned_html
        input_data["cleaned_file"] = str(output_path)
        return input_data
