# crew_formatter_agent.py

import pdfkit
from pathlib import Path
from urllib.parse import urlparse
from crewai import Agent

# Output folder
OUTPUT_DIR = Path("regulatory_outputs/site_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class FormatterAgent(Agent):
    def __init__(self):
        super().__init__(
            name="FormatterAgent",
            role="PDF Formatter",
            goal="Convert cleaned HTML content into a formatted PDF for structured analysis.",
            backstory="You take clean HTML and turn it into a uniform format to help downstream agents extract key data."
        )

    def run(self, input_data: dict) -> dict:
        html_path = input_data.get("cleaned_file")
        url = input_data.get("url")

        if not html_path or not url:
            raise ValueError("Missing 'cleaned_file' or 'url' in input_data")

        # Prepare output file path
        domain = urlparse(url).netloc.replace('.', '_')
        pdf_output_path = OUTPUT_DIR / f"{domain}_formatted.pdf"

        # wkhtmltopdf config
        wkhtmltopdf_path = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
        config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)

        options = {
            'encoding': 'UTF-8',
            'enable-local-file-access': ''
        }

        try:
            pdfkit.from_file(html_path, str(pdf_output_path), configuration=config, options=options)
            print(f"✅ PDF created: {pdf_output_path}")
        except Exception as e:
            print(f"❌ PDF conversion failed for {html_path}: {e}")
            input_data["pdf_error"] = str(e)
            return input_data

        input_data["pdf_file"] = str(pdf_output_path)
        return input_data
