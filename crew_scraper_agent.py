# crew_scraper_agent.py

import asyncio
import sys
import os
from urllib.parse import urljoin, urlparse
from pathlib import Path
from playwright.async_api import async_playwright
from crewai import Agent

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Ensure output directory exists
OUTPUT_DIR = Path("regulatory_outputs/site_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class ScraperAgent(Agent):
    def __init__(self):
        super().__init__(
            name="ScraperAgent",
            role="Web Scraper",
            goal="Extract raw HTML from regulatory websites",
            backstory="You automate the collection of online regulatory content for further processing by downstream agents."
        )

    async def _convert_links_to_absolute(self, page, base_url):
        await page.evaluate(
            """(base) => {
                document.querySelectorAll('a[href]').forEach(a => {
                    const href = a.getAttribute('href');
                    if (href && !href.startsWith('http')) {
                        a.setAttribute('href', new URL(href, base).href);
                    }
                });
            }""",
            base_url
        )

    async def _scrape_site(self, url: str) -> str:
        print(f"🔍 Scraping: {url}")
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=60000)
                await page.wait_for_timeout(5000)

                await self._convert_links_to_absolute(page, url)

                html = await page.content()
                await browser.close()
                return html
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")
            return f"<html><body><h1>Error scraping {url}</h1><p>{e}</p></body></html>"

    async def run(self, input_data: dict) -> dict:
        url = input_data.get("url")
        if not url:
            raise ValueError("Missing 'url' in input_data")

        html_output = await self._scrape_site(url)

        # Save output
        domain = urlparse(url).netloc.replace('.', '_')
        output_path = OUTPUT_DIR / f"{domain}_scraped.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_output)
        print(f"✅ HTML saved to {output_path}")

        input_data["scraped_html"] = html_output
        input_data["scraped_file"] = str(output_path)
        return input_data
