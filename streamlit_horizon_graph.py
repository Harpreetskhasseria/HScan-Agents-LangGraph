# streamlit_horizon_graph.py

import streamlit as st
import asyncio
import pandas as pd
from horizon_graph import run_horizon_scan

st.set_page_config(page_title="Horizon Scanner", layout="wide")
st.title("🧠 Horizon Scanning - Multi-Site LangGraph Runner")

st.markdown("Enter up to 11 regulatory site URLs (comma-separated):")

url_input = st.text_area(
    label="",
    height=100,
    placeholder="https://www.whitehouse.gov/briefings-statements/, https://www.bis.org/press/pressrels.htm"
)

if st.button("▶️ Run Horizon Scan"):
    urls = [u.strip() for u in url_input.split(",") if u.strip()]
    if not urls:
        st.error("❌ Please enter at least one URL.")
    else:
        st.info("⏳ Running agents and processing sites...")
        with st.spinner("Processing..."):
            try:
                updates = asyncio.run(run_horizon_scan(urls, return_updates=True))
                if updates:
                    df = pd.DataFrame(updates)

                    st.success("✅ Scan complete. See results below:")

                    # ✅ Site update counts
                    st.subheader("🌐 Update Counts by Website")
                    if "source_url" in df.columns:
                        site_counts = (
                            df["source_url"]
                            .value_counts()
                            .rename_axis("Website")
                            .reset_index(name="Update Count")
                        )
                        st.dataframe(site_counts, use_container_width=True)
                    else:
                        st.warning("⚠️ 'source_url' column missing in results.")

                    # Final updates table
                    st.subheader("📋 Final Extracted Updates")
                    st.dataframe(df, use_container_width=True)

                    # Download button
                    csv_data = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name="horizon_scan_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("⚠️ No updates found.")
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
