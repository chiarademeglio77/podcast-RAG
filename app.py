import streamlit as st
from src.query import Searcher
import src.config as config
import os
from datetime import datetime, date

# Page configuration
st.set_page_config(
    page_title="RAG Article Assistant",
    page_icon="🎙️",
    layout="wide"
)

# Initialize searcher with cache clearing capability
@st.cache_resource
def get_searcher(cache_key=0):
    return Searcher()

# Sidebar for Filters and Cache
st.sidebar.header("🔍 Filters & System")

# Button to force refresh the database index
if st.sidebar.button("🔄 Refresh Database", help="Click if search doesn't show your latest files"):
    st.cache_resource.clear()
    st.rerun()

searcher = get_searcher()

all_sources = searcher.get_all_sources()
selected_source = st.sidebar.selectbox("Filter by Source:", ["All"] + all_sources)

# Ensure both are date objects to avoid comparison errors
date_range = st.sidebar.date_input(
    "Date Range:",
    value=[date(2025, 1, 1), date.today()],
    help="Select start and end dates"
)

st.sidebar.markdown("---")
st.sidebar.header("⚡ Preset Actions")

if st.sidebar.button("📊 Daily Summary", help="Summarize top 3 most recent articles"):
    recent_docs = searcher.get_recent_documents(n=3)
    if recent_docs:
        st.subheader("📋 Recent Daily Summary")
        for doc in recent_docs:
            with st.expander(f"**{doc.metadata.get('date', 'N/A')}** - {doc.metadata.get('source', 'N/A')}: *{doc.metadata.get('title', 'N/A')}*"):
                st.write(doc.page_content[:500] + "...")
                st.info(f"Source: {doc.metadata.get('source')} | Date: {doc.metadata.get('date')}")
    else:
        st.warning("No recent documents found.")

preset_query = st.sidebar.selectbox(
    "Common Questions:",
    [
        "Select a question...",
        "Summarize recent trends",
        "What are the top 3 topics this week?",
        "Show latest updates on Iran",
        "Identify geopolitical risks mentioned recently"
    ]
)

# Main Search Logic
st.title("🎙️ RAG Article Assistant")
st.markdown("---")
st.subheader("💬 Ask a Question")
query_text = st.text_input("", placeholder="Type your query here...")

# Handle Preset Query selection
if preset_query != "Select a question...":
    query_text = preset_query

if query_text:
    with st.spinner("Analyzing articles..."):
        # Build filters dictionary
        filters = {}
        if selected_source != "All":
            filters["source"] = selected_source
            
        results = searcher.query(query_text, k=10, filters=filters)
        
        if results:
            filtered_results = []
            # Safety check for date_range length
            start_date = date_range[0] if isinstance(date_range, (list, tuple)) else date_range
            end_date = date_range[1] if isinstance(date_range, (list, tuple)) and len(date_range) > 1 else date.today()

            for res in results:
                doc_date_str = res.metadata.get('date', 'unknown')
                if doc_date_str != 'unknown':
                    try:
                        dt = date.fromisoformat(doc_date_str)
                        if not (start_date <= dt <= end_date):
                            continue
                    except Exception:
                        pass
                filtered_results.append(res)

            if filtered_results:
                st.markdown(f"### Found {len(filtered_results)} relevant citations:")
                for i, res in enumerate(filtered_results[:5]):
                    st.markdown(f"""
                    ---
                    #### CITATION {i+1}
                    **Source:** {res.metadata.get('source')} | **Date:** {res.metadata.get('date')}
                    **Title:** {res.metadata.get('title')}
                    
                    {res.page_content}
                    """)
            else:
                st.info("No documents match your query within the selected date range.")
        else:
            st.info("No documents match your query and filters.")

# System Status
st.sidebar.markdown("---")
if os.path.exists(config.FAISS_INDEX_PATH):
    st.sidebar.success(f"✅ System Ready ({len(all_sources)} Sources)")
else:
    st.sidebar.error("❌ Database Missing")
