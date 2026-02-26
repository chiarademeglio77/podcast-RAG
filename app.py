import streamlit as st
from src.query import Searcher
import src.config as config
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="RAG Article Assistant",
    page_icon="🎙️",
    layout="wide"
)

# Initialize searcher
@st.cache_resource
def get_searcher():
    return Searcher()

searcher = get_searcher()

# Title and Description
st.title("🎙️ RAG Article Assistant")
st.markdown("---")

# Sidebar for Filters
st.sidebar.header("🔍 Filters")

all_sources = searcher.get_all_sources()
selected_source = st.sidebar.selectbox("Filter by Source:", ["All"] + all_sources)

date_range = st.sidebar.date_input(
    "Date Range:",
    value=[datetime(2025, 1, 1), datetime.now()],
    help="Select start and end dates"
)

st.sidebar.markdown("---")
st.sidebar.header("⚡ Preset Actions")

if st.sidebar.button("📊 Daily Summary", help="Summarize top 3 most recent articles"):
    recent_docs = searcher.get_recent_documents(n=3)
    if recent_docs:
        st.subheader("📋 Recent Daily Summary")
        for doc in recent_docs:
            with st.expander(f"**{doc.metadata['date']}** - {doc.metadata['source']}: *{doc.metadata['title']}*"):
                st.write(doc.page_content[:500] + "...")
                st.info(f"Source: {doc.metadata['source']} | Date: {doc.metadata['date']}")
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
            
        results = searcher.query(query_text, k=5, filters=filters)
        
        if results:
            st.markdown(f"### Found {len(results)} relevant citations:")
            
            for i, res in enumerate(results):
                # Filter by date range manually (results were pre-filtered by source in Searcher.query)
                doc_date = res.metadata.get('date', 'unknown')
                if doc_date != 'unknown':
                    dt = datetime.fromisoformat(doc_date).date()
                    if not (date_range[0] <= dt <= date_range[1]):
                        continue

                st.markdown(f"""
                ---
                #### CITATION {i+1}
                **Source:** {res.metadata['source']} | **Date:** {res.metadata['date']}
                **Title:** {res.metadata['title']}
                
                {res.page_content}
                """)
        else:
            st.info("No documents match your query and filters.")

# System Status
st.sidebar.markdown("---")
if os.path.exists(config.FAISS_INDEX_PATH):
    st.sidebar.success("✅ System Ready")
else:
    st.sidebar.error("❌ Database Missing")
