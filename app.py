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
def get_searcher():
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

# Handle Daily Summary via Synthesis
if st.sidebar.button("📊 Daily Summary", help="Summarize top 3 most recent articles"):
    recent_docs = searcher.get_recent_documents(n=3)
    if recent_docs:
        with st.spinner("Generating summary of recent news..."):
            summary = searcher.synthesize_answer("Cosa è successo di importante negli ultimi articoli caricati?", recent_docs)
            st.subheader("📋 Analisi AI: Riepilogo Recente")
            st.markdown(summary)
            
            with st.expander("Vedi fonti originali"):
                for doc in recent_docs:
                    st.write(f"**{doc.metadata.get('date', 'N/A')}** - {doc.metadata.get('source', 'N/A')}: *{doc.metadata.get('title', 'N/A')}*")
                    st.info(doc.page_content[:300] + "...")
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
st.subheader("💬 Chiedi all'AI")
query_text = st.text_input("", placeholder="Cosa dice Anthropic riguardo all'educazione?")

# Handle Preset Query selection
if preset_query != "Select a question...":
    query_text = preset_query

if query_text:
    with st.spinner("Analizzando i documenti e filtrando le pubblicità..."):
        # 1. Retrieve fragments
        filters = {}
        if selected_source != "All":
            filters["source"] = selected_source
            
        raw_results = searcher.query(query_text, k=10, filters=filters)
        
        if raw_results:
            # Filter by date range manually
            start_date = date_range[0] if isinstance(date_range, (list, tuple)) else date_range
            end_date = date_range[1] if isinstance(date_range, (list, tuple)) and len(date_range) > 1 else date.today()

            valid_results = []
            for res in raw_results:
                doc_date_str = res.metadata.get('date', 'unknown')
                if doc_date_str != 'unknown':
                    try:
                        dt = date.fromisoformat(doc_date_str)
                        if not (start_date <= dt <= end_date):
                            continue
                    except Exception:
                        pass
                valid_results.append(res)

            if valid_results:
                # 2. Synthesize organic answer
                organic_answer = searcher.synthesize_answer(query_text, valid_results)
                
                st.markdown("### 🤖 Analisi AI")
                st.info(organic_answer)
                
                # 3. Show citations in collapsible section
                with st.expander("🔍 Vedi Citazioni e Fonti (Dati Grezzi)"):
                    for i, res in enumerate(valid_results):
                        st.markdown(f"**FONTE {i+1}**: {res.metadata.get('source')} ({res.metadata.get('date')}) - *{res.metadata.get('title')}*")
                        st.write(res.page_content)
                        st.markdown("---")
            else:
                st.info("Nessun documento trovato nel periodo selezionato.")
        else:
            st.info("Nessun documento trovato per questa ricerca.")

# System Status
st.sidebar.markdown("---")
if os.path.exists(config.FAISS_INDEX_PATH):
    all_filenames = searcher.get_all_filenames()
    total_docs = searcher.get_total_documents()
    
    st.sidebar.success(f"✅ Sistema Online")
    st.sidebar.info(f"📄 **Articoli**: {len(all_filenames)} file caricati")
    st.sidebar.info(f"🎙️ **Podcast**: {len(all_sources)} canali diversi")
    st.sidebar.info(f"🧩 **Database**: {total_docs} frammenti analizzabili")
    
    # Check for API Key presence for user feedback
    has_api_key = False
    if searcher.model:
        has_api_key = True
    
    if not has_api_key:
        st.sidebar.warning("⚠️ AI Disattivata: Inserisci la Google API Key nei 'Secrets' di Streamlit.")
    else:
        st.sidebar.caption("✨ AI Synthesis Attiva (Gemini 1.5 Flash)")
else:
    st.sidebar.error("❌ Database Mancante")


