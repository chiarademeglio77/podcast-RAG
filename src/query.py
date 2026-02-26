import warnings
# Suppress the harmless Pydantic V1 warning for Python 3.14
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import src.config as config
import os
import re
from typing import List, Dict, Optional, Tuple
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

class Searcher:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
        if os.path.exists(config.FAISS_INDEX_PATH):
            self.vector_db = FAISS.load_local(
                str(config.FAISS_INDEX_PATH), 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_db = None
        
        # Access Google API Key from Environment (.env local) or Secrets (Streamlit Cloud)
        api_key = os.getenv("GOOGLE_API_KEY") 
        if not api_key:
            try:
                api_key = st.secrets.get("GOOGLE_API_KEY")
            except Exception:
                api_key = None
        
        if api_key:
            genai.configure(api_key=api_key)
            # Use 1.5-flash for better stability in the free tier
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None


    def get_total_documents(self) -> int:
        """Get total count of chunks/documents in the index."""
        if not self.vector_db:
            return 0
        return len(self.vector_db.index_to_docstore_id)

    def query(self, text: str, k: int = 10, filters: Optional[Dict] = None):
        """Search for relevant documents with keyword boosting for titles."""
        if not self.vector_db:
            return []
        
        initial_k = k * 3
        results_with_scores = self.vector_db.similarity_search_with_score(text, k=initial_k)
        
        query_words = set(re.findall(r'\w+', text.lower()))
        
        ranked_results = []
        for doc, score in results_with_scores:
            if filters:
                match = True
                for key, value in filters.items():
                    if key in doc.metadata and doc.metadata[key] != value:
                        match = False
                        break
                if not match:
                    continue

            boost = 0
            title = doc.metadata.get('title', '').lower()
            filename = doc.metadata.get('filename', '').lower()
            
            for word in query_words:
                if len(word) > 3:
                    if word in title or word in filename:
                        boost += 0.2
            
            final_score = score - boost
            ranked_results.append((doc, final_score))

        ranked_results.sort(key=lambda x: x[1])
        return [res[0] for res in ranked_results[:k]]

    def synthesize_answer(self, query: str, contexts: List[any]) -> str:
        """Use Gemini to filter ads and synthesize a coherent answer."""
        if not self.model:
            return "Errore: Google API Key non configurata. Configurala nei Secrets di Streamlit Cloud."
        
        if not contexts:
            return "Non ho trovato informazioni pertinenti."

        # Prepare context text with metadata labels
        context_text = ""
        for i, doc in enumerate(contexts):
            context_text += f"\n--- DOCUMENT {i+1} (Source: {doc.metadata.get('source')}, Date: {doc.metadata.get('date')}) ---\n"
            context_text += doc.page_content + "\n"

        prompt = f"""
Sei un assistente esperto. Il tuo compito è rispondere alla domanda dell'utente basandoti ESCLUSIVAMENTE sui frammenti di testo forniti sotto.

REGOLE STRETTE:
1. PULIZIA AUTOMATICA: Riconosci ed elimina assolutamente ogni segmento che sembra un annuncio pubblicitario, sponsorizzazione o promozione (es. Philip Morris, carte di credito, promozioni di altri podcast, broker, ecc.).
2. RERANKING: Usa SOLO i frammenti che parlano realmente del soggetto richiesto. Ignora quelli irrilevanti.
3. SINTESI ORGANICA: Riassumi in modo coerente e discorsivo le informazioni. Non limitarti a copiare e incollare.
4. STRICT CONTEXT: Se nei documenti non c'è una risposta chiara al soggetto richiesto, rispondi ESATTAMENTE con: 'Non ho trovato informazioni pertinenti'. Non provare a inventare nulla usando la pubblicità o conoscenze esterne.
5. LINGUA: Rispondi in Italiano.

TESTI FORNITI:
{context_text}

DOMANDA: {query}

RISPOSTA ORGANICA:
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Errore durante la sintesi AI: {str(e)}"

    def get_all_sources(self) -> List[str]:
        """Extract unique podcast sources from the index metadata."""
        if not self.vector_db:
            return []
        sources = set()
        for doc_id in self.vector_db.index_to_docstore_id.values():
            doc = self.vector_db.docstore.search(doc_id)
            if doc and 'source' in doc.metadata:
                sources.add(doc.metadata['source'])
        return sorted(list(sources))

    def get_all_filenames(self) -> List[str]:

        """Extract unique filenames from the index metadata."""
        if not self.vector_db:
            return []
        files = set()
        for doc_id in self.vector_db.index_to_docstore_id.values():
            doc = self.vector_db.docstore.search(doc_id)
            if doc and 'filename' in doc.metadata:
                files.add(doc.metadata['filename'])
        return sorted(list(files))

    def get_recent_documents(self, n: int = 5) -> List[Dict]:

        if not self.vector_db:
            return []
        docs = []
        for doc_id in self.vector_db.index_to_docstore_id.values():
            doc = self.vector_db.docstore.search(doc_id)
            if doc:
                docs.append(doc)
        sorted_docs = sorted(docs, key=lambda x: x.metadata.get('date', '0000-00-00'), reverse=True)
        return sorted_docs[:n]
