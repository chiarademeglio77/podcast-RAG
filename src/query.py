from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import src.config as config
import os
import re
from typing import List, Dict, Optional

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

    def query(self, text: str, k: int = 10, filters: Optional[Dict] = None):
        """Search for relevant documents with keyword boosting for titles."""
        if not self.vector_db:
            return []
        
        # 1. Retrieve more candidates than requested to allow for re-ranking
        initial_k = k * 3
        results_with_scores = self.vector_db.similarity_search_with_score(text, k=initial_k)
        
        # 2. Extract words from query for boosting
        query_words = set(re.findall(r'\w+', text.lower()))
        
        ranked_results = []
        for doc, score in results_with_scores:
            # Apply metadata filters first
            if filters:
                match = True
                for key, value in filters.items():
                    if key in doc.metadata and doc.metadata[key] != value:
                        match = False
                        break
                if not match:
                    continue

            # 3. Calculate Boost Score
            # If query keywords are in the Title or Filename, we lower the "score" 
            # (since lower score = more relevant in vector search distance)
            boost = 0
            title = doc.metadata.get('title', '').lower()
            filename = doc.metadata.get('filename', '').lower()
            
            for word in query_words:
                if len(word) > 3: # Only boost non-trivial words
                    if word in title or word in filename:
                        boost += 0.2 # Substantial boost percentage
            
            # Adjusted score (FAISS distance - boost)
            final_score = score - boost
            ranked_results.append((doc, final_score))

        # 4. Sort by the new boosted score
        ranked_results.sort(key=lambda x: x[1])
        
        # Return only the documents
        return [res[0] for res in ranked_results[:k]]


    def get_all_sources(self) -> List[str]:
        """Extract unique sources from the index metadata."""
        if not self.vector_db:
            return []
        sources = set()
        # Access original docs from FAISS (docstore)
        for doc_id in self.vector_db.index_to_docstore_id.values():
            doc = self.vector_db.docstore.search(doc_id)
            if doc and 'source' in doc.metadata:
                sources.add(doc.metadata['source'])
        return sorted(list(sources))

    def get_recent_documents(self, n: int = 5) -> List[Dict]:
        """Fetch n most recent documents based on metadata date."""
        if not self.vector_db:
            return []
        
        docs = []
        for doc_id in self.vector_db.index_to_docstore_id.values():
            doc = self.vector_db.docstore.search(doc_id)
            if doc:
                docs.append(doc)
        
        # Sort by date descending
        # Dates are stored as ISO strings "YYYY-MM-DD"
        sorted_docs = sorted(docs, key=lambda x: x.metadata.get('date', '0000-00-00'), reverse=True)
        return sorted_docs[:n]

if __name__ == "__main__":
    s = Searcher()
    print("Sources:", s.get_all_sources())
