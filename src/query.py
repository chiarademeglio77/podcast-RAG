from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import src.config as config
import os
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

    def query(self, text: str, k: int = 5, filters: Optional[Dict] = None):
        """Search for relevant documents with optional metadata filtering."""
        if not self.vector_db:
            return []
        
        # In FAISS, filtering is typically done post-retrieval for simplicity in light loads
        # However, we can use metadata filter if specific keys are provided
        results = self.vector_db.similarity_search(text, k=k*2) # Retrieve more to allow for filtering
        
        if filters:
            filtered_results = []
            for res in results:
                match = True
                for key, value in filters.items():
                    if key in res.metadata and res.metadata[key] != value:
                        match = False
                        break
                if match:
                    filtered_results.append(res)
            return filtered_results[:k]
        
        return results[:k]

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
