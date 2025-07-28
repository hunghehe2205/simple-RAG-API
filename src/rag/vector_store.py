from typing import Union
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorDB:
    """
    Wrapper for building and querying a vector database using Chroma or FAISS in LangChain.

    Initializes a vector store from a list of documents and an embedding model,
    and provides a retriever interface for similarity search.

    Args:
        documents (List[Document], optional): List of documents to embed and store.
        vector_db (Union[Chroma, FAISS], optional): Vector store class to use (default: Chroma).
        embedding (Embeddings, optional): Embedding model for converting text to vectors.

    Methods:
        get_retriever(search_type, search_kwargs): Returns a retriever for semantic search.
    """
    def __init__(self,
                 documents = None,
                 vector_db: Union[Chroma, FAISS] = Chroma,
                 embedding = HuggingFaceEmbeddings(),
                 ) -> None:
        
        self.vector_db = vector_db
        self.embedding = embedding
        self.db = self._build_db(documents)

    def _build_db(self, documents):
        db = self.vector_db.from_documents(documents=documents, 
                                          embedding=self.embedding) # type : Chroma
        return db

    def get_retriever(self, 
                      search_type: str = "similarity", 
                      search_kwargs: dict = None
                      ):
        """Get standard retriever for similarity search."""
        if search_kwargs is None:
            search_kwargs = {"k": 2}  # Default to top 2 sources
            
        retriever = self.db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        return retriever