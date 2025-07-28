from pydantic import BaseModel, Field
from typing import List, Optional

from src.rag.file_loader import Loader
from src.rag.vector_store import VectorDB
from src.rag.rag_chain import Offline_RAG

class InputQA(BaseModel):
    """
    Input model for the RAG system.

    Attributes:
        question (str): The question to be answered.
        documents (list): List of documents to be used for answering the question.
    """
    question: str = Field(..., description="The question to be answered.")
    

class SourceMetadata(BaseModel):
    """Metadata for each source document used in the response."""
    source: str = Field(..., description="Source document path/name")
    page: Optional[int] = Field(default=None, description="Page number if applicable")


class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")
    sources: List[SourceMetadata] = Field(..., title="Source documents used")
    
    
def build_rag_chain(llm, data_dir, data_type):
    doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers=2)
    retriever = VectorDB(documents = doc_loaded).get_retriever(search_kwargs={"k": 3})
    rag_chain = Offline_RAG(llm).get_chain(retriever)
    return rag_chain