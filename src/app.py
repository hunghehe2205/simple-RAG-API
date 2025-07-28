import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes

from src.base.llm_api import llm
from src.rag.main import build_rag_chain, InputQA, OutputQA

gen_ai_docs = "./data_source"

# -----------Chains-----------
rag_chains = build_rag_chain(llm, data_dir=gen_ai_docs, data_type="pdf")

#----------FastAPI App-----------
app = FastAPI(
    title="RAG API with Top 2 Sources",
    description="RAG API that returns answers with top 2 most relevant source documents",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# --------- Routes - FastAPI ----------------

@app.get("/check")
async def check():
    return {"status": "ok", "message": "Enhanced RAG API is running"}


@app.post("/generative_ai", response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    """
    Generate an answer with top 2 most relevant source documents.
    
    Returns:
        - answer: The generated response
        - sources: Top 2 most relevant source documents
    """
    try:
        # Invoke the enhanced RAG chain
        result = rag_chains.invoke(inputs.question)
        
        # The result now contains answer and sources
        return OutputQA(
            answer=result["answer"],
            sources=result["sources"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing question: {str(e)}"
        )


@app.get("/stats")
async def get_api_stats():
    """Get basic statistics about the RAG system."""
    return {
        "data_source_directory": gen_ai_docs,
        "file_type": "pdf",
        "status": "ready"
    }