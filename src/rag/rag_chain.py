import re
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Dict, List, Any


class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()
        
    def parse(self, text: str) -> str:
        return self.extract_answer(text)
    
    def extract_answer(self,
                       text_response: str,
                       pattern: str = r"Answer:\s*(.*)"
                       ) -> str:
        
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response


class Enhanced_OutputParser:
    """Enhanced output parser that preserves both answer and source documents."""
    
    def __init__(self):
        self.str_parser = Str_OutputParser()
    
    def parse(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the inputs containing both LLM response and retrieved documents."""
        llm_response = inputs["llm_response"]
        retrieved_docs = inputs["retrieved_docs"]
        
        # Handle AIMessage object - extract content if it's an AIMessage
        if hasattr(llm_response, 'content'):
            response_text = llm_response.content
        else:
            response_text = str(llm_response)
        
        # Extract the answer from LLM response
        answer = self.str_parser.parse(response_text)
        
        # Process retrieved documents to extract source metadata only
        sources = []
        seen_sources = set()  # To avoid duplicate sources
        
        for i, doc in enumerate(retrieved_docs):
            # Extract source information
            source_path = doc.metadata.get('source', f'Document_{i}')
            page_num = doc.metadata.get('page', None)
            
            # Create a unique identifier for this source
            source_id = f"{source_path}_{page_num}"
            
            # Only add if we haven't seen this source+page combination
            if source_id not in seen_sources:
                source_info = {
                    "source": source_path,
                    "page": page_num
                }
                sources.append(source_info)
                seen_sources.add(source_id)
        
        return {
            "answer": answer,
            "sources": sources
        }


class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Use the following context to answer the question:
            {context}

            Question: {question}
            Answer:"""
            )
        self.enhanced_parser = Enhanced_OutputParser()

    def get_chain(self, retriever):
        def retrieve_and_format(question: str) -> Dict[str, Any]:
            """Retrieve documents and format context while preserving document metadata."""
            # Get documents with similarity scores
            retrieved_docs = retriever.invoke(question)
            
            # Format context for the prompt
            context = self.format_docs(retrieved_docs)
            
            return {
                "context": context,
                "question": question,
                "retrieved_docs": retrieved_docs
            }
        
        def generate_response(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Generate LLM response and combine with retrieved documents."""
            # Generate LLM response
            prompt_inputs = {
                "context": inputs["context"],
                "question": inputs["question"]
            }
            llm_response = (self.prompt | self.llm).invoke(prompt_inputs)
            
            # Combine LLM response with retrieved documents
            return {
                "llm_response": llm_response,
                "retrieved_docs": inputs["retrieved_docs"]
            }
        
        # Build the enhanced chain
        rag_chain = (
            RunnableLambda(retrieve_and_format)
            | RunnableLambda(generate_response) 
            | RunnableLambda(self.enhanced_parser.parse)
        )
        
        return rag_chain

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)