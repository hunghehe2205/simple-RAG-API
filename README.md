# Simple RAG API with Top 2 Sources

A comprehensive Retrieval-Augmented Generation (RAG) system built with FastAPI and LangChain that provides intelligent question-answering capabilities with source attribution. The system processes PDF and HTML documents, creates vector embeddings, and returns answers with the top 2 most relevant source documents. In this project, I have to 2 options of using model: using a deployed model on Azure AI Foundry or a HuggingFace model.


## 🚀 Features

- **Multi-format Document Support**: Process PDF and HTML files
- **Intelligent Document Retrieval**: Vector-based similarity search using Chroma or FAISS
- **Source Attribution**: Returns top 2 most relevant source documents with each answer
- **Scalable Processing**: Multi-threaded document loading with progress tracking
- **Flexible LLM Support**: Compatible with Azure OpenAI and Hugging Face models
- **RESTful API**: FastAPI-based web service with automatic documentation
- **CORS Support**: Cross-origin resource sharing enabled
## 📁 Project Structure

```
.
├── data_source/
│   └── download.py          # Script to download research papers
├── requirements.txt         # Python dependencies
└── src/
    ├── app.py              # FastAPI application entry point
    ├── base/
    │   ├── llm_api.py      # Azure OpenAI LLM configuration
    │   └── llm_hugging.py  # Hugging Face LLM configuration
    └── rag/
        ├── file_loader.py  # Document loading and text splitting
        ├── main.py         # Main RAG chain builder and data models
        ├── rag_chain.py    # RAG chain implementation
        ├── utils.py        # Utility functions
        └── vector_store.py # Vector database wrapper
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd simple-RAG-API
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the root directory:
```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2023-12-01-preview
```

4. **Download sample documents**
```bash
cd data_source
python download.py
```

## 🚀 Usage

### Starting the Server

```bash
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Health Check
```http
GET /check
```
Returns the API status.

#### 2. Question Answering
```http
POST /generative_ai
Content-Type: application/json

{
    "question": "What is attention mechanism in transformers?"
}
```

**Response:**
```json
{
    "answer": "The attention mechanism in transformers...",
    "sources": [
        {
            "source": "./data_source/Attention Is All You Need.pdf",
            "page": 3
        },
        {
            "source": "./data_source/BERT- Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf",
            "page": 2
        }
    ]
}
```

#### 3. System Statistics
```http
GET /stats
```
Returns basic information about the RAG system configuration.

### Interactive API Documentation

Visit `http://localhost:8000/docs` for Swagger UI documentation or `http://localhost:8000/redoc` for ReDoc documentation.

## 🏗️ Architecture

### Core Components

1. **Document Loader** (`file_loader.py`)
   - Supports PDF and HTML file processing
   - Multi-threaded loading with progress tracking
   - Text preprocessing and UTF-8 normalization
   - Configurable text chunking

2. **Vector Store** (`vector_store.py`)
   - Wrapper for Chroma and FAISS vector databases
   - Uses HuggingFace embeddings by default
   - Configurable similarity search parameters

3. **RAG Chain** (`rag_chain.py`)
   - Enhanced output parser that preserves source metadata
   - Custom prompt template for context-aware responses
   - Retrieval and generation pipeline

4. **LLM Integration** (`base/`)
   - Azure OpenAI integration with chat models
   - Hugging Face models with quantization support
   - Flexible model switching

### Data Flow

1. **Document Processing**: PDF/HTML files → Text chunks → Vector embeddings
2. **Query Processing**: User question → Vector similarity search → Top-k documents
3. **Answer Generation**: Retrieved context + Question → LLM → Answer + Sources
4. **Response Formatting**: Structured output with source attribution

## ⚙️ Configuration

### Document Processing
```python
# Text splitting configuration
split_kwargs = {
    "chunk_size": 300,
    "chunk_overlap": 0,
}

# Vector search configuration
search_kwargs = {"k": 3}  # Retrieve top 3 documents
```

### LLM Configuration

**Azure OpenAI:**
- Temperature: 0 (deterministic responses)
- Max retries: 2
- Configurable timeout and token limits

**Hugging Face:**
- Default model: `Qwen/Qwen2.5-3B-Instruct`
- 4-bit quantization support
- GPU acceleration with device mapping

## 📊 Sample Documents

The system comes with a curated collection of AI/ML research papers:

- Attention Is All You Need (Transformers)
- BERT: Pre-training of Deep Bidirectional Transformers
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- Instruction Tuning for Large Language Models: A Survey
- Llama 2: Open Foundation and Fine-Tuned Chat Models
- DeepSeek LLM series (V2, V3, R1)

## 🔧 Customization

### Adding New Document Types
Extend the `BaseLoader` class in `file_loader.py`:

```python
class CustomLoader(BaseLoader):
    def __call__(self, files: List[str], **kwargs):
        # Implement custom loading logic
        pass
```

### Custom Embedding Models
Modify the `VectorDB` initialization in `vector_store.py`:

```python
from langchain_community.embeddings import OpenAIEmbeddings

embedding = OpenAIEmbeddings()
vector_db = VectorDB(embedding=embedding)
```

### Custom Prompts
Update the prompt template in `rag_chain.py`:

```python
self.prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Your custom prompt template here..."
)
```

## 🐛 Troubleshooting

### Common Issues

1. **TOKENIZERS_PARALLELISM Warning**: Already handled by setting `os.environ["TOKENIZERS_PARALLELISM"] = "false"`

2. **Memory Issues with Large Documents**: Reduce `chunk_size` or use fewer workers in document loading

3. **Slow Response Times**: Consider using FAISS instead of Chroma for larger document collections
   
4. **Azure OpenAI Rate Limits**: Implement retry logic or use Hugging Face models as fallback

---

**Note**: Make sure to keep your API keys secure and never commit them to version control. Use environment variables or secure key management services in production.
