from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import os
import dotenv
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables from .env file
dotenv.load_dotenv()

# Retrieve Azure OpenAI credentials
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Set environment variables
os.environ["AZURE_OPENAI_API_KEY"] = azure_openai_api_key
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_openai_endpoint



# Initialize the LLM
llm = AzureChatOpenAI(
    azure_deployment=azure_openai_deployment_name,
    api_version=azure_openai_api_version,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

