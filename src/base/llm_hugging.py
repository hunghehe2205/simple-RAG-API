import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
from langchain.llms.huggingface_pipeline import HuggingFacePipeline


nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

def get_huggingface_llm(model_name: str = "Qwen/Qwen2.5-3B-Instruct",
                        max_new_token = 1024,
                        **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )
    llm = HuggingFacePipeline(pipeline=model_pipeline, model_kwargs=kwargs)
    return llm