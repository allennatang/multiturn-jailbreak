from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_id = "deepseek-ai/DeepSeek-V2-Lite-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)