from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig



model_path = "../FlagAlpha--Llama2-Chinese-7b-Chat/"


quantized_model_dir = "/shared/gptq_output"
tokenizer = AutoTokenizer.from_pretrained(model_path)


model = AutoGPTQForCausalLM.from_quantized(
    "/shared/gptq_output/",
    device="cuda:0", 
)

print(tokenizer.decode(model.generate(
    **tokenizer("<s>Human: 介绍一下中国\n</s><s>Assistant: ", return_tensors="pt").to(model.device))[0]))