"""
这里需要注意理解：

1. DataCollatorForCompletionOnlyLM 到底做了什么事情？
    怎么把 prompt 的 loss 忽略掉？（如果需要自己实现，需要怎么做，当然一般不自己实现）

2. data_path 之类的东西，请大家自行修改

"""


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig,TrainingArguments, HfArgumentParser
import torch
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from peft import LoraConfig

parser = HfArgumentParser(TrainingArguments)

training_args = parser.parse_args_into_dataclasses()[0]
# print(training_args)
# training_args.report_to=None

data_path = "../input/openassistant-guanaco-valid.csv"
model_path_or_name = "../../FlagAlpha--Llama2-Chinese-7b-Chat/"

ds_imdb = load_dataset("csv", data_files=data_path,  split="train[:2%]")


lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)
 
model = AutoModelForCausalLM.from_pretrained(
    model_path_or_name,
    load_in_4bit=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path_or_name
)

instruction_template = "### Human:"
response_template = "### Assistant:"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)



trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=ds_imdb,
    eval_dataset=ds_imdb,
    dataset_text_field="text",
    max_seq_length=512,
    peft_config=lora_config
)

trainer.train()
