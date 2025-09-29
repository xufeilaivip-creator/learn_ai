"""
这个版本是使用 trl 跑通 SFT

记住： SFT 和 FT 在 LM 任务下是一样的。只是一般来说，大家说的 SFT是 instruction tuning.


小任务：怎么定义自己的 metrics ??? （有的时候，我们不是只想看 loss, 有可能看其他的指标，比如 acc)


1. 导入需要的包
2. 定义参数（parser arguments)
3. 导入模型相关的东西；
    p1. data
    p2. model
    p3. tokenizer
4. 定义trainer, trainer.run()
"""


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig,TrainingArguments, HfArgumentParser
import torch
from datasets import load_dataset
from trl import SFTTrainer

from peft import LoraConfig

parser = HfArgumentParser(TrainingArguments)

training_args = parser.parse_args_into_dataclasses()[0]
# print(training_args)
# training_args.report_to=None

data_path = "./imdb.csv"
model_path_or_name = "/openbayes/input/input0"

ds_imdb = load_dataset("csv", data_files=data_path,  split="train")

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
# print(model)
# print(ds_imdb[0])

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
