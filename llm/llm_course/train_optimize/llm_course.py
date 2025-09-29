from dataclasses import dataclass
from datasets import load_dataset, DatasetDict
import os

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from sklearn.metrics import accuracy_score

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)


MAX_SEQ_LEN = 512
MODEL_PATH = "../FlagAlpha--Llama2-Chinese-7b-Chat/"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

dataset_input_name = "prompt"
dataset_output_name = "completion"


@dataclass
class ModelArguments:
    model_name_or_path: str = MODEL_PATH
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"]


@dataclass
class DataTrainingArguments:
    data_file: str


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Text
ds = DatasetDict()
ds["train"] = load_dataset("json", data_files=data_args.data_file, split="train[:85%]")
ds["valid"] = load_dataset("json", data_files=data_args.data_file, split="train[:15%]")
for v in ds["train"]:
    print(v)
    break

# Dataset(Token)
def tokenize(item):
    def _tokenize(prompt):
        inputs = tokenizer(
            prompt,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        if (
            inputs["input_ids"][-1] != tokenizer.eos_token_id
            and len(inputs["input_ids"]) < MAX_SEQ_LEN
        ):
            inputs["input_ids"].append(tokenizer.eos_token_id)
            inputs["attention_mask"].append(1)
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    inp = item["prompt"] + item["completion"]
    inputs = _tokenize(inp)
    prompt_inputs = _tokenize(item["prompt"])
    prompt_len = len(prompt_inputs["input_ids"])
    if prompt_inputs["input_ids"][-1] == tokenizer.eos_token_id:
        prompt_len -= 1
    inputs["labels"] = [-100] * prompt_len + inputs["labels"][prompt_len:]
    return inputs

dataset = ds.map(
    tokenize, 
    remove_columns=[dataset_input_name, dataset_output_name]
)
for v in dataset["train"]:
    print(v)
    break

# DataLoader
data_collator=DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

def build_dataloader(dataset):
    return DataLoader(
        dataset, batch_size=2, 
        shuffle=True, collate_fn=data_collator, pin_memory=True
    )
train_dataloader = build_dataloader(dataset["train"])
valid_dataloader = build_dataloader(dataset["valid"])
for v in train_dataloader:
    print(v)
    break


# def compute_metrics(eval_preds):
#     # 约定包含predictions和labels
#     logits, labels = eval_preds
#     preds = logits.argmax(dim=-1)
#     y_true = labels[:, 1:].reshape(-1)
#     y_pred = preds[:, :-1].reshape(-1)
#     return accuracy_score(y_true, y_pred)

# lora_config = LoraConfig(
#     r=model_args.lora_r,
#     lora_alpha=model_args.lora_alpha,
#     target_modules=model_args.target_modules,
#     lora_dropout=model_args.lora_dropout,
#     task_type="CAUSAL_LM",
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_args.model_name_or_path,
#     torch_dtype=torch.float16,
#     load_in_8bit=False,
#     device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
# )
# model.add_adapter(lora_config)

# training_args.remove_unused_columns = False
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["valid"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )
# model.config.use_cache = False
# trainer.train()




# print(model_args)
# print(data_args)
# print(training_args)
