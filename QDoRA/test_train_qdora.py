import os
import torch

from datasets import load_dataset
from huggingface_hub import login

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)


MODEL = "google/gemma-3-4b-it"



hf_token = os.getenv("HF_TOKEN")

if hf_token:
    login(hf_token)
else:
    print("Warning: HF_TOKEN not found in environment variables.")


print("Uzkraunamas duomenu rinkinys...")

ds = load_dataset("uonlp/CulturaX", "lt", split="train")

ds = ds.shuffle(seed=2026).select(range(100000))

ds = ds.remove_columns([
    c for c in ds.column_names
    if c != "text"
])


split_ds = ds.train_test_split(test_size=0.1, seed=2026)

train_ds = split_ds["train"]
eval_ds = split_ds["test"]

print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")



tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"



def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

    labels = tokens["input_ids"].copy()

    labels = [
        -100 if token_id == tokenizer.pad_token_id else token_id
        for token_id in labels
    ]

    tokens["labels"] = labels

    tokens["token_type_ids"] = [0] * len(tokens["input_ids"])

    return tokens


print("Tokenizuojama train duomenu aibe...")

train_ds = train_ds.map(
    tokenize,
    remove_columns=train_ds.column_names,
    num_proc=4,
)

print("Tokenizuojama eval duomenu aibe...")

eval_ds = eval_ds.map(
    tokenize,
    remove_columns=eval_ds.column_names,
    num_proc=4,
)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)



print("Uzkraunamas modelis...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

model.config.torch_dtype = torch.float16



peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_dora=True,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()



training_args = TrainingArguments(
    output_dir="./qdora-culturax-lt",

    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,

    num_train_epochs=1,
    learning_rate=2e-4,

    logging_steps=10,

    save_strategy="steps",
    save_steps=500,

    eval_strategy="steps",
    eval_steps=500,

    fp16=True,
    bf16=False,

    optim="paged_adamw_8bit",

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },

    lr_scheduler_type="linear",
    warmup_steps=5,

    report_to="none",
    remove_unused_columns=False,

    weight_decay=0.01,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=DefaultDataCollator(),
)


print("Pradedamas QDoRA modelio adaptavimas...")

trainer.train()


print("Issaugomas modelis...")

trainer.save_model("./qdora-culturax-lt/final_checkpoint")
tokenizer.save_pretrained("./qdora-culturax-lt/final_checkpoint")

print("Modelis ir tokenizeris issaugoti.")
