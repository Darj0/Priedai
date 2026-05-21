import logging
import os
 
import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log"),
    ],
)
log = logging.getLogger(__name__)
 
MODEL_NAME = "google/gemma-3-4b-it"
OUTPUT_DIR = "./lora-culturax-lt"       # Pakeistas katalogo pavadinimas į lora
FINAL_DIR  = "./lora-culturax-lt-final" # Pakeistas katalogo pavadinimas į lora
 
COMPUTE_DTYPE = torch.bfloat16
 
MAX_LENGTH = 512        
LORA_R     = 32          
LORA_ALPHA = 32       
LEARNING_RATE = 2e-4
WARMUP_STEPS  = 5    
GRAD_ACC_STEPS = 4       
MIN_TEXT_CHARS = 200     

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)
    log.info("HuggingFace prisijungimas sėkmingas.")
else:
    log.warning("HF_TOKEN nerastas – gali nepavykti atsisiųsti modelio.")

# 1. Duomenų paruošimas
log.info("Kraunamas CulturaX (lt) dataset...")
ds = load_dataset("uonlp/CulturaX", "lt", split="train")
 
ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
 
before = len(ds)
ds = ds.filter(lambda ex: len(ex["text"]) >= MIN_TEXT_CHARS, num_proc=4)
log.info(f"Filtruota: {before:,} → {len(ds):,} tekstų (pašalinta {before - len(ds):,})")
 
ds = ds.shuffle(seed=2026).select(range(100_000))
ds = ds.train_test_split(test_size=0.1, seed=2026)
log.info(f"Train: {len(ds['train']):,} | Val: {len(ds['test']):,}")

# 2. Modelio užkrovimas (Standartinis LoRA - be 4-bit kvantizacijos)
log.info("Kraunamas modelis standartiniu 16-bit formatu...")
 
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=COMPUTE_DTYPE,
    device_map="auto",
    attn_implementation="eager",    
)
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
 
# [PAŠALINTA] prepare_model_for_kbit_training nebereikia, nes modelis nėra kvantizuotas
model.config.use_cache = False

# 3. Tokenizavimas
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_token_type_ids=True,
    )
    
    if "token_type_ids" not in tokens:
        tokens["token_type_ids"] = [0] * len(tokens["input_ids"])
        
    labels = [
        -100 if t == tokenizer.pad_token_id else t
        for t in tokens["input_ids"]
    ]
    tokens["labels"] = labels
    return tokens
 
log.info("Tokenizuojami duomenys...")
ds["train"] = ds["train"].map(
    tokenize,
    remove_columns=ds["train"].column_names,
    num_proc=4,
    desc="Tokenizuojamas train",
)
ds["test"] = ds["test"].map(
    tokenize,
    remove_columns=ds["test"].column_names,
    num_proc=4,
    desc="Tokenizuojamas test",
)

# 4. LoRA konfigūracija
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  
        "gate_proj", "up_proj", "down_proj",      
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5. Mokymo parametrai
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=GRAD_ACC_STEPS,  
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    num_train_epochs=1,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",   
    warmup_steps=WARMUP_STEPS,
    bf16=True,
    fp16=False,
    # Naudojame standartinį adamw_torch. 
    # Jei trūksta VRAM, galima pakeisti atgal į "adamw_8bit" arba "paged_adamw_8bit"
    optim="adamw_torch",
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,           
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    remove_unused_columns=False,
    weight_decay=0.01
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=DefaultDataCollator(),
)
 
log.info("Pradedamas treniravimas...")
trainer.train(
    resume_from_checkpoint="output_dir/checkpoint-2000"
)
 
log.info(f"Treniravimas baigtas. Išsaugoma į: {FINAL_DIR}")
trainer.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
log.info("Viskas išsaugota.")
