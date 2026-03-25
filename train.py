import os
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from unsloth import is_bfloat16_supported

# 1. Configurable Parameters for RTX 3060 12GB
model_name = "unsloth/Qwen3.5-0.8B" # Smaller Qwen 3.5 for 12GB VRAM
# model_name = "unsloth/Llama-3.2-1B" # Alternative Llama model

max_seq_length = 2048 # Adjust based on dataset complexity
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage.

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # LoRA rank, keep low for memory (8, 16, 32, 64)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Optimized to 0 for unsloth
    bias = "none",    # Optimized to "none" for unsloth
    use_gradient_checkpointing = "unsloth", # Use unsloth's optimized version
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 4. Data Preparation
# Example formatting function
PROMPT_STYLE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = PROMPT_STYLE.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

# Load your local dataset (jsonl)
if os.path.exists("dataset.jsonl"):
    print("Loading local 'dataset.jsonl'...")
    dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
else:
    print("Local dataset not found. Using Alpaca-cleaned sample...")
    dataset = load_dataset("yahma/alpaca-cleaned", split = "train[:100]") # Small slice for test

dataset = dataset.map(formatting_prompts_func, batched = True,)

# 5. Training Setup
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Total training steps, adjust based on size
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Set to "wandb" or "tensorboard" if configured
    ),
)

# 6. Run Training
trainer_stats = trainer.train()

# 7. Save Model
model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
# model.save_pretrained_gguf("model.gguf", tokenizer, quantization_method = "q4_k_m") # GGUF format for Ollama/LMStudio

print(f"Training completed. Memory used: {torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024:.2f} GB")
