import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME = "HuggingFaceH4/no_robots"
OUTPUT_DIR = "outputs/qwen-no-robots-qlora"


def keep_messages_only(example):
    return {"messages": example["messages"]}


def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)

    train_dataset = (
        dataset["train"]
        .select(range(100))
        .map(
            keep_messages_only,
            remove_columns=dataset["train"].column_names,
        )
    )
    eval_dataset = (
        dataset["test"]
        .select(range(50))
        .map(
            keep_messages_only,
            remove_columns=dataset["test"].column_names,
        )
    )

    print("Train columns:", train_dataset.column_names)
    print("Eval columns:", eval_dataset.column_names)
    print("Sample example:", train_dataset[0])

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )

    model.config.use_cache = False

    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
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

    print("Attaching LoRA adapters...")
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    all_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    trainable_dtypes = sorted(
        {str(p.dtype) for p in model.parameters() if p.requires_grad}
    )
    print("All parameter dtypes:", all_dtypes)
    print("Trainable parameter dtypes:", trainable_dtypes)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=25,
        save_steps=25,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        max_length=1024,
        packing=False,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving adapter...")
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")

    print("Done.")


if __name__ == "__main__":
    main()
