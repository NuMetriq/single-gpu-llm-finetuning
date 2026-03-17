import torch
import unsloth
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"
DATASET_NAME = "HuggingFaceH4/no_robots"
OUTPUT_DIR = "outputs/qwen-no-robots-qlora-unsloth"
MAX_SEQ_LENGTH = 1024


def format_chat(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    train_dataset = dataset["train"].select(range(100))
    eval_dataset = dataset["test"].select(range(50))

    print("Formatting dataset...")
    train_dataset = train_dataset.map(lambda x: format_chat(x, tokenizer))
    eval_dataset = eval_dataset.map(lambda x: format_chat(x, tokenizer))

    print("Sample text:")
    print(train_dataset[0]["text"][:500])

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=10,
        save_steps=10,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        optim="adamw_8bit",
        report_to="none",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
    )

    print("Starting training...")
    trainer.train()

    print("Saving adapter...")
    model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    print("Done.")


if __name__ == "__main__":
    main()
