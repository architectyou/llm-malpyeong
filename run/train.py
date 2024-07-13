from .module import *


def parse_args():
    parser = argparse.ArgumentParser(
        prog="train", description="Training about Conversational Context Inference."
    )

    # Common Parameters
    parser.add_argument("--model_id", type=str, required=True, help="Model file path")
    parser.add_argument("--tokenizer", type=str, help="Huggingface tokenizer path")
    parser.add_argument(
        "--save_dir", type=str, default="../resource/results", help="Model save path"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size (both train and eval)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument("--warmup_steps", type=int, help="Scheduler warmup steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epoch", type=int, default=5, help="Training epoch")
    return parser.parse_args()


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if args.tokenizer is None:
        args.tokenizer = args.model_id
        
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    # Set the current directory to the script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(current_dir, '../resource/data/대화맥락추론_train.json')
    valid_file = os.path.join(current_dir, '../resource/data/대화맥락추론_dev.json')

    train_dataset = CustomDataset(train_file, tokenizer)
    valid_dataset = CustomDataset(valid_file, tokenizer)

    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        'labels': train_dataset.label,
    })
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        'labels': valid_dataset.label,
    })
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    training_args = SFTConfig(
        output_dir=args.save_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=0.1,
        num_train_epochs=args.epoch,
        max_steps=-1,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=5,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=1024,
        packing=True,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        args=training_args,
    )

    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
