from .module import *


class Trainer:
    def __init__(self, 
            tokenizer, 
            model, 
            train_dataset, 
            valid_dataset,
            data_collator,
            args):
        self.__tokenizer = tokenizer
        self.__model, self.__peft_config = self._prepare_model(model)
        self.__train_dataset = train_dataset
        self.__valid_dataset = valid_dataset
        self.__data_collator = data_collator
        self.__args = args
        self.__training_args = self._get_training_args()
        self.__trainer = self._get_sfttrainer()
        
    def _prepare_model(self, model): # lora config
        model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        return model, peft_config

    def _get_training_args(self):
        return SFTConfig(
            output_dir=self.__args.save_dir,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            evaluation_strategy="epoch",
            per_device_train_batch_size=self.__args.batch_size,
            per_device_eval_batch_size=self.__args.batch_size,
            gradient_accumulation_steps=self.__args.gradient_accumulation_steps,
            learning_rate=self.__args.lr,
            weight_decay=0.1,
            num_train_epochs=self.__args.epoch,
            max_steps=-1,
            lr_scheduler_type="cosine",
            warmup_steps=self.__args.warmup_steps,
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=5,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_seq_length=1024,
            packing=True,
            seed=42,
            load_best_model_at_end=True,
            optim="paged_adamw_8bit"
        )

    def _get_sfttrainer(self):
        return SFTTrainer(
            model=self.__model,
            tokenizer=self.__tokenizer,
            train_dataset=self.__train_dataset,
            eval_dataset=self.__valid_dataset,
            data_collator=self.__data_collator,
            args=self.__training_args, 
            peft_config=self.__peft_config
        )
    
    def run(self):
        self.__trainer.train()