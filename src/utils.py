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
        self.__model = model
        self.__train_dataset = train_dataset
        self.__valid_dataset = valid_dataset
        self.__data_collator = data_collator
        self.__args = args
        self.__training_args = self._get_training_args()
        self.__trainer = self._get_sfttrainer()

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
        )

    def _get_sfttrainer(self):
        return SFTTrainer(
            model=self.__model,
            tokenizer=self.__tokenizer,
            train_dataset=self.__train_dataset,
            eval_dataset=self.__valid_dataset,
            data_collator=self.__data_collator,
            args=self.__training_args, 
        )
    
    def run(self):
        self.__trainer.train()