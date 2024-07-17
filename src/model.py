from .module import *


class ModelZoo:
    def __init__(self, args):
        """
        지원하였으면 하는 모델
        1. BLOOSOM-8B
        2. Llama-3
        """
        if args.tokenizer is None:
            args.tokenizer = args.model_id
        self.tokenizer = self._get_tokenizer(args)
        self.model     = self._get_model(args)

    def _get_tokenizer(self, args):
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def _get_model(self, args):
        # 4-bit quantizing
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            # quantization_config=quantization_config,
            device_map="auto",
            # quantization_config = BitsAndBytesConfig(load_in_8bit=True), #H00 8bit 양자화 오류로 사용하지 않음.
        )
        return model