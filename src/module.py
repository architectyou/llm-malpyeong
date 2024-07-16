import os
import json
import tqdm
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import (
    SFTTrainer, 
    SFTConfig
)
from peft import (
    prepare_model_for_kbit_training, 
    LoraConfig, 
    get_peft_model
)