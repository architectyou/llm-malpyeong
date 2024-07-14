import argparse


def train_args():
    parser = argparse.ArgumentParser(
        prog="train", description="Training about Conversational Context Inference."
    )

    # Common Parameters
    parser.add_argument("--model_id", type=str, required=True, help="Model file path")
    parser.add_argument("--tokenizer", type=str, help="Huggingface tokenizer path")
    parser.add_argument(
        "--save_dir", type=str, default="resource/results", help="Model save path"
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


def test_args():
    parser = argparse.ArgumentParser(
        prog="test", description="Testing about Conversational Context Inference."
    )
    
    g = parser.add_argument_group("Common Parameter")
    g.add_argument("--output", type=str, required=True, help="Output filename")
    g.add_argument("--model_id", type=str, required=True, help="Huggingface model ID")
    g.add_argument("--tokenizer", type=str, help="Huggingface tokenizer")
    g.add_argument("--device", type=str, required=True, help="Device to load the model")
    return parser.parse_args()