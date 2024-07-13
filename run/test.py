from .module import *


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="test", description="Testing about Conversational Context Inference."
    )
    
    g = parser.add_argument_group("Common Parameter")
    g.add_argument("--output", type=str, required=True, help="Output filename")
    g.add_argument("--model_id", type=str, required=True, help="Huggingface model ID")
    g.add_argument("--tokenizer", type=str, help="Huggingface tokenizer")
    g.add_argument("--device", type=str, required=True, help="Device to load the model")
    return parser.parse_args()


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )
    model.eval()

    if args.tokenizer is None:
        args.tokenizer = args.model_id
        
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = CustomDataset("resource/data/대화맥락추론_test.json", tokenizer)

    answer_dict = {
        0: "inference_1",
        1: "inference_2",
        2: "inference_3"
    }

    with open("resource/data/대화맥락추론_test.json", "r") as f:
        result = json.load(f)

    for idx in tqdm.tqdm(range(len(dataset))):
        inp, _ = dataset[idx]
        outputs = model(inp.to(args.device).unsqueeze(0))
        logits = outputs.logits[:, -1].flatten()
        probs = torch.nn.functional.softmax(
            torch.tensor([
                logits[tokenizer.vocab['A']],
                logits[tokenizer.vocab['B']],
                logits[tokenizer.vocab['C']]
            ]),
            dim=0
        ).detach().cpu().numpy()

        result[idx]["output"] = answer_dict[np.argmax(probs)]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)