from src.data import *
from src.model import *
from src.utils import *
from config import test_args


def main(args):
    # 모델, 토크나이저 불러오기
    zoo = ModelZoo(args)
    tokenizer = zoo.tokenizer
    model = zoo.model
    model.eval()

    # 데이터셋 불러오기
    dataset = CustomDataset("resource/data/대화맥락추론_test.json", tokenizer)

    answer_dict = {
        0: "inference_1",
        1: "inference_2",
        2: "inference_3"
    }

    with open("resource/data/대화맥락추론_test.json", "r") as f:
        result = json.load(f)

    for idx in tqdm.tqdm(range(len(dataset))):
        inp, _  = dataset[idx]
        outputs = model(inp.to(args.device).unsqueeze(0))
        logits  = outputs.logits[:, -1].flatten()
        probs   = torch.nn.functional.softmax(
            torch.tensor([
                logits[tokenizer.vocab['A']],
                logits[tokenizer.vocab['B']],
                logits[tokenizer.vocab['C']]
            ]),
            dim=0
        )
        probs = probs.detach().cpu().numpy()
        result[idx]["output"] = answer_dict[np.argmax(probs)]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = test_args()
    main(args)