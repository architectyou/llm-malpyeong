from src.data import *
from src.model import *
from src.utils import *
from config import train_args


def main(args):
    # ModelZoo 클래스로부터 모델과 토크나이저 정의
    zoo       = ModelZoo(args)
    tokenizer = zoo.tokenizer
    model     = zoo.model

    # KoreanDataset 클래스로부터 국립국어원 데이터셋 정의
    dataset = KoreanDataset(args, tokenizer)
    train_dataset = dataset.train_dataset
    valid_dataset = dataset.valid_dataset
    data_collator = dataset.data_collator

    # 필요한 것: 모델, 토크나이저, 데이터셋, 데이터콜레이터, 학습 설정

    # Trainer 클래스로부터 학습 진행
    trainer = Trainer(
        tokenizer, 
        model, 
        train_dataset, 
        valid_dataset, 
        data_collator, 
        args
    )
    trainer.run()


if __name__ == "__main__":
    args = train_args()
    main(args)