## Contributer
이 레포지토리는 아래 세 분의 기여를 통해 완성되어가고 있습니다.
- Hyeongcheol Geum (oppenheimer1223@outlook.com)
- SeungAh Son (gongsoonyee@gmail.com)
- Donghun Choi(ln996975@gmail.com)

## To do list
- [ ] 현 베이스 학습/튜닝/추론 코드에서 정량적 evaluation 코드 추가
- [ ] 블라썸 8B 결과 및 다른 모델 2B ~ 7, 8B 베이스 모델로 정량적 결과 내보기
- [ ] 어떻게 하면 성능을 향상시킬 수 있을까?
- [ ] 필요한 부가 기능 구현 및 코드 정리

## Repository Structure
```
resource/                 # 학습에 필요한 리소스들을 보관하는 디렉토리
└── data/                 # 데이터 파일들이 저장되는 폴더

run/                      # 실행 가능한 Python 스크립트를 보관하는 디렉토리
├── test.py               # 테스트 스크립트
└── train.py              # 학습 스크립트

src/                      # 학습에 사용될 커스텀 함수들을 보관하는 디렉토리
├── data.py               # 커스텀 데이터셋 클래스
└── utils.py              # 유틸리티 함수들
```

## Dataset Structure
```
{
    "id": "nikluge-2024-일상 대화 요약-train-000001",
    "input": {
        "conversation": [
            {
                "speaker": "SD2000001",
                "utterance": "저는 여행 다니는 것을 굉장히 좋아하는데요. 그래가지고 스페인이나 뭐 영국 유럽 아니면 국내에서도 뭐 강릉이나 전주 같은 데를 많이 다녔는데"
            },
            {
                "speaker": "SD2000001",
                "utterance": "혹시 여행 다니는 거 좋아하시나요?"
            },
            {
                "speaker": "SD2000002",
                "utterance": "저 여행 다니는 거 되게 좋아해서 대학교 내내 여행을 엄청 많이 다녔었는데요."
            },
            ...
            ...
            ...
        ],
        "subject_keyword": [
            "해외여행"
        ]
    },
    "output": "이 대화에서 화자들은 좋았던 여행지와 기억나는 주요 명소에 대해 이야기했습니다. SD2000001은 여행을 좋아하여 국내, 해외 여행을 많이 다녔다고 말했습니다. 특히 기억에 남는 여행지로 스페인 마드리드의 톨레도를 소개했습니다. 그 중 화려하게 꾸며진 대성당과 디저트가 인상적이었다고 이야기했습니다. SD2000002는 대학교에 진학한 후 해외여행을 자주 다녔고, 스페인과 포루투갈이 가장 기억에 남는 여행지라고 말했습니다. 그리고 톨레도도 다녀왔지만 날씨가 더워서 제대로 구경하지 못했다는 경험을 이야기했습니다."
}
```

## 실행 방법
### 베이스 학습
```
bash base_train.sh
```
### 베이스 추론
```
bash base_inference.sh
```

## Experiments
추가 예정

## Reference
- huggingface/transformers (https://github.com/huggingface/transformers)  
- Bllossome (Teddysum) (https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)  
- 국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  
