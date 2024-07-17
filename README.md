# 2024년 국립국어원 인공지능 한국어 능력평가 - 대화 맥락추론(가)

## Contributer
이 레포지토리는 아래 레포지토리에서 Baseline Code를 참조하였습니다.
<br/>
https://github.com/russellgeum/LLM-Korean-CCI-2024


## Repository Structure
```
resource/                 # 학습에 필요한 리소스들을 보관하는 디렉토리
└── data.py/              # 데이터 파일들이 저장되는 폴더
src/                      # 학습에 사용될 커스텀 함수들을 보관하는 디렉토리
└── module.py             # 소스 파일들에 필요한 모듈
├── data.py               # 커스텀 데이터셋 클래스
├── model.py              # 커스텀 데이터셋 클래스
└── utils.py              # 유틸리티 함수들
resuls/                   # 학습 완료 후 추론 결과가 저장되는 디렉토리

test.py                   # 테스트 스크립트
train.py                  # 학습 스크립트
```

## Dataset Structure
```
{
        "id": "nikluge-2024-대화 맥락 추론-train-000001",
        "input": {
            "conversation": [
                {
                    "speaker": 2,
                    "utterance": "진짜 신의 한수",
                    "utterance_id": "MDRW2100003410.1.1"
                },
                {
                    "speaker": 1,
                    "utterance": "이사하자마자 비 많이 와서 베란다 물 많이 새는 거 알았잖아",
                    "utterance_id": "MDRW2100003410.1.2"
                },
                {
                    "speaker": 2,
                    "utterance": "글치 계속 해떴으면 몰랐겠지",
                    "utterance_id": "MDRW2100003410.1.3"
                },
                {
                    "speaker": 1,
                    "utterance": "그 때 물새는 거 알고 코킹작업해소 다행이다",
                    "utterance_id": "MDRW2100003410.1.4"
                },
                {
                    "speaker": 2,
                    "utterance": "ㅇㅇ 안그랬으면 오늘처럼 비 많이 내리는 날 물바다됐을거야",
                    "utterance_id": "MDRW2100003410.1.5"
                },
                {
                    "speaker": 1,
                    "utterance": "요 아래 씽크홀 공사하던데 괜찮을라나",
                    "utterance_id": "MDRW2100003410.1.6"
                },
                {
                    "speaker": 2,
                    "utterance": "그러게 저번에도 비 많이 와서 땅꺼진 건데 큰일이네",
                    "utterance_id": "MDRW2100003410.1.7"
                },
                {
                    "speaker": 1,
                    "utterance": "하수도 공사도 같이 하더만 물 안빠져서",
                    "utterance_id": "MDRW2100003410.1.8"
                },
                {
                    "speaker": 2,
                    "utterance": "새로 지은 곳인데도 그러네",
                    "utterance_id": "MDRW2100003410.1.9"
                },
                {
                    "speaker": 1,
                    "utterance": "부실공사지 뭐",
                    "utterance_id": "MDRW2100003410.1.10"
                },
                {
                    "speaker": 2,
                    "utterance": "비 많이 올 때는 그쪽으로 다니지 말아야겠다",
                    "utterance_id": "MDRW2100003410.1.11"
                },
                {
                    "speaker": 1,
                    "utterance": "ㅇㅇ 조심해",
                    "utterance_id": "MDRW2100003410.1.12"
                },
                {
                    "speaker": 1,
                    "utterance": "저번에 지나가다 보니 좀 무섭더라",
                    "utterance_id": "MDRW2100003410.1.13"
                },
                {
                    "speaker": 2,
                    "utterance": "나도 봤는데 씽크홀 크기가 엄청나더라",
                    "utterance_id": "MDRW2100003410.1.14"
                },
                {
                    "speaker": 1,
                    "utterance": "오늘 비가 엄청 많이 내리네",
                    "utterance_id": "MDRW2100003410.1.15"
                }
            ],
            "reference_id": [
                "MDRW2100003410.1.11"
            ],
            "category": "원인",
            "inference_1": "화자2가 사는 곳 근처에서 베란다 보수 공사가 진행되고 있다.",
            "inference_2": "화자2가 사는 곳 근처에서 싱크홀 보수 공사가 진행되고 있다.",
            "inference_3": "화자2가 사는 곳 근처에서 싱크홀 보수 공사가 중단되었다."
        },
        "output": "inference_2"
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
- 국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  
- 국립국어원 대화맥락추론 가형 베이스 코드 (https://github.com/teddysum/Korean_CCI_2024)
- Bllossome (Teddysum) (https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)