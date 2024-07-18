from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os, json, re
from time import time
import pdb

start_time = time()
# env load
load_dotenv()
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# model
model = genai.GenerativeModel('gemini-1.5-flash')
safety_settings = {
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
}

# def generate_safe_content(model=model, prompt, safety_settings=safety_settings, max_attempts = 3):
#     for attempt in range(max_attempts):
#         try : 
#             response = model.generate_content(prompt, safety_settings)
            
#             if response.candidates:
#                 result = response.text.split(',')
#                 return result[0], result[1]
#             else : 
#                 print(f"Attempt {attempt+1} : Content blocked. Regenerating...")
#         except ValueError as e : 
#             print(f"Attempt {attempt+1} : Error occurred : {e}. Regenerating...")
        
#         if attempt == max_attempts -1 :
#             print("Max attempts reached. Returning safe default content.")
#             return "Safe default content 1", "Safe default content 2"

def generate_conversation():
    prompt = """
        1. 한국어로 2명의 화자 간 15~20턴의 대화를 생성해주세요.
        2. 대화 주제는 일상적 대화 주제를 바탕으로 합니다. 유해한 대화는 만들지 마세요.
        3. 생성하는 대화는 구어체, 문어체, 인터넷 용어가 섞여도 괜찮습니다. 상황에 맞게 만들어주세요. (단, 이모지는 넣지 않습니다.)
        4. 구문 강조 (볼드체)는 사용하지 않습니다.
        5. 대화를 진행하는 화자는 익명입니다. 'speaker1', 'speaker2' 혹은 '화자1', '화자2'로 표시하세요.
        6. 아래의 example의 형식을 참고하여 답변을 생성해주세요.
        7. 답변을 절대 Markdown으로 생성하지 않습니다 / Do not use markdown formatting under any circumstances.

    **example**
    'speaker1: 아침부터 비가 엄청 오네. 덕분에 출근길이 헬이었어. ㅠㅠ\n\nspeaker2: 헐, 나도! 버스 엄청 밀려서 겨우 도착했어. 덕분에 지각할 뻔 했네. \n\nspeaker1: 나도 지각할 뻔 했어. 택시 잡으려고 했는데, 택시도 없고… 결국 뛰어서 왔어. \n\nspeaker2: 헐 대박. 뛰어서 오다니, 진짜 힘들었겠다. \n\nspeaker1: 응, 힘들긴 했는데 그래도 지각은 안해서 다행이야. \n\nspeaker2: 그래도 무사히 도착해서 다행이네. 오늘 점심 뭐 먹을지 정했어? \n\nspeaker1: 아직 안 정했는데. 너 뭐 먹고 싶어? \n\nspeaker2: 난 칼칼한 국물이 땡기는데, 혹시 칼국수 괜찮아? \n\nspeaker1: 칼국수 좋아! 근데 나 칼국수 먹으면 속 더부룩해서… \n\nspeaker2: 아, 그렇구나. 그럼 뭐가 좋을까? \n\nspeaker1: 샐러드는 어때? \n\nspeaker2: 샐러드도 괜찮긴 한데, 뭔가 든든한 게 먹고 싶어. \n\nspeaker1: 그럼 돈까스? \n\nspeaker2: 돈까스도 좋네! 그럼 돈까스로 결정! \n\nspeaker1: 오케이! 돈까스로 하자! \n\nspeaker2: 좋아! 빨리 퇴근하고 맛있는 돈까스 먹자. \n\nspeaker1: 응! 빨리 퇴근하고 싶다! \n'
    
    """
    response = model.generate_content(prompt,
                                      safety_settings = safety_settings)
    # pdb.set_trace()
    return response.text

def generate_inferences(conversation):
    prompt = f"""
        1. {conversation} 다음 대화에 대해 다음과 같은 추론문 유형의 정의가 있습니다. 
        2. 추론문 유형의 정의는 랜덤으로 "한 가지"만 선택합니다.
        3. 추론문 유형의 정의를 한 가지 선택 후 이에 대한 추론 예시를 3가지 생성해주세요.
        4. 문장은 '\n'를 기준으로 3가지를 생성하면 됩니다.
        5. 추론 예시 중 '2가지'는 '가짜 추론'예시, '오직 1가지'만 '진짜 추론예시'로 생성합니다.
        5. '추론문 유형', '추론 예시'만 'output example' 예제와 같이 return 합니다.
        6. output을 *절대* Markdown 형태로 출력하지 마세요. / Do not use markdown formatting under any circumstances.
        7. 추론문 유형은 '추론문 유형 정의'에 정의 되어 있는 유형으로만 답하세요. / Do not write "추론문 유형".
        
        
    **추론문 유형 정의** 
    - "원인(Cause)" : 대상 발화의 사건을 유발하는 사건,
    - "후행 사건(subsequent event)" : 대상 발화 이후에 일어날 수 있는 사건,
    - "전제 조건(prerequisite)" : 대상 발화의 사건을 가능하게 하는 상태 혹은 사건,
    - "내적 동기(motivation)" : 대상 발화를 일으키는 '화자'의 감정이나 기본 욕구,
    - "감정 반응(emotional reaction)" : 대상 발화 사건에 대해 '청자'가 보일 수 있는 감정 반응
        
    **output example**
    ex1. 원인(Cause),화자2가 친구와 영화를 보기로 한 것은 화자2가 영화가 재밌다고 생각하기 때문이다.,화자1이 화자2에게 연락을 하면 화자1과 화자2는 함께 저녁 식사를 할 것이다.,화자1이 영화를 보기 위해서는 화자1에게 내일 시간이 있어야 한다.
    ex3. 원인(Cause),화자2가 친구와 영화를 보기로 한 것은 화자2가 영화가 재밌다고 생각하기 때문이다.,화자1이 화자2에게 연락을 하면 화자1과 화자2는 함께 저녁 식사를 할 것이다.,화자1이 영화를 보기 위해서는 화자1에게 내일 시간이 있어야 한다.
    ex2. 내적 동기(motivation),화자1이 볶음밥을 먹고 싶어하는 것은 화자1이 볶음밥을 좋아하기 때문이다.,화자1이 볶음밥을 먹기로 결정한 것은 화자1의 냉장고에 볶음밥 재료가 있기 때문이다.,화자1이 볶음밥을 먹고 싶어하는 것은 화자1이 배고픔을 느끼기 때문이다.
    """
    response = model.generate_content(prompt,
                                      safety_settings = safety_settings)
    # pdb.set_trace()
    response_list = response.text.split(',')
    return response_list

def get_inferences(conversation, category, inference_1, inference_2, inference_3):
    prompt = f"""
    1. {conversation}와 {category}를 참조하여 {inference_1}, {inference_2}, {inference_3} 중 가장 가까운 결과를 output으로 리턴하세요.
    2. 이 때 {inference_1}은 'inference_1', {inference_2}는 'inference_2', {inference_3}은 'inference_3' 입니다.
    3. 대화를 보고 논리적인 이유로 판단하여 답을 결정합니다.
    4. output example을 참조하여 답변합니다.
    5. 제일 논리적인 이유 하나만을 선택하여 답변합니다.
    
    **output example**
    ex1. inference_1,'화자2가 화자1에게 저녁 7시에 만나 닭갈비를 먹으러 가자고 제안했고, 화자1이 이에 동의했기 때문입니다.'
    ex2. inference_3,'화자1은 대화 초반에 "밤에 잠을 제대로 못 잤어"라고 말하며 졸린 이유를 밤잠 부족으로 명확하게 설명합니다. 햇살 때문에 졸리다는 말은 후반부에 나오지만, 이는 잠 못 이룬 것과는 별개로 햇살에 대한 추가적인 언급일 뿐입니다. 따라서 화자1의 졸림은 밤잠 부족이라는 직접적인 이유에 더 가깝습니다.'
    """    
    
    response = model.generate_content(prompt,
                                      safety_settings = safety_settings)
    # pdb.set_trace()
    # result = json.loads(response.text)
    result = response.text.split(',')
    return result[0], result[1]
    

def create_dataset_entry(index):
    conversation = generate_conversation()
    inferences = generate_inferences(conversation)
    
    # pdb.set_trace()
    # conversation = clean_dialogue(conversation)
    inference_answer, inference_reason = get_inferences(conversation, inferences[0],
                                      inferences[1], inferences[2], inferences[3])
    
    structured_conversation = [
        {"speaker": i % 2 + 1, "utterance": utterance.strip(), "utterance_id": f"MDRW2100003410.1.{i+1}"}
        for i, utterance in enumerate(conversation.split('\n\n'))
    ]

    entry = {
        "id": f"nikluge-2024-대화 맥락 추론-train-{index:06d}",
        "input": {
            "conversation": structured_conversation,
            "reference_id": ["MDRW2100003410.1.11"],
            "category": inferences[0],
            "inference_1": inferences[1],
            "inference_2": inferences[2],
            "inference_3": inferences[3]
        },
        "output": inference_answer,
        "output_answer" : inference_reason
    }
    
    # pdb.set_trace()
    return entry
    
dataset = [create_dataset_entry(i) for i in range(1,201)]

with open('gemini(flash)_generated_dataset_100.json', 'w', encoding = 'utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)
    
end_time = time()
print(f"inference time : {end_time - start_time}")