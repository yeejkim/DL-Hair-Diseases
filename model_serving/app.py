import os
import numpy as np
import cv2
import gradio as gr
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import time
import google.generativeai as genai

# Google Gemini API 설정
GOOGLE_API_KEY = "AIzaSyB55b9Da-I11QNiHfjXyvpSzX-xzC8BSmM"
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # Gemini 모델 인스턴스

# 질병 리스트 (영어, 한글)
disease_list = [
    ('Alopecia Areata', '원형 탈모증'),
    ('Contact Dermatitis', '접촉 피부염'),
    ('Folliculitis', '모낭염'),
    ('Head Lice', '머리 이'),
    ('Lichen Planus', '편평 태선'),
    ('Male Pattern Baldness', '남성형 탈모'),
    ('Psoriasis', '건선'),
    ('Seborrheic Dermatitis', '지루성 피부염'),
    ('Telogen Effluvium', '탈모 증상'),
    ('Tinea Capitis', '두피 백선')
]

# 모델 정의
class MultiBinaryClassificationModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MultiBinaryClassificationModel, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.classifier = nn.Linear(1000, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classification_model = MultiBinaryClassificationModel()  # 모델 인스턴스 생성
classification_model.load_state_dict(torch.load('C:/Users/a0707/PycharmProjects/DL/multi_best_model_state_dict.pth', map_location=device))
classification_model.eval()  # 평가 모드로 전환

# 이미지 크기
IMG_SIZE = 256  # 모델에 맞는 크기로 설정

# 이미지 전처리 함수
def preprocess_image(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img)
    median_filtered_image = cv2.medianBlur(img_np, 3)
    norm_img = np.asarray(np.float32(median_filtered_image)) / 255.0
    norm_img = norm_img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)로 변환
    return norm_img[np.newaxis, ...]  # 배치 차원 추가

# 예측 함수
def predict(image):
    if image is None:
        return "No image provided."

    try:
        image_tensor = preprocess_image(image)  # 이미지 전처리
        image_tensor = torch.tensor(image_tensor).to(device)  # 텐서로 변환 및 GPU로 이동
        with torch.no_grad():
            outputs = classification_model(image_tensor)  # 모델 예측
        predictions = (outputs > 0.5).cpu().numpy().flatten()  # 이진 예측

        # Positive인 질병만 필터링하여 결과 생성
        positive_diseases = [f"{disease[0]} ({disease[1]})" for disease, pred in zip(disease_list, predictions) if pred]

        # Gemini API 호출
        if positive_diseases:
            user_prompt = f"두피 질환 중 {', '.join(positive_diseases)}에 대한 치료법을 간단하게 한국어로 알려주세요."
            start_time = time.time()
            response = gemini_model.generate_content(  # 올바른 객체에서 호출
                user_prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    stop_sequences=['x'],
                    temperature=1.0)
            )
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"실행 시간: {execution_time:.2f} 초")

            # 예측 결과와 Gemini API 응답 결합하여 반환
            return "\n".join(positive_diseases) + "\n\n" + response.text
        else:
            return "지금처럼 건강한 습관 유지하세요!"  # 건강한 경우 메시지 반환

    except Exception as e:
        return f"Error: {str(e)}"  # 오류 메시지 반환

import gradio as gr

# CSS 스타일 정의
css = """
#header {
    text-align: center;
    margin-bottom: 20px;
    color: #007BFF; /* 헤더 텍스트 색상 */
}

#upload-header, #result-header {
    font-weight: bold;
    font-size: 18px; /* 헤더 글자 크기 */
    margin-bottom: 10px;
}

#image-input {
    border: 2px dashed #007BFF; /* 이미지 업로드 박스 스타일 */
    border-radius: 10px;
    padding: 10px;
}

#predict-button {
    background-color: #007BFF; /* 버튼 색상 */
    color: white; /* 버튼 글자 색상 */
    border: none; /* 버튼 테두리 없음 */
    border-radius: 5px; /* 버튼 둥글기 */
    padding: 10px 20px; /* 버튼 패딩 */
    font-size: 16px; /* 버튼 글자 크기 */
    cursor: pointer; /* 커서 포인터 */
}

#predict-button:hover {
    background-color: #0056b3; /* 버튼 호버 색상 */
}

#footer {
    text-align: center;
    margin-top: 20px;
    font-size: 12px; /* 푸터 글자 크기 */
    color: gray;
}

#result-output {
    font-size: 20px; /* 진단 결과 기본 글자 크기 */
    font-weight: bold; /* 진단 결과 두껍게 */
}

#disease-name {
    font-size: 24px; /* 질병명 글자 크기 증가 */
    color: #0056b3; /* 질병명 색상 */
    font-weight: bold; /* 질병명 두껍게 */
    margin-bottom: 10px; /* 질병명 아래 여백 */
}
"""

iface = gr.Blocks()

with iface:
    # CSS 추가
    gr.Markdown(f"<style>{css}</style>")

    # 헤더
    with gr.Row():
        gr.Markdown(
            """
            # 두피 질환 진단 모델
            이미지를 업로드하여 두피 질환을 진단하세요.  
            예측 결과와 간단한 치료법을 확인할 수 있습니다.
            """,
            elem_id="header"
        )

    # 입력 영역
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 이미지를 업로드하세요:", elem_id="upload-header")
            image_input = gr.Image(type="pil", label="두피 이미지 업로드", elem_id="image-input")

    # 출력 영역
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 진단 결과:", elem_id="result-header")
            # 질병명과 결과를 별도로 표시하기 위한 마크다운 추가
            disease_name = gr.Markdown(label="**Telogen Effluvium (탈모 증상)**", elem_id="disease-name")
            markdown_output = gr.Markdown(label="탈모의 원인이 되는 기저 질환을 치료하는 것이 가장 중요합니다. 스트레스, 영양결핍, 출산 등 원인이 되는 요소를 파악하고 개선하는 것이 핵심입니다. 특별한 치료법은 없지만, 원인을 제거하면 대부분 자연적으로 회복됩니다.\n\n다만, 심각하거나 오래 지속될 경우, 의사의 진찰을 받아 철분제, 비타민 등의 영양제 처방이나, 필요에 따라 탈모 치료제를 고려할 수 있습니다. 자가 치료보다는 전문의의 진료를 받는 것이 좋습니다.", elem_id="result-output")

    # 버튼 영역
    with gr.Row():
        predict_button = gr.Button("진단 시작", elem_id="predict-button")

    # 이벤트 연결
    def format_markdown_output(prediction_text):
        """텍스트 형식으로 전체 결과 반환"""
        return prediction_text

    predict_button.click(
        lambda image: format_markdown_output(predict(image)),
        inputs=image_input,
        outputs=markdown_output
    )

    # 푸터
    with gr.Row():
        gr.Markdown(
            """
            **주의**: 이 모델은 참고용으로만 사용되며, 정확한 진단과 치료는 반드시 의료 전문가와 상담하세요. 
            """,
            elem_id="footer"
        )

# Gradio 앱 실행
iface.launch()
