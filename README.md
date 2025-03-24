# 💇‍♀️ ResNet50 기반 두피 질환 분류 모델

이 Repository는 2024학년도 2학기 **'딥러닝 기반 데이터 분석'** 수업에서 진행한 **기말 과제**를 기록한 공간입니다.  
ResNet50을 활용하여 **10가지 두피 질환을 다중 라벨 분류**하는 모델을 개발하였습니다. 🚀  

> **🔍 주요 내용**
> - 📌 **10가지 두피 질환 데이터셋 학습**
> - 🛠️ **ResNet50 기반 분류 모델 설계**
> - 🤖 **데이터 전처리 및 Augmentation**
> - 🎯 **98.83% 정확도로 두피 질환 분류**
> - 🌐 **Gradio & Gemini API 활용한 모델 서빙**  

📄 **[👩‍🏫 발표자료 보러 가기‼️](./[딥러닝]발표자료_9_김예진.pdf)**  

---

## 🛠️ Architecture  
<p align="center">
  <img alt="아키텍처" src="https://github.com/user-attachments/assets/1f66371d-55f5-4365-a079-96ca32e858f2"/>
</p>

---

## 🧬 Dataset
- 📂 **데이터 출처**: [Hair Diseases Dataset (Kaggle)](https://www.kaggle.com/datasets/sundarannamalai/hair-diseases/data)  
- 🏷️ **총 10가지 두피 질환 카테고리**
- 🖼️ **총 12,000개의 이미지**
- 📊 **데이터 분할 비율**
  ```bash
  Train : Validation : Test = 8 : 1 : 1

## 🤖 Data Preprocessing 
1️⃣ 크기 조정 (Resize)
- 모든 이미지를 256x256 크기로 조정 <br>

2️⃣ 노이즈 제거 (Median Filtering)
- cv2.medianBlur를 사용하여 노이즈 제거 <br>

3️⃣ 정규화 (Normalization)
- 픽셀 값을 [0, 1] 사이로 변환 <br>

4️⃣ 채널 변환 (Channel Reordering)
- OpenCV에서 불러온 이미지를 (H, W, C) → (C, H, W) 로 변경 (PyTorch 호환) <br>

5️⃣ 라벨 처리 (One-Hot Encoding) <br>
- 범주형 데이터를 이진 벡터로 변환하는 방법
- 다중 질환 판단 가능하도록 다중 라벨 적용
  - 해당 클래스의 인덱스만 1로 설정, 나머지는 0으로 설정

- 최종적으로, **각 이미지에 대한 두피 질환 라벨을 독립적인 이진 벡터로 변환**
- 전처리된 이미지 경로와 라벨을 포함하여 csv 파일 형태로 저장

<br>

## 🌳 Modeling
### 🎯 Resnet50 기반 다중 이진 분류 모델 

🔍 왜 ResNet50을 선택했을까?
- ResNet50은 Residual Connection을 활용하여 깊은 신경망 학습 시 기울기 소실 문제를 해결
- 50개의 계층을 거쳐 이미지 특징을 효과적으로 추출
- 다양한 Computer Vision 태스크에서 우수한 성능을 보장
- **전이 학습(Transfer Learning)**을 활용해 빠른 학습 가능
- 이러한 특징을 가진 Resnet50을 기반으로 하여 **MultiBinaryClassificationModel** 설계


💡 모델 구조
- ResNet50의 Feature Extractor를 활용하여 Feature Map 추출
- Fully Connected Layer를 추가하여 출력 차원을 10개 클래스에 맞게 조정
- 최종적으로 각 이미지에 대해 다중 질환 판단이 가능하도록 설계



<br>

## 🧐 Evaluation
- Test Dataset을 통해 평가하고, True label과 비교한 결과, **Accuracy 98.83%** 확인

<br>

## ⭐ Model Serving 
- Gradio 사용
- 모델의 결과를 Gemini에 넣어 진단법과 함께 진단 결과 확인 가능

<img alt="시연" src="https://github.com/user-attachments/assets/1f2a4090-b199-4705-a574-427d11b8740d" />
