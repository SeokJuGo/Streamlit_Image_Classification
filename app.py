import streamlit as st
import torch
from PIL import Image
import numpy as np
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False) # deprecation 표시 안함 
st.sidebar.text("슬라이드바 테스트 입니다.")
# 타이틀
st.title("잎사귀 다중분류")

# 아무 이미지 출력
image2 = Image.open('./Test_1.jpg')
st.image(image2, caption='multiple_disease(1)',use_column_width=True)

# csv파일 훑어보기
train = pd.read_csv('./train.csv')
st.write(train.head())
test = pd.read_csv('./test.csv')
st.write(test.head())


# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
# 모델 로드
model = torch.load("model.pt", map_location=device)
st.write(model)

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
size = (224, 224)

image2 = image2.resize((size))


image_array = np.asarray(image2)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
st.write(prediction)