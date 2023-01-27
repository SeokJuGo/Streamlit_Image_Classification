import streamlit as st
import torch
from PIL import Image

st.set_option('deprecation.showfileUploaderEncoding', False) # deprecation 표시 안함 
st.title("머신러닝 이용 뇌종양 MRI 사진 판독 서비스")
st.markdown("""
뇌종양 MRI 사진을 분류합니다. 
이 서비스는 의사들의 뇌 MRI 사진 판독의 도우미일 뿐입니다. 
정확한 최종 진단 결과는 반드시 전문 담당 의사의 확인과 승인을 거치십시요.""")

image2 = Image.open('./Test_1.jpg')
st.image(image2, caption='질병식물',use_column_width=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("model.pt", map_location=device)
st.write(model)
image = Image.open('test_photo.jpg')