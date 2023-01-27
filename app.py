import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from io import StringIO

st.title('*식물 건강 테스트*')
st.write('🌿이 식물은 건강할까? 지금 바로 확인해봅시다!')

st.sidebar.subheader("File upload")
file_up = st.sidebar.file_uploader("식물 사진을 업로드해주세요.", type=['jpeg', 'png', 'jpg', 'webp'])

def predict(image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load("model.pt", map_location=device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
        )])     

    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)
    model.eval()
    out = model(batch_t)

    with open('labels.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0]*100
    _, indices = torch.sort(out, descending = True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:1]]

if file_up is not None:
        image = Image.open(file_up)
        st.image(image, caption = 'Uploaded Image.', use_column_width = True)
        st.write("")
        with st.spinner('Wait for it...'):
            time.sleep(3)
            st.success('Done!')
        labels = predict(file_up)
        st.write(labels)
        print(labels)
        for i in labels[0][0]:
                if '1' in i:
                        st.write("***결과 : 건강합니다!***")
                if '2' == i:
                        st.write("***결과 : 여러 질병에 걸린 상태입니다!***")
                if '3' == i:
                        st.write("***결과 : 녹병균에 걸린 상태입니다!***")
                if '4' == i:
                        st.write("***결과 : 붉은 곰팡이병에 걸린 상태입니다!***")
