# 🍃식물 건강 테스트🍃
> 2023년 1월 30일

학습된 모델이 식물 이미지를 입력받고, 식물이 어떤 상태인지 예측합니다.  
이것으로 식물의 건강 상태를 확인할 수 있습니다!

<div align='center'>
  <img src="https://user-images.githubusercontent.com/106129152/215032337-e3d9c5d6-2b2c-47ff-9076-96b388df22b6.png" width="750">
</div>
<br/>

**Link** : https://plant-diseases-check.streamlit.app/
## 팀원 💁
- 고석주
- 이재영
- 전현준
- 김건영



## 목차 :bookmark_tabs:

- [개요](#개요)
- [모델](#모델)
- [구동](#구동)
- [TEST2](#TEST2)


## 개요


식물의 상태는 총 4가지로 분류됩니다.
- 건강함
- 여러 질병에 걸림
- 녹병균
* 녹병균이란? 양치식물이나 종자식물에 기생하여 녹병을 일으키는데, 
  녹병이 발생된 곳은 세포 분열이나 가지치기가 촉진되어 녹과 같은 모양이 되거나 혹 모양의 기형이 되기도 한다.
- 붉은 곰팡이병
* 붉은 곰팡이병이란? 밀·보리에 감염되는 병해. 적미병(赤微病)이라고도 한다. 
이 병은 온난다습한 남부지방에서 많이 발생하며, 병든 종자가 사료에 혼합되면 가축이 중독(中毒)을 일으키고, 
사람이 먹었을 경우에는 구토를 유발시키기도 한다.


## 시연영상
![plant](https://user-images.githubusercontent.com/116260619/215047900-0cb0e739-7d99-4242-b897-1eac00f3d3cf.gif)


## 코드

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from io import StringIO
import time
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    
    .block-container.css-91z34k.egzxvld4 {{
        background-color: rgb(255, 255, 255, 0.7);
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('image.jpg')      

result=[]
images=[]
st.title('*식물 건강 테스트*')
st.write('🌿이 식물은 건강할까? 지금 바로 확인해봅시다!')

st.sidebar.subheader("File upload")
file_up = st.sidebar.file_uploader("식물 사진을 업로드해주세요.", type=['jpeg', 'png', 'jpg', 'webp'])
import streamlit as st
import time



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
        images.append(image)
        st.image(image, caption = 'Uploaded Image.', use_column_width = True)
        st.write("")
        my_bar = st.progress(0)

        with st.spinner('Wait for it...'):
            time.sleep(3)



        labels = predict(file_up)
        print(labels)
        # result.append[labels]
        for i in labels[0][0]:
                print(i)
        for i in labels[0][0]:
                if '1' in i:
                        st.write("***결과 : 건강합니다!***")
                        st.write("당신의 노력이 식물을 건강하게 키워냈습니다!")
                if '2' == i:
                        st.write("***결과 : 여러 질병에 걸린 상태입니다!***")
                        st.write("식물도 인간처럼 질병에 걸린답니다.")
                        st.write("농업기술포털의 식물병에 대한 [정보](https://www.nongsaro.go.kr/portal/ps/psv/psvr/psvrc/rdaInterDtl.ps?&menuId=PS00063&cntntsNo=208859) 를 참고해보세요.")
                if '3' == i:
                        st.write("***결과 : 녹병균에 걸린 상태입니다!***")
                        st.write("녹병균은 종류도, 그 예방법도 정말 다양하답니다.")
                        st.write("산림청의 [부분별진단검색](https://www.forest.go.kr/kfsweb/mer/fip/search/selectSrchPartDgnosisList.do?dbhisTpcd=10&dbhisPartActoDgnssCd=01&mn=NKFS_02_02_02_02_06) 을 통해 더 정확한 정보를 찾아보세요.")
                if '4' == i:
                        st.write("***결과 : 붉은 곰팡이병에 걸린 상태입니다!***")
                        st.write("농업기술포털의 [붉은 곰팡이병](https://www.nongsaro.go.kr/portal/ps/pss/pssa/sicknsSearchDtl.ps?pageIndex=1&pageSize=10&&sicknsCode=D00000753&menuId=PS00202) 에 대한 문서를 참고해보세요.")

        st.write("Accuracy!")

        for percent_complete in range(int(labels[0][1])):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
        st.write(str(int(labels[0][1]))+"%") 
