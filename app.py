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
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('image.png')    

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
