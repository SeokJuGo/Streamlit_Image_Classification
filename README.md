# ๐์๋ฌผ ๊ฑด๊ฐ ํ์คํธ๐
> ๐๏ธ 23.01.27(๊ธ) ~ 23.01.30(์)

ํ์ต๋ ๋ชจ๋ธ์ด ์๋ฌผ ์ด๋ฏธ์ง๋ฅผ ์๋ ฅ๋ฐ๊ณ , ์๋ฌผ์ด ์ด๋ค ์ํ์ธ์ง ์์ธกํฉ๋๋ค.  
์ด๊ฒ์ผ๋ก ์๋ฌผ์ ๊ฑด๊ฐ ์ํ๋ฅผ ํ์ธํ  ์ ์์ต๋๋ค!

<div align='center'>
  <img src="https://user-images.githubusercontent.com/106129152/215032337-e3d9c5d6-2b2c-47ff-9076-96b388df22b6.png" width="750">
</div>
<br/>

**Link** : https://plant-diseases-check.streamlit.app/
## ํ์ ๐
- ๊ณ ์์ฃผ
- ์ด์ฌ์
- ์ ํ์ค
- ๊น๊ฑด์



## ๋ชฉ์ฐจ :bookmark_tabs:

- [๊ฐ์](#๊ฐ์)
- [๊ตฌ๋](#๊ตฌ๋)
- [TEST2](#TEST2)




## ๊ฐ์


์๋ฌผ์ ์ํ๋ ์ด 4๊ฐ์ง๋ก ๋ถ๋ฅ๋ฉ๋๋ค.
- ๊ฑด๊ฐํจ
- ์ฌ๋ฌ ์ง๋ณ์ ๊ฑธ๋ฆผ
- ๋น๋ณ๊ท 
* ๋น๋ณ๊ท ์ด๋? ์์น์๋ฌผ์ด๋ ์ข์์๋ฌผ์ ๊ธฐ์ํ์ฌ ๋น๋ณ์ ์ผ์ผํค๋๋ฐ, 
  ๋น๋ณ์ด ๋ฐ์๋ ๊ณณ์ ์ธํฌ ๋ถ์ด์ด๋ ๊ฐ์ง์น๊ธฐ๊ฐ ์ด์ง๋์ด ๋น๊ณผ ๊ฐ์ ๋ชจ์์ด ๋๊ฑฐ๋ ํน ๋ชจ์์ ๊ธฐํ์ด ๋๊ธฐ๋ ํ๋ค.
- ๋ถ์ ๊ณฐํก์ด๋ณ
* ๋ถ์ ๊ณฐํก์ด๋ณ์ด๋? ๋ฐยท๋ณด๋ฆฌ์ ๊ฐ์ผ๋๋ ๋ณํด. ์ ๋ฏธ๋ณ(่ตคๅพฎ็)์ด๋ผ๊ณ ๋ ํ๋ค. 
์ด ๋ณ์ ์จ๋๋ค์ตํ ๋จ๋ถ์ง๋ฐฉ์์ ๋ง์ด ๋ฐ์ํ๋ฉฐ, ๋ณ๋  ์ข์๊ฐ ์ฌ๋ฃ์ ํผํฉ๋๋ฉด ๊ฐ์ถ์ด ์ค๋(ไธญๆฏ)์ ์ผ์ผํค๊ณ , 
์ฌ๋์ด ๋จน์์ ๊ฒฝ์ฐ์๋ ๊ตฌํ ๋ฅผ ์ ๋ฐ์ํค๊ธฐ๋ ํ๋ค.


## ์์ฐ์์
![plant](https://user-images.githubusercontent.com/116260619/215047900-0cb0e739-7d99-4242-b897-1eac00f3d3cf.gif)

## Trouble Shooting
- ๋ชจ๋ธ ํ์ฅ์

- ํ์ดํ ์น ๋ผ์ด๋ธ๋ฌ๋ฆฌ ํจ์ predict

## ์ฝ๋

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
st.title('*์๋ฌผ ๊ฑด๊ฐ ํ์คํธ*')
st.write('๐ฟ์ด ์๋ฌผ์ ๊ฑด๊ฐํ ๊น? ์ง๊ธ ๋ฐ๋ก ํ์ธํด๋ด์๋ค!')

st.sidebar.subheader("File upload")
file_up = st.sidebar.file_uploader("์๋ฌผ ์ฌ์ง์ ์๋ก๋ํด์ฃผ์ธ์.", type=['jpeg', 'png', 'jpg', 'webp'])
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
                        st.write("***๊ฒฐ๊ณผ : ๊ฑด๊ฐํฉ๋๋ค!***")
                        st.write("๋น์ ์ ๋ธ๋ ฅ์ด ์๋ฌผ์ ๊ฑด๊ฐํ๊ฒ ํค์๋์ต๋๋ค!")
                if '2' == i:
                        st.write("***๊ฒฐ๊ณผ : ์ฌ๋ฌ ์ง๋ณ์ ๊ฑธ๋ฆฐ ์ํ์๋๋ค!***")
                        st.write("์๋ฌผ๋ ์ธ๊ฐ์ฒ๋ผ ์ง๋ณ์ ๊ฑธ๋ฆฐ๋ต๋๋ค.")
                        st.write("๋์๊ธฐ์ ํฌํธ์ ์๋ฌผ๋ณ์ ๋ํ [์ ๋ณด](https://www.nongsaro.go.kr/portal/ps/psv/psvr/psvrc/rdaInterDtl.ps?&menuId=PS00063&cntntsNo=208859) ๋ฅผ ์ฐธ๊ณ ํด๋ณด์ธ์.")
                if '3' == i:
                        st.write("***๊ฒฐ๊ณผ : ๋น๋ณ๊ท ์ ๊ฑธ๋ฆฐ ์ํ์๋๋ค!***")
                        st.write("๋น๋ณ๊ท ์ ์ข๋ฅ๋, ๊ทธ ์๋ฐฉ๋ฒ๋ ์ ๋ง ๋ค์ํ๋ต๋๋ค.")
                        st.write("์ฐ๋ฆผ์ฒญ์ [๋ถ๋ถ๋ณ์ง๋จ๊ฒ์](https://www.forest.go.kr/kfsweb/mer/fip/search/selectSrchPartDgnosisList.do?dbhisTpcd=10&dbhisPartActoDgnssCd=01&mn=NKFS_02_02_02_02_06) ์ ํตํด ๋ ์ ํํ ์ ๋ณด๋ฅผ ์ฐพ์๋ณด์ธ์.")
                if '4' == i:
                        st.write("***๊ฒฐ๊ณผ : ๋ถ์ ๊ณฐํก์ด๋ณ์ ๊ฑธ๋ฆฐ ์ํ์๋๋ค!***")
                        st.write("๋์๊ธฐ์ ํฌํธ์ [๋ถ์ ๊ณฐํก์ด๋ณ](https://www.nongsaro.go.kr/portal/ps/pss/pssa/sicknsSearchDtl.ps?pageIndex=1&pageSize=10&&sicknsCode=D00000753&menuId=PS00202) ์ ๋ํ ๋ฌธ์๋ฅผ ์ฐธ๊ณ ํด๋ณด์ธ์.")

        st.write("Accuracy!")

        for percent_complete in range(int(labels[0][1])):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
        st.write(str(int(labels[0][1]))+"%") 



## ์ถ๊ฐํ๊ณ  ์ถ์ ๊ธฐ๋ฅ
- ํ์ผ ์๋ก๋๋ฅผ ํ๊บผ๋ฒ์ ํด์ ์์ธกํ๊ธฐ
- 
- ์์ธกํ ๊ฒฐ๊ณผ๋ค๊ณผ ์ด๋ฏธ์ง๋ค์ ๋์ ํด์ ํ์ผ๋ก ๋ด๋ณด๋ด๊ธฐ
