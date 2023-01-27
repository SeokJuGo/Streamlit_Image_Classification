import streamlit as st
import torch
from PIL import Image

st.set_option('deprecation.showfileUploaderEncoding', False) # deprecation 표시 안함 
st.title("잎사귀 다중분류")
image2 = Image.open('./Test_1.jpg')
st.image(image2, caption='질병식물',use_column_width=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("model.pt", map_location=device)
st.write(model)
