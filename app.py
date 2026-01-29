import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image

st.set_page_config(page_title="QuickCanvas", layout="centered")
st.title("🎨 QuickCanvas")
st.caption("Fast AI Image Generator (SD Turbo)")

@st.cache_resource
def load_model():
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float32
    )
    return pipe

pipe = load_model()

prompt = st.chat_input("Describe your image...")

if prompt:
    with st.spinner("Generating..."):
        img = pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]
    st.image(img, use_column_width=True)
