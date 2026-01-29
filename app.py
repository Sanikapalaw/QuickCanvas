import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image

st.set_page_config(page_title="AI Image Chat", layout="centered")
st.title("🧠 AI Image Chat")
st.caption("Fast & Free Image Generator (SD Turbo)")

@st.cache_resource
def load_model():
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float32,   # CPU safe
        use_safetensors=True
    )
    pipe = pipe.to("cpu")           # Force CPU
    return pipe

pipe = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.image(msg["image"], use_column_width=True)
        else:
            st.markdown(msg["content"])

prompt = st.chat_input("Describe your image...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Generating image..."):
            image = pipe(
                prompt,
                num_inference_steps=4,
                guidance_scale=0.0
            ).images[0]

        st.image(image, use_column_width=True)
        st.session_state.messages.append({"role": "assistant", "image": image})
