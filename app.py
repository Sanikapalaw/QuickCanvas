import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="ImageGen Chat",
    page_icon="🎨",
    layout="centered"
)

# 2. CSS to tweak the interface (Optional: makes it look cleaner like ChatGPT)
st.markdown("""
<style>
    .stChatInput {
        position: fixed;
        bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Generative AI Chat 🎨")
st.caption("Powered by Stable Diffusion")

# 3. Load the Model (Cached to avoid reloading on every run)
@st.cache_resource
def load_pipeline():
    # Using Stable Diffusion v1.5
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Check for GPU (CUDA) or MPS (Mac) availability
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16 # Use float16 for GPU to save memory
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=dtype
    )
    pipe = pipe.to(device)
    
    # Enable memory optimization if using low VRAM
    if device == "cuda":
        pipe.enable_attention_slicing()
        
    return pipe

# Load the model with a spinner
with st.spinner("Loading AI Model... (this may take a minute)"):
    pipe = load_pipeline()

# 4. Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I can generate images for you. What would you like to see?", "type": "text"}
    ]

# 5. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["type"] == "text":
            st.markdown(message["content"])
        elif message["type"] == "image":
            st.image(message["content"], caption=message.get("caption"))

# 6. Chat Input Logic
if prompt := st.chat_input("Describe the image you want to generate..."):
    
    # Add User Message to History & Display
    st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("Thinking and generating..."):
                # Run the diffusion model
                image = pipe(prompt).images[0]
                
                # Display the Image
                st.image(image, caption=f"Generated: {prompt}")
                
                # Add Assistant Response to History
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": image, 
                    "type": "image",
                    "caption": f"Generated: {prompt}"
                })
        except Exception as e:
            st.error(f"An error occurred: {e}")
