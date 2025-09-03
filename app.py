import streamlit as st
import model
import psutil, os, sys

# === Website Setup ===

st.set_page_config(page_title="Anime Face Generator", layout="wide")

st.markdown("<h1 style='text-align: center;'>Anime Face Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center'>Click the button to generate random anime faces!</p>", unsafe_allow_html=True)

# === Generator Setup ===

@st.cache_resource
def load_generator():
    generator = model.ImageGenerator()
    return generator

generator = load_generator()

# Initial image when the website loads
if "img" not in st.session_state:
    with st.spinner("Generating face..."):
        st.session_state.img = generator.generate()

# ==== UI Setup ===

# To centralize the button
col1, col2, col3 = st.columns([5, 2, 5])
with col2:
    if st.button("Generate image", use_container_width=True):
        
        # Generate new image
        with st.spinner("Generating face..."):
            st.session_state.img = generator.generate()

# === Update image ===

# Show the image using html to make the image cenetred
st.markdown(
    f"""<div style="text-align: center;"><img src="data:image/jpeg;base64,{st.session_state.img}" style="border-radius: 5%; border: 4px solid #888;" width="256"></div>""",
    unsafe_allow_html=True
)

# For extra gap
for _ in range(2):
    st.write("")

# === Footer ===

st.markdown("<p style='text-align: center'>By <b>Mevan</b></p>", unsafe_allow_html=True)