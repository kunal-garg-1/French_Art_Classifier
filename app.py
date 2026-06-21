import streamlit as st
from fastai.vision.all import *
import pathlib
import platform
import torch
import torchvision.transforms as T
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="French Art AI", page_icon="🎨", layout="wide")

# --- Windows Path Hack ---
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

# --- Load the Model ---
@st.cache_resource
def load_model():
    return load_learner('french_art_model.pkl')

learn = load_model()
vocab = learn.dls.vocab  # Extract the painter names

# --- SIDEBAR DESIGN ---
st.sidebar.title("🖼️ The AI Curator")
st.sidebar.write("This neural network was trained to differentiate between the subtle brushstrokes of four French Masters:")
st.sidebar.markdown("- **Claude Monet**\n- **Pierre-Auguste Renoir**\n- **Edgar Degas**\n- **Paul Cézanne**")
st.sidebar.divider()
st.sidebar.caption("Engineered from scratch using PyTorch & fast.ai.")

# --- MAIN DASHBOARD UI ---
st.title("🎨 French Impressionist Classifier")
st.markdown("Upload a painting, and the deep learning model will analyze the image to determine the master behind the brushstrokes.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Submit Artwork")
    uploaded_file = st.file_uploader("Drop a JPG or PNG here...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with col1:
        st.image(uploaded_file, caption="Your Uploaded Artwork", use_container_width=True)
    
    with col2:
        st.subheader("2. AI Analysis")
        
        # --- THE ULTIMATE BYPASS (PURE PYTORCH) ---
        # fast.ai's predict pipeline has a known crash in Python 3.13.
        # So, we bypass fast.ai entirely and extract the raw PyTorch model to do the math ourselves!
        
        with st.spinner("Analyzing brushstrokes and color palettes..."):
            # 1. Load image and force RGB
            img = Image.open(uploaded_file).convert('RGB')
            
            # 2. Recreate fast.ai's exact data pipeline using pure PyTorch/Torchvision
            transform = T.Compose([
                T.Resize(128),
                T.CenterCrop(128),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Apply transforms and add the batch dimension [1, 3, 128, 128]
            img_tensor = transform(img).unsqueeze(0)
            
            # 3. Extract the raw PyTorch model and set it to Evaluation mode
            model = learn.model
            model.eval()
            
            # 4. Feed the tensor directly into the neural network!
            with torch.no_grad():
                output = model(img_tensor)
                # Convert the raw logit outputs into percentage probabilities
                probs = torch.nn.functional.softmax(output[0], dim=0)
            
            # 5. Extract the winning prediction
            pred_idx = torch.argmax(probs).item()
            pred = vocab[pred_idx]
            confidence = probs[pred_idx].item()
        
        # Clean up the output text
        clean_name = pred.replace(" painting", "")
        
        # Display the main verdict
        st.success(f"**The Verdict:** This looks like a **{clean_name}**!")
        
        # Display confidence metric
        st.metric(label="Match Confidence", value=f"{confidence * 100:.2f}%")
        st.progress(confidence)
        
        # Let the user see the exact math
        with st.expander("See all raw calculations"):
            for i, painter_name in enumerate(vocab):
                st.write(f"- {painter_name}: {probs[i].item() * 100:.2f}%")