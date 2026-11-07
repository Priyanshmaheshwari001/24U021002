import streamlit as st
import joblib
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Dual-Model Humour Detection System", page_icon="🤖", layout="centered")

st.title("🎭 Dual-Model Humour Detection & Reasoning System")
st.write("Compare the blazing-fast speed of an **Offline ML Model** with the deep reasoning of a **Cloud API CoT Model**!")

# --- LOAD OFFLINE MODEL & VECTORIZER ---
@st.cache_resource
def load_offline_models():
    model = joblib.load("offline_humour_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

offline_model, vectorizer = load_offline_models()

# --- SIDEBAR: MODEL SELECTION ---
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox(
    "Choose Architecture",
    ["Offline ML Model (Fast & Local)", "Cloud API CoT Model (Smart Reasoning)"]
)

# API Key configuration for Gemini (if using cloud model)
api_key = None
if model_choice == "Cloud API CoT Model (Smart Reasoning)":
    default_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.sidebar.text_input("Enter Gemini API Key", value=default_key, type="password")
    if api_key:
        genai.configure(api_key=api_key)

# --- MAIN INPUT SECTION ---
user_text = st.text_area("Enter a sentence, joke, or statement to analyze:", placeholder="Type something here...")

if st.button("Detect Humour 🚀"):
    if not user_text.strip():
        st.warning("Please enter some text first!")
    else:
        if model_choice == "Offline ML Model (Fast & Local)":
            # --- MODEL 1: OFFLINE LOGIC ---
            start_time = time.time()
            
            # Transform text and predict
            transformed_input = vectorizer.transform([user_text])
            prediction = offline_model.predict(transformed_input)[0]
            
            # Get probability/confidence if available
            probabilities = offline_model.predict_proba(transformed_input)[0]
            confidence = round(max(probabilities) * 100, 2)
            
            end_time = time.time()
            latency = round(end_time - start_time, 4)
            
            st.markdown("---")
            st.subheader("📊 Offline ML Results:")
            if prediction == 1:
                st.success(f"🎉 **Result: HUMOROUS (Label: 1)**")
            else:
                st.info(f"ℹ️ **Result: NOT HUMOROUS (Label: 0)**")
                
            st.write(f"🔍 **Confidence Score:** {confidence}%")
            st.caption(f"⚡ Execution Latency: {latency}s | Mode: Local TF-IDF + Logistic Regression")
            
        else:
            # --- MODEL 2: CLOUD API CoT LOGIC ---
            if not api_key:
                st.error("Please enter your Gemini API Key in the sidebar first!")
            else:
                with st.spinner("Generating Chain-of-Thought reasoning via Gemini API..."):
                    try:
                        generation_config = {"temperature": 0.3}
                        env_model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-lite")
                        model = genai.GenerativeModel(env_model_name, generation_config=generation_config)
                        
                        prompt = f"""
                        Analyze the following text step-by-step to determine if it is humorous or not.
                        Text: "{user_text}"
                        
                        Provide your response in this format:
                        1. Step-by-step reasoning (Chain-of-Thought).
                        2. Final Classification: [HUMOROUS] or [NOT HUMOROUS].
                        """
                        
                        response = model.generate_content(prompt)
                        
                        st.markdown("---")
                        st.subheader("🧠 Cloud API CoT Results:")
                        st.markdown(response.text)
                        st.caption(f"☁️ Mode: Cloud Generative AI ({env_model_name}) with Step-by-Step Reasoning")
                        
                    except Exception as e:
                        st.error(f"An error occurred with the API: {e}")