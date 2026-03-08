import streamlit as st
from predict import predict

st.set_page_config(page_title="SMS Spam Detector", page_icon="🛡️")

st.title("🛡️ SMS Spam Detector")
st.write("Type any SMS message below to check if it's spam or ham.")

message = st.text_area("Enter your message:", height=150)

if st.button("Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        label, confidence = predict(message)
        if label == "spam":
            st.error(f"🚨 SPAM — {confidence}% confident")
        else:
            st.success(f"✅ HAM (Not Spam) — {confidence}% confident")