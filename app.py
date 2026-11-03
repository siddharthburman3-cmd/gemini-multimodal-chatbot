import streamlit as st
from PIL import Image
from gemini_bot import get_gemini_response

st.set_page_config(page_title="Gemini Multimodal Chatbot", page_icon="🤖", layout="wide")
st.title("💬 Gemini Multimodal Chatbot")
st.caption("Ask questions using text or upload an image and ask Gemini about it.")

with st.sidebar:
    st.header("Chat controls")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.write("**Model:** `gemini-3.6-flash`")
    st.write("**Modes:** Text + Image")
    st.info("Your API key is read from the project's env file. Never share it.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("image") is not None:
            st.image(message["image"], caption="Uploaded image", width=320)
        st.markdown(message["content"])

uploaded_file = st.file_uploader(
    "Optional: upload an image",
    type=["png", "jpg", "jpeg", "webp"],
    help="Upload an image, then type a question below."
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image ready for Gemini", width=360)
else:
    image = None

prompt = st.chat_input("Type your message or ask a question about the image...")

if prompt:
    user_message = {"role": "user", "content": prompt, "image": image}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)
        if image is not None:
            st.image(image, width=320)
            
    with st.chat_message("assistant"):
        with st.spinner("Gemini is thinking..."):
            try:
                history = [
                    {"role": item["role"], "content": item["content"]}
                    for item in st.session_state.messages[:-1]
                ]
                answer = get_gemini_response(prompt, image, history)
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "image": None}
                )
            except Exception as error:
                st.error(f"Something went wrong: {error}")