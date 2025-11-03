import streamlit as st
from fitobot import get_fitobot_response

# Streamlit Page Config
st.set_page_config(page_title="Fitobot Chat", page_icon="💬", layout="centered")

st.title("💬 Fitobot — Your Workout Buddy")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display past messages
for chat in st.session_state["messages"]:
    role, content = chat["role"], chat["content"]
    if role == "user":
        st.markdown(f"**🧍‍♂️ You:** {content}")
    else:
        st.markdown(f"**🤖 Fitobot:** {content}")

# Chat input box
user_input = st.chat_input("Ask me anything about fitness...")

if user_input:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.markdown(f"**🧍‍♂️ You:** {user_input}")

    # Get bot response
    bot_response = get_fitobot_response(user_input)

    # Add and display bot response
    st.session_state["messages"].append({"role": "bot", "content": bot_response})
    st.markdown(f"**🤖 Fitobot:** {bot_response}")
