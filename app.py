import streamlit as st
from rag import research_assistant
from dotenv import load_dotenv

load_dotenv()

# Configure page
st.set_page_config(
    page_title="Financial Research Assistant",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Financial Research Assistant")
st.markdown("Ask questions about the provided financial documents (Hybrid Search + Reranking + LLM).")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is the revenue growth this quarter?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Searching and analyzing documents..."):
            try:
                response = research_assistant(prompt)
                st.markdown(response)
            except Exception as e:
                response = f"**Error:** {str(e)}"
                st.error(response)
                
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
