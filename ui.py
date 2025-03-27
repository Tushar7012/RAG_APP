## Creating the Frontend
import streamlit as st
import requests

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="RAG App", page_icon="🔍",layout = "wide")

# Streamlit UI
st.title("RAG Chatbot 🤖")
st.write("Ask me anything!")

# User Input
query = st.text_input("Enter your question:")


if st.button("Ask AI") and query:
    with st.spinner("Fetching response..."):
        # Send request to FastAPI backend
        response = requests.get(API_URL, params={"question": query})
        
        if response.status_code == 200:
            data = response.json()
            
            # Display Retrieved Documents
            st.subheader("📚 Retrieved Documents")
            for i, doc in enumerate(data["retrieved_docs"], 1):
                st.markdown(f"**{i}. {doc}**")
                
            # Display AI Response
            st.subheader("🤖 AI Response")
            st.write(data["answer"])
        else:
            st.error("Failed to fetch response. Please try again!")
