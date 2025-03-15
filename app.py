import streamlit as st
from query_engine import Chatbot
from document_processor import DocumentProcessor
import time
import google.generativeai as genai

# Set page configuration
st.set_page_config(
    page_title="Knowledge Graph Chatbot",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Load secrets from Streamlit
NEO4J_URI = st.secrets["neo4j"]["uri"]
NEO4J_USER = st.secrets["neo4j"]["user"]
NEO4J_PASSWORD = st.secrets["neo4j"]["password"]

# ✅ Initialize chatbot & document processor
@st.cache_resource
def get_chatbot():
    return Chatbot(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)

@st.cache_resource
def get_processor():
    return DocumentProcessor(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)

# Initialize these objects early
chatbot = get_chatbot()
processor = get_processor()

# Check Gemini model availability
@st.cache_data(ttl=3600)  # Cache for 1 hour
def check_gemini_models():
    try:
        # Configure API - assuming API key is already set in secrets
        if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
            genai.configure(api_key=st.secrets["gemini"]["api_key"])
            models = genai.list_models()
            return [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        return []
    except Exception as e:
        st.error(f"Error checking Gemini models: {str(e)}")
        return []

# App title and description
st.title("🧠 Knowledge Graph Chatbot")
st.markdown("""
Upload PDF documents and ask questions about their content. 
The system will extract information and store it in a knowledge graph for intelligent retrieval.
""")

# Display available Gemini models (in expander to not clutter UI)
with st.expander("🤖 API Model Status"):
    available_models = check_gemini_models()
    if available_models:
        st.success(f"✅ Available Gemini models: {', '.join(available_models)}")
    else:
        st.warning("⚠️ No Gemini models detected. Check your API key configuration.")

# Create two columns for layout
col1, col2 = st.columns([1, 2])

# 📂 File Upload Section (left column)
with col1:
    st.subheader("📂 Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        if st.button("Process Document"):
            try:
                with st.spinner("🔄 Processing document..."):
                    success = processor.process_pdf(uploaded_file)
                    
                    if success:
                        st.success("✅ Document processed successfully!")
                        st.session_state.chat_history.append(
                            {"role": "system", "content": f"Document '{uploaded_file.name}' has been processed and added to the knowledge graph."}
                        )
                    else:
                        st.error("❌ Failed to process document. See messages above for details.")
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")

    # Display system status
    st.subheader("System Status")
    try:
        with st.spinner("Checking connection..."):
            with chatbot.driver.session() as session:
                result = session.run("RETURN 1 as test").single()
                if result and result["test"] == 1:
                    st.success("✅ Connected to Neo4j")
                    
                    # Check if database has content
                    count_result = session.run("MATCH (c:TextChunk) RETURN count(c) as count").single()
                    count = count_result["count"] if count_result else 0
                    if count > 0:
                        st.success(f"✅ Database contains {count} text chunks")
                    else:
                        st.warning("⚠️ Database is empty. Please upload a document.")
                else:
                    st.error("❌ Neo4j connection issue")
    except Exception as e:
        st.error(f"❌ Neo4j connection error: {str(e)}")

# 💬 Chat Section (right column)
with col2:
    st.subheader("💬 Ask Questions About Your Documents")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            elif message["role"] == "assistant":
                st.chat_message("assistant").write(message["content"])
            elif message["role"] == "system":
                st.info(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask a question about your documents...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        st.chat_message("user").write(user_input)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    start_time = time.time()
                    response_text = chatbot.chat(user_input)
                    end_time = time.time()
                    
                    st.write(response_text)
                    
                    # Add debug info if response took too long
                    response_time = end_time - start_time
                    if response_time > 5:
                        st.caption(f"Response time: {response_time:.2f} seconds")
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                response_text = "I encountered a technical issue. Please try again later."
                st.write(response_text)
        
        # Add assistant message to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

# Footer
st.markdown("---")
st.caption("Knowledge Graph Chatbot powered by Neo4j and Gemini AI")
