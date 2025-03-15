import streamlit as st
from neo4j import GraphDatabase
import json
import google.generativeai as genai
import time
import re

class Chatbot:
    def __init__(self, uri, user, password, database="neo4j"):
    """Initialize Neo4j connection"""
    self.driver = GraphDatabase.driver(uri, auth=(user, password))
    self.database = database
    self.gemini_model = None
    
    # Configure Gemini API
    try:
        GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Check available models and select one
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            st.info(f"Available Gemini models: {available_models}")
            
            # Try these models in order of preference - UPDATED to prefer newer models
            preferred_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro", "gemini-pro-latest"]
            
            selected_model = None
            for model_name in preferred_models:
                if model_name in available_models:
                    selected_model = model_name
                    break
            
            if not selected_model:
                # Use first available model that supports text generation
                selected_model = available_models[0] if available_models else "gemini-1.5-flash"
            
            st.success(f"✅ Using Gemini model: {selected_model}")
            self.gemini_model = selected_model
            
            # Verify selected model works
            model = genai.GenerativeModel(selected_model)
            _ = model.generate_content("Test")
            st.success("✅ Gemini API configured and tested successfully")
        except Exception as e:
            st.error(f"⚠️ Gemini API test failed: {str(e)}")
            # Fall back to a newer recommended model
            self.gemini_model = "gemini-1.5-flash"
    except Exception as e:
        st.error(f"❌ Error configuring Gemini API: {str(e)}")

    def close(self):
        """Close Neo4j connection"""
        self.driver.close()

    def chat(self, user_input):
        """Process user query, retrieve knowledge from Neo4j, and generate chatbot response"""
        if not user_input or user_input.strip() == "":
            return "Please ask a question."
        
        # Debug info
        st.info(f"Searching for information about: '{user_input}'")
        
        # Check if database has any content
        db_has_content = self._check_database_has_content()
        if not db_has_content:
            return "The knowledge base appears to be empty. Please upload a document first."
            
        # First try exact keyword search
        text_chunks = self._find_relevant_text_exact(user_input)
        if text_chunks:
            st.info(f"Found {len(text_chunks)} chunks with exact match")
        
        # If no results, try fuzzy search with individual keywords
        if not text_chunks:
            st.info("No exact matches found, trying keyword search...")
            text_chunks = self._find_relevant_text_keywords(user_input)
            if text_chunks:
                st.info(f"Found {len(text_chunks)} chunks with keyword search")
            
        # If still no results, try searching for acronyms or abbreviations
        if not text_chunks and len(user_input) <= 5:
            st.info("Trying acronym/abbreviation search...")
            text_chunks = self._find_text_with_acronym(user_input)
            if text_chunks:
                st.info(f"Found {len(text_chunks)} chunks with acronym search")
            
        # If still no results, get some random chunks as context
        if not text_chunks:
            st.info("No specific matches found, retrieving sample chunks...")
            text_chunks = self._get_sample_chunks()
            if text_chunks:
                st.info(f"Retrieved {len(text_chunks)} sample chunks")
        
        # Generate response using Gemini
        try:
            response = self._generate_gemini_response(user_input, text_chunks)
            return response
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return f"⚠️ Error generating response: {str(e)}"

    def _check_database_has_content(self):
        """Check if the database has any content"""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (c:TextChunk) RETURN count(c) as count").single()
                count = result["count"] if result else 0
                st.info(f"Found {count} chunks in the database")
                return count > 0
        except Exception as e:
            st.error(f"Error checking database content: {str(e)}")
            return False

    def _find_relevant_text_exact(self, query_text):
        """Retrieve relevant text chunks from Neo4j using exact match"""
        try:
            with self.driver.session() as session:
                results = session.run(
                    """
                    MATCH (c:TextChunk)
                    WHERE toLower(c.text) CONTAINS toLower($query_text)
                    RETURN c.text AS text, c.id AS id
                    LIMIT 5
                    """,
                    query_text=query_text
                )
                chunks = []
                for record in results:
                    chunks.append(record["text"])
                    st.info(f"Found match in chunk {record['id']}")
                return chunks
        except Exception as e:
            st.error(f"Error querying Neo4j: {str(e)}")
            return []

    def _find_relevant_text_keywords(self, query_text):
        """Retrieve relevant text chunks from Neo4j using keywords"""
        # Extract meaningful keywords (words longer than 3 characters)
        keywords = [word.strip().lower() for word in query_text.split() if len(word.strip()) > 3]
        
        # If no meaningful keywords, use all words
        if not keywords:
            keywords = [word.strip().lower() for word in query_text.split() if len(word.strip()) > 0]
            
        st.info(f"Searching with keywords: {keywords}")
        
        results = []
        
        try:
            with self.driver.session() as session:
                for keyword in keywords:
                    query_results = session.run(
                        """
                        MATCH (c:TextChunk)
                        WHERE toLower(c.text) CONTAINS toLower($keyword)
                        RETURN c.text AS text, c.id AS id
                        LIMIT 3
                        """,
                        keyword=keyword
                    )
                    for record in query_results:
                        if record["text"] not in results:
                            st.info(f"Found keyword '{keyword}' in chunk {record['id']}")
                            results.append(record["text"])
                            if len(results) >= 5:  # Limit to 5 chunks
                                return results
                return results
        except Exception as e:
            st.error(f"Error querying Neo4j with keywords: {str(e)}")
            return []

    def _find_text_with_acronym(self, acronym):
        """Search for text chunks that might contain the expanded form of an acronym"""
        try:
            with self.driver.session() as session:
                # First try to find exact acronym
                results = session.run(
                    """
                    MATCH (c:TextChunk)
                    WHERE toLower(c.text) CONTAINS toLower($acronym)
                    RETURN c.text AS text, c.id AS id
                    LIMIT 5
                    """,
                    acronym=acronym
                )
                
                chunks = []
                for record in results:
                    st.info(f"Found acronym '{acronym}' in chunk {record['id']}")
                    chunks.append(record["text"])
                
                # If found, return these chunks
                if chunks:
                    return chunks
                    
                # If not found, get all chunks to search for potential matches
                # (This is inefficient but works for small datasets)
                all_results = session.run(
                    """
                    MATCH (c:TextChunk)
                    RETURN c.text AS text, c.id AS id
                    LIMIT 20
                    """
                )
                
                # Look for patterns where the acronym might be defined
                # e.g., "All Star Driver Education (ASDE)"
                chunks = []
                pattern1 = re.compile(r'([A-Za-z\s]+)\s*\(' + re.escape(acronym) + r'\)', re.IGNORECASE)
                pattern2 = re.compile(r'([A-Za-z\s]+)\s*\(' + ''.join([f'{c}[A-Za-z]*' for c in acronym]) + r'\)', re.IGNORECASE)
                
                for record in all_results:
                    text = record["text"]
                    if pattern1.search(text) or pattern2.search(text) or acronym.lower() in text.lower():
                        st.info(f"Found potential acronym match in chunk {record['id']}")
                        chunks.append(text)
                        if len(chunks) >= 5:
                            break
                
                return chunks
                
        except Exception as e:
            st.error(f"Error searching for acronym: {str(e)}")
            return []

    def _get_sample_chunks(self, limit=5):
        """Get sample chunks from Neo4j when no relevant chunks are found"""
        try:
            with self.driver.session() as session:
                results = session.run(
                    """
                    MATCH (c:TextChunk)
                    RETURN c.text AS text, c.id AS id
                    LIMIT $limit
                    """,
                    limit=limit
                )
                chunks = []
                for record in results:
                    st.info(f"Retrieved sample chunk {record['id']}")
                    chunks.append(record["text"])
                return chunks
        except Exception as e:
            st.error(f"Error getting sample chunks: {str(e)}")
            return []

    def _generate_gemini_response(self, user_input, text_chunks):
        """Generate chatbot response using Gemini AI"""
        if not text_chunks:
            return "I don't have enough information to answer that question. Please upload relevant documents."
            
        # Prepare context from chunks (limit total size)
        max_context_length = 8000  # Reduced to avoid token limits
        context = ""
        for chunk in text_chunks:
            if len(context) + len(chunk) < max_context_length:
                context += chunk + "\n\n"
            else:
                break
                
        prompt = f"""
        You are a helpful assistant answering questions based on the provided document context.
        
        CONTEXT:
        {context}
        
        USER QUESTION: {user_input}
        
        Answer the user's question based ONLY on the information in the context provided.
        If the answer cannot be found in the context, say "I don't have enough information to answer that question."
        Be concise and accurate.
        """

        try:
            # Add retry logic with better error handling
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Use a more reliable model configuration
                    generation_config = {
                        "temperature": 0.2,
                        "top_p": 0.8,
                        "top_k": 40,
                        "max_output_tokens": 1024,
                    }
                    
                    safety_settings = [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        }
                    ]
                    
                    # Use the model name selected during initialization
                    model_name = self.gemini_model if self.gemini_model else "gemini-pro"
                    st.info(f"Using Gemini model: {model_name}")
                    
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    
                    response = model.generate_content(prompt)
                    
                    if response and hasattr(response, "text"):
                        return response.text
                    else:
                        st.warning(f"Empty response from Gemini (attempt {attempt+1})")
                        time.sleep(1)  # Wait before retry
                except Exception as e:
                    st.warning(f"Gemini API error (attempt {attempt+1}): {str(e)}")
                    time.sleep(2)  # Wait before retry
                    
            # Fallback response if all retries fail
            return "I'm having trouble connecting to my knowledge base. Please try a different question or try again later."
        except Exception as e:
            st.error(f"Error with Gemini API: {str(e)}")
            return "I encountered a technical issue. Please check your API configuration and try again."
