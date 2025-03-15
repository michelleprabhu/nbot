import streamlit as st
from neo4j import GraphDatabase
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

            # Fetch available models (Removing 'models/' prefix from names)
            available_models = [m.name.split('/')[-1] for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]

            if not available_models:
                raise ValueError("No Gemini models are available. Check API key configuration.")

            # Select from supported models ONLY
            preferred_models = ["gemini-2.0-flash", "gemini-2.0-pro-exp", "gemini-1.5-flash", "gemini-1.5-pro"]
            selected_model = next((m for m in preferred_models if m in available_models), None)

            if not selected_model:
                raise ValueError(f"❌ No supported models found. Available models: {available_models}")

            st.success(f"✅ Using Gemini model: {selected_model}")
            self.gemini_model = selected_model

            # Test API connectivity
            model = genai.GenerativeModel(selected_model)
            model.generate_content("Test")
            st.success("✅ Gemini API configured and tested successfully")

        except Exception as e:
            st.error(f"⚠️ Gemini API setup failed: {str(e)}")
            self.gemini_model = "gemini-2.0-flash"  # Default fallback

    def close(self):
        """Close Neo4j connection"""
        self.driver.close()

    def chat(self, user_input):
        """Process user query, retrieve knowledge from Neo4j, and generate chatbot response"""
        if not user_input.strip():
            return "Please ask a question."

        st.info(f"Searching for: '{user_input}'")

        if not self._check_database_has_content():
            return "Knowledge base is empty. Please upload a document."

        # Try various retrieval methods
        text_chunks = (self._find_relevant_text_exact(user_input) or
                       self._find_relevant_text_keywords(user_input) or
                       self._get_sample_chunks())

        # Generate response using Gemini
        try:
            return self._generate_gemini_response(user_input, text_chunks)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return f"⚠️ Error: {str(e)}"

    def _check_database_has_content(self):
        """Check if database contains text chunks"""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (c:TextChunk) RETURN count(c) AS count").single()
                return result["count"] > 0 if result else False
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return False

    def _find_relevant_text_exact(self, query_text):
        """Find exact matches in Neo4j"""
        return self._query_neo4j(f"""
            MATCH (c:TextChunk)
            WHERE toLower(c.text) CONTAINS toLower($query_text)
            RETURN c.text AS text
            LIMIT 5
        """, {"query_text": query_text})

    def _find_relevant_text_keywords(self, query_text):
        """Find text based on keywords"""
        keywords = [word.lower() for word in query_text.split() if len(word) > 3]
        return [self._query_neo4j(f"""
            MATCH (c:TextChunk)
            WHERE toLower(c.text) CONTAINS toLower($keyword)
            RETURN c.text AS text
            LIMIT 3
        """, {"keyword": keyword}) for keyword in keywords][:5]

    def _get_sample_chunks(self):
        """Get random text chunks if no matches"""
        return self._query_neo4j("""
            MATCH (c:TextChunk) RETURN c.text AS text LIMIT 5
        """)

    def _query_neo4j(self, query, params=None):
        """Run Neo4j query & return results"""
        try:
            with self.driver.session() as session:
                return [record["text"] for record in session.run(query, params)]
        except Exception as e:
            st.error(f"Neo4j Query Error: {str(e)}")
            return []

    def _generate_gemini_response(self, user_input, text_chunks):
        """Generate response using Gemini AI"""
        if not text_chunks:
            return "I don't have enough information to answer that question."

        context = "\n\n".join(text_chunks)[:8000]
        prompt = f"""
        CONTEXT:
        {context}

        USER QUESTION: {user_input}

        Answer using ONLY the above context. If the answer isn't in the context, say:
        "I don't have enough information."
        """

        try:
            model = genai.GenerativeModel(self.gemini_model)
            response = model.generate_content(prompt)
            return response.text if hasattr(response, "text") else "Response was empty."
        except Exception as e:
            st.error(f"Gemini API Error: {str(e)}")
            return "⚠️ Gemini API encountered an issue."
