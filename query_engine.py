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

            # Fetch available models
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]

            if not available_models:
                raise ValueError("No Gemini models are available. Check API key configuration.")

            # Select an available model from the preferred list
            preferred_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro", "gemini-pro-latest"]
            selected_model = next((m for m in preferred_models if m in available_models), None)

            if not selected_model:
                selected_model = available_models[0]  # Default to the first available model

            st.success(f"✅ Using Gemini model: {selected_model}")
            self.gemini_model = selected_model

            # Test API connectivity
            model = genai.GenerativeModel(selected_model)
            _ = model.generate_content("Test")
            st.success("✅ Gemini API configured and tested successfully")
        except Exception as e:
            st.error(f"⚠️ Gemini API test failed: {str(e)}")
            self.gemini_model = "gemini-1.5-flash"  # Fallback model

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

        # Try exact keyword search
        text_chunks = self._find_relevant_text_exact(user_input)
        if text_chunks:
            st.info(f"Found {len(text_chunks)} chunks with exact match")

        # If no results, try fuzzy search with individual keywords
        if not text_chunks:
            st.info("No exact matches found, trying keyword search...")
            text_chunks = self._find_relevant_text_keywords(user_input)
            if text_chunks:
                st.info(f"Found {len(text_chunks)} chunks with keyword search")

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
                chunks = [record["text"] for record in results]
                return chunks
        except Exception as e:
            st.error(f"Error querying Neo4j: {str(e)}")
            return []

    def _find_relevant_text_keywords(self, query_text):
        """Retrieve relevant text chunks from Neo4j using keywords"""
        keywords = [word.strip().lower() for word in query_text.split() if len(word.strip()) > 3]

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
                    results.extend(record["text"] for record in query_results)
                return results[:5]
        except Exception as e:
            st.error(f"Error querying Neo4j with keywords: {str(e)}")
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
                return [record["text"] for record in results]
        except Exception as e:
            st.error(f"Error getting sample chunks: {str(e)}")
            return []

    def _generate_gemini_response(self, user_input, text_chunks):
        """Generate chatbot response using Gemini AI"""
        if not text_chunks:
            return "I don't have enough information to answer that question. Please upload relevant documents."

        # Prepare context from chunks
        max_context_length = 8000
        context = "\n\n".join(text_chunks)[:max_context_length]

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
            model = genai.GenerativeModel(self.gemini_model)
            response = model.generate_content(prompt)
            return response.text if hasattr(response, "text") else "I couldn't generate a response."
        except Exception as e:
            st.error(f"Error with Gemini API: {str(e)}")
            return "I encountered a technical issue. Please check your API configuration and try again."
