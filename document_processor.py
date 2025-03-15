import PyPDF2
from neo4j import GraphDatabase
import io
import streamlit as st
import re

class DocumentProcessor:
    def __init__(self, uri, user, password):
        """Initialize Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def process_pdf(self, uploaded_file):
        """Extract text from a Streamlit uploaded PDF, chunk it, and store in Neo4j"""
        try:
            # Verify the uploaded file
            if uploaded_file is None:
                st.error("No file was uploaded")
                return False
                
            # Debug information
            st.info(f"Processing file: {uploaded_file.name}, Size: {uploaded_file.size} bytes")
            
            # Extract text from the PDF
            text = self._extract_text_from_streamlit_file(uploaded_file)
            
            if not text or len(text.strip()) == 0:
                st.warning("⚠️ No text could be extracted from the PDF!")
                return False

            st.info(f"Extracted {len(text)} characters of text")
            
            # Clean the text
            text = self._clean_text(text)
            st.info(f"Text cleaned, now {len(text)} characters")
                
            # Chunk the text
            chunks = self._chunk_text(text)
            if not chunks or len(chunks) == 0:
                st.warning("⚠️ No chunks created from the text!")
                return False
                
            st.info(f"Created {len(chunks)} text chunks")

            # Clear existing chunks before adding new ones
            self._clear_existing_chunks()

            # Store chunks in Neo4j
            success = self._store_chunks_in_neo4j(chunks)
            return success
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return False

    def _extract_text_from_streamlit_file(self, uploaded_file):
        """Extract text from a PDF file uploaded via Streamlit"""
        text = ""
        try:
            # Read the file content
            bytes_data = uploaded_file.getvalue()
            
            # Create a file-like object
            file_stream = io.BytesIO(bytes_data)
            
            # Use PyPDF2 to extract text
            try:
                reader = PyPDF2.PdfReader(file_stream)
                
                # Debug info
                st.info(f"PDF has {len(reader.pages)} pages")
                
                # Extract text from each page
                for i, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        st.warning(f"Error extracting text from page {i+1}: {str(e)}")
                
            except Exception as e:
                st.error(f"Error reading PDF: {str(e)}")
                return ""
                
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            return ""
        
        return text
        
    def _clean_text(self, text):
        """Clean the extracted text"""
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r' +', ' ', text)
        
        # Remove any non-printable characters
        text = ''.join(c for c in text if c.isprintable() or c == '\n')
        
        return text

    def _chunk_text(self, text, chunk_size=800, overlap=200):
        """Split text into overlapping chunks"""
        chunks = []
        if not text or len(text.strip()) == 0:
            return chunks
            
        # Try to chunk at paragraph boundaries first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                # If the current chunk has content, add it to chunks
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Start a new chunk with overlap
                if len(current_chunk) > overlap:
                    # Find the last period in the overlap region
                    overlap_text = current_chunk[-overlap:]
                    last_period = overlap_text.rfind('.')
                    
                    if last_period != -1:
                        # Start new chunk from the last sentence in the overlap
                        current_chunk = current_chunk[-(overlap-last_period):] + para + "\n\n"
                    else:
                        # No good break point, just use the overlap
                        current_chunk = current_chunk[-overlap:] + para + "\n\n"
                else:
                    current_chunk = para + "\n\n"
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        # If no chunks were created (maybe no paragraph breaks), fall back to character-based chunking
        if not chunks:
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                
                # Try to end at a sentence or paragraph break
                if end < len(text):
                    # Look for paragraph break
                    para_break = text.rfind('\n\n', start, end)
                    if para_break != -1 and para_break > start + chunk_size // 2:
                        end = para_break + 2
                    else:
                        # Look for sentence break
                        sentence_break = text.rfind('. ', start, end)
                        if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                            end = sentence_break + 2
                
                chunk = text[start:end]
                if chunk.strip():  # Only add non-empty chunks
                    chunks.append(chunk.strip())
                
                # Calculate next start position with overlap
                if end == len(text):
                    break
                    
                # Try to start at a sentence beginning
                overlap_start = max(end - overlap, start)
                sentence_start = text.find('. ', overlap_start, end)
                if sentence_start != -1:
                    start = sentence_start + 2
                else:
                    start = overlap_start
        
        return chunks
        
    def _clear_existing_chunks(self):
        """Clear existing chunks from Neo4j"""
        try:
            with self.driver.session() as session:
                session.run("MATCH (c:TextChunk) DETACH DELETE c")
                st.info("🧹 Cleared existing chunks from Neo4j")
                return True
        except Exception as e:
            st.error(f"Error clearing existing chunks: {str(e)}")
            return False
    
    def _store_chunks_in_neo4j(self, chunks):
        """Store text chunks in Neo4j"""
        success = False
        try:
            with self.driver.session() as session:
                # Create a transaction function to batch the inserts
                def create_chunks_tx(tx, chunk_batch, start_idx):
                    for i, chunk in enumerate(chunk_batch):
                        chunk_id = f"chunk-{start_idx + i}"
                        tx.run(
                            """
                            CREATE (c:TextChunk {id: $id, text: $text})
                            """,
                            id=chunk_id,
                            text=chunk
                        )
                
                # Process in smaller batches to avoid transaction timeouts
                batch_size = 10
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    session.execute_write(create_chunks_tx, batch, i)
                    st.info(f"Stored batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                
                st.success(f"✅ Successfully stored {len(chunks)} in Neo4j")
                success = True
        except Exception as e:
            st.error(f"Error storing chunks in Neo4j: {str(e)}")
            success = False
        
        return success
