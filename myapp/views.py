import os
import tiktoken
import PyPDF2
import docx
import openai
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from sentence_transformers import SentenceTransformer
import chromadb
from .models import Document, DocumentChunk
from .serializers import DocumentSerializer, ChunkSerializer

# Constants for chunk creation
CHUNK_LENGTH = 25  # tokens
CHUNK_OVERLAP = 5  # tokens

# Initialize OpenAI API (make sure to set your API key)
openai.api_key = os.getenv('OPENAI_API_KEY')  # Set this in your environment variables

# Initialize model and vector DB
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()

# Create two collections in ChromaDB
document_collection = client.get_or_create_collection("Document")
document_chunk_collection = client.get_or_create_collection("DocumentChunk")

def split_in_chunks(document_text: str) -> list[str]:
    """
    Split document text into chunks based on token count using tiktoken.
    
    Args:
        document_text (str): The text to be chunked
        
    Returns:
        list[str]: List of text chunks
    """
    chunks = []
    
    # Initialize tiktoken encoder for GPT models 
    # #cl100k_base is used by GPT-3.5 and GPT-4
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Convert text to tokens
    tokens = encoding.encode(document_text)
    
    # Create chunks with overlap
    start_idx = 0
    while start_idx < len(tokens):
        # Define end index for current chunk
        end_idx = min(start_idx + CHUNK_LENGTH, len(tokens))
        
        # Extract chunk tokens
        chunk_tokens = tokens[start_idx:end_idx]
        
        # Convert tokens back to text
        chunk_text = encoding.decode(chunk_tokens)
        
        # Add chunk to array
        chunks.append(chunk_text)
        
        # Move start index forward, considering overlap
        # If we're at the end, break to avoid infinite loop
        if end_idx >= len(tokens):
            break
            
        start_idx = end_idx - CHUNK_OVERLAP
        
        # Ensure we don't go backwards (in case CHUNK_OVERLAP >= CHUNK_LENGTH)
        if start_idx <= 0:
            start_idx = end_idx
    
    return chunks

def determine_file_type(file_path: str) -> str:
    """
    Determine file type based on file extension.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: File type ('txt', 'pdf', 'docx', or 'unknown')
    """
    _, ext = os.path.splitext(file_path.lower())
    
    if ext == '.txt':
        return 'txt'
    elif ext == '.pdf':
        return 'pdf'
    elif ext in ['.docx', '.doc']:
        return 'docx'
    else:
        return 'unknown'

def read_document_content(file_path: str, file_type: str) -> tuple[str, int]:
    """
    Read document content based on file type.
    
    Args:
        file_path (str): Path to the file
        file_type (str): Type of file ('txt', 'pdf', 'docx')
        
    Returns:
        tuple[str, int]: Document content and page count
    """
    content = ""
    pages = 1
    
    try:
        if file_type == 'txt':
            # Read text file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                pages = 1  # Text files considered as 1 page
                
        elif file_type == 'pdf':
            # Read PDF file
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pages = len(pdf_reader.pages)
                
                for page_num in range(pages):
                    page = pdf_reader.pages[page_num]
                    content += page.extract_text() + "\n"
                    
        elif file_type == 'docx':
            # Read DOCX file
            doc = docx.Document(file_path)
            pages = 1  # DOCX page count is complex, defaulting to 1
            
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
                
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        raise Exception(f"Error reading {file_type} file: {str(e)}")
    
    return content.strip(), pages

def make_search_query(chat_history: list) -> str:
    """
    Generate a search query from chat history using GPT-4o.
    
    Args:
        chat_history (list): List of chat messages with 'role' and 'content'
        
    Returns:
        str: Generated search query
    """
    try:
        # Create a prompt to extract search intent from chat history
        system_prompt = """You are a search query generator. Based on the chat history provided, generate a concise search query that captures the user's information need. 

Rules:
1. Extract the main topic/question from the latest user message
2. Consider the context from previous messages if relevant
3. Generate a clear, specific search query (2-10 words)
4. Focus on keywords that would help find relevant document chunks
5. Do not include conversational words like "please", "can you", "I want to know"

Examples:
User: "What is machine learning?" → Query: "machine learning definition"
User: "Can you explain neural networks?" → Query: "neural networks explanation"
User: "How does photosynthesis work in plants?" → Query: "photosynthesis process plants"

Return only the search query, nothing else."""

        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Chat history: {chat_history}\n\nGenerate a search query:"}
        ]
        
        # Call OpenAI GPT-4o
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=50,
            temperature=0.3
        )
        
        search_query = response.choices[0].message.content.strip()
        return search_query
        
    except Exception as e:
        # Fallback: extract query from last user message
        print(f"Error generating search query: {str(e)}")
        
        # Simple fallback - get last user message
        for message in reversed(chat_history):
            if message.get('role') == 'user':
                return message.get('content', '')[:100]  # Limit length
        
        return "search query"

def find_similar_chunks(search_query: str, n_results: int = 5) -> list:
    """
    Find similar chunks from ChromaDB using the search query.
    
    Args:
        search_query (str): Query to search for
        n_results (int): Number of results to return
        
    Returns:
        list: List of similar chunks with metadata
    """
    try:
        # Generate embedding for the search query
        query_embedding = model.encode([search_query])[0]
        
        # Search in DocumentChunk collection
        results = document_chunk_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        similar_chunks = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                chunk_data = {
                    'chunk_id': results['ids'][0][i] if 'ids' in results else f"chunk_{i}",
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'similarity_score': 1 - results['distances'][0][i] if results['distances'] else 0.0,  # Convert distance to similarity
                    'document_name': results['metadatas'][0][i].get('document_name', 'Unknown') if results['metadatas'] else 'Unknown'
                }
                similar_chunks.append(chunk_data)
        
        return similar_chunks
        
    except Exception as e:
        print(f"Error finding similar chunks: {str(e)}")
        return []

@method_decorator(csrf_exempt, name='dispatch')
class AskView(APIView):
    def post(self, request):
        try:
            # Step 1: Accept chat history from user
            chat_history = request.data.get('chat_history', [])
            
            if not chat_history:
                return Response({
                    'error': 'No chat history provided'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            print(f"✅ Step 1: Received chat history with {len(chat_history)} messages")
            
            # Step 2: Generate search query from chat history
            search_query = make_search_query(chat_history)
            print(f"✅ Step 2: Generated search query: '{search_query}'")
            
            # Step 3: Store search query in variable (already done above)
            
            # Step 4: Find similar chunks using the search query
            similar_chunks = find_similar_chunks(search_query, n_results=5)
            print(f"✅ Step 4: Found {len(similar_chunks)} similar chunks")
            
            # Step 5: Similar chunks retrieved (already done above)
            
            # Step 6 & 7: Use OpenAI GPT-4o with the specified prompt
            context_text = "\n\n".join([chunk['text'] for chunk in similar_chunks])
            
            # The prompt from the end of document (you'll need to provide this)
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided document context. 

Instructions:
1. Use only the information provided in the context to answer questions
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Provide accurate, detailed answers when possible
4. Always cite which document chunks you're referencing
5. Be concise but thorough in your responses

Context from documents:
{context}

Chat History:
{chat_history}

Please answer the user's latest question based on the provided context."""

            # Prepare messages for final response generation
            messages = [
                {
                    "role": "system", 
                    "content": system_prompt.format(
                        context=context_text,
                        chat_history=chat_history
                    )
                }
            ]
            
            # Add chat history to messages
            for msg in chat_history:
                if msg.get('role') in ['user', 'assistant']:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
            
            # Call GPT-4o for final response
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            print(f"✅ Step 7: Generated AI response")
            
            # Step 8: Return response with citations
            return Response({
                'response': ai_response,
                'search_query': search_query,
                'citations': similar_chunks,
                'metadata': {
                    'chunks_found': len(similar_chunks),
                    'model_used': 'gpt-4o',
                    'chat_history_length': len(chat_history)
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                'error': f'Error processing chat request: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@method_decorator(csrf_exempt, name='dispatch')
class DocumentUploadView(APIView):
    def post(self, request):
        try:
            # Step 1: Get file and name from request
            uploaded_file = request.FILES.get('file')
            document_name = request.data.get('name', uploaded_file.name if uploaded_file else 'Untitled')
            
            if not uploaded_file:
                return Response({
                    'error': 'No file uploaded'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            print(f"✅ Step 1: File received - {document_name}")
            
            # Save file to storage first
            document = Document.objects.create(
                name=document_name,
                file=uploaded_file,
                file_type=uploaded_file.content_type,
                size=uploaded_file.size,
                pages=1  # Will be updated after reading
            )
            
            file_path = document.file.path
            print(f"✅ File saved to: {file_path}")
            
            # Step 2: Determine file type
            file_type = determine_file_type(file_path)
            if file_type == 'unknown':
                document.delete()
                return Response({
                    'error': 'Unsupported file type. Only TXT, PDF, and DOCX files are supported.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            print(f"✅ Step 2: File type determined - {file_type}")
            
            # Step 3: Read document content
            document_content, total_pages = read_document_content(file_path, file_type)
            
            # Update document with correct page count
            document.pages = total_pages
            document.save()
            
            print(f"✅ Step 3: Document content read - {len(document_content)} characters, {total_pages} pages")
            
            # Step 4: Split content into chunks
            chunks = split_in_chunks(document_content)
            print(f"✅ Step 4: Document split into {len(chunks)} chunks")
            
            # Step 5: ChromaDB is already setup (collections created at top)
            print("✅ Step 5: ChromaDB collections ready")
            
            # Step 6 & 8: Create document object in ChromaDB Document collection
            chroma_doc_id = f"doc_{document.id}"
            document_collection.add(
                ids=[chroma_doc_id],
                documents=[f"Document: {document_name}"],
                metadatas=[{
                    'name': document_name,
                    'total_pages': total_pages,
                    'file_type': file_type,
                    'mysql_id': document.id
                }]
            )
            
            print(f"✅ Step 8: Document created in ChromaDB with ID: {chroma_doc_id}")
            
            # Step 9: Process each chunk
            chunk_ids = []
            for chunk_index, chunk_text in enumerate(chunks):
                if chunk_text.strip():  # Skip empty chunks
                    # Step 9.1: Create embedding using SentenceTransformer
                    embedding = model.encode([chunk_text])[0]
                    
                    # Step 9.2: Create object in DocumentChunk collection
                    chunk_number = chunk_index + 1  # Index + 1 as requested
                    chroma_chunk_id = f"doc_{document.id}_chunk_{chunk_number}"
                    
                    document_chunk_collection.add(
                        embeddings=[embedding.tolist()],
                        documents=[chunk_text],
                        ids=[chroma_chunk_id],
                        metadatas=[{
                            'document_id': chroma_doc_id,
                            'chunk_number': chunk_number,
                            'document_name': document_name,
                            'mysql_doc_id': document.id
                        }]
                    )
                    
                    chunk_ids.append(chroma_chunk_id)
                    
                    # Also store chunk in MySQL for tracking
                    DocumentChunk.objects.create(
                        document=document,
                        chunk_index=chunk_index,
                        page_number=1,  # Simplified for now
                        text=chunk_text,
                        embedding_id=chroma_chunk_id
                    )
            
            print(f"✅ Step 9: Processed {len(chunk_ids)} chunks with embeddings")
            
            # Update document record with ChromaDB ID
            document.chroma_id = chroma_doc_id
            document.save()
            
            # Step 10: Return success response
            serializer = DocumentSerializer(document)
            return Response({
                'message': 'Document uploaded and processed successfully',
                'document': serializer.data,
                'chroma_document_id': chroma_doc_id,
                'chunks_created': len(chunk_ids),
                'total_pages': total_pages,
                'file_type': file_type,
                'processing_steps_completed': [
                    '✅ File uploaded and saved',
                    '✅ File type determined',
                    '✅ Document content extracted',
                    '✅ Text split into chunks',  
                    '✅ ChromaDB collections ready',
                    '✅ Document stored in ChromaDB',
                    '✅ All chunks processed with embeddings',
                    '✅ MySQL records created'
                ]
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            # Clean up if error occurs
            try:
                if 'document' in locals():
                    document.delete()
            except:
                pass
                
            return Response({
                'error': f'Error processing document: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class DocumentListView(APIView):
    def get(self, request):
        try:
            documents = Document.objects.all().order_by('-created_at')
            serializer = DocumentSerializer(documents, many=True)
            return Response({
                'documents': serializer.data
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                'error': f'Error retrieving documents: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class DocumentDetailView(APIView):
    def get(self, request, doc_id):
        try:
            document = Document.objects.get(id=doc_id)
            chunks = DocumentChunk.objects.filter(document=document).order_by('chunk_index')
            
            doc_serializer = DocumentSerializer(document)
            chunk_serializer = ChunkSerializer(chunks, many=True)
            
            return Response({
                'document': doc_serializer.data,
                'chunks': chunk_serializer.data
            }, status=status.HTTP_200_OK)
            
        except Document.DoesNotExist:
            return Response({
                'error': 'Document not found'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'Error retrieving document: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def delete(self, request, doc_id):
        try:
            document = Document.objects.get(id=doc_id)
            
            # Get all chunks for this document
            chunks = DocumentChunk.objects.filter(document=document)
            
            # Delete embeddings from both ChromaDB collections
            chunk_ids = [chunk.embedding_id for chunk in chunks]
            if chunk_ids:
                document_chunk_collection.delete(ids=chunk_ids)
            
            # Delete document from ChromaDB Document collection
            if hasattr(document, 'chroma_id') and document.chroma_id:
                document_collection.delete(ids=[document.chroma_id])
            
            # Delete file from storage
            if document.file:
                file_path = document.file.path
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Delete document (cascades to chunks)
            document.delete()
            
            return Response({
                'message': 'Document deleted successfully'
            }, status=status.HTTP_200_OK)
            
        except Document.DoesNotExist:
            return Response({
                'error': 'Document not found'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'Error deleting document: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def home(request):
    return JsonResponse({
        'message': 'Document Intelligence Platform API',
        'endpoints': {
            'upload': 'POST /document',
            'list': 'GET /document/list', 
            'detail': 'GET /document/<id>',
            'delete': 'DELETE /document/<id>',
            'ask': 'POST /ask'
        }
    })