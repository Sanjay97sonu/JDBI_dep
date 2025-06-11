# ✅ COMPLETE WEBSITE SCRAPER WITH PDF EXTRACTION - POSTGRESQL VERSION
# pip install beautifulsoup4 requests nltk sentence-transformers faiss-cpu anthropic flask PyMuPDF psycopg2-binary

import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import anthropic
from flask import Flask, request, jsonify, render_template_string
import threading
import time
import re
from collections import deque
import fitz  # PyMuPDF for PDF extraction
import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
import base64
import pickle

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# ✅ POSTGRESQL CONFIGURATION
# Update these settings to match your PostgreSQL server
DB_CONFIG = {
    'host': '192.168.0.37',   # Your PostgreSQL host
    'database': 'cloude-team', # Your database name
    'user': 'cloude-team',     # Your PostgreSQL username
    'password': 'cloude-team123', # Add your actual password here
    'port': 5432               # Your PostgreSQL port
}

# ✅ STEP 2: Initialize Flask app
app = Flask(__name__)

# Global variables to store the RAG components
model = None
index = None
all_chunks = []
client = None
system_ready = False
db_pool = None
chat_history = []  # Added for chat context

# ✅ PERSISTENT STORAGE CONFIGURATION
PDF_DIR = "pdfs"  # Directory for downloaded PDFs

# Create directories if they don't exist
os.makedirs(PDF_DIR, exist_ok=True)

# ✅ POSTGRESQL DATABASE FUNCTIONS
def init_database():
    """Initialize PostgreSQL connection pool and create tables"""
    global db_pool
    try:
        # Create connection pool
        db_pool = SimpleConnectionPool(
            1, 20,  # min and max connections
            **DB_CONFIG
        )
        
        # Create tables
        conn = db_pool.getconn()
        cursor = conn.cursor()
        
        # Check if tables exist and create/update them
        print("🔧 Checking and creating database tables...")
        
        # Drop existing tables if they have wrong structure (optional - be careful!)
        try:
            cursor.execute("DROP TABLE IF EXISTS content_chunks CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS embeddings_data CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS pdf_files CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS crawl_sessions CASCADE;")
            print("🗑️ Dropped existing tables to recreate with correct structure")
        except Exception as e:
            print(f"⚠️ No existing tables to drop: {e}")
        
        # Create fresh tables
        cursor.execute("""
            CREATE TABLE crawl_sessions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                pages_crawled INTEGER DEFAULT 0,
                pdfs_processed INTEGER DEFAULT 0,
                chunks_created INTEGER DEFAULT 0,
                base_url TEXT,
                status VARCHAR(50) DEFAULT 'active'
            );
        """)
        print("✅ Created crawl_sessions table")
        
        cursor.execute("""
            CREATE TABLE content_chunks (
                id SERIAL PRIMARY KEY,
                session_id INTEGER REFERENCES crawl_sessions(id) ON DELETE CASCADE,
                chunk_text TEXT NOT NULL,
                source_url TEXT,
                source_type VARCHAR(10) CHECK (source_type IN ('web', 'pdf')),
                source_title TEXT,
                chunk_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("✅ Created content_chunks table")
        
        cursor.execute("""
            CREATE TABLE embeddings_data (
                id SERIAL PRIMARY KEY,
                session_id INTEGER REFERENCES crawl_sessions(id) ON DELETE CASCADE,
                embeddings_binary BYTEA NOT NULL,
                dimension INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("✅ Created embeddings_data table")
        
        cursor.execute("""
            CREATE TABLE pdf_files (
                id SERIAL PRIMARY KEY,
                session_id INTEGER REFERENCES crawl_sessions(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                original_url TEXT,
                file_path TEXT,
                text_content TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("✅ Created pdf_files table")
        
        # Create indexes for better performance
        cursor.execute("""
            CREATE INDEX idx_content_chunks_session 
            ON content_chunks(session_id);
        """)
        
        cursor.execute("""
            CREATE INDEX idx_content_chunks_source_type 
            ON content_chunks(source_type);
        """)
        
        cursor.execute("""
            CREATE INDEX idx_embeddings_session 
            ON embeddings_data(session_id);
        """)
        
        print("✅ Created database indexes")
        
        conn.commit()
        cursor.close()
        db_pool.putconn(conn)
        
        print("✅ PostgreSQL database initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            db_pool.putconn(conn)
        return False

def save_data_to_db(chunks, embeddings, crawl_info, pdf_data=None):
    """Save all data to PostgreSQL database"""
    try:
        print("💾 Saving complete website and PDF data to PostgreSQL...")
        
        conn = db_pool.getconn()
        cursor = conn.cursor()
        
        # Create new crawl session
        cursor.execute("""
            INSERT INTO crawl_sessions (pages_crawled, pdfs_processed, chunks_created, base_url)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
        """, (
            crawl_info.get('pages_crawled', 0),
            crawl_info.get('pdfs_processed', 0),
            len(chunks),
            crawl_info.get('base_url', '')
        ))
        
        session_id = cursor.fetchone()[0]
        
        # Save content chunks with proper text cleaning
        for i, chunk in enumerate(chunks):
            # Clean chunk text for database storage
            clean_chunk = clean_text_for_db(chunk)
            
            # Extract source info from chunk
            source_type = 'pdf' if 'PDF:' in clean_chunk else 'web'
            source_url = ''
            source_title = ''
            
            if 'SOURCE:' in clean_chunk:
                source_part = clean_chunk.split('SOURCE:')[1].split('\n')[0] if 'SOURCE:' in clean_chunk else ''
                if '(' in source_part and ')' in source_part:
                    source_title = source_part.split('(')[0].strip()
                    source_url_part = source_part.split('(')[1].split(')')[0]
                    if ':' in source_url_part:
                        source_url = source_url_part.split(':', 1)[1].strip()
            
            # Clean source fields
            source_url = clean_text_for_db(source_url) if source_url else ''
            source_title = clean_text_for_db(source_title) if source_title else ''
            
            cursor.execute("""
                INSERT INTO content_chunks (session_id, chunk_text, source_url, source_type, source_title, chunk_index)
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (session_id, clean_chunk, source_url, source_type, source_title, i))
        
        # Save embeddings as binary data
        embeddings_binary = pickle.dumps(embeddings)
        cursor.execute("""
            INSERT INTO embeddings_data (session_id, embeddings_binary, dimension)
            VALUES (%s, %s, %s);
        """, (session_id, embeddings_binary, embeddings.shape[1]))
        
        # Save PDF file information with cleaned text
        if pdf_data:
            for pdf in pdf_data:
                # Clean all text fields
                filename = clean_text_for_db(pdf.get('filename', ''))
                original_url = clean_text_for_db(pdf.get('url', ''))
                file_path = clean_text_for_db(pdf.get('filepath', ''))
                text_content = clean_text_for_db(pdf.get('content', ''))
                
                cursor.execute("""
                    INSERT INTO pdf_files (session_id, filename, original_url, file_path, text_content)
                    VALUES (%s, %s, %s, %s, %s);
                """, (session_id, filename, original_url, file_path, text_content))
        
        conn.commit()
        cursor.close()
        db_pool.putconn(conn)
        
        print(f"✅ Saved session {session_id} with {len(chunks)} chunks to database")
        return True
        
    except Exception as e:
        print(f"❌ Database save error: {e}")
        import traceback
        traceback.print_exc()
        if 'conn' in locals():
            conn.rollback()
            cursor.close()
            db_pool.putconn(conn)
        return False

def load_data_from_db():
    """Load the most recent data from PostgreSQL database"""
    try:
        conn = db_pool.getconn()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get the most recent session
        cursor.execute("""
            SELECT * FROM crawl_sessions 
            WHERE status = 'active'
            ORDER BY timestamp DESC 
            LIMIT 1;
        """)
        
        session = cursor.fetchone()
        if not session:
            print("📂 No saved data found in database")
            db_pool.putconn(conn)
            return None, None, None
        
        session_id = session['id']
        print(f"📂 Loading saved data from session {session_id}...")
        
        # Load content chunks
        cursor.execute("""
            SELECT chunk_text FROM content_chunks 
            WHERE session_id = %s 
            ORDER BY chunk_index;
        """, (session_id,))
        
        chunks = [row['chunk_text'] for row in cursor.fetchall()]
        
        # Load embeddings
        cursor.execute("""
            SELECT embeddings_binary FROM embeddings_data 
            WHERE session_id = %s;
        """, (session_id,))
        
        embeddings_row = cursor.fetchone()
        if not embeddings_row:
            print("❌ No embeddings found")
            db_pool.putconn(conn)
            return None, None, None
        
        embeddings = pickle.loads(embeddings_row['embeddings_binary'])
        
        # Create crawl info
        crawl_info = {
            'timestamp': session['timestamp'].timestamp(),
            'pages_crawled': session['pages_crawled'],
            'pdfs_processed': session['pdfs_processed'],
            'chunks_created': session['chunks_created'],
            'base_url': session['base_url']
        }
        
        cursor.close()
        db_pool.putconn(conn)
        
        print(f"✅ Loaded {len(chunks)} chunks from database session {session_id}")
        return chunks, embeddings, crawl_info
        
    except Exception as e:
        print(f"❌ Database load error: {e}")
        if conn:
            cursor.close()
            db_pool.putconn(conn)
        return None, None, None

def clear_database():
    """Clear all data from database (for rebuild)"""
    try:
        conn = db_pool.getconn()
        cursor = conn.cursor()
        
        # Mark old sessions as inactive instead of deleting
        cursor.execute("UPDATE crawl_sessions SET status = 'inactive';")
        
        # Or completely delete everything (uncomment if you prefer):
        # cursor.execute("DELETE FROM pdf_files;")
        # cursor.execute("DELETE FROM embeddings_data;")
        # cursor.execute("DELETE FROM content_chunks;")
        # cursor.execute("DELETE FROM crawl_sessions;")
        
        conn.commit()
        cursor.close()
        db_pool.putconn(conn)
        
        print("✅ Database cleared successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database clear error: {e}")
        if conn:
            conn.rollback()
            cursor.close()
            db_pool.putconn(conn)
        return False

# ✅ STEP 3: HTML Template with enhanced features
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
    <title>J D Birla Institute - AI Assistant (PostgreSQL)</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .status {
            padding: 20px;
            text-align: center;
            font-weight: bold;
        }
        .status.loading {
            background: #fff3cd;
            color: #856404;
        }
        .status.ready {
            background: #d4edda;
            color: #155724;
        }
        .chat-container {
            height: 400px;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 15px;
            padding: 15px;
            border-radius: 15px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .user-message {
            background: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .bot-message {
            background: white;
            color: #333;
            border: 1px solid #dee2e6;
            line-height: 1.6;
        }
        .bot-message p {
            margin: 10px 0;
        }
        .bot-message ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        .bot-message li {
            margin: 8px 0;
            line-height: 1.5;
        }
        .bot-message strong {
            color: #007bff;
            font-weight: 600;
        }
        .bot-message a {
            color: #007bff;
            text-decoration: underline;
            margin-right: 10px;
        }
        .bot-message a:hover {
            color: #0056b3;
            text-decoration: none;
        }
        .input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #dee2e6;
            display: flex;
            gap: 10px;
        }
        input[type="text"] {
            flex: 1;
            padding: 15px;
            border: 2px solid #dee2e6;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
        }
        input[type="text"]:focus {
            border-color: #007bff;
        }
        button {
            padding: 15px 30px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s;
        }
        button:hover:not(:disabled) {
            background: #0056b3;
        }
        button:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        .rebuild-btn, .update-btn {
            margin: 5px;
            padding: 10px 20px;
            font-size: 14px;
        }
        .rebuild-btn {
            background: #dc3545;
        }
        .update-btn {
            background: #28a745;
        }
        .rebuild-btn:hover:not(:disabled) {
            background: #c82333;
        }
        .update-btn:hover:not(:disabled) {
            background: #218838;
        }
        .db-indicator {
            background: #17a2b8;
            color: white;
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 12px;
            margin-bottom: 10px;
            display: inline-block;
        }
        .suggestions-container {
            margin: 10px 0;
            text-align: center;
        }
        .suggestion-btn {
            margin: 5px;
            padding: 8px 15px;
            background: #f8f9fa;
            color: #007bff;
            border: 1px solid #007bff;
            border-radius: 20px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.3s;
        }
        .suggestion-btn:hover {
            background: #007bff;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 J D Birla Institute - AI Assistant</h1>
            <p>Complete institute knowledge base with PDFs</p>
            <div class="db-indicator">🗄️ PostgreSQL Database</div>
        </div>
        
        <div id="status" class="status loading">
            🔄 System starting up... Please wait.
        </div>
        
        <div style="text-align: center; margin: 10px 0;">
            <button onclick="updateWebsiteData()" id="update-button" class="update-btn" disabled>
                🔄 Update Website Data
            </button>
            <button onclick="rebuildDatabase()" id="rebuild-button" class="rebuild-btn" disabled>
                🗑️ Complete Rebuild
            </button>
        </div>
        
        <div id="chat-container" class="chat-container">
            <div class="message bot-message">
                Hello! I am Edwin, your AI Assistant. Ask me anything — let’s make your JDBI journey smoother from the very first step!
            </div>
        </div>
        
        <div class="input-container">
            <input type="text" id="question-input" placeholder="Ask me anything specific..." disabled>
            <button onclick="askQuestion()" id="ask-button" disabled>Send</button>
        </div>
    </div>

    <script>
        function checkStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const statusDiv = document.getElementById('status');
                    const questionInput = document.getElementById('question-input');
                    const askButton = document.getElementById('ask-button');
                    const rebuildButton = document.getElementById('rebuild-button');
                    const updateButton = document.getElementById('update-button');
                    
                    if (data.ready) {
                        statusDiv.className = 'status ready';
                        const lastUpdated = data.last_updated ? 
                            new Date(data.last_updated * 1000).toLocaleDateString() : 'Unknown';
                        statusDiv.innerHTML = `✅ System ready! ${data.chunks_count} chunks from ${data.pages_crawled} pages + ${data.pdfs_processed} PDFs<br>
                                             <small>Last updated: ${lastUpdated} | Source: ${data.data_source} | 🗄️ PostgreSQL</small>`;
                        questionInput.disabled = false;
                        askButton.disabled = false;
                        rebuildButton.disabled = false;
                        updateButton.disabled = false;
                    } else {
                        statusDiv.innerHTML = `🔄 ${data.message || 'Loading...'}`;
                        setTimeout(checkStatus, 3000);
                    }
                })
                .catch(error => {
                    document.getElementById('status').innerHTML = '❌ Connection error. Retrying...';
                    setTimeout(checkStatus, 5000);
                });
        }

        function updateWebsiteData() {
            if (!confirm('This will update the website data by crawling for new/changed content and PDFs. Continue?')) return;
            
            const statusDiv = document.getElementById('status');
            const updateButton = document.getElementById('update-button');
            const rebuildButton = document.getElementById('rebuild-button');
            
            statusDiv.className = 'status loading';
            statusDiv.innerHTML = '🔄 Updating website data and PDFs... Crawling for new content.';
            updateButton.disabled = true;
            rebuildButton.disabled = true;
            
            fetch('/update', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        statusDiv.innerHTML = '✅ Update started! This may take a few minutes...';
                        setTimeout(checkStatus, 3000);
                    } else {
                        statusDiv.innerHTML = '❌ Update failed: ' + (data.error || 'Unknown error');
                        updateButton.disabled = false;
                        rebuildButton.disabled = false;
                    }
                })
                .catch(error => {
                    statusDiv.innerHTML = '❌ Update error: ' + error.message;
                    updateButton.disabled = false;
                    rebuildButton.disabled = false;
                });
        }

        function rebuildDatabase() {
            if (!confirm('This will completely rebuild the database from scratch, deleting all saved data. Continue?')) return;
            
            const statusDiv = document.getElementById('status');
            const updateButton = document.getElementById('update-button');
            const rebuildButton = document.getElementById('rebuild-button');
            
            statusDiv.className = 'status loading';
            statusDiv.innerHTML = '🔄 Rebuilding complete database... This will take 5-10 minutes.';
            updateButton.disabled = true;
            rebuildButton.disabled = true;
            
            fetch('/rebuild', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        statusDiv.innerHTML = '✅ Rebuild started! Complete refresh in progress...';
                        setTimeout(() => location.reload(), 3000);
                    } else {
                        statusDiv.innerHTML = '❌ Rebuild failed: ' + (data.error || 'Unknown error');
                        updateButton.disabled = false;
                        rebuildButton.disabled = false;
                    }
                })
                .catch(error => {
                    statusDiv.innerHTML = '❌ Rebuild error: ' + error.message;
                    updateButton.disabled = false;
                    rebuildButton.disabled = false;
                });
        }

        function askQuestion() {
            const input = document.getElementById('question-input');
            const question = input.value.trim();
            if (!question) return;
            
            // Remove any existing suggestions
            const existingSuggestions = document.querySelectorAll('.suggestions-container');
            existingSuggestions.forEach(s => s.remove());
            
            addMessage(question, 'user');
            input.value = '';
            
            const loadingMsg = addMessage('🤔 Thinking...', 'bot');
            
            fetch('/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: question})
            })
            .then(response => response.json())
            .then(data => {
                loadingMsg.remove();
                const answerDiv = addMessage(data.error || data.answer, 'bot');
                
                // Add suggestions if available
                if (data.suggestions && data.suggestions.length > 0) {
                    addSuggestions(data.suggestions);
                }
            })
            .catch(error => {
                loadingMsg.remove();
                addMessage('❌ Error: ' + error.message, 'bot');
            });
        }

        function addSuggestions(suggestions) {
            const container = document.getElementById('chat-container');
            const suggestionsDiv = document.createElement('div');
            suggestionsDiv.className = 'suggestions-container';
            
            suggestions.forEach(suggestion => {
                const button = document.createElement('button');
                button.textContent = suggestion;
                button.className = 'suggestion-btn';
                button.onclick = () => {
                    document.getElementById('question-input').value = suggestion;
                    askQuestion();
                };
                suggestionsDiv.appendChild(button);
            });
            
            container.appendChild(suggestionsDiv);
            container.scrollTop = container.scrollHeight;
        }

        function addMessage(message, type) {
            const container = document.getElementById('chat-container');
            const div = document.createElement('div');
            div.className = `message ${type}-message`;
            
            if (type === 'bot') {
                div.innerHTML = formatBotMessage(message);
            } else {
                div.textContent = message;
            }
            
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
            return div;
        }

        function formatBotMessage(message) {
            let formatted = message
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\n\n/g, '</p><p>')
                .replace(/\n/g, '<br>');
            
            // Convert markdown links to HTML links
            formatted = formatted.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
            
            if (!formatted.includes('<p>')) {
                formatted = '<p>' + formatted + '</p>';
            }
            
            formatted = formatted.replace(/<p><\/p>/g, '');
            formatted = formatted.replace(/<p>\s*<\/p>/g, '');
            
            return formatted;
        }

        document.getElementById('question-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') askQuestion();
        });

        checkStatus();
    </script>
</body>
</html>
"""

# ✅ STEP 4: PDF EXTRACTION FUNCTIONS (same as before)
def extract_pdf_text(pdf_path):
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text_content = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():  # Only add non-empty pages
                # Clean the text to remove NUL characters and other problematic chars
                text = clean_text_for_db(text)
                text_content.append(text)
        
        doc.close()
        full_text = "\n".join(text_content)
        return full_text if full_text.strip() else None
        
    except Exception as e:
        print(f"❌ Error extracting PDF {pdf_path}: {e}")
        return None

def clean_text_for_db(text):
    """Clean text to remove problematic characters for PostgreSQL"""
    if not text:
        return text
    
    # Remove NUL characters (0x00) that cause PostgreSQL errors
    text = text.replace('\x00', '')
    
    # Remove other control characters that might cause issues
    import re
    # Keep only printable characters, newlines, tabs, and carriage returns
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Replace multiple newlines with double newline
    text = re.sub(r'\n+', '\n\n', text)
    
    return text.strip()

def download_pdf(url, filename):
    """Download PDF from URL"""
    try:
        response = requests.get(url, timeout=30, stream=True)
        if response.status_code == 200:
            filepath = os.path.join(PDF_DIR, filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Downloaded PDF: {filename}")
            return filepath
        else:
            print(f"❌ Failed to download PDF: {url} (Status: {response.status_code})")
            return None
    except Exception as e:
        print(f"❌ Error downloading PDF {url}: {e}")
        return None

# ✅ STEP 5: COMPLETE WEBSITE CRAWLER WITH PDF EXTRACTION (same crawling logic)
def setup_rag_system(force_rebuild=False):
    global model, index, all_chunks, client, system_ready
    
    try:
        print("🚀 Setting up COMPLETE JDBI website scraping system with PostgreSQL...")
        
        # Initialize database
        if not init_database():
            print("❌ Database initialization failed!")
            return
        
        # Initialize Claude
        client = anthropic.Anthropic(api_key="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")
        print("✅ Claude initialized")
        
        # Load embedding model
        print("🧠 Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model loaded")
        
        # Try loading saved data from database first
        if not force_rebuild:
            chunks, embeddings, crawl_info = load_data_from_db()
            if chunks and embeddings is not None:
                all_chunks = chunks
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                system_ready = True
                print("🎉 System ready from saved PostgreSQL data!")
                return
        
        print("🌐 Starting COMPLETE JDBI website crawl with PDF extraction...")
        
        # COMPLETE WEBSITE CRAWLER WITH PDF DETECTION (same as before)
        def crawl_entire_website_with_pdfs():
            # Same crawling logic as original code...
            base_url = "https://www.jdbikolkata.in/"
            domain = urlparse(base_url).netloc
            
            visited_urls = set()
            page_data = []
            pdf_links = set()
            url_queue = deque([base_url])
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            print("🔍 PHASE 1: Discovering all URLs and PDFs...")
            
            sitemap_urls = discover_from_sitemaps(base_url, headers)
            if sitemap_urls:
                print(f"📄 Found {len(sitemap_urls)} URLs from sitemaps")
                url_queue.extend(sitemap_urls)
            
            max_pages = 1000
            processed_count = 0
            
            while url_queue and processed_count < max_pages:
                current_url = url_queue.popleft()
                
                if current_url in visited_urls:
                    continue
                
                if urlparse(current_url).netloc != domain:
                    continue
                
                visited_urls.add(current_url)
                processed_count += 1
                
                try:
                    print(f"🔍 [{processed_count}/{max_pages}] Crawling: {current_url}")
                    
                    response = requests.get(current_url, timeout=20, headers=headers)
                    
                    if response.status_code != 200:
                        print(f"❌ Status {response.status_code} for {current_url}")
                        continue
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    text_content = extract_comprehensive_text(soup, current_url)
                    
                    if text_content and len(text_content) > 100:
                        page_data.append({
                            'url': current_url,
                            'content': text_content,
                            'title': soup.find('title').get_text() if soup.find('title') else 'No Title'
                        })
                        print(f"✅ Extracted content from: {current_url}")
                    
                    new_urls, new_pdfs = discover_all_links_and_pdfs(soup, current_url, domain)
                    
                    for url in new_urls:
                        if url not in visited_urls and url not in url_queue:
                            url_queue.append(url)
                    
                    pdf_links.update(new_pdfs)
                    
                    time.sleep(0.5)
                    
                except requests.exceptions.Timeout:
                    print(f"⏰ Timeout for {current_url}")
                except requests.exceptions.RequestException as e:
                    print(f"❌ Request error for {current_url}: {e}")
                except Exception as e:
                    print(f"❌ Error processing {current_url}: {e}")
            
            print(f"🎯 COMPLETE CRAWL FINISHED!")
            print(f"📊 Total pages crawled: {len(page_data)}")
            print(f"🔗 Total URLs discovered: {len(visited_urls)}")
            print(f"📄 Total PDFs found: {len(pdf_links)}")
            
            return page_data, pdf_links
        
        # Execute complete crawl
        page_data, pdf_links = crawl_entire_website_with_pdfs()
        
        # Step 3: Download and extract PDFs
        print("📄 PHASE 2: Downloading and extracting PDFs...")
        pdf_data = []
        pdfs_processed = 0
        
        for pdf_url in pdf_links:
            try:
                filename = os.path.basename(urlparse(pdf_url).path)
                if not filename.endswith('.pdf'):
                    filename = f"document_{pdfs_processed + 1}.pdf"
                
                pdf_path = download_pdf(pdf_url, filename)
                
                if pdf_path:
                    pdf_text = extract_pdf_text(pdf_path)
                    
                    if pdf_text:
                        pdf_data.append({
                            'url': pdf_url,
                            'filename': filename,
                            'filepath': pdf_path,
                            'content': pdf_text,
                            'title': filename.replace('.pdf', '').replace('_', ' ').title()
                        })
                        pdfs_processed += 1
                        print(f"✅ Processed PDF: {filename}")
                    else:
                        print(f"⚠️ No text extracted from: {filename}")
                
            except Exception as e:
                print(f"❌ Error processing PDF {pdf_url}: {e}")
        
        print(f"✅ Successfully processed {pdfs_processed} PDFs")
        
        # Step 4: Create comprehensive chunks from both web pages and PDFs
        all_chunks = create_comprehensive_chunks_with_pdfs(page_data, pdf_data)
        print(f"✅ Created {len(all_chunks)} comprehensive chunks from website + PDFs")
        
        if not all_chunks:
            print("❌ No content found! Using fallback...")
            all_chunks = ["No content was successfully crawled from the website or PDFs."]
        
        # Create embeddings
        print("🧠 Creating embeddings for all content...")
        embeddings = model.encode(all_chunks, show_progress_bar=True)
        
        # Build FAISS index
        print("🔍 Building comprehensive search index...")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        # Save everything to PostgreSQL
        crawl_info = {
            'timestamp': time.time(),
            'pages_crawled': len(page_data),
            'pdfs_processed': pdfs_processed,
            'chunks_created': len(all_chunks),
            'base_url': "https://www.jdbikolkata.in/"
        }
        
        save_data_to_db(all_chunks, embeddings, crawl_info, pdf_data)
        
        system_ready = True
        print("🎉 COMPLETE JDBI website + PDF RAG system ready with PostgreSQL!")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        system_ready = False

def discover_from_sitemaps(base_url, headers):
    """Try to discover URLs from sitemap.xml"""
    sitemap_urls = []
    possible_sitemaps = [
        urljoin(base_url, '/sitemap.xml'),
        urljoin(base_url, '/sitemap_index.xml'),
        urljoin(base_url, '/robots.txt')
    ]
    
    for sitemap_url in possible_sitemaps:
        try:
            response = requests.get(sitemap_url, headers=headers, timeout=10)
            if response.status_code == 200:
                if 'sitemap' in sitemap_url:
                    soup = BeautifulSoup(response.text, 'xml')
                    urls = soup.find_all('loc')
                    sitemap_urls.extend([url.get_text() for url in urls])
                elif 'robots.txt' in sitemap_url:
                    for line in response.text.split('\n'):
                        if line.startswith('Sitemap:'):
                            sitemap_url = line.replace('Sitemap:', '').strip()
                            sitemap_urls.extend(discover_from_sitemaps(sitemap_url, headers))
        except:
            continue
    
    return list(set(sitemap_urls))

def extract_comprehensive_text(soup, url):
    """Extract ALL meaningful text from a webpage"""
    
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.extract()
    
    text_elements = []
    
    for tag in soup.find_all(['title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        text = tag.get_text(strip=True)
        if text and len(text) > 3:
            text_elements.append(f"HEADING: {text}")
    
    for tag in soup.find_all(['p', 'li', 'td', 'th', 'div', 'span', 'article', 'section']):
        text = tag.get_text(strip=True)
        if text and len(text) > 20:
            text_elements.append(text)
    
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc:
        text_elements.append(f"META: {meta_desc.get('content', '')}")
    
    for img in soup.find_all('img', alt=True):
        alt_text = img.get('alt', '').strip()
        if alt_text and len(alt_text) > 5:
            text_elements.append(f"IMAGE: {alt_text}")
    
    full_text = "\n".join(text_elements)
    full_text = re.sub(r'\s+', ' ', full_text)
    full_text = re.sub(r'\n+', '\n', full_text)
    
    # Clean text for database storage
    full_text = clean_text_for_db(full_text)
    
    return full_text.strip()

def discover_all_links_and_pdfs(soup, current_url, domain):
    """Find ALL internal links AND PDF links on a page"""
    web_links = set()
    pdf_links = set()
    
    for link in soup.find_all('a', href=True):
        href = link['href']
        absolute_url = urljoin(current_url, href)
        parsed = urlparse(absolute_url)
        
        if absolute_url.lower().endswith('.pdf'):
            pdf_links.add(absolute_url)
            print(f"📄 Found PDF: {absolute_url}")
        elif parsed.netloc == domain:
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                clean_url += f"?{parsed.query}"
            
            if not any(clean_url.lower().endswith(ext) for ext in ['.jpg', '.png', '.gif', '.zip', '.doc', '.docx']):
                web_links.add(clean_url)
    
    script_tags = soup.find_all('script')
    for script in script_tags:
        if script.string:
            pdf_matches = re.findall(r'["\']([^"\']*\.pdf)["\']', script.string)
            for match in pdf_matches:
                absolute_url = urljoin(current_url, match)
                if urlparse(absolute_url).netloc == domain:
                    pdf_links.add(absolute_url)
    
    return list(web_links), list(pdf_links)

def create_comprehensive_chunks_with_pdfs(page_data, pdf_data, max_tokens=600):
    """Create chunks from both web pages and PDF content"""
    chunks = []
    
    print("📝 Creating chunks from web pages...")
    for page in page_data:
        url = page['url']
        content = page['content']
        title = page['title']
        
        try:
            sentences = sent_tokenize(content)
            current_chunk = [f"SOURCE: {title} (WEB: {url})"]
            current_length = len(' '.join(current_chunk).split())
            
            for sentence in sentences:
                sentence_length = len(sentence.split())
                
                if current_length + sentence_length <= max_tokens:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                else:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [f"SOURCE: {title} (WEB: {url})", sentence]
                    current_length = len(' '.join(current_chunk).split())
            
            if current_chunk and len(current_chunk) > 1:
                chunks.append(" ".join(current_chunk))
                
        except Exception as e:
            print(f"❌ Error chunking web page {url}: {e}")
            words = content.split()
            for i in range(0, len(words), max_tokens):
                chunk = " ".join(words[i:i+max_tokens])
                if len(chunk.strip()) > 50:
                    chunks.append(f"SOURCE: {title} (WEB) - {chunk}")
    
    print("📄 Creating chunks from PDFs...")
    for pdf in pdf_data:
        url = pdf['url']
        content = pdf['content']
        title = pdf['title']
        filename = pdf['filename']
        
        try:
            sentences = sent_tokenize(content)
            current_chunk = [f"SOURCE: {title} (PDF: {filename})"]
            current_length = len(' '.join(current_chunk).split())
            
            for sentence in sentences:
                sentence_length = len(sentence.split())
                
                if current_length + sentence_length <= max_tokens:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                else:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [f"SOURCE: {title} (PDF: {filename})", sentence]
                    current_length = len(' '.join(current_chunk).split())
            
            if current_chunk and len(current_chunk) > 1:
                chunks.append(" ".join(current_chunk))
                
        except Exception as e:
            print(f"❌ Error chunking PDF {filename}: {e}")
            words = content.split()
            for i in range(0, len(words), max_tokens):
                chunk = " ".join(words[i:i+max_tokens])
                if len(chunk.strip()) > 50:
                    chunks.append(f"SOURCE: {title} (PDF: {filename}) - {chunk}")
    
    unique_chunks = []
    seen = set()
    for chunk in chunks:
        if chunk not in seen:
            unique_chunks.append(chunk)
            seen.add(chunk)
    
    print(f"📊 Created {len(chunks)} total chunks, {len(unique_chunks)} unique chunks")
    return unique_chunks

# ✅ STEP 6: Enhanced Helper Functions with Links, History & Suggestions
def search_chunks(query, k=3):
    if not (model and index and all_chunks):
        return []
    try:
        q_embed = model.encode([query])
        D, I = index.search(q_embed, k)
        return [all_chunks[i] for i in I[0] if i < len(all_chunks)]
    except:
        return []

def ask_claude_with_history(context_question, original_question):
    if not client:
        return "System not ready"
    
    try:
        context_chunks = search_chunks(original_question)
        context = "\n\n".join(context_chunks)
        
        # Extract source URLs from chunks for links
        source_urls = []
        for chunk in context_chunks:
            if 'SOURCE:' in chunk and '(WEB:' in chunk:
                try:
                    url_part = chunk.split('(WEB:')[1].split(')')[0].strip()
                    if url_part and url_part.startswith('http'):
                        source_urls.append(url_part)
                except:
                    continue
        
        unique_urls = list(dict.fromkeys(source_urls))[:3]
        
        prompt = f"""You are an AI assistant for the website. Answer questions directly and conversationally.

Context from website and documents:
{context}

Question with context: {context_question}

INSTRUCTIONS:
1. Give PRECISE, TO-THE-POINT answers only
2. Answer EXACTLY what is asked - don't add extra information
3. Use a natural, conversational tone
4. If this question relates to previous context, acknowledge it naturally
5. Keep responses SHORT and focused
6. Maximum 2-3 sentences for simple questions

Answer naturally and directly."""

        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer = response.content[0].text
        formatted_answer = format_answer(answer)
        
        if unique_urls:
            links_text = "\n\n**Helpful Links:**\n"
            for i, url in enumerate(unique_urls, 1):
                url_name = url.replace('https://', '').replace('http://', '').split('/')[0]
                if len(url.split('/')) > 3:
                    page_name = url.split('/')[-1] or url.split('/')[-2]
                    if page_name and page_name != url_name:
                        url_name = f"{url_name} - {page_name[:20]}"
                
                links_text += f"{i}. [{url_name}]({url})\n"
            
            formatted_answer += links_text
        
        return formatted_answer
        
    except Exception as e:
        return f"Error: {str(e)}"

def generate_suggestions(question, answer):
    """Generate follow-up question suggestions"""
    try:
        suggestions_prompt = f"""Based on this Q&A, suggest 3 short follow-up questions (max 6 words each):

Question: {question}
Answer: {answer}

Generate 3 relevant follow-up questions as a simple list:"""

        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=100,
            messages=[{"role": "user", "content": suggestions_prompt}]
        )
        
        suggestions_text = response.content[0].text
        suggestions = [s.strip().replace('- ', '').replace('• ', '') for s in suggestions_text.split('\n') if s.strip()][:3]
        return suggestions
        
    except:
        return ["Tell me more", "What about fees?", "How to apply?"]

def format_answer(text):
    """Format answer to be conversational and concise"""
    text = text.strip()
    
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith(('- ', '* ', '• ')):
            line = line[2:].strip()
        elif line.startswith(('1. ', '2. ', '3. ', '4. ', '5. ')):
            line = line[3:].strip()
        
        line = line.replace('**', '')
        
        if line:
            formatted_lines.append(line)
    
    result = ' '.join(formatted_lines)
    
    while '  ' in result:
        result = result.replace('  ', ' ')
    
    result = result.replace('. ', '. ')
    
    return result.strip()

# ✅ STEP 7: Routes
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    crawl_info = {}
    
    try:
        conn = db_pool.getconn()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT * FROM crawl_sessions 
            WHERE status = 'active'
            ORDER BY timestamp DESC 
            LIMIT 1;
        """)
        
        session = cursor.fetchone()
        if session:
            crawl_info = {
                'timestamp': session['timestamp'].timestamp(),
                'pages_crawled': session['pages_crawled'],
                'pdfs_processed': session['pdfs_processed'],
                'chunks_created': session['chunks_created'],
                'base_url': session['base_url']
            }
        
        cursor.close()
        db_pool.putconn(conn)
        
    except Exception as e:
        print(f"❌ Error getting status from database: {e}")
    
    return jsonify({
        "ready": system_ready,
        "chunks_count": len(all_chunks),
        "pages_crawled": crawl_info.get('pages_crawled', 0),
        "pdfs_processed": crawl_info.get('pdfs_processed', 0),
        "last_updated": crawl_info.get('timestamp', None),
        "data_source": "PostgreSQL database" if crawl_info else "fresh crawl",
        "message": "Complete J D Birla Institute system ready with PostgreSQL!" if system_ready else "Loading complete institute data from PostgreSQL..."
    })

@app.route('/update', methods=['POST'])
def update_website():
    """Update website data by crawling for new/changed content and PDFs"""
    global system_ready
    try:
        print("🔄 Starting J D Birla Institute data update with PostgreSQL...")
        
        system_ready = False
        
        def update_process():
            global system_ready
            try:
                print("📊 Checking for institute updates...")
                
                old_timestamp = None
                try:
                    conn = db_pool.getconn()
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT timestamp FROM crawl_sessions 
                        WHERE status = 'active'
                        ORDER BY timestamp DESC 
                        LIMIT 1;
                    """)
                    result = cursor.fetchone()
                    if result:
                        old_timestamp = result[0].timestamp()
                        print(f"📅 Last update: {time.ctime(old_timestamp)}")
                    cursor.close()
                    db_pool.putconn(conn)
                except:
                    pass
                
                print("🌐 Crawling institute website and PDFs for updates...")
                setup_rag_system(force_rebuild=True)
                
                if system_ready:
                    print("✅ J D Birla Institute data update completed successfully!")
                else:
                    print("❌ J D Birla Institute data update failed!")
                    
            except Exception as e:
                print(f"❌ Update process failed: {e}")
                import traceback
                traceback.print_exc()
        
        update_thread = threading.Thread(target=update_process, daemon=True)
        update_thread.start()
        
        return jsonify({"success": True, "message": "Institute update started"})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/rebuild', methods=['POST'])
def rebuild():
    """Complete rebuild - clears database and starts fresh"""
    global system_ready
    try:
        print("🗑️ Starting complete J D Birla Institute database rebuild...")
        
        clear_database()
        
        if os.path.exists(PDF_DIR):
            for pdf_file in os.listdir(PDF_DIR):
                pdf_path = os.path.join(PDF_DIR, pdf_file)
                if os.path.isfile(pdf_path):
                    os.remove(pdf_path)
                    print(f"🗑️ Deleted PDF: {pdf_file}")
        
        system_ready = False
        
        def rebuild_process():
            print("🚀 Starting complete institute rebuild from scratch...")
            setup_rag_system(force_rebuild=True)
            if system_ready:
                print("✅ Complete J D Birla Institute rebuild finished successfully!")
            else:
                print("❌ Complete J D Birla Institute rebuild failed!")
        
        rebuild_thread = threading.Thread(target=rebuild_process, daemon=True)
        rebuild_thread.start()
        
        return jsonify({"success": True, "message": "Complete institute rebuild started"})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/ask', methods=['POST'])
def ask():
    global chat_history
    
    if not system_ready:
        return jsonify({"error": "System still loading complete J D Birla Institute data from PostgreSQL"})
    
    data = request.get_json()
    question = data.get('question', '').strip()
    if not question:
        return jsonify({"error": "No question provided"})
    
    # Add context from chat history for related questions
    context_question = question
    if len(chat_history) > 0:
        recent_context = " ".join([f"Previous: {item['question']} Answer: {item['answer'][:100]}" for item in chat_history[-2:]])
        context_question = f"Context: {recent_context}\n\nCurrent question: {question}"
    
    answer = ask_claude_with_history(context_question, question)
    
    # Store in chat history (keep last 5 exchanges)
    chat_history.append({"question": question, "answer": answer})
    if len(chat_history) > 5:
        chat_history.pop(0)
    
    # Generate suggested follow-up questions
    suggestions = generate_suggestions(question, answer)
    
    return jsonify({"answer": answer, "suggestions": suggestions})

# ✅ STEP 8: Database Analytics Routes (Optional - for monitoring)
@app.route('/stats')
def database_stats():
    """Get database statistics"""
    try:
        conn = db_pool.getconn()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("SELECT COUNT(*) as total_sessions FROM crawl_sessions;")
        total_sessions = cursor.fetchone()['total_sessions']
        
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT session_id) as active_sessions,
                SUM(CASE WHEN source_type = 'web' THEN 1 ELSE 0 END) as web_chunks,
                SUM(CASE WHEN source_type = 'pdf' THEN 1 ELSE 0 END) as pdf_chunks,
                COUNT(*) as total_chunks
            FROM content_chunks 
            WHERE session_id IN (
                SELECT id FROM crawl_sessions WHERE status = 'active'
            );
        """)
        chunk_stats = cursor.fetchone()
        
        cursor.execute("""
            SELECT COUNT(*) as total_pdfs 
            FROM pdf_files 
            WHERE session_id IN (
                SELECT id FROM crawl_sessions WHERE status = 'active'
            );
        """)
        pdf_count = cursor.fetchone()['total_pdfs']
        
        cursor.close()
        db_pool.putconn(conn)
        
        return jsonify({
            "total_sessions": total_sessions,
            "active_sessions": chunk_stats['active_sessions'],
            "web_chunks": chunk_stats['web_chunks'],
            "pdf_chunks": chunk_stats['pdf_chunks'],
            "total_chunks": chunk_stats['total_chunks'],
            "total_pdfs": pdf_count
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

# ✅ STEP 9: Main
if __name__ == "__main__":
    print("🚀 Starting J D Birla Institute Complete AI Assistant with PostgreSQL...")
    print("🎓 This will crawl EVERY PAGE and PDF of the institute website")
    print("🗄️ All data will be stored in PostgreSQL database")
    print("📱 Server will be available at: http://localhost:5000")
    print("\n📋 POSTGRESQL CONFIGURATION:")
    print(f"   Host: {DB_CONFIG['host']}")
    print(f"   Database: {DB_CONFIG['database']}")
    print(f"   User: {DB_CONFIG['user']}")
    print(f"   Port: {DB_CONFIG['port']}")
    print("\n⚠️  IMPORTANT: Make sure your PostgreSQL server is running!")
    print("⚠️  Update DB_CONFIG at the top of this file with your credentials!")
    print("⏳ Setting up complete institute scraping system...")
    
    setup_thread = threading.Thread(target=setup_rag_system, daemon=True)
    setup_thread.start()
    
    time.sleep(1)
    
    print("🌟 Flask server starting...")
    print("🎯 Open your browser and go to: http://localhost:5000")
    print("📊 The system will crawl the ENTIRE institute website + PDFs on first run")
    print("🔄 Complete crawling may take 5-15 minutes")
    print("⚡ Subsequent runs will load instantly from PostgreSQL")
    print("🎓 NEW ENHANCED FEATURES:")
    print("   • Supporting links extracted from content")
    print("   • Chat history for contextual conversations")
    print("   • Suggested follow-up questions after each answer")
    print("   • PostgreSQL database storage")
    print("   • Discovers URLs from sitemaps")
    print("   • Crawls every institute page")
    print("   • Downloads and extracts ALL PDFs")
    print("   • Creates comprehensive knowledge base")
    print("   • Smart update vs rebuild options")
    print("   • Database analytics at /stats")
    print("\n" + "="*60)
    
    try:
        app.run(host='127.0.0.1', port=5000, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        if db_pool:
            db_pool.closeall()
    except OSError as e:
        if "Address already in use" in str(e):
            print("❌ Port 5000 is already in use!")
            print("💡 Try: python -c \"import os; os.system('lsof -ti:5000 | xargs kill -9')\"")
            print("💡 Or change the port in the code")
        else:
            print(f"❌ Server error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if db_pool:
            db_pool.closeall()
            print("🗄️ Database connections closed")
