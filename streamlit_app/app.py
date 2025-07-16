import streamlit as st
import sys
import os
from datetime import datetime
import time

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.research_navigator import ResearchNavigator
from src.config import Config

# Configure Streamlit page
st.set_page_config(
    page_title="Research Navigator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        max-width: 800px;
    }
    
    .user-message {
        background-color: #f0f2f6;
        margin-left: auto;
        margin-right: 0;
        text-align: right;
    }
    
    .assistant-message {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        margin-left: 0;
        margin-right: auto;
    }
    
    .research-status {
        background-color: #e8f4fd;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    
    .error-message {
        background-color: #ffe6e6;
        border-left: 4px solid #ff4444;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    
    .success-message {
        background-color: #e6ffe6;
        border-left: 4px solid #44ff44;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 1rem;
        border-top: 1px solid #e0e0e0;
        z-index: 1000;
    }
    
    .main-content {
        padding-bottom: 120px;
    }
    
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    
    .example-prompts {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .example-prompt {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 0.9rem;
    }
    
    .example-prompt:hover {
        background-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'navigator' not in st.session_state:
    st.session_state.navigator = None
if 'research_topics' not in st.session_state:
    st.session_state.research_topics = set()
if 'is_researching' not in st.session_state:
    st.session_state.is_researching = False

def initialize_navigator():
    """Initialize the research navigator"""
    if st.session_state.navigator is None:
        with st.spinner("Initializing Research Navigator..."):
            config = Config(
                paper_cache_days=30,
                refresh_threshold_days=7,
                max_papers_per_topic=50
            )
            st.session_state.navigator = ResearchNavigator(config)
        st.success("Research Navigator initialized successfully!")

def add_message(role, content, message_type="normal"):
    """Add a message to the chat history"""
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "type": message_type,
        "timestamp": datetime.now()
    })

def display_message(message):
    """Display a single message with appropriate styling"""
    role = message["role"]
    content = message["content"]
    msg_type = message.get("type", "normal")
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
    
    elif role == "assistant":
        if msg_type == "research":
            st.markdown(f"""
            <div class="research-status">
                <strong>Research Update:</strong> {content}
            </div>
            """, unsafe_allow_html=True)
        elif msg_type == "error":
            st.markdown(f"""
            <div class="error-message">
                <strong>Error:</strong> {content}
            </div>
            """, unsafe_allow_html=True)
        elif msg_type == "success":
            st.markdown(f"""
            <div class="success-message">
                <strong>Success:</strong> {content}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>Research Assistant:</strong><br>{content}
            </div>
            """, unsafe_allow_html=True)

def is_research_command(text):
    """Check if the input is a research command"""
    research_keywords = [
        "research", "download papers", "find papers", "study", 
        "investigate", "explore topic", "papers about", "literature on"
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in research_keywords)

def extract_research_topic(text):
    """Extract research topic from user input"""
    # Simple extraction - in practice, you might want more sophisticated NLP
    text_lower = text.lower()
    
    # Remove common research command phrases
    for phrase in ["research", "download papers about", "find papers on", "study", "investigate", "explore"]:
        text_lower = text_lower.replace(phrase, "").strip()
    
    # Remove question words
    for word in ["what", "how", "why", "when", "where", "papers", "about", "on"]:
        text_lower = text_lower.replace(word, "").strip()
    
    return text_lower.strip() or text.strip()

def process_research_request(topic):
    """Process a research request"""
    st.session_state.is_researching = True
    
    try:
        # Add research status message
        add_message("assistant", f"Starting research on: {topic}", "research")
        
        # Create progress placeholder
        progress_placeholder = st.empty()
        
        with progress_placeholder.container():
            st.markdown(f"""
            <div class="research-status">
                <strong>Research in Progress:</strong> Searching and downloading papers on "{topic}"...
            </div>
            """, unsafe_allow_html=True)
        
        # Perform research
        result = st.session_state.navigator.research_topic(topic, max_papers=5)
        
        # Clear progress and add success message
        progress_placeholder.empty()
        add_message("assistant", result, "success")
        st.session_state.research_topics.add(topic)
        
        # Add follow-up suggestions
        suggestions = [
            f"How does {topic} work?",
            f"What are the latest developments in {topic}?",
            f"What are the applications of {topic}?",
            f"What are the challenges in {topic}?"
        ]
        
        suggestion_text = "You can now ask questions like:\n" + "\n".join([f"• {s}" for s in suggestions])
        add_message("assistant", suggestion_text, "normal")
        
    except Exception as e:
        add_message("assistant", f"Research failed: {str(e)}", "error")
    
    finally:
        st.session_state.is_researching = False

def process_question(question):
    """Process a regular question"""
    try:
        if not st.session_state.research_topics:
            add_message("assistant", 
                       "I haven't researched any topics yet. Please start by asking me to research a topic like 'research machine learning' or 'find papers on neural networks'.", 
                       "normal")
            return
        
        # Show thinking status
        with st.spinner("Thinking..."):
            answer = st.session_state.navigator.ask_question(question)
        
        add_message("assistant", answer, "normal")
        
    except Exception as e:
        add_message("assistant", f"Failed to answer question: {str(e)}", "error")

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Research Navigator</h1>
        <p>AI-powered research assistant that downloads and analyzes academic papers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize navigator
    initialize_navigator()
    
    # Main content area
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Display example prompts if no messages
    if not st.session_state.messages:
        st.markdown("### Getting Started")
        st.markdown("Try one of these example prompts:")
        
        examples = [
            "Research transformer attention mechanisms",
            "Find papers on reinforcement learning",
            "Study deep learning optimization",
            "Investigate neural network architectures",
            "Research natural language processing"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}"):
                    # Add user message and process directly
                    add_message("user", example)
                    topic = extract_research_topic(example)
                    process_research_request(topic)
                    st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        display_message(message)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area (fixed at bottom)
    with st.container():
        st.markdown("---")
        
        # Create columns for input and button
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Use a counter to reset the input field
            if 'input_counter' not in st.session_state:
                st.session_state.input_counter = 0
            
            user_input = st.text_input(
                "Message Research Navigator...",
                key=f"chat_input_{st.session_state.input_counter}",
                placeholder="Ask me to research a topic or ask questions about researched papers",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("Send", type="primary", disabled=st.session_state.is_researching)
    
    # Process input
    if (send_button or user_input) and user_input and not st.session_state.is_researching:
        # Add user message
        add_message("user", user_input)
        
        # Determine if this is a research request or question
        if is_research_command(user_input):
            topic = extract_research_topic(user_input)
            process_research_request(topic)
        else:
            process_question(user_input)
        
        # Clear input by incrementing counter (creates new widget)
        st.session_state.input_counter += 1
        st.rerun()
    
    # Sidebar with research status
    with st.sidebar:
        st.header("Research Status")
        
        if st.session_state.research_topics:
            st.success(f"Researched Topics: {len(st.session_state.research_topics)}")
            for topic in st.session_state.research_topics:
                st.write(f"• {topic}")
        else:
            st.info("No topics researched yet")
        
        st.header("How to Use")
        st.markdown("""
        **Research Mode:**
        - "Research [topic]"
        - "Find papers on [topic]"
        - "Study [topic]"
        
        **Question Mode:**
        - Ask questions about researched topics
        - Get answers based on downloaded papers
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()