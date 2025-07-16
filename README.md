# **README.md - Agentic Research Navigator**

> **A modern AI-powered document analysis system that learns and evolves**

Research Navigator transforms how you interact with documents by combining Retrieval-Augmented Generation (RAG) with cutting-edge AI technologies. Built from the ground up to demonstrate best practices in AI development.

---

## What It Does

**Ask questions about your documents and get intelligent answers.**

Upload PDFs, Word documents, or text files, and Research Navigator will:
- Understand your document content at a deep level
- Answer complex questions with precise citations
- Provide context-aware responses based on your specific documents
- Learn and improve with each interaction

---

## Built With Modern AI Stack

**Local-First Architecture**
- **LangChain & LangGraph** - AI workflow orchestration
- **Ollama** - Local language models (privacy-focused)
- **ChromaDB** - Intelligent vector storage
- **Sentence Transformers** - Advanced document understanding

**Developer-Friendly**
- **Python 3.11+** with full type safety
- **Pydantic** for robust data validation
- **Streamlit** for interactive demonstrations
- **Comprehensive testing** with pytest

---

## Quick Start

### System Requirements
- **macOS** (M4 optimized) or Linux
- **Python 3.11+**
- **4GB+ RAM** recommended

### Installation

```bash
# Clone and setup
git clone https://github.com/yourusername/research-navigator.git
cd research-navigator

# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Local AI
```bash
# Install Ollama (one-time setup)
curl -fsSL https://ollama.ai/install.sh | sh

# Download AI model
ollama pull llama3.2
```

### Launch Demo
```bash
streamlit run streamlit_app/app.py
```

---

## Project Architecture

```
research-navigator/
├── src/core/          # Document processing & AI components
├── src/rag/           # Question-answering engine
├── src/models/        # Data structures & validation
├── tests/             # Comprehensive test suite
├── streamlit_app/     # Interactive demo interface
└── data/              # Your documents & AI knowledge base
```

---

## Learning Journey

This project documents the complete process of building a production-ready AI system. Each version introduces new capabilities while maintaining clean, understandable code.

**Current Focus**: Establishing robust RAG foundations with local AI infrastructure.

**Development Philosophy**: Learn by building real solutions, one feature at a time.

---

## Why Research Navigator?

**Privacy First**: Your documents never leave your machine  
**Performance**: Optimized for Apple Silicon and modern hardware  
**Educational**: Clean code that teaches AI development patterns  
**Practical**: Solves real document analysis problems  

---

## Development Status

**Version**: `0.1.0-dev`  
**Phase**: Foundation & Core RAG Implementation

The system follows incremental development where each phase delivers a complete, working solution before adding complexity.

---

## License

MIT License - Build upon this project freely

---

**Ready to explore intelligent document analysis?** Follow the Quick Start guide above or dive into the code to see how modern AI systems are built.