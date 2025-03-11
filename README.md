# Suspicious Account Activity and Billing Support Chatbot

Internal chatbot for support agents to quickly access information about suspicious account activity and billing procedures.

## 🏗️ Architecture Overview

This solution uses a Retrieval-Augmented Generation (RAG) approach to create a context-aware chatbot that provides accurate responses based on the provided PDF documentation. The architecture includes:

1. **Document Ingestion Pipeline**: Extracts text and images from PDFs, chunks content, and creates vector embeddings for semantic search.
2. **Vector Database**: Stores embeddings for efficient similarity search.
3. **Conversational AI**: LLM-based chatbot with context management and conversation memory.
4. **Web Interface**: Simple and intuitive UI for support agents.
5. **Evaluation System**: Measures accuracy, response time, and relevance metrics.

### Architecture Diagram

```
┌─────────────────┐     ┌────────────────┐     ┌───────────────────┐
│                 │     │                │     │                   │
│  PDF Documents  │─────▶  Text & Image  │─────▶  Vector Database  │
│                 │     │  Extraction    │     │    (FAISS)        │
└─────────────────┘     └────────────────┘     └─────────┬─────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌────────────────┐     ┌───────────────────┐
│                 │     │                │     │                   │
│   Web Frontend  │◀────▶   Flask API    │◀────▶  LLM-based Chat   │
│                 │     │                │     │  with Context     │
└─────────────────┘     └────────────────┘     └───────────────────┘
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key
- PDF documentation files

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/account-billing-chatbot.git
   cd account-billing-chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env file with your OpenAI API key and other settings
   ```

5. Place your PDF files in the `data/` directory.

### Usage

1. Ingest your PDF documents to create the vector database:
   ```bash
   python src/ingest.py
   ```

2. Run the Flask web application:
   ```bash
   python app/app.py
   ```

3. Access the chatbot interface at http://localhost:5000

## 🔍 Evaluation

The system includes a comprehensive evaluation module for assessing performance against predefined test cases. Run the evaluation with:

```bash
python src/evaluate.py
```

The evaluation results, including accuracy, precision, recall, F1-score, and response times, will be saved in the `evaluation_results/` directory.

## 📦 Deployment

The project includes a Dockerfile and a deployment script for Google Cloud Run:

```bash
./deploy.sh
```

This will build and deploy the application to Google Cloud Run, providing a publicly accessible URL to the chatbot.

## 🧪 Testing

You can run the test suite as follows:

```bash
pytest tests/
```

## 📚 Project Structure

```
account-billing-chatbot/
├── app/
│   └── app.py                # Flask application
├── data/                     # PDF documents
├── models/
│   └── vector_store/         # FAISS vector store
├── src/
│   ├── ingest.py             # Document ingestion
│   ├── chatbot.py            # Core chatbot functionality
│   └── evaluate.py           # Evaluation system
├── static/
│   └── images/               # Extracted images from PDFs
├── templates/
│   └── index.html            # Web interface
├── tests/
│   └── test_cases.json       # Test cases for evaluation
├── .env.example              # Example environment variables
├── Dockerfile                # For containerization
├── deploy.sh                 # Deployment script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔒 Security Measures

- Input sanitization to prevent prompt injection attacks
- Limited scope of responses to only the provided documentation
- No storage or processing of sensitive user information
- Regular expression filtering for potentially harmful content

## ✨ Additional Features

- Context-aware responses with support for follow-up questions
- Image inclusion in responses when relevant
- Conversation memory for better contextual understanding
- Source citation in responses for accountability
- Response time optimization (<10 seconds per query)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or support, contact omaria.palacios@konecta.com