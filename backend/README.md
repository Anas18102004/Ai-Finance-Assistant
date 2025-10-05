# AI Financial Assistant

A high-performance AI Financial Assistant built with HuggingFace embeddings, Chroma vector database, and Gemini summarization.

## 🚀 Features

- **AI-Powered Data Generation**: Faker (Python) + Gemini AI for realistic transactions
- **Smart Categories**: Food, Shopping, Rent, Salary, Utilities, Entertainment, Travel, Others
- **Realistic Descriptions**: AI-generated using Gemini or realistic Indian company names
- **Vector Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **Vector Database**: Chroma with persistent storage and cosine similarity
- **Smart Retrieval**: Context-aware filtering by user, category, date, amount
- **AI Summarization**: Google Gemini Flash for intelligent financial insights
- **Fast API**: Async endpoints with <500ms response times
- **Query Caching**: In-memory cache for repeated queries

## 📋 Requirements

- Python 3.8+
- Gemini API key (for summarization)

## 🛠️ Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run setup and start server**:
```bash
python setup_and_run.py
```

This will:
- Generate synthetic transaction data
- Build the vector index
- Start the API server at `http://localhost:8000`

## 🔑 API Usage

### Get Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Use it in the `X-Gemini-API-Key` header

### Endpoints

#### 1. Query Transactions
```bash
POST /query
```

**Headers**:
```
Content-Type: application/json
X-Gemini-API-Key: your_api_key_here
```

**Request**:
```json
{
  "user_id": "user_001",
  "query": "Show me my top 5 expenses this month",
  "summarize": true,
  "top_k": 5,
  "filters": {}
}
```

**Response**:
```json
{
  "status": "success",
  "query": "Show me my top 5 expenses this month",
  "user_id": "user_001",
  "intent": "top_expenses",
  "retrieved": [...],
  "summary": "Based on your transactions, you spent ₹45,230 across 23 transactions...",
  "metadata": {
    "total_time_ms": 245.67
  }
}
```

#### 2. Generate New Data
```bash
POST /generate
```

#### 3. Rebuild Index
```bash
POST /index/build
```

#### 4. Health Check
```bash
GET /health
```

## 💡 Example Queries

- `"Show me my top 5 expenses this month"`
- `"How much did I spend on food in September?"`
- `"Find transactions above ₹5000"`
- `"Show my entertainment expenses"`
- `"What are my recent shopping transactions?"`

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Query Parser  │───▶│    Retriever     │───▶│   Summarizer    │
│                 │    │                  │    │                 │
│ • Intent        │    │ • Vector Search  │    │ • Gemini AI     │
│ • Filters       │    │ • Chroma DB      │    │ • Smart Summary │
│ • Categories    │    │ • Cosine Sim     │    │ • Follow-up Q   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
backend/
├── api/
│   └── app.py              # FastAPI application
├── data/
│   └── transactions.json   # Synthetic transaction data
├── services/
│   └── embeddings.py       # HuggingFace embedding service
├── index_build/
│   └── build_index.py      # Chroma index builder
├── nodes/
│   ├── query_parser.py     # NLP query parser
│   ├── retriever.py        # Vector retrieval
│   ├── summarizer.py       # Gemini summarization
│   └── graph_orchestrator.py # LangGraph-style workflow
├── chroma_db/              # Vector database (auto-created)
├── requirements.txt        # Dependencies
├── setup_and_run.py       # Setup script
└── README.md              # This file
```

## ⚡ Performance

- **Embedding Generation**: Batched processing (64 transactions/batch)
- **Vector Search**: <150ms for 10k transactions
- **End-to-End Latency**: <500ms (excluding Gemini)
- **Query Caching**: In-memory cache for repeated queries
- **Async Processing**: Non-blocking FastAPI endpoints

## 🎯 Supported Query Types

1. **Top Expenses**: "top 5 expenses", "highest spending"
2. **Sum Spent**: "total spent", "how much did I spend"
3. **Filter**: "transactions in September", "food expenses"
4. **Compare**: "compare spending", "more than ₹1000"
5. **General**: Natural language financial queries

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Your Gemini API key (optional, can use header)
- `CHROMA_DB_PATH`: Custom path for Chroma database (default: ./chroma_db)

### Performance Tuning
- Adjust `batch_size` in `embeddings.py` for your hardware
- Modify `top_k` limits in retriever for different result sizes
- Configure cache size in retriever for memory management

## 🐛 Troubleshooting

### Common Issues

1. **"Service not initialized"**
   - Run `python setup_and_run.py` to initialize all components

2. **"Gemini API key required"**
   - Add `X-Gemini-API-Key` header with valid API key

3. **Slow performance**
   - Check if index is built: `GET /health`
   - Clear cache: `POST /cache/clear`
   - Reduce `top_k` in queries

4. **Empty results**
   - Generate data: `POST /generate`
   - Rebuild index: `POST /index/build`
   - Check user_id matches generated data

## 📊 API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🧪 Testing

```bash
# Test health
curl http://localhost:8000/health

# Test query (replace API key)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-Gemini-API-Key: your_api_key" \
  -d '{
    "user_id": "user_001",
    "query": "Show me my top expenses",
    "summarize": true
  }'
```

## 🔮 Future Enhancements

- Redis caching for production
- User authentication
- Real-time transaction ingestion
- Advanced analytics dashboard
- Multi-language support
- Custom embedding fine-tuning

---

**Built with ❤️ using FastAPI, HuggingFace, Chroma, and Gemini**
