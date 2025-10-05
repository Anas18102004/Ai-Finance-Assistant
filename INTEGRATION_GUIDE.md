# Backend-Frontend Integration Guide

## Overview
This guide explains how to connect your AI Financial Assistant backend with the React frontend chatbot.

## Architecture
- **Backend**: FastAPI server running on `http://localhost:8000`
- **Frontend**: React/Vite application with TypeScript
- **Communication**: REST API calls using fetch

## Setup Instructions

### 1. Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies (if not already done):
   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend server:
   ```bash
   python setup_and_run.py
   ```

   This will:
   - Generate synthetic transaction data
   - Initialize the embedding service
   - Build the vector index
   - Start the API server on port 8000

4. Verify the backend is running by visiting:
   - API docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

### 2. Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and set:
   ```
   VITE_API_URL=http://localhost:8000
   VITE_GEMINI_API_KEY=your_gemini_api_key_here  # Optional
   ```

4. Start the frontend development server:
   ```bash
   npm run dev
   ```

5. Open your browser to `http://localhost:5173`

## Features Integrated

### 1. Chat Interface (`/`)
- Real-time communication with the AI backend
- Natural language query processing
- Error handling and connection status
- User-friendly error messages

### 2. AI Dashboard (`/ai-dashboard`)
- Natural language query input with "Ask AI" button
- Real-time query processing with loading states
- Results display with formatted output
- Integration with existing dashboard components

### 3. Backend Status Monitoring
- Real-time connection status indicator
- Health check monitoring
- Service status breakdown
- Connection troubleshooting instructions

## API Integration Details

### Service Layer (`src/services/api.ts`)
- Centralized API communication
- Type-safe request/response handling
- Error handling and retry logic
- Environment-based configuration

### Key Endpoints Used
- `POST /query` - Natural language transaction queries
- `GET /health` - Backend health monitoring
- `GET /examples` - Sample queries for testing
- `POST /generate` - Generate synthetic data
- `POST /index/build` - Rebuild vector index

## Testing the Integration

### 1. Basic Connection Test
1. Start both backend and frontend
2. Check the backend status indicator (should show "Connected")
3. Try a simple query: "Show me my expenses"

### 2. Sample Queries to Test
- "Show me my top 5 expenses this month"
- "How much did I spend on food in September?"
- "Find transactions above ₹5000"
- "What's my biggest spending category?"
- "Show my recent shopping transactions"

### 3. Error Scenarios
- Stop the backend server and test error handling
- Try queries with invalid data
- Test with and without Gemini API key

## Troubleshooting

### Backend Not Starting
- Check if port 8000 is available
- Verify all dependencies are installed
- Check for missing API keys or configuration

### Frontend Connection Issues
- Verify VITE_API_URL in .env file
- Check browser console for CORS errors
- Ensure backend is running and accessible

### Query Processing Issues
- Check if synthetic data was generated
- Verify vector index was built successfully
- Test with simpler queries first

## Environment Variables

### Backend
- `GEMINI_API_KEY` - For AI summarization (optional)
- `LOG_LEVEL` - Logging level (default: INFO)

### Frontend
- `VITE_API_URL` - Backend API URL (default: http://localhost:8000)
- `VITE_GEMINI_API_KEY` - Gemini API key for enhanced features (optional)

## Development Notes

### Adding New Features
1. Add API endpoints in `backend/api/app.py`
2. Update API service in `frontend/src/services/api.ts`
3. Implement UI components as needed
4. Test integration thoroughly

### Deployment Considerations
- Update API URLs for production
- Configure CORS settings appropriately
- Set up proper environment variables
- Consider API rate limiting and authentication

## Support

If you encounter issues:
1. Check the backend logs for errors
2. Verify all services are healthy via `/health` endpoint
3. Test API endpoints directly using `/docs`
4. Check browser console for frontend errors
