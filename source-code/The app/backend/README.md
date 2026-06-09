# L6S (Lean Six Sigma) Chatbot Backend

## Overview
This backend powers the L6S (Lean Six Sigma) Expert Advisor chatbot. It uses a graph RAG approach with CSF (Critical Success Factors) data to provide personalized recommendations for achieving performance goals.

## Folder Structure
```
backend_for_colleague/
├── api/                    # API integration files
│   ├── l6s_api_optimized.py   # Main API with process-level caching
│   └── server.js              # Express.js server for React frontend integration
├── core/                   # Core brain logic
│   └── neo.py                 # Main L6SBrain class (replace placeholder)
├── data/                   # Data files
│   └── unified_lss_data_final.json   # Graph RAG data (replace placeholder)
├── utils/                  # Utility files
│   └── l6s_prompt_templates.json   # Prompt templates (replace placeholder)
├── requirements.txt        # Python dependencies
└── README.md             # This file
```

## Required Files to Replace
The following placeholder files need to be replaced with the actual files from the parent directory:

1. `core/neo.py` - Contains the main `L6SBrain` class
2. `data/unified_lss_data_final.json` - Graph RAG data for the chatbot
3. `utils/l6s_prompt_templates.json` - Prompt templates used by the system

## How the Backend Works

### Architecture
- **React Frontend** → **Express.js Server** → **Python Backend** 
- The `server.js` file acts as a bridge between the React UI and Python backend
- `l6s_api_optimized.py` provides process-level caching for better performance
- The `L6SBrain` class from `neo.py` handles all the core logic

### Key Features
- Goal-based recommendations (e.g., "How to achieve 85% performance?")
- CSF (Critical Success Factors) extraction from user input
- Performance analytics and charts generation
- Multi-language support (English/French)
- Graph RAG-based responses using network analysis

### API Endpoints
- `POST /api/chat` - Main chat endpoint
- `GET /api/statistics` - CSF statistics endpoint (currently mocked)

### CSF Categories
- **IL**: Lean Critical Success Factors (IL1-IL7)
- **IS**: Six Sigma Critical Success Factors (IS1-IS7) 
- **M**: Maturity Factors (M1-M7)

## Setup Instructions

1. **Replace placeholder files** with actual files from parent directory
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Install Node.js dependencies** (if integrating with React):
   ```bash
   cd path/to/react/app
   npm install
   ```
4. **Start the Express server**:
   ```bash
   node server.js
   ```
5. The server will run on `http://localhost:5001`

## Integration with New Frontend

To integrate with your new frontend:

1. Make POST requests to `http://localhost:5001/api/chat` with the following body:
   ```json
   {
     "message": "Your message here",
     "useSidebarValues": true/false,
     "sidebarScores": {"IL1": 3, "IS2": 4, ...},
     "conversationHistory": [...],
     "modelConfig": {
       "local_model": "llama3.2",
       "api_model": "gemini-pro", 
       "temperature": 0.3,
       "api_key": "...",
       "use_api": false
     }
   }
   ```

2. The response will be in the format:
   ```json
   {
     "content": "Response text",
     "structuredOutput": {
       "type": "text|chart|dataframe",
       "data": {...},
       "confidence": 95,
       "sampleSize": 156
     },
     "performanceTable": {...}
   }
   ```

## Key Python Classes and Functions

- `L6SBrain` class: Core logic for processing queries and generating recommendations
- `get_reverse_recommendation(target_perf)`: Gets CSF values needed to achieve target performance
- `extract_scores_from_text(text)`: Extracts CSF scores from user input
- `is_l6s_related(message)`: Determines if message is L6S-related
- `generate_performance_breakdown_chart()`: Creates performance charts
- `generate_subcriteria_barchart()`: Creates subcriteria charts

## Dependencies
- Python 3.8+
- Node.js 16+
- LangChain for LLM orchestration
- NetworkX for graph operations
- Matplotlib/Seaborn for chart generation

## Troubleshooting
- Ensure all three main files (neo.py, unified_lss_data_final.json, l6s_prompt_templates.json) are properly replaced
- Check that Python path is correctly configured in server.js
- Verify that the Python virtual environment is activated
- Check server logs for any error messages