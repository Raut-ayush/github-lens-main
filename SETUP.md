# GitHub Profile Intelligence - Setup Guide

## Frontend Setup

1. **Install dependencies:**
```bash
npm install
```

2. **Configure API endpoint:**
Create a `.env` file in the root directory:
```bash
VITE_API_URL=http://localhost:8000
```

3. **Run the development server:**
```bash
npm run dev
```

The frontend will be available at `http://localhost:8080`

## Backend Setup

1. **Navigate to your backend directory** (where you extracted backend.zip)

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the FastAPI server:**
```bash
uvicorn app:app --reload --port 8000
```

The backend API will be available at `http://localhost:8000`

## Usage Flow

1. **Start both servers:**
   - Backend: `uvicorn app:app --reload --port 8000`
   - Frontend: `npm run dev`

2. **Open the app** in your browser: `http://localhost:8080`

3. **Enter a GitHub username** (e.g., `torvalds`, `gaearon`, etc.)

4. **Wait for analysis** - The backend will:
   - Fetch data from GitHub API
   - Parse repositories and extract skills
   - Generate analysis and store results in `output/<username>/`
   - Return summary JSON

5. **View dashboard** with:
   - Repository metrics and analytics
   - Skills intelligence and language breakdown
   - ML-powered insights
   - Activity timelines

## API Endpoints

### Analyze User
```
GET /api/analyze?username=<username>
```
Fetches and analyzes GitHub profile data.

### Clear Cache
```
POST /api/clear-cache/<username>
```
Clears cached data for a specific user.

### Health Check
```
GET /health
```
Checks if the backend is running.

## Features

- **Landing Page:** Username input with CSV upload option (coming soon)
- **Overview Dashboard:** Total stats, timeline, languages, activity metrics
- **Repositories Explorer:** Searchable table with full repo details
- **Skills Intelligence:** Skill cloud, language growth, framework usage
- **ML Insights:** AI-powered developer scoring (using mock data until ML backend ready)

## Development Notes

- The frontend uses React + Vite + TypeScript
- Real-time data fetching from FastAPI backend
- Responsive design with Tailwind CSS
- Chart visualizations with Recharts
- Toast notifications for user feedback

## Troubleshooting

**CORS Issues:** If you encounter CORS errors, make sure your FastAPI backend has CORS middleware configured:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Connection Refused:** Ensure both servers are running on the correct ports (backend on 8000, frontend on 8080).
