# CoherixLive

This app provides a live market coherence chart using FastAPI and Plotly, fetching real-time data from Finnhub.

## Setup

1. Create and activate a virtual environment:
   ```
   python -m venv crptlv
   # On Windows:
   crptlv\Scripts\activate
   # On macOS/Linux:
   source crptlv/bin/activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   uvicorn main:app --reload
   ```

4. Open your browser to [http://localhost:8000](http://localhost:8000)

## Directory Structure

- `main.py` - FastAPI app entry point
- `utils/binance_utils.py` - Binance data fetching utility
- `templates/live_charts.html` - Main HTML template
- `static/.keep` - Keeps the static directory in version control

## Docker

See `Dockerfile` for containerization instructions.
