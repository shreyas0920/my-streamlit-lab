# My Streamlit Lab Submission

## Original Lab
Iris prediction UI with JSON upload and FastAPI backend.

## My Modifications
- Adapted for Wine dataset (13 features) to match FastAPI backend.
- Added prediction history table and clear button.
- Included feature importance bar chart from RandomForest model.
- Why: Align with Wine model, enhance monitoring, and improve interpretability.

## Learnings/Challenges
Learned Streamlit dataframes and plotting; fixed JSON and model path issues.

## Prior Knowledge
Streamlit basics from tutorials.

## Run Instructions
1. Activate: streamlit-env\Scripts\activate
2. Backend: cd backend/src && uvicorn main:app --reload
3. App: cd src && streamlit run Dashboard.py
4. View: http://localhost:8501