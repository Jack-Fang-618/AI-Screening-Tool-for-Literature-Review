# Streamlit Cloud Deployment Issue

## Problem
Backend thread dies immediately (< 2s) without capturing any logs.

## Root Cause
Streamlit Cloud doesn't support running FastAPI (uvicorn) in the same process as Streamlit.

## Solution Options

### Option 1: Use Streamlit Alone (Recommended for Streamlit Cloud)
Remove FastAPI completely, call backend functions directly from Streamlit pages.

**Pros:**
- Simple, no threading needed
- Works perfectly on Streamlit Cloud
- No port conflicts

**Cons:**
- No REST API
- Harder to test backend independently

### Option 2: Deploy Backend Separately
Deploy FastAPI on a separate service (Railway, Render, Heroku) and have Streamlit call it via HTTP.

**Pros:**
- Keeps API architecture
- Backend can be tested independently

**Cons:**
- Need 2 deployments
- More complex setup
- Need to configure CORS

### Option 3: Use Streamlit + Background Tasks
Keep Streamlit-only approach but use `st.session_state` + background threads for long-running tasks.

**Pros:**
- Works on Streamlit Cloud
- Can still do async processing

**Cons:**
- No REST API
- Need to rewrite API client logic

## Recommendation
For quick deployment to Streamlit Cloud: **Option 1**
Create `streamlit_cloud.py` that imports backend logic directly without FastAPI/uvicorn.
