# AI Copilot Instructions - PRISMA-ScR Toolkit

## Project Status: ✅ Core Features Complete (v2.0.0)

**Completed Features:**
- ✅ **Data Management**: Upload (Excel/CSV/RIS), LLM field mapping, merge, smart deduplication
- ✅ **AI Screening**: Parallel processing (8 workers), cost tracking, checkpoint/resume support
- ✅ **Results & Export**: Interactive table with filters, CSV export with full abstracts
- ✅ **Database**: SQLite with SQLAlchemy ORM, persistent storage across restarts
- ✅ **Frontend**: Streamlit pages for all workflows (Data Management, AI Screening, Results, Settings)
- ✅ **Backend**: FastAPI with async processing, task management, error handling

**Known Working Test Data:**
- Mock diabetes dataset (89 articles)
- Task IDs: `b4c7954a-2a90-4d96-be12-91d664041a5e`, `0723cf09-3af3-43b5-8a6b-63f8ab632c6e`
- All abstracts preserved and exported correctly

## Project Overview

**✅ Production-ready AI-powered systematic review screening toolkit** with hybrid FastAPI + Streamlit architecture. Processes 5,000+ articles in ~30 min using parallel AI screening (8 workers). All core features complete: data management, intelligent deduplication, LLM-based field mapping, and AI screening with full abstract export.

## Architecture Essentials

### Two-Server Pattern
- **Backend**: FastAPI (port 8000) - async processing, task management, AI integration
- **Frontend**: Streamlit (port 8501) - polling-based UI, no complex state management
- **Communication**: REST API with polling (2-sec intervals for long tasks)
- **Start**: Run `python start_all.py` (launches both) or separately via `start_backend.py`/`start_frontend.py`

### Key Data Flow
```
Upload → Field Mapping (LLM auto-detect) → Merge → Smart Dedup → AI Screening → PRISMA Export
```

**Data Storage**: 
- **Database** (SQLAlchemy ORM): Persistent storage for datasets, tasks, screening results
- **Legacy in-memory**: `datasets_storage` dict (being migrated to database)
- **Task tracking**: SQLAlchemy `Task` and `ScreeningResult` models with checkpoint/resume support

## Critical Patterns

### 1. Database & Session Management (✅ Complete)
**SQLAlchemy ORM** for persistent storage. Always use context manager or dependency injection:
```python
# Context manager (preferred for scripts/background tasks)
from backend.db import get_db
with get_db() as db:
    dataset = db.query(Dataset).filter_by(id=dataset_id).first()
    db.add(new_record)
    # Auto-commits on exit

# Dependency injection (for FastAPI endpoints)
from backend.db import get_db_session
@router.get("/datasets")
def get_datasets(db: Session = Depends(get_db_session)):
    return db.query(Dataset).all()
```

**Models**: `Dataset`, `Task`, `ScreeningResult`, `DeduplicationRecord`, `SystemLog` in `backend/models/database.py`

**Init database**: `python init_db.py` (creates tables), `python init_db.py --reset` (drops & recreates), `python init_db.py --seed` (adds test data)

### 2. Environment Variables - XAI API Key
**Always** use `XAI_API_KEY` (not `GROK_API_KEY`). Load from `.env` in project root:
```python
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')
```

### 3. Column Name Handling
- **Pre-mapping**: Original columns (e.g., `TI`, `Article Title`, `AB`)
- **Post-mapping**: Standardized lowercase (`title`, `abstract`, `authors`, `journal`, `year`, `doi`)
- Always check both forms with `_find_column()` pattern:
```python
title_col = 'title' if 'title' in df.columns else self._find_column(df, ['Title', 'TI', 'Article Title'])
```

### 4. Smart Deduplication Workflow (✅ Complete)
**5-stage intelligent process** (see `backend/core/smart_deduplicator.py`):
1. Quality check (remove invalid records)
2. DOI exact matching
3. Title similarity (TF-IDF, default 0.85 threshold)
4. **Metadata validation** - if titles similar but authors/journal/year differ → flag for manual review
5. Return cleaned dataset + review dataset (not just remove everything)

**Key**: Don't auto-remove title-similar records - validate metadata first or flag for review.

### 5. Parallel Screening with Cost Tracking (✅ Complete)
Uses `ThreadPoolExecutor` (8 workers default) with:
- **Token tracking**: `input_tokens`, `output_tokens`, `reasoning_tokens` (for grok-4-fast-reasoning)
- **Cost calculation**: HKD pricing (USD × 7.78), supports prompt caching
- **Progress callbacks**: Update `TaskManager` + cost tracking in real-time
- **Checkpointing**: Save every 100 articles to `data/checkpoints/`

Model preference: `grok-4-fast-reasoning` (cheapest + best for screening).

### 6. API Response Structure (✅ Complete)
Pydantic models in `backend/api/`. Example pattern:
```python
# Request
class StartScreeningRequest(BaseModel):
    data_id: str
    criteria: ScreeningCriteria
    limit: Optional[int] = None  # Test mode

# Progress (for polling)
class ScreeningProgress(BaseModel):
    task_id: str
    status: TaskStatus
    processed: int
    progress_percent: float
    current_cost: float
    estimated_total_cost: float
```

### 7. Frontend-Backend Integration (✅ Complete)
Streamlit uses `frontend/utils/api_client.py`. Pattern:
```python
# Start task
response = api_client.start_screening(data_id, criteria, limit=100)
task_id = response['task_id']

# Poll status
while True:
    status = api_client.get_screening_status(task_id)
    if status['status'] == 'completed': break
    time.sleep(2)
```

**Important**: Streamlit pages are in `frontend/pages/` with naming `1_Data_Management.py`. Navigation via `st.switch_page()`.

## Development Workflows

### Running Tests
```powershell
# Standalone tests (no pytest framework used)
python test_smart_dedup.py
python test_grok_connection.py
python test_all_endpoints.py  # Requires backend running
```

### Testing New Endpoints
1. Add to `backend/api/[module].py`
2. Update `frontend/utils/api_client.py` with client method
3. Test via `test_all_endpoints.py` or Swagger UI (`http://localhost:8000/docs`)

### Debugging Task State
```python
# In backend
from backend.tasks.task_manager import task_manager
task = task_manager.get_task(task_id)
print(task.status, task.processed_items, task.result)

# Check checkpoints
checkpoint_file = Path("data/checkpoints") / f"checkpoint_{n}_{total}.json"
```

### Adding New LLM Models
1. Update pricing in `AIScreener.MODEL_PRICING` (backend/core/screener.py)
2. Update `GrokClient` if API changes (backend/services/grok_client.py)
3. Test cost estimation: `api_client.estimate_screening_cost(data_id, model="new-model")`

## Project-Specific Conventions

### Logging
Use module-level logger with emoji prefixes for clarity:
```python
logger.info("🚀 Starting screening...")
logger.warning("⚠️ DOI column not found")
logger.error("❌ API call failed")
```

### File Handling
- **Uploads**: Stored temporarily, converted to pandas DataFrame
- **Results**: JSON in `data/screening_results/{task_id}.json`
- **Checkpoints**: `data/checkpoints/checkpoint_{current}_{total}.json`

### Error Handling
- **Backend**: Raise `HTTPException` with status codes (404, 400, 500)
- **Frontend**: Catch in `try/except`, display with `st.error()`
- **Retry logic**: Exponential backoff in `GrokClient` (3 retries default)

### Cost Reporting
Always show in **HKD** (Hong Kong Dollars). Pricing updated Nov 2025 in screener.py. Include:
- Per-article cost
- Total cost
- Token breakdown (input/cached/output/reasoning)

## Known Gotchas

1. **Don't use `GROK_API_KEY`** - code expects `XAI_API_KEY`
2. **Session state**: Streamlit session state persists across page changes - use `st.session_state` carefully
3. **DataFrames**: Always use `.copy()` when slicing to avoid SettingWithCopyWarning
4. **Task cleanup**: Tasks stored in database; use `init_db.py --reset` to clear (dev only)
5. **Deduplication**: `smart_deduplicator` returns 2 datasets (cleaned + review), not 1
6. **Database sessions**: Always close sessions - use `with get_db()` or dependency injection
7. **JSON serialization**: DataFrame data stored as JSON in database - use `pd.read_json()` to deserialize
8. **Abstract field**: Ensure all API response paths include `abstract=db_result.abstract` or `abstract=d.get('abstract')` for JSON fallback

## Key Files for Common Tasks

- **Add API endpoint**: `backend/api/[screening|data_management|results].py`
- **Add frontend page**: `frontend/pages/N_PageName.py`
- **Modify screening logic**: `backend/core/screener.py`
- **Change dedup behavior**: `backend/core/smart_deduplicator.py`
- **Update field mapping**: `backend/core/llm_field_mapper.py`
- **Cost calculations**: `backend/core/screener.py` (MODEL_PRICING dict)

## Testing Checklist for New Features

1. Backend: Add route in `backend/api/`, test via `/docs`
2. Client: Add method to `APIClient`, test connection
3. Frontend: Add UI in appropriate page, test with real data
4. Error cases: Test with missing data, wrong types, API failures
5. Documentation: Update relevant .md files in `docs/`

## Quick Debugging Commands

```powershell
# Check backend health
curl http://localhost:8000/health

# View API docs
# Open browser: http://localhost:8000/docs

# Test Grok connection
python test_grok_connection.py

# Database operations
python init_db.py --info          # Show database info
python init_db.py --check         # Health check
python init_db.py --reset --seed  # Reset and add test data (dev only)

# Check dataset storage (legacy in-memory or database)
# Add breakpoint in backend/api/data_management.py
```
