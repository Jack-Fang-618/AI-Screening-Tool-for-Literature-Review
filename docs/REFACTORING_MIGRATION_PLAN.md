# 🔄 AI Scoping Review - Core Function Refactoring & Migration Plan

**Version:** 2.0.0  
**Date:** November 7, 2025  
**Status:** Planning Phase  
**Architecture:** FastAPI Backend + Streamlit Frontend  
**Purpose:** Streamline and modularize the AI Scoping Review toolkit into a focused, efficient, standalone system

---

## 📋 Executive Summary

### 🎯 Critical Enhancements (November 2025 Update)

#### 1. **Cross-Database Field Standardization** (Week 2 Implementation)
**Problem:** Researchers export data from multiple databases (PubMed, Scopus, Web of Science, Embase), each with different field naming conventions:
- PubMed: `TI`, `AB`, `AU`, `TA`
- Scopus: `Article Title`, `Abstract`, `Authors`, `Source title`
- Web of Science: `TI`, `AB`, `AU`, `SO`
- Embase: `Title`, `Abstract`, `Author`, `Source`

**Solution:** Intelligent auto-detection + manual refinement workflow
> "自動+人工輔助，將這些數據結構（比如説column的順序和命名不同）統一標準化"
- Auto-detect database source based on column patterns
- Apply predefined mapping rules (e.g., `TI` → `title`, `AB` → `abstract`)
- Interactive UI for manual adjustment of uncertain mappings
- Save custom mappings for future reuse
- Unified schema for merged datasets

**Implementation:** Field Mapper enhancement (Week 2) - see Section 2.2

---

#### 2. **Rubric-based Dual-LLM Evaluation System** (Phase 2 - Months 3-6)
**Problem:** Current screening confidence is "intuitive" and lacks transparency
> "給他加一個evaluation system...讓另一個llm去eval...可以變成一個matrix"

**Solution:** Second LLM validates primary screening decisions using structured rubric
- Primary LLM: Screens article (Include/Exclude decision)
- Evaluation LLM: Assesses quality of decision using 5-criterion rubric
  - Relevance to PCC (35% weight)
  - Logical Reasoning (25% weight)
  - Citation Evidence (20% weight)
  - Consistency (15% weight)
  - Completeness (5% weight)
- Matrix-based confidence calculation:
  - `Final Confidence = 0.40×Primary + 0.35×Eval Score + 0.15×Agreement + 0.10×Reasoning Quality`
- Auto-flag disagreements for manual review

**Implementation:** Phase 2 Enhancement (Months 3-6) - see Future Enhancements section

---

### Current State Analysis
The existing `modern_app.py` has grown to **8,103 lines**, becoming increasingly difficult to maintain, debug, and extend. The application includes many features that are not core to the primary use case.

### Target State
A streamlined, **hybrid architecture** system focused on **three core workflows**:
1. **Data Management**: Merge, clean, and deduplicate articles from multiple database exports
   - **NEW: Cross-Database Field Standardization** - Auto-detect and map heterogeneous field names from PubMed, Scopus, Web of Science, Embase
   - **NEW: Manual Refinement UI** - Interactive interface for adjusting auto-detected field mappings
2. **AI Screening**: Parallel AI-powered title/abstract screening with 8 concurrent workers (FastAPI backend)
   - **FUTURE: Dual-LLM Evaluation** - Second LLM validates screening decisions using rubric-based quality matrix
3. **Results Visualization**: PRISMA flow diagrams and screening analytics

### Architecture Decision
**FastAPI Backend + Streamlit Frontend** for optimal performance:
- **Streamlit**: Simple, intuitive UI for researchers (no login complexity)
- **FastAPI**: Asynchronous backend for long-running tasks (screening, deduplication)
- **Separation of Concerns**: Core logic independent of UI framework
- **Future-proof**: Easy to add React frontend or CLI interface later

### Key Principles
- ✅ **Simplicity First**: Remove user login, project management complexity
- ✅ **Performance**: 8-channel parallel screening with async backend
- ✅ **Modularity**: Separate concerns into independent, testable modules
- ✅ **Scalability**: FastAPI backend supports multiple concurrent users
- ✅ **Standalone**: Independent project with own environment and dependencies
- ✅ **Optimization**: Preserve excellent deduplication logic, improve UI/UX

---

## 🎯 Design Philosophy & User Persona

### Target User Profile
**Dr. Sarah Chen** - Medical researcher conducting systematic reviews
- Already collected 5,000+ articles from PubMed, Scopus, Web of Science, Embase
- Has Excel/CSV/RIS files exported from each database
- Needs to: merge datasets → remove duplicates → screen with AI → generate PRISMA report
- **Does NOT need**: Complex user authentication, multi-project management, advanced database searching

### Core User Journey
```
1. Launch App (no login)
   ↓
2. Upload Files (Excel/CSV/RIS from PubMed, Scopus, Web of Science, Embase)
   ↓
3. Auto-Detect Database Source (PubMed/Scopus/WoS/Embase)
   ↓
4. Auto-Map Fields + Preview
   - System automatically maps: "TI" → "title", "AB" → "abstract", etc.
   - Shows confidence scores for each mapping
   - Highlights uncertain mappings for manual review
   ↓
5. Manual Field Refinement (Interactive UI)
   - Review auto-detected mappings
   - Drag-and-drop to adjust incorrect mappings
   - Save custom mappings for future use
   ↓
6. Merge Datasets (Standardized Schema)
   - All files converted to unified field names
   - Source tracking preserved
   ↓
7. Clean & Deduplicate (intelligent algorithms)
   ↓
8. AI Screening (parallel, 8 workers)
   ↓
9. Review Results + Export PRISMA Diagram
```

**Total time goal**: < 30 minutes for 5,000 articles

**Key Innovation - Data Merging Workflow:**
> "自動+人工輔助，將這些數據結構（比如説column的順序和命名不同）統一標準化"
- Different databases export with different field names
- System auto-detects source and applies standard mapping
- User reviews and refines mappings interactively
- Final merged dataset has unified schema for downstream processing

---

## 📁 New Project Structure (FastAPI + Streamlit)

```
Core function/                          # 🆕 Standalone Project Root
│
├── docs/                              # 📚 Documentation
│   ├── REFACTORING_MIGRATION_PLAN.md # This file
│   ├── API_REFERENCE.md              # FastAPI endpoint documentation
│   ├── USER_GUIDE.md                 # End-user guide
│   └── DEVELOPMENT_GUIDE.md          # Developer setup guide
│
├── backend/                           # � FastAPI Backend
│   ├── __init__.py
│   ├── main.py                       # FastAPI application entry point
│   ├── api/                          # API endpoints
│   │   ├── __init__.py
│   │   ├── data_management.py        # Data upload/processing endpoints
│   │   ├── screening.py              # Screening task endpoints
│   │   └── results.py                # Results retrieval endpoints
│   │
│   ├── core/                         # Core business logic
│   │   ├── __init__.py
│   │   ├── data_processor.py         # Data processing engine
│   │   ├── deduplicator.py          # Duplicate detection
│   │   ├── field_mapper.py          # Field mapping logic
│   │   └── screener.py              # AI screening engine
│   │
│   ├── models/                       # Data models (Pydantic)
│   │   ├── __init__.py
│   │   ├── article.py               # Article data model
│   │   ├── screening_request.py     # Screening request model
│   │   └── screening_result.py      # Screening result model
│   │
│   ├── services/                     # External services
│   │   ├── __init__.py
│   │   ├── grok_client.py           # Grok API client
│   │   └── storage.py               # File storage service
│   │
│   └── tasks/                        # Background tasks
│       ├── __init__.py
│       ├── screening_task.py         # Async screening tasks
│       └── task_manager.py           # Task state management
│
├── frontend/                          # 🎨 Streamlit Frontend
│   ├── __init__.py
│   ├── app.py                        # Main Streamlit application
│   ├── pages/                        # Streamlit pages
│   │   ├── 1_📊_Data_Management.py
│   │   ├── 2_🤖_AI_Screening.py
│   │   └── 3_📈_Results.py
│   │
│   ├── components/                   # Reusable UI components
│   │   ├── __init__.py
│   │   ├── file_uploader.py
│   │   ├── progress_tracker.py
│   │   └── result_viewer.py
│   │
│   └── utils/                        # Frontend utilities
│       ├── __init__.py
│       ├── api_client.py            # Backend API client
│       └── session_manager.py       # Session state management
│
├── shared/                            # � Shared between backend/frontend
│   ├── __init__.py
│   ├── config.py                     # Configuration management
│   ├── constants.py                  # Shared constants
│   └── validators.py                 # Data validation
│
├── utils/                             # 🛠️ General Utilities
│   ├── __init__.py
│   ├── logger.py                     # Logging utilities
│   └── helpers.py                    # Helper functions
│
├── tests/                             # 🧪 Test Suite
│   ├── __init__.py
│   ├── test_backend/                 # Backend tests
│   │   ├── test_api_endpoints.py
│   │   ├── test_data_processor.py
│   │   └── test_screener.py
│   ├── test_frontend/                # Frontend tests
│   │   └── test_components.py
│   └── fixtures/                     # Test data
│   ├── test_ai_screening.py
│   ├── test_screening_results.py
│   └── fixtures/                     # Test data
│
├── app.py                             # 🚀 Main Streamlit Application
├── requirements.txt                   # 📦 Python Dependencies
├── .env.example                       # 🔐 Environment Template
├── README.md                          # 📖 Project README
├── setup.py                           # 📦 Package Setup (optional)
└── pyproject.toml                     # 🔧 Project Configuration
```

---

## 🔍 Current Code Analysis

### Module Breakdown from `modern_app.py` (8,103 lines)

#### 1️⃣ **Data Management Functions** (Lines ~2900-4700)
**Current Implementation:**
- `show_data_management_with_project()` - Main entry point
- `create_unified_file_registry()` - File discovery
- `auto_assign_fields()` - Basic field mapping
- **Deduplication Logic** (Lines 4650-4720):
  - ✅ DOI-based exact matching
  - ✅ TF-IDF title similarity (cosine ≥ 0.85)
  - ✅ Author-Year-Journal combination
  - ✅ sklearn-based vectorization
  
**Migration Priority:** HIGH - Core functionality, preserve deduplication logic

**Key Features to Preserve:**
```python
# Excellence in current deduplication:
1. Multi-strategy approach (DOI, title, metadata)
2. TF-IDF vectorization with cosine similarity
3. Configurable thresholds (currently 0.85)
4. Intelligent title cleaning (regex, lowercase, punctuation removal)
5. Graceful fallback when advanced features unavailable
```

**Improvements Needed:**
- Currently single-file focused → needs multi-file batch processing
- Field mapping logic scattered → centralize in dedicated module
- Limited error handling → add robust exception management
- No progress indicators → add real-time progress tracking

---

#### 2️⃣ **AI Screening Functions** (Lines ~6800-7200)
**Current Implementation:**
- `show_ai_screening_with_project()` - Main entry point
- Single-threaded processing (sequential)
- Grok API integration
- Basic cost tracking
- Manual review threshold logic

**Migration Priority:** CRITICAL - Requires major optimization

**Current Performance Issues:**
```python
# Problem: Sequential processing
for article in articles:
    result = screen_article(article)  # Blocks until complete
    # ~2-5 seconds per article
    # 5,000 articles = 2.7-7 hours total!
```

**Target: 8-Channel Parallel Architecture:**
```python
# Solution: Concurrent processing
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(screen_article, art): art 
               for art in articles}
    for future in as_completed(futures):
        result = future.result()
        # 8 articles processing simultaneously
        # 5,000 articles = ~20-50 minutes (est.)
```

**Design Considerations:**
- **Rate Limiting**: Grok API may have rate limits (check documentation)
- **Error Handling**: Individual failures shouldn't crash entire batch
- **Progress Tracking**: Real-time updates on completion rate
- **Cost Control**: Pre-calculate estimated cost, allow user confirmation
- **Resumability**: Save checkpoints to resume after interruption

---

#### 3️⃣ **Screening Results Functions** (Lines ~200-600)
**Current Implementation:**
- `display_screening_results_section()` - Results viewer
- `show_screening_results_with_project()` - Project-specific view
- PRISMA diagram generation
- Export functionality

**Migration Priority:** MEDIUM - Functional but needs UI improvement

**Key Features to Preserve:**
- PRISMA-ScR compliant flow diagrams
- Interactive data tables
- Export to Excel with multiple sheets

**Improvements Needed:**
- Better visualization (Plotly/Matplotlib)
- Real-time screening progress dashboard
- Confidence score distribution charts
- Inter-rater reliability metrics (if manual review exists)

---

#### 4️⃣ **Configuration & Settings** (Lines ~various)
**Current Implementation:**
- YAML-based configuration
- Project-specific AI config
- Model selection (Grok-3, Grok-3-mini-fast, Grok-4)
- Confidence thresholds

**Migration Priority:** LOW - Simplify significantly

**Changes:**
- Remove project management complexity
- Use single global config file
- Environment variables for API keys
- Simple UI for runtime adjustments

---

## 🚀 Migration Strategy

### Phase 1: Foundation Setup (Week 1)
**Goal:** Establish standalone project infrastructure

#### Tasks:
1. ✅ **Create directory structure** (COMPLETED)
2. 🔲 **Setup independent environment**
   ```bash
   cd "Core function"
   python -m venv venv
   venv\Scripts\activate
   ```
3. 🔲 **Create requirements.txt** (minimal dependencies)
4. 🔲 **Write README.md** (quickstart guide)
5. 🔲 **Setup .env.example** (API key template)
6. 🔲 **Initialize git** (version control for new module)

#### Deliverables:
- Functional Python environment
- Documentation skeleton
- Basic project configuration

---

### Phase 2: Data Management Module (Week 2)
**Goal:** Extract and optimize data handling logic

#### 2.1 File Handler (`file_handler.py`)
**Source:** Lines 3080-3300 from `modern_app.py`

**Responsibilities:**
- Parse Excel (.xlsx, .xls), CSV, RIS files
- Handle multiple sheets in Excel
- Detect encoding for CSV files
- Extract structured data

**Key Functions:**
```python
class FileHandler:
    def parse_file(file_path: Path) -> pd.DataFrame
    def parse_excel(file_path: Path, sheet_name: str) -> pd.DataFrame
    def parse_csv(file_path: Path) -> pd.DataFrame
    def parse_ris(file_path: Path) -> pd.DataFrame
    def detect_file_type(file_path: Path) -> str
    def get_excel_sheets(file_path: Path) -> List[str]
```

**Enhancements:**
- Async file parsing for large files
- Progress callbacks for UI updates
- Validation on parse (detect corrupted files early)

---

#### 2.2 Field Mapper (`field_mapper.py`) ⭐ CRITICAL FOR DATA MERGING
**Source:** Lines 2814-2900 from `modern_app.py`

**Purpose:** **Heterogeneous Database Field Standardization**
> **Key Requirement:** "自動+人工輔助，將這些數據結構（比如説column的順序和命名不同）統一標準化"

**Challenge:** Different databases use different field names:
- **PubMed**: `TI` (title), `AB` (abstract), `AU` (authors), `TA` (journal)
- **Scopus**: `Article Title`, `Abstract`, `Authors`, `Source title`
- **Web of Science**: `TI`, `AB`, `AU`, `SO`
- **Embase**: `Title`, `Abstract`, `Author`, `Source`

**Current Logic (PRESERVE & ENHANCE):**
```python
def auto_assign_fields(available_columns, essential_fields):
    # Pattern matching for common field names
    field_patterns = {
        'title': ['title', 'article title', 'paper title', 'TI', 'ti', ...],
        'abstract': ['abstract', 'summary', 'description', 'AB', 'ab', ...],
        'authors': ['author', 'authors', 'creator', 'AU', 'au', ...],
        'journal': ['journal', 'publication', 'source', 'TA', 'SO', 'source title', ...],
        'year': ['year', 'date', 'publication year', 'PY', 'publication_year', ...],
        'doi': ['doi', 'digital object identifier', 'DI', ...],
        'keywords': ['keywords', 'tags', 'subject', 'KW', 'DE', ...]
    }
```

**CRITICAL ENHANCEMENTS - Data Merging Workflow:**

1. **Cross-Database Field Mapping Matrix:**
```python
DATABASE_FIELD_MAPPING = {
    'pubmed': {
        'TI': 'title',
        'AB': 'abstract',
        'AU': 'authors',
        'TA': 'journal',
        'DP': 'year',
        'AID': 'doi'
    },
    'scopus': {
        'Article Title': 'title',
        'Abstract': 'abstract',
        'Authors': 'authors',
        'Source title': 'journal',
        'Year': 'year',
        'DOI': 'doi'
    },
    'web_of_science': {
        'TI': 'title',
        'AB': 'abstract',
        'AU': 'authors',
        'SO': 'journal',
        'PY': 'year',
        'DI': 'doi'
    },
    'embase': {
        'Title': 'title',
        'Abstract': 'abstract',
        'Author': 'authors',
        'Source': 'journal',
        'Publication Year': 'year',
        'DOI': 'doi'
    }
}
```

2. **Auto-Detection + Manual Refinement UI:**
```python
# Step 1: Auto-detect database source and apply mapping
def detect_database_source(columns: List[str]) -> str:
    """Detect which database the file came from based on column patterns"""
    
# Step 2: Auto-apply standard mapping
def apply_database_mapping(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Standardize columns based on detected database"""
    
# Step 3: Manual adjustment interface (Streamlit UI)
def show_field_mapping_ui(auto_mapping: Dict, columns: List[str]):
    """
    Interactive UI for users to:
    - Review auto-detected mappings
    - Manually adjust incorrect mappings
    - Add mappings for unrecognized fields
    - Save custom mapping for future use
    """
```

3. **Intelligent Enhancements:**
- **Fuzzy Matching**: Use `difflib.SequenceMatcher` or `fuzzywuzzy` for similar field names
- **Learning System**: Remember user corrections, build custom mapping library
- **Confidence Scores**: Return mapping confidence (0-1 scale) for each field
- **Multi-language Support**: Detect Chinese/Spanish field names
- **Validation Warnings**: Alert users to missing critical fields

**New Functions:**
```python
class FieldMapper:
    def detect_database_source(columns: List[str]) -> Tuple[str, float]
    def auto_map_fields(columns: List[str], source: str = None) -> Dict[str, str]
    def suggest_mapping(column: str, target_field: str) -> float
    def validate_mapping(mapping: Dict) -> List[str]  # Returns warnings
    def get_required_fields() -> List[str]
    def get_optional_fields() -> List[str]
    def save_custom_mapping(source_name: str, mapping: Dict) -> None
    def load_custom_mappings() -> Dict[str, Dict]
    def merge_with_standard_schema(df: pd.DataFrame, mapping: Dict) -> pd.DataFrame
```

**UI Workflow for Data Merging:**
```
1. User uploads file → Auto-detect database (PubMed/Scopus/WoS/Embase)
2. Apply standard mapping → Preview mapped columns
3. Show confidence scores → Highlight uncertain mappings (< 0.8)
4. Manual refinement UI → Drag-and-drop field assignment
5. Save mapping → Option to reuse for future files from same database
6. Merge datasets → Standardized schema for all files
```

---

#### 2.3 Data Merger (`data_merger.py`)
**Source:** Lines 4400-4550 from `modern_app.py`

**Current Logic:**
```python
# Merge multiple datasets with field mappings
for file_key, mapping in mappings.items():
    df = load_file(file_key)
    standardized = apply_mapping(df, mapping)
    standardized['source_file'] = file_name
    merged_datasets.append(standardized)

combined = pd.concat(merged_datasets, ignore_index=True)
```

**Enhancements:**
- **Conflict Resolution**: Handle overlapping records across sources
- **Source Tracking**: Preserve provenance for each record
- **Schema Validation**: Ensure all datasets match standard schema

**New Functions:**
```python
class DataMerger:
    def merge_datasets(datasets: List[pd.DataFrame]) -> pd.DataFrame
    def validate_schemas(datasets: List[pd.DataFrame]) -> bool
    def resolve_conflicts(records: List[Dict]) -> Dict
    def track_provenance(df: pd.DataFrame, source: str) -> pd.DataFrame
```

---

#### 2.4 Data Cleaner (`data_cleaner.py`)
**Source:** Lines 4580-4650 from `modern_app.py`

**Current Quality Filters (PRESERVE):**
```python
# Remove records without valid titles
cleaned = df[
    df['title'].notna() & 
    (df['title'].str.strip() != '') &
    (df['title'].str.len() >= 10)
]

# Remove records without abstracts
cleaned = cleaned[
    cleaned['abstract'].notna() & 
    (cleaned['abstract'].str.strip() != '')
]
```

**Enhancements:**
- **Configurable Rules**: User-defined cleaning rules
- **Data Imputation**: Fill missing journal/year with heuristics
- **Language Detection**: Filter non-English articles (optional)

**New Functions:**
```python
class DataCleaner:
    def remove_empty_titles(df: pd.DataFrame) -> pd.DataFrame
    def remove_empty_abstracts(df: pd.DataFrame) -> pd.DataFrame
    def remove_short_titles(df: pd.DataFrame, min_length: int) -> pd.DataFrame
    def standardize_years(df: pd.DataFrame) -> pd.DataFrame
    def clean_author_names(df: pd.DataFrame) -> pd.DataFrame
    def validate_dois(df: pd.DataFrame) -> pd.DataFrame
```

---

#### 2.5 Deduplicator (`deduplicator.py`) ⭐ CRITICAL
**Source:** Lines 4650-4720 from `modern_app.py`

**Current Algorithm (EXCELLENT - PRESERVE FULLY):**
```python
# Strategy 1: DOI-based (exact match)
doi_duplicates = df[df['doi'] != ''].duplicated(subset=['doi'], keep='first')

# Strategy 2: Title similarity (TF-IDF + Cosine)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(cleaned_titles)
cosine_sim = cosine_similarity(tfidf_matrix)

# Threshold: 0.85 similarity
for i in range(len(cosine_sim)):
    for j in range(i + 1, len(cosine_sim)):
        if cosine_sim[i][j] >= 0.85:
            duplicates.append(j)

# Strategy 3: Author-Year-Journal combination
df['combined_key'] = (
    df['authors'].str[:50].str.lower() + "_" +
    df['year'].astype(str) + "_" +
    df['journal'].str[:30].str.lower()
)
combined_duplicates = df.duplicated(subset=['combined_key'], keep='first')
```

**Enhancements:**
- **Performance Optimization**: 
  - Batch processing for large datasets (>10k records)
  - Parallel title comparison using multiprocessing
  - Caching of TF-IDF matrices
- **User Review Interface**: 
  - Show duplicate pairs with confidence scores
  - Allow manual override (keep/remove decisions)
- **Advanced Algorithms**:
  - Levenshtein distance for title comparison
  - BioBERT embeddings for semantic similarity (medical papers)

**New Functions:**
```python
class Deduplicator:
    def find_doi_duplicates(df: pd.DataFrame) -> List[int]
    def find_title_duplicates(df: pd.DataFrame, threshold: float) -> List[Tuple[int, int, float]]
    def find_metadata_duplicates(df: pd.DataFrame) -> List[int]
    def remove_duplicates(df: pd.DataFrame, indices: List[int]) -> pd.DataFrame
    def generate_duplicate_report(df: pd.DataFrame) -> Dict
    def interactive_review(duplicate_pairs: List) -> List[int]  # UI function
```

---

### Phase 3: AI Screening Module (Week 3) ⚡ HIGH PRIORITY
**Goal:** Implement 8-channel parallel screening system

#### 3.1 Parallel Screener (`parallel_screener.py`)
**Current Issue:** Sequential processing (1 article at a time)
**Solution:** ThreadPoolExecutor with 8 workers

**Architecture:**
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading

class ParallelScreener:
    def __init__(self, num_workers: int = 8):
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.results_queue = Queue()
        self.progress_lock = threading.Lock()
        self.total_processed = 0
    
    def screen_batch(self, articles: List[Dict], 
                    criteria: Dict, 
                    progress_callback: Callable = None) -> List[Dict]:
        """
        Screen multiple articles in parallel
        
        Args:
            articles: List of article dictionaries
            criteria: Screening criteria (PCC)
            progress_callback: Function to call with progress updates
            
        Returns:
            List of screening results
        """
        futures = {}
        results = []
        
        # Submit all tasks
        for article in articles:
            future = self.executor.submit(
                self._screen_single, 
                article, 
                criteria
            )
            futures[future] = article
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result(timeout=30)  # 30s timeout per article
                results.append(result)
                
                # Update progress
                with self.progress_lock:
                    self.total_processed += 1
                    if progress_callback:
                        progress_callback(
                            self.total_processed, 
                            len(articles)
                        )
            
            except TimeoutError:
                article = futures[future]
                results.append({
                    'article_id': article.get('id'),
                    'status': 'timeout',
                    'decision': 'manual_review',
                    'confidence': 0.0
                })
            
            except Exception as e:
                article = futures[future]
                results.append({
                    'article_id': article.get('id'),
                    'status': 'error',
                    'error': str(e),
                    'decision': 'manual_review',
                    'confidence': 0.0
                })
        
        return results
    
    def _screen_single(self, article: Dict, criteria: Dict) -> Dict:
        """Screen a single article (called by worker threads)"""
        # Will use AIClient to make API call
        pass
```

**Performance Estimation:**
```
Current:  5,000 articles × 3 seconds = 4.2 hours
Parallel: 5,000 articles ÷ 8 workers × 3 seconds = 31 minutes (8× faster!)
```

**Rate Limiting Strategy:**
```python
from ratelimit import limits, sleep_and_retry

class RateLimiter:
    # Assume Grok allows 100 requests/minute
    @sleep_and_retry
    @limits(calls=100, period=60)
    def make_api_call(self, payload: Dict) -> Dict:
        # API call here
        pass
```

---

#### 3.2 AI Client (`ai_client.py`)
**Source:** Current Grok integration in `modern_app.py`

**Responsibilities:**
- Wrap Grok API calls
- Handle authentication (API keys)
- Retry logic for failed requests
- Response parsing

**Key Functions:**
```python
class AIClient:
    def __init__(self, api_key: str, model: str = "grok-3"):
        self.api_key = api_key
        self.model = model
        self.session = requests.Session()
    
    def screen_article(self, title: str, abstract: str, 
                      criteria: Dict) -> Dict:
        """
        Send screening request to Grok API
        
        Returns:
            {
                'decision': 'include' | 'exclude' | 'manual_review',
                'confidence': float (0-1),
                'reasoning': str,
                'cost': float
            }
        """
        pass
    
    def batch_screen(self, articles: List[Dict], 
                    criteria: Dict) -> List[Dict]:
        """Screen multiple articles (if API supports batch)"""
        pass
```

---

#### 3.3 Screening Queue (`screening_queue.py`)
**Purpose:** Manage large screening jobs with checkpointing

**Features:**
- Save progress every N articles
- Resume from checkpoint if interrupted
- Prioritize articles (e.g., by journal impact factor)

**Key Functions:**
```python
class ScreeningQueue:
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.queue = []
        self.completed = []
    
    def add_articles(self, articles: List[Dict]):
        """Add articles to screening queue"""
        pass
    
    def save_checkpoint(self):
        """Save current progress"""
        pass
    
    def load_checkpoint(self) -> int:
        """Load progress, return number of completed articles"""
        pass
    
    def get_next_batch(self, size: int) -> List[Dict]:
        """Get next batch of articles to screen"""
        pass
```

---

#### 3.4 Result Aggregator (`result_aggregator.py`)
**Purpose:** Collect and summarize screening results

**Key Functions:**
```python
class ResultAggregator:
    def aggregate_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate screening results
        
        Returns:
            {
                'total': int,
                'included': int,
                'excluded': int,
                'manual_review': int,
                'avg_confidence': float,
                'total_cost': float,
                'processing_time': float
            }
        """
        pass
    
    def categorize_by_confidence(self, results: List[Dict]) -> Dict:
        """Group results by confidence level"""
        pass
    
    def identify_edge_cases(self, results: List[Dict], 
                           threshold: float = 0.6) -> List[Dict]:
        """Find articles near decision boundary"""
        pass
```

---

#### 3.5 Cost Tracker (`cost_tracker.py`)
**Purpose:** Estimate and track API costs

**Grok Pricing (estimated - verify with actual docs):**
```python
PRICING = {
    'grok-3': {
        'input': 0.000001,   # per token
        'output': 0.000002
    },
    'grok-3-mini-fast': {
        'input': 0.0000005,
        'output': 0.000001
    },
    'grok-4': {
        'input': 0.000002,
        'output': 0.000004
    }
}
```

**Key Functions:**
```python
class CostTracker:
    def estimate_cost(self, num_articles: int, 
                     avg_title_length: int,
                     avg_abstract_length: int,
                     model: str) -> float:
        """Estimate total cost before screening"""
        pass
    
    def track_usage(self, response: Dict) -> float:
        """Track actual usage from API response"""
        pass
    
    def get_summary(self) -> Dict:
        """Get cost summary"""
        pass
```

---

### Phase 4: Screening Results Module (Week 4)
**Goal:** Enhanced visualization and export

#### 4.1 Result Viewer (`result_viewer.py`)
**Source:** Lines 200-600 from `modern_app.py`

**Enhancements:**
- Real-time progress dashboard during screening
- Interactive data tables with filtering
- Confidence score distribution histogram
- Decision breakdown pie chart

**Visualization Examples:**
```python
import plotly.graph_objects as go
import plotly.express as px

class ResultViewer:
    def create_progress_dashboard(self, results: Dict) -> None:
        """Real-time screening progress"""
        pass
    
    def plot_confidence_distribution(self, results: List[Dict]) -> go.Figure:
        """Histogram of confidence scores"""
        pass
    
    def plot_decision_breakdown(self, results: Dict) -> go.Figure:
        """Pie chart of include/exclude/review"""
        pass
    
    def create_interactive_table(self, df: pd.DataFrame) -> None:
        """Filterable results table"""
        pass
```

---

#### 4.2 PRISMA Generator (`prisma_generator.py`)
**Source:** PRISMA diagram logic from `modern_app.py`

**Current Implementation:** Basic PRISMA-ScR flow
**Enhancements:**
- Export as PNG/SVG/PDF
- Customizable styling
- Multi-language support

**Key Functions:**
```python
class PRISMAGenerator:
    def generate_flow_diagram(self, screening_data: Dict) -> Image:
        """Generate PRISMA-ScR compliant flow diagram"""
        pass
    
    def export_diagram(self, format: str = 'png') -> Path:
        """Export diagram to file"""
        pass
    
    def generate_report(self) -> str:
        """Generate markdown report"""
        pass
```

---

#### 4.3 Export Manager (`export_manager.py`)
**Purpose:** Export results in multiple formats

**Key Functions:**
```python
class ExportManager:
    def export_to_excel(self, results: pd.DataFrame, 
                       output_path: Path) -> None:
        """
        Export with multiple sheets:
        - All results
        - Included articles
        - Excluded articles
        - Manual review needed
        - Summary statistics
        """
        pass
    
    def export_to_csv(self, results: pd.DataFrame, 
                     output_path: Path) -> None:
        pass
    
    def export_for_covidence(self, results: pd.DataFrame) -> Path:
        """Export in format compatible with Covidence"""
        pass
    
    def export_for_rayyan(self, results: pd.DataFrame) -> Path:
        """Export in format compatible with Rayyan"""
        pass
```

---

### Phase 5: Integration & Testing (Week 5)
**Goal:** Connect all modules and create main app

#### 5.1 Main Application (`app.py`)
**Structure:**
```python
import streamlit as st
from data_management import FileHandler, FieldMapper, DataMerger, DataCleaner, Deduplicator
from ai_screening import ParallelScreener, AIClient
from screening_results import ResultViewer, PRISMAGenerator, ExportManager

def main():
    st.set_page_config(page_title="AI Scoping Review", layout="wide")
    
    # Sidebar navigation (simple, no login)
    page = st.sidebar.radio("Navigation", [
        "🏠 Home",
        "📊 Data Management",
        "🤖 AI Screening",
        "📈 Results"
    ])
    
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Data Management":
        show_data_management()
    elif page == "🤖 AI Screening":
        show_ai_screening()
    elif page == "📈 Results":
        show_results()

def show_home():
    """Welcome page with quick start guide"""
    pass

def show_data_management():
    """Data upload, merge, clean, deduplicate"""
    pass

def show_ai_screening():
    """Parallel AI screening interface"""
    pass

def show_results():
    """Results visualization and export"""
    pass

if __name__ == "__main__":
    main()
```

---

#### 5.2 Testing Strategy

**Unit Tests:**
```
tests/
├── test_file_handler.py        # Test all file parsers
├── test_field_mapper.py         # Test mapping logic
├── test_deduplicator.py         # Test deduplication algorithms
├── test_parallel_screener.py    # Test parallel processing
└── test_prisma_generator.py     # Test diagram generation
```

**Integration Tests:**
```
tests/integration/
├── test_full_workflow.py        # End-to-end test
├── test_data_pipeline.py        # Upload → Clean → Deduplicate
└── test_screening_pipeline.py   # Screen → Aggregate → Export
```

**Test Data:**
```
tests/fixtures/
├── sample_pubmed.csv
├── sample_scopus.xlsx
├── sample_wos.ris
├── expected_merged.csv
└── expected_deduplicated.csv
```

---

## 📊 Performance Targets

### Current vs Target Performance

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| **Startup Time** | ~10s (with login) | <3s | 3.3× faster |
| **File Upload (1000 records)** | ~5s | <2s | 2.5× faster |
| **Deduplication (5000 records)** | ~45s | <30s | 1.5× faster |
| **AI Screening (5000 articles)** | ~4.2 hours | ~30 min | 8× faster |
| **Results Export** | ~10s | <5s | 2× faster |
| **Memory Usage** | ~500MB | <300MB | 40% reduction |
| **Code Lines** | 8,103 | <3,000 | 63% reduction |

### Scalability Targets

| Dataset Size | Upload | Deduplicate | Screen (8 workers) | Total Time |
|--------------|--------|-------------|-------------------|------------|
| 1,000 articles | 1s | 5s | 6 min | ~7 min |
| 5,000 articles | 3s | 30s | 30 min | ~31 min |
| 10,000 articles | 6s | 90s | 60 min | ~62 min |
| 50,000 articles | 20s | 10 min | 5 hours | ~5.2 hours |

---

## 🔐 Security & Configuration

### Environment Variables (`.env`)
```bash
# API Configuration
GROK_API_KEY=your_api_key_here
GROK_MODEL=grok-3
GROK_MAX_WORKERS=8

# Screening Configuration
CONFIDENCE_THRESHOLD=0.8
MANUAL_REVIEW_THRESHOLD=0.6

# Performance Settings
MAX_BATCH_SIZE=100
CHECKPOINT_INTERVAL=50

# Optional: Rate Limiting
MAX_REQUESTS_PER_MINUTE=100
```

### Configuration File (`config/default.yaml`)
```yaml
# Data Management
data_management:
  supported_formats: [xlsx, xls, csv, ris, txt]
  max_file_size_mb: 100
  deduplication:
    title_similarity_threshold: 0.85
    use_doi_matching: true
    use_metadata_matching: true

# AI Screening
ai_screening:
  default_model: grok-3
  num_workers: 8
  timeout_seconds: 30
  retry_attempts: 3
  checkpoint_every: 50

# Results
results:
  prisma_format: scr
  export_formats: [xlsx, csv]
  include_confidence_scores: true
```

---

## 🚨 Risk Management

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **API Rate Limits** | High | Medium | Implement adaptive rate limiting, queue system |
| **Memory Issues (Large Files)** | High | Medium | Stream processing, chunking |
| **Parallel Processing Bugs** | Medium | Low | Extensive testing, fallback to sequential |
| **Data Loss During Screening** | High | Low | Frequent checkpointing, autosave |
| **Deduplication False Positives** | Medium | Medium | User review interface, adjustable thresholds |

### Migration Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Missing Features** | Medium | Medium | Comprehensive feature audit before migration |
| **Performance Regression** | High | Low | Benchmarking, performance tests |
| **User Confusion** | Low | High | Clear documentation, migration guide |
| **Data Compatibility** | Medium | Medium | Support old data formats during transition |

---

## 📅 Timeline & Milestones (FastAPI + Streamlit Architecture)

### Week 1: Foundation & Backend Setup ✅ COMPLETE
- [x] Create directory structure (backend/, frontend/, shared/)
- [x] Write migration plan with FastAPI architecture
- [x] Create backend/main.py skeleton
- [x] Create frontend/app.py skeleton
- [x] Setup basic project structure
- [x] Create .env file with XAI API configuration

**Deliverables:** Complete project skeleton with 11 files

---

### Week 2: Backend Core - Data Management ✅ COMPLETE
**Focus: FastAPI endpoints + core processing logic**

- [x] **Backend Data Models** (Pydantic)
  - [x] Article model
  - [x] UploadRequest/Response models
  - [x] ProcessingStatus models

- [x] **Core Processing Logic** (backend/core/)
  - [x] DataProcessor class (parse files) - 420 lines
  - [x] FieldMapper class (intelligent mapping) - From Week 1
  - [x] Deduplicator class ⭐ (DOI + TF-IDF logic) - 450 lines
  - [x] DataMerger class - 430 lines
  
- [x] **API Endpoints** (backend/api/data_management.py) - 500+ lines
  - [x] POST /upload (file upload)
  - [x] GET /parse/{id} (parse and validate)
  - [x] POST /auto-map (auto field detection)
  - [x] POST /map-fields (apply field mappings)
  - [x] POST /merge (merge datasets)
  - [x] POST /deduplicate (run deduplication)
  - [x] GET /datasets (list uploaded datasets)
  - [x] GET /preview/{id} (preview dataset)
  - [x] GET /columns/{id} (get column names)
  - [x] GET /export/{id} (export dataset)
  - [x] DELETE /dataset/{id} (delete dataset)
  
- [ ] **Unit Tests** (Pending)
  - [ ] Test file parsing (Excel, CSV, RIS)
  - [ ] Test field mapping accuracy
  - [ ] Test deduplication algorithms

**Deliverables:** Complete data management backend (11 API endpoints)

---

### Week 3: Backend AI Screening + Task Management ✅ COMPLETE
**Focus: Asynchronous screening with 8 parallel workers**

- [x] **AI Screening Engine** (backend/core/screener.py) - 550+ lines
  - [x] AIScreener class with ThreadPoolExecutor (8 workers)
  - [x] Individual article screening logic
  - [x] Confidence scoring and decision logic
  - [x] Error handling and retry mechanism
  - [x] Checkpoint/resume functionality
  - [x] Cost tracking (HKD pricing)
  
- [x] **Grok API Client** (backend/services/grok_client.py) - 280 lines
  - [x] API authentication and request handling
  - [x] Rate limiting implementation (100ms delay)
  - [x] Cost tracking per request
  - [x] Timeout and error handling
  - [x] Exponential backoff retry logic
  - [x] Support for Grok 4 Fast models

- [x] **Task Manager** (backend/tasks/task_manager.py) - 300 lines
  - [x] In-memory task state storage
  - [x] Task creation and ID generation
  - [x] Progress tracking (current/total)
  - [x] Checkpoint/resume functionality
  - [x] Thread-safe operations with lock
  
- [x] **Screening API Endpoints** (backend/api/screening.py) - 600+ lines
  - [x] POST /start (initiate screening task)
  - [x] GET /status/{task_id} (poll progress)
  - [x] GET /results/{task_id} (retrieve results)
  - [x] POST /cancel/{task_id} (cancel running task)
  - [x] GET /checkpoint/{task_id} (get checkpoint data)
  - [x] POST /resume/{task_id} (resume from checkpoint)
  - [x] GET /estimate-cost/{task_id} (cost estimation)
  
- [ ] **Integration Tests** (Pending)
  - [ ] Test parallel screening with mock data
  - [ ] Test task lifecycle (start → progress → complete)
  - [ ] Test checkpoint/resume functionality

**Deliverables:** Complete AI screening backend (7 API endpoints, 8-worker parallel processing)

---

### Week 4: Frontend Development ✅ COMPLETE (100%)
**Focus: Streamlit UI connecting to FastAPI backend**

- [x] **Frontend API Client** (frontend/utils/api_client.py) - COMPLETE
  - [x] HTTP client wrapper (requests)
  - [x] 19 API endpoint methods (data + screening)
  - [x] Error handling and retry logic
  - [x] Response parsing
  - [x] Fixed file upload to handle Streamlit UploadedFile objects
  - [x] Added content-type detection for different file formats

- [x] **Streamlit Pages** - ALL 5 PAGES COMPLETE ✨
  - [x] **Home page** (app.py) - COMPLETE (200+ lines)
    - [x] Beautiful gradient header
    - [x] 4 feature cards (Data Management, AI Screening, Results, Settings)
    - [x] Quick start guide with workflow steps
    - [x] Performance benchmarks display
    - [x] System status checker
    - [x] Help section with requirements
    - [x] Clean UI without HTML feature-card divs
  
  - [x] **Data Management page** (pages/1_Data_Management.py) - COMPLETE (500+ lines)
    - [x] File upload component with progress bars
    - [x] Field mapping UI with auto-detection
    - [x] Merge interface with checkbox selection
    - [x] Deduplicate interface with strategy toggles
    - [x] Beautiful gradient styling (no emojis)
    - [x] 4-step workflow with progress indicators
  
  - [x] **AI Screening page** (pages/2_AI_Screening.py) - COMPLETE (550+ lines) ✨
    - [x] Inclusion/Exclusion criteria input (replaced PCC framework)
    - [x] Criteria preview with parsed format display
    - [x] Auto-load default criteria from Settings
    - [x] Model selection (Grok 4 Fast reasoning/non-reasoning)
    - [x] Worker configuration slider (1-8 workers)
    - [x] Dataset selection from uploaded files
    - [x] Cost estimation button
    - [x] Real-time progress tracking with auto-refresh (2s interval)
    - [x] Live decision breakdown (relevant/irrelevant/uncertain)
    - [x] Animated progress bars and metrics
    - [x] Cost accumulation display (HKD)
    - [x] Pause/Resume/Cancel controls
    - [x] Completion handling with navigation to Results
    - [x] Criteria parsing function (text → list of strings)
  
  - [x] **Results page** (pages/3_Results.py) - COMPLETE (400+ lines)
    - [x] Task ID input for loading results
    - [x] Summary metrics (4 color-coded cards)
    - [x] Interactive results table
    - [x] Filters (decision type, confidence, keyword search)
    - [x] Screening details in expandable section
    - [x] Export placeholders (Excel/CSV - coming soon)
    - [x] PRISMA diagram placeholder (coming soon)
    - [x] Navigation buttons (Start New Screening, Back to Data Management)
  
  - [x] **Settings page** (pages/4_Settings.py) - COMPLETE (470+ lines) ✨
    - [x] API Configuration tab
      - [x] XAI API key input (password field)
      - [x] Auto-detect existing key from .env
      - [x] Save API key to .env file
      - [x] Display masked key for security
      - [x] Instructions on getting API key
      - [x] Multi-user deployment notes
    - [x] Default Criteria tab
      - [x] Inclusion criteria text area
      - [x] Exclusion criteria text area
      - [x] Save/Clear/Load Example buttons
      - [x] Tips for writing good criteria
      - [x] Best practices guide
    - [x] Preferences tab
      - [x] Default AI model selection
      - [x] Default workers slider
      - [x] Auto-save toggle
      - [x] Reset to defaults button
    - [x] Settings persistence (config/user_settings.json)

- [x] **Session Management** - COMPLETE
  - [x] Dataset tracking in session state
  - [x] Current step progression
  - [x] Task ID persistence for screening
  - [x] Settings loading from JSON file
  - [x] Criteria auto-population from settings

**Current Status:** All 5 frontend pages complete with beautiful UI, functional Settings page with API key management

**Key Achievements:**
- ✅ Replaced PCC framework with standard Inclusion/Exclusion criteria
- ✅ Added comprehensive Settings page for API keys and default criteria
- ✅ Implemented criteria parsing (text → array) for backend compatibility
- ✅ Fixed file upload to support Streamlit UploadedFile objects
- ✅ Removed unused HTML feature-card div tags
- ✅ Created cohesive UI design across all pages
- ✅ Real-time progress tracking for AI screening
- ✅ Beautiful gradient styling throughout

**Week 4 Deliverables:** ✅ 5 complete Streamlit pages with beautiful UI and Settings management

---

### Week 5: Testing & Integration 🔄 IN PROGRESS (30%)
**Focus: System integration and comprehensive testing**

- [x] **Environment Setup**
  - [x] Created virtual environment in Core function directory
  - [x] Installed all 81 packages (fastapi, streamlit, pandas, scikit-learn, etc.)
  - [x] Configured Python 3.13.9
  - [x] Added python-multipart for file uploads
  - [x] Fixed import errors (FieldMappingResult)

- [x] **Backend Testing**
  - [x] Backend server starts without errors
  - [x] All 25 API endpoints accessible (/docs verified)
  - [x] Debug logging middleware working
  - [x] Request tracking functional (logs method, path, headers, timing)
  - [x] Health check endpoint responding (200 OK)
  
- [x] **Frontend Testing**
  - [x] Streamlit server starts successfully
  - [x] Home page loads with 4 feature cards
  - [x] Navigation between pages working
  - [x] Settings page loads and displays
  - [x] Fixed file upload (now handles UploadedFile objects)
  
- [ ] **End-to-End Integration** (IN PROGRESS)
  - [x] Frontend connects to backend successfully
  - [ ] Full workflow test (upload → map → merge → deduplicate → screen → export)
  - [ ] Cross-page navigation testing
  - [ ] Session state persistence testing
  - [ ] Error recovery testing
  
- [ ] **Unit Testing** (PENDING)
  - [ ] Test data processing modules (parser, mapper, deduplicator)
  - [ ] Test screening engine and task manager
  - [ ] Test API client methods
  - [ ] Target: >70% code coverage
  
- [ ] **Integration Testing** (PENDING)
  - [ ] Test API endpoints with real data
  - [ ] Test parallel screening (8 workers)
  - [ ] Test checkpoint/resume functionality
  - [ ] Test cost calculation accuracy
  
- [ ] **Performance Testing** (PENDING)
  - [ ] Benchmark file upload (1k, 5k, 10k records)
  - [ ] Benchmark deduplication (1k, 5k, 10k records)
  - [ ] Benchmark screening speed (target: 5k articles in <1 hour)
  - [ ] Memory profiling
  
- [ ] **Bug Fixes & Refinements**
  - [x] Fixed file upload to handle Streamlit UploadedFile
  - [x] Added content-type detection for file uploads
  - [x] Fixed API client Content-Type header for multipart uploads
  - [x] Removed unused CSS (feature-card, step-number)
  - [ ] UI/UX improvements based on testing feedback
  - [ ] Performance optimizations

**Current Status:** Backend and frontend both running, testing file upload functionality

**Recent Fixes:**
- Fixed UploadedFile upload error by updating API client to handle both file paths and UploadedFile objects
- Added proper content-type handling for different file formats (xlsx, xls, csv, ris)
- Removed Content-Type header conflict for multipart/form-data uploads

**Next Steps:**
1. Test file upload with real datasets
2. Test complete data management workflow
3. Test AI screening with small dataset
4. Performance benchmarking
5. Bug fixes and refinements

**Week 5 Deliverables:** Fully tested, integrated system ready for deployment

---

### Week 6: Documentation, Deployment & Launch 🎯 PENDING
**Focus: Production readiness and documentation**

- [ ] **Documentation**
  - [ ] Complete README.md with installation guide
  - [ ] API documentation (FastAPI auto-docs)
  - [ ] User guide with screenshots
  - [ ] Video tutorial (optional)
  - [ ] Troubleshooting guide
  
- [ ] **Deployment Preparation**
  - [ ] Create requirements.txt (finalize dependencies)
  - [ ] Docker containerization (optional)
  - [ ] Deployment scripts (start backend + frontend)
  - [ ] Environment setup guide
  
- [ ] **Beta Testing**
  - [ ] Test with real datasets from users
  - [ ] Collect feedback on UI/UX
  - [ ] Identify edge cases
  - [ ] Performance validation
  
- [ ] **Final Polish**
  - [ ] Code cleanup and refactoring
  - [ ] Remove debug code
  - [ ] Final UI improvements
  - [ ] Error message improvements
  
- [ ] **Launch Preparation**
  - [ ] Create release notes
  - [ ] Prepare demo dataset
  - [ ] Training materials for users
  - [ ] Migration guide from old system

**Week 6 Deliverables:** Production-ready system with complete documentation

---

## 📊 Progress Summary

### ✅ Completed (Weeks 1-3)
- **17 files created** (~5,000+ lines of code)
- **Backend API**: 18 endpoints fully implemented (11 data + 7 screening)
- **Core modules**: DataProcessor, FieldMapper, DataMerger, Deduplicator, AIScreener, GrokClient, TaskManager
- **Frontend**: APIClient + 1 complete page (Data Management)
- **Features**: 8-worker parallel screening, checkpoint/resume, cost tracking (HKD), multi-strategy deduplication

### 🔄 In Progress (Week 4)
- **Frontend UI Development** (15% complete)
  - ✅ Data Management page
  - 🔜 AI Screening page (next priority)
  - 🔜 Results page
  - 🔜 Home page

### ⏳ Pending (Weeks 5-6)
- Testing & Integration
- Documentation
- Deployment preparation
- Beta testing

---

## 🎯 Immediate Next Steps (Week 4 Continuation)

### Priority 1: AI Screening Page (2-3 hours)
Create `frontend/pages/2_AI_Screening.py` with:
- PCC criteria input (Population, Concept, Context)
- Model selection dropdown (Grok 4 Fast reasoning/non-reasoning)
- Worker configuration slider (1-8 workers)
- Dataset selection (from uploaded/merged datasets)
- Cost estimation before starting
- Start screening button
- Real-time progress tracking:
  - Animated progress bar
  - Current/total articles counter
  - Live decision breakdown (relevant/irrelevant/uncertain)
  - Estimated time remaining
  - Cost accumulation (HKD)
- Pause/Resume/Cancel controls
- Beautiful styling matching Data Management page

### Priority 2: Results Page (2-3 hours)
Create `frontend/pages/3_Results.py` with:
- Screening task selector (view completed screenings)
- Results summary cards:
  - Total articles processed
  - Relevant/Irrelevant/Uncertain counts
  - Total cost (HKD)
  - Processing time
- Results table with filters:
  - Filter by decision (relevant/irrelevant/uncertain)
  - Filter by confidence range
  - Search by title/keywords
  - Sortable columns
- Visualization charts:
  - Confidence score distribution (histogram)
  - Decision breakdown (pie chart)
  - Processing timeline (line chart)
- PRISMA diagram generation button
- Export options:
  - Excel (multiple sheets)
  - CSV (all results)
  - Filtered results

### Priority 3: Home Page Enhancement (1 hour)
Update `frontend/app.py` with:
- Welcome message and project overview
- Quick start guide (step-by-step instructions)
- System status indicators
- Recent activity log
- Navigation buttons to main pages
- Feature highlights

### Priority 4: Testing & Refinement (2 hours)
- Test complete workflow end-to-end
- Fix any bugs discovered
- UI/UX improvements
- Error message refinement

---

## 📚 Documentation Requirements

### User Documentation
1. **README.md**: Quick start, installation, basic usage
2. **USER_GUIDE.md**: Comprehensive user manual with screenshots
3. **FAQ.md**: Common questions and troubleshooting
4. **CHANGELOG.md**: Version history and updates

### Developer Documentation
1. **API_REFERENCE.md**: Module APIs and function signatures
2. **DEVELOPMENT_GUIDE.md**: Setup dev environment, contribution guidelines
3. **ARCHITECTURE.md**: System design and data flow diagrams
4. **TESTING_GUIDE.md**: How to run tests, add new tests

### Video Tutorials (Optional)
1. Data upload and merging (5 min)
2. AI screening workflow (10 min)
3. Results analysis and export (5 min)

---

## 🎯 Success Criteria

### Functional Requirements ✅
- [ ] Upload and parse Excel/CSV/RIS files
- [ ] Auto-map fields with >90% accuracy
- [ ] Merge multiple datasets preserving metadata
- [ ] Remove duplicates with <1% false positive rate
- [ ] Screen 5,000 articles in <1 hour (8 workers)
- [ ] Generate PRISMA-ScR compliant diagrams
- [ ] Export results in Excel/CSV formats

### Non-Functional Requirements ✅
- [ ] Startup time <3 seconds
- [ ] Support datasets up to 50,000 articles
- [ ] Memory usage <500MB for 10,000 articles
- [ ] 99.9% uptime during screening
- [ ] Graceful error handling (no crashes)
- [ ] Mobile-responsive UI (basic support)

### Code Quality ✅
- [ ] >80% test coverage
- [ ] All modules have docstrings
- [ ] Type hints for all public functions
- [ ] Pass flake8 linting
- [ ] <500 lines per module (average)

---

## 🔄 Backward Compatibility

### Data Migration Strategy
**Goal:** Users with existing projects can migrate to new system

**Migration Script:** `scripts/migrate_from_v1.py`
```python
def migrate_project(old_project_path: Path, new_data_dir: Path):
    """
    Migrate old project data to new format
    
    Steps:
    1. Load old standardized_data
    2. Load old screening_results
    3. Convert to new schema
    4. Save in new format
    """
    pass
```

**Supported Migrations:**
- `standardized_data.xlsx` → new merged data format
- `screening_results.xlsx` → new results format
- `ai_config.yaml` → new config format

---

## 💡 Future Enhancements (Post-Launch)

### Phase 2 Features (3-6 months)

#### 1. **Dual-LLM Deliberative Evaluation System with Iterative Discussion** ⭐ HIGH PRIORITY
> **Key Requirement:** "Two LLMs discuss and iteratively refine screening decisions through structured dialogue"

**Purpose:** Add a second layer of quality assurance with LLM-to-LLM discussion mechanism for consensus-building

---

## 📋 Implementation Plan Overview

This feature will be implemented in **5 major sections** over 3-6 months:

1. **System Architecture & Data Models** (Weeks 1-2)
2. **Core Evaluation Engine** (Weeks 3-5)
3. **Discussion Loop Mechanism** (Weeks 6-8)
4. **Frontend "Evaluation" Page** (Weeks 9-10)
5. **Testing, Optimization & Deployment** (Weeks 11-12)

---

## 📐 Section 1: System Architecture & Data Models (Weeks 1-2)

### 1.1 Conceptual Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ARTICLE + CRITERIA INPUT                          │
│              (Title, Abstract, Inclusion/Exclusion Criteria)         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PRIMARY LLM (Screener)                            │
│                    Model: grok-4-fast-reasoning                      │
│                                                                       │
│  Task: Initial screening decision                                    │
│  Output: {decision, reasoning, confidence}                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 EVALUATION LLM (Quality Checker)                     │
│                    Model: grok-4-fast-reasoning                      │
│                                                                       │
│  Input: Article + Criteria + Primary Decision                        │
│  Task: Evaluate decision quality using rubric                        │
│  Output: {                                                           │
│    evaluation_result: "agree" | "disagree" | "human_needed",        │
│    reasoning: "Why agree/disagree",                                  │
│    concerns: ["List of specific concerns"],                          │
│    suggestion: "What primary LLM should reconsider"                  │
│  }                                                                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
         ┌─────────┐   ┌──────────┐   ┌────────────┐
         │ AGREE   │   │ DISAGREE │   │   HUMAN    │
         │         │   │          │   │   NEEDED   │
         └────┬────┘   └─────┬────┘   └──────┬─────┘
              │              │                │
              │              │                │
              ▼              ▼                ▼
         ┌─────────┐   ┌──────────────────────────────┐
         │ ACCEPT  │   │  DISCUSSION LOOP             │
         │ RESULT  │   │  (Max 3 iterations)          │
         └─────────┘   │                              │
                       │  ┌────────────────────────┐  │
                       │  │ Round 1:               │  │
                       │  │ - Eval gives feedback  │  │
                       │  │ - Primary reconsiders  │  │
                       │  │ - Check agreement      │  │
                       │  └───────────┬────────────┘  │
                       │              │ No consensus  │
                       │  ┌───────────▼────────────┐  │
                       │  │ Round 2:               │  │
                       │  │ - More detailed debate │  │
                       │  │ - Primary adjusts      │  │
                       │  │ - Re-evaluate          │  │
                       │  └───────────┬────────────┘  │
                       │              │ No consensus  │
                       │  ┌───────────▼────────────┐  │
                       │  │ Round 3 (Final):       │  │
                       │  │ - Final arguments      │  │
                       │  │ - Force decision or    │  │
                       │  │ - Flag for human       │  │
                       │  └────────────────────────┘  │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │   FINAL OUTPUT:              │
                       │   - Consensus decision       │
                       │   - Discussion history       │
                       │   - Confidence score         │
                       │   - Human review flag        │
                       └──────────────────────────────┘
```

### 1.2 Database Schema Design

**New Tables to Add:**

```sql
-- Table 1: Evaluation Sessions
CREATE TABLE evaluation_sessions (
    id TEXT PRIMARY KEY,                    -- UUID
    screening_result_id TEXT NOT NULL,      -- FK to screening_results
    article_title TEXT NOT NULL,
    article_abstract TEXT NOT NULL,
    criteria TEXT NOT NULL,                 -- JSON: {inclusion: [], exclusion: []}
    
    -- Primary LLM output
    primary_decision TEXT NOT NULL,         -- "include" | "exclude"
    primary_reasoning TEXT NOT NULL,
    primary_confidence REAL NOT NULL,
    
    -- Evaluation result
    evaluation_result TEXT NOT NULL,        -- "agree" | "disagree" | "human_needed"
    
    -- Final outcome
    final_decision TEXT NOT NULL,           -- Final consensus decision
    final_confidence REAL NOT NULL,         -- Final confidence (0-1)
    requires_human_review BOOLEAN NOT NULL, -- Flag for manual review
    discussion_rounds INTEGER DEFAULT 0,    -- Number of discussion rounds (0-3)
    
    -- Metadata
    total_cost REAL DEFAULT 0.0,           -- Total API cost (HKD)
    processing_time REAL DEFAULT 0.0,      -- Total time (seconds)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (screening_result_id) REFERENCES screening_results(id)
);

-- Table 2: Discussion History (for iterative loop)
CREATE TABLE discussion_rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_session_id TEXT NOT NULL,
    round_number INTEGER NOT NULL,         -- 1, 2, or 3
    
    -- Round input
    input_decision TEXT NOT NULL,          -- Current decision being evaluated
    input_reasoning TEXT NOT NULL,         -- Current reasoning
    
    -- Evaluator response
    evaluator_result TEXT NOT NULL,        -- "agree" | "disagree" | "human_needed"
    evaluator_reasoning TEXT NOT NULL,     -- Why evaluator agrees/disagrees
    evaluator_concerns TEXT,               -- JSON: ["concern1", "concern2", ...]
    evaluator_suggestion TEXT,             -- Specific suggestion for reconsideration
    
    -- Primary LLM response (if disagreed)
    primary_response TEXT,                 -- New reasoning after reconsideration
    primary_new_decision TEXT,             -- New decision (may change)
    primary_new_confidence REAL,           -- New confidence
    
    -- Round outcome
    consensus_reached BOOLEAN NOT NULL,    -- Did they agree this round?
    
    -- Metadata
    round_cost REAL DEFAULT 0.0,
    round_time REAL DEFAULT 0.0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (evaluation_session_id) REFERENCES evaluation_sessions(id)
);

-- Table 3: Evaluation Rubric Scores
CREATE TABLE rubric_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_session_id TEXT NOT NULL,
    
    -- Rubric criteria (0-5 scale each)
    relevance_score REAL NOT NULL,         -- How relevant to criteria
    reasoning_quality_score REAL NOT NULL, -- Quality of reasoning
    evidence_score REAL NOT NULL,          -- Supporting evidence from abstract
    consistency_score REAL NOT NULL,       -- Internal consistency
    completeness_score REAL NOT NULL,      -- Completeness of analysis
    
    -- Weighted scores
    weighted_total REAL NOT NULL,          -- Sum of weighted scores
    
    -- Evaluator notes
    relevance_notes TEXT,
    reasoning_notes TEXT,
    evidence_notes TEXT,
    consistency_notes TEXT,
    completeness_notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (evaluation_session_id) REFERENCES evaluation_sessions(id)
);
```

### 1.3 Pydantic Data Models

**File: `backend/models/evaluation.py`** (NEW)

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
from enum import Enum

class EvaluationResult(str, Enum):
    """Possible evaluation outcomes"""
    AGREE = "agree"
    DISAGREE = "disagree"
    HUMAN_NEEDED = "human_needed"

class DecisionType(str, Enum):
    """Screening decisions"""
    INCLUDE = "include"
    EXCLUDE = "exclude"
    UNCERTAIN = "uncertain"

# ============================================
# Core Models
# ============================================

class PrimaryDecision(BaseModel):
    """Primary LLM screening decision"""
    decision: DecisionType
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time: float = 0.0
    cost: float = 0.0

class RubricScore(BaseModel):
    """Evaluation rubric scores"""
    relevance_score: float = Field(ge=0.0, le=5.0, description="Relevance to criteria")
    reasoning_quality_score: float = Field(ge=0.0, le=5.0, description="Quality of reasoning")
    evidence_score: float = Field(ge=0.0, le=5.0, description="Supporting evidence")
    consistency_score: float = Field(ge=0.0, le=5.0, description="Internal consistency")
    completeness_score: float = Field(ge=0.0, le=5.0, description="Completeness of analysis")
    
    # Notes for each criterion
    relevance_notes: Optional[str] = None
    reasoning_notes: Optional[str] = None
    evidence_notes: Optional[str] = None
    consistency_notes: Optional[str] = None
    completeness_notes: Optional[str] = None
    
    # Weights (should sum to 1.0)
    WEIGHTS = {
        'relevance': 0.35,
        'reasoning_quality': 0.25,
        'evidence': 0.20,
        'consistency': 0.15,
        'completeness': 0.05
    }
    
    @property
    def weighted_total(self) -> float:
        """Calculate weighted total score"""
        return (
            self.relevance_score * self.WEIGHTS['relevance'] +
            self.reasoning_quality_score * self.WEIGHTS['reasoning_quality'] +
            self.evidence_score * self.WEIGHTS['evidence'] +
            self.consistency_score * self.WEIGHTS['consistency'] +
            self.completeness_score * self.WEIGHTS['completeness']
        )

class EvaluatorResponse(BaseModel):
    """Evaluator LLM response"""
    result: EvaluationResult
    reasoning: str
    concerns: List[str] = Field(default_factory=list)
    suggestion: Optional[str] = None
    rubric_scores: RubricScore
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time: float = 0.0
    cost: float = 0.0

class DiscussionRound(BaseModel):
    """Single round of discussion between LLMs"""
    round_number: int = Field(ge=1, le=3)
    
    # Input to this round
    input_decision: DecisionType
    input_reasoning: str
    input_confidence: float
    
    # Evaluator response
    evaluator_response: EvaluatorResponse
    
    # Primary LLM counter-response (if disagreed)
    primary_response: Optional[str] = None
    primary_new_decision: Optional[DecisionType] = None
    primary_new_confidence: Optional[float] = None
    
    # Round outcome
    consensus_reached: bool = False
    round_cost: float = 0.0
    round_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)

class EvaluationSession(BaseModel):
    """Complete evaluation session with discussion history"""
    id: str
    screening_result_id: str
    
    # Article info
    article_title: str
    article_abstract: str
    criteria: dict  # {inclusion: [], exclusion: []}
    
    # Primary decision
    primary_decision: PrimaryDecision
    
    # Discussion rounds (0-3)
    discussion_rounds: List[DiscussionRound] = Field(default_factory=list)
    
    # Final outcome
    final_decision: DecisionType
    final_confidence: float = Field(ge=0.0, le=1.0)
    final_reasoning: str
    requires_human_review: bool = False
    
    # Metadata
    total_cost: float = 0.0
    processing_time: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)

# ============================================
# API Request/Response Models
# ============================================

class StartEvaluationRequest(BaseModel):
    """Request to start evaluation"""
    screening_result_id: str
    max_discussion_rounds: int = Field(default=3, ge=1, le=5)
    enable_discussion: bool = True  # If False, single evaluation only
    
class EvaluationProgressResponse(BaseModel):
    """Progress update during evaluation"""
    evaluation_session_id: str
    status: Literal["running", "completed", "failed"]
    current_round: int
    max_rounds: int
    consensus_reached: bool
    estimated_time_remaining: Optional[float] = None
    current_cost: float = 0.0

class EvaluationResultResponse(BaseModel):
    """Final evaluation result"""
    session: EvaluationSession
    summary: dict  # Summary statistics
    visualization_data: Optional[dict] = None  # Data for charts
```

### 1.4 Configuration Settings

**File: `backend/core/evaluation_config.py`** (NEW)

```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class EvaluationConfig:
    """Configuration for evaluation system"""
    
    # Models
    primary_model: str = "grok-4-fast-reasoning"
    evaluator_model: str = "grok-4-fast-reasoning"
    
    # Discussion loop settings
    max_discussion_rounds: int = 3
    enable_discussion: bool = True
    consensus_threshold: float = 0.8  # Agreement score needed for consensus
    
    # Rubric weights (must sum to 1.0)
    rubric_weights: Dict[str, float] = None
    
    # Confidence thresholds
    high_confidence_threshold: float = 0.8  # Auto-accept
    low_confidence_threshold: float = 0.6   # Auto-review
    
    # Cost control
    max_cost_per_article: float = 0.50  # HKD
    
    # Timeouts
    llm_timeout_seconds: int = 30
    max_total_time_seconds: int = 120
    
    def __post_init__(self):
        if self.rubric_weights is None:
            self.rubric_weights = {
                'relevance': 0.35,
                'reasoning_quality': 0.25,
                'evidence': 0.20,
                'consistency': 0.15,
                'completeness': 0.05
            }
        
        # Validate weights sum to 1.0
        total = sum(self.rubric_weights.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Rubric weights must sum to 1.0, got {total}")

# Default configuration
DEFAULT_EVAL_CONFIG = EvaluationConfig()
```

### 1.5 Database Migration Script

**File: `backend/db/migrations/add_evaluation_tables.py`** (NEW)

```python
"""
Migration: Add evaluation system tables
Run with: python -m backend.db.migrations.add_evaluation_tables
"""

from backend.db.config import get_engine
from sqlalchemy import text

def upgrade():
    """Add evaluation tables"""
    engine = get_engine()
    
    with engine.begin() as conn:
        # Evaluation sessions table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS evaluation_sessions (
                id TEXT PRIMARY KEY,
                screening_result_id TEXT NOT NULL,
                article_title TEXT NOT NULL,
                article_abstract TEXT NOT NULL,
                criteria TEXT NOT NULL,
                
                primary_decision TEXT NOT NULL,
                primary_reasoning TEXT NOT NULL,
                primary_confidence REAL NOT NULL,
                
                evaluation_result TEXT NOT NULL,
                
                final_decision TEXT NOT NULL,
                final_confidence REAL NOT NULL,
                requires_human_review BOOLEAN NOT NULL,
                discussion_rounds INTEGER DEFAULT 0,
                
                total_cost REAL DEFAULT 0.0,
                processing_time REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (screening_result_id) REFERENCES screening_results(id)
            )
        """))
        
        # Discussion rounds table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS discussion_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_session_id TEXT NOT NULL,
                round_number INTEGER NOT NULL,
                
                input_decision TEXT NOT NULL,
                input_reasoning TEXT NOT NULL,
                
                evaluator_result TEXT NOT NULL,
                evaluator_reasoning TEXT NOT NULL,
                evaluator_concerns TEXT,
                evaluator_suggestion TEXT,
                
                primary_response TEXT,
                primary_new_decision TEXT,
                primary_new_confidence REAL,
                
                consensus_reached BOOLEAN NOT NULL,
                
                round_cost REAL DEFAULT 0.0,
                round_time REAL DEFAULT 0.0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (evaluation_session_id) REFERENCES evaluation_sessions(id)
            )
        """))
        
        # Rubric scores table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS rubric_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_session_id TEXT NOT NULL,
                
                relevance_score REAL NOT NULL,
                reasoning_quality_score REAL NOT NULL,
                evidence_score REAL NOT NULL,
                consistency_score REAL NOT NULL,
                completeness_score REAL NOT NULL,
                
                weighted_total REAL NOT NULL,
                
                relevance_notes TEXT,
                reasoning_notes TEXT,
                evidence_notes TEXT,
                consistency_notes TEXT,
                completeness_notes TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (evaluation_session_id) REFERENCES evaluation_sessions(id)
            )
        """))
        
        print("✅ Evaluation tables created successfully")

def downgrade():
    """Remove evaluation tables"""
    engine = get_engine()
    
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS rubric_scores"))
        conn.execute(text("DROP TABLE IF EXISTS discussion_rounds"))
        conn.execute(text("DROP TABLE IF EXISTS evaluation_sessions"))
        
        print("✅ Evaluation tables removed successfully")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "downgrade":
        downgrade()
    else:
        upgrade()
```

### 1.6 Week 1-2 Deliverables Checklist

- [ ] **Database Schema** (1 day)
  - [ ] Create migration script
  - [ ] Test table creation
  - [ ] Verify foreign key constraints
  - [ ] Add indexes for performance

- [ ] **Pydantic Models** (1 day)
  - [ ] Define all data models
  - [ ] Add validation rules
  - [ ] Write model tests
  - [ ] Document model fields

- [ ] **Configuration** (0.5 days)
  - [ ] Create evaluation config
  - [ ] Define default settings
  - [ ] Add config validation

- [ ] **Documentation** (0.5 days)
  - [ ] Document architecture
  - [ ] Create data flow diagrams
  - [ ] Write usage examples

- [ ] **Testing** (2 days)
  - [ ] Write unit tests for models
  - [ ] Test database operations
  - [ ] Validate configuration
  - [ ] Integration tests

**Total Estimated Time: Week 1-2 (5 working days)**

---

## 🔄 Next Sections Preview

**Section 2: Core Evaluation Engine** (Coming next)
- Evaluator LLM prompt engineering
- Rubric scoring implementation
- Single-pass evaluation (no discussion)
- Cost tracking and optimization

**Section 3: Discussion Loop Mechanism** (Week 6-8)
- Iterative discussion protocol
- Consensus detection algorithm
- Loop termination logic
- Human intervention triggers

**Section 4: Frontend "Evaluation" Page** (Week 9-10)
- Evaluation dashboard UI
- Real-time discussion viewer
- Results visualization
- Manual review interface

**Section 5: Testing & Deployment** (Week 11-12)
- End-to-end testing
- Performance optimization
- Cost analysis
- Production deployment 
- Screening confidence is "intuitive" - single LLM makes subjective judgments
- No systematic quality control on screening decisions
- Confidence scores lack transparency (how is 0.85 confidence calculated?)

**Proposed Architecture: Dual-LLM System**

```
┌─────────────────────────────────────────────────────────┐
│                     Article Input                        │
│              (Title + Abstract + Criteria)               │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│              PRIMARY SCREENING LLM (Grok-3)             │
│                                                          │
│  Task: Screen article based on PCC criteria             │
│  Output:                                                 │
│    - Decision: Include/Exclude                          │
│    - Reasoning: Why this decision was made              │
│    - Initial Confidence: 0.0-1.0                        │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│           EVALUATION LLM (Grok-3 or Grok-4)             │
│                                                          │
│  Task: Evaluate screening decision using rubric         │
│  Input:                                                  │
│    - Article (title, abstract)                          │
│    - Screening criteria (PCC)                           │
│    - Primary LLM decision + reasoning                   │
│                                                          │
│  Rubric Evaluation Matrix:                              │
│  ┌────────────────────────────────────────────────┐    │
│  │ Criterion          │ Score (0-5) │ Weight │ Notes │    │
│  ├────────────────────────────────────────────────┤    │
│  │ Relevance to PCC   │     4       │  0.35  │ ... │    │
│  │ Logical Reasoning  │     5       │  0.25  │ ... │    │
│  │ Citation Support   │     3       │  0.20  │ ... │    │
│  │ Consistency        │     4       │  0.15  │ ... │    │
│  │ Completeness       │     5       │  0.05  │ ... │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  Output:                                                 │
│    - Evaluation Score: Weighted average (0-5)           │
│    - Agreement: Agree/Disagree with decision            │
│    - Quality Matrix: Breakdown of scores                │
│    - Recommendations: Suggest manual review if needed   │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│              FINAL CONFIDENCE CALCULATION                │
│                                                          │
│  Matrix-based Confidence Formula:                       │
│  ┌────────────────────────────────────────────────┐    │
│  │ Component              │ Weight │ Contribution│    │
│  ├────────────────────────────────────────────────┤    │
│  │ Primary LLM Confidence │  0.40  │   0.85 → 0.34 │    │
│  │ Evaluation Score (0-5) │  0.35  │   4.2 → 0.29  │    │
│  │ LLM Agreement (Y/N)    │  0.15  │   Yes → 0.15  │    │
│  │ Reasoning Quality      │  0.10  │   High → 0.09 │    │
│  │───────────────────────────────────────────────│    │
│  │ FINAL CONFIDENCE       │        │   0.87        │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  Decision Logic:                                         │
│    - Confidence ≥ 0.80 → Auto-accept                    │
│    - 0.60 ≤ Confidence < 0.80 → Manual review           │
│    - Confidence < 0.60 → Flag for expert review         │
│    - LLM Disagreement → Automatic manual review         │
└─────────────────────────────────────────────────────────┘
```

**Rubric Framework Design:**

```python
EVALUATION_RUBRIC = {
    'relevance_to_criteria': {
        'weight': 0.35,
        'description': 'How well the article matches PCC criteria',
        'scoring': {
            5: 'Perfect match for all PCC components',
            4: 'Strong match with minor gaps',
            3: 'Moderate match, some criteria met',
            2: 'Weak match, few criteria met',
            1: 'Minimal relevance',
            0: 'No relevance'
        }
    },
    'logical_reasoning': {
        'weight': 0.25,
        'description': 'Quality of reasoning in screening decision',
        'scoring': {
            5: 'Clear, comprehensive, evidence-based reasoning',
            4: 'Good reasoning with minor logical gaps',
            3: 'Adequate reasoning, some unclear points',
            2: 'Weak reasoning, significant gaps',
            1: 'Poor reasoning, mostly unsupported claims',
            0: 'No clear reasoning provided'
        }
    },
    'citation_evidence': {
        'weight': 0.20,
        'description': 'Evidence from abstract supporting decision',
        'scoring': {
            5: 'Strong evidence directly quoted from abstract',
            4: 'Good evidence with clear connections',
            3: 'Moderate evidence, some inferences needed',
            2: 'Weak evidence, mostly assumptions',
            1: 'Minimal evidence',
            0: 'No supporting evidence'
        }
    },
    'consistency': {
        'weight': 0.15,
        'description': 'Internal consistency of decision and reasoning',
        'scoring': {
            5: 'Perfectly consistent throughout',
            4: 'Mostly consistent, minor contradictions',
            3: 'Generally consistent, some issues',
            2: 'Inconsistent in multiple places',
            1: 'Highly inconsistent',
            0: 'Contradictory'
        }
    },
    'completeness': {
        'weight': 0.05,
        'description': 'Completeness of analysis',
        'scoring': {
            5: 'All PCC components addressed',
            4: 'Most components addressed',
            3: 'Some components missing',
            2: 'Many components missing',
            1: 'Minimal coverage',
            0: 'Incomplete analysis'
        }
    }
}
```

**Implementation Plan:**

**Phase 2.1: Rubric Design & Testing (Month 3-4)**
```python
# backend/core/evaluator.py
class ScreeningEvaluator:
    def __init__(self, evaluator_llm: str = "grok-3"):
        self.evaluator = AIClient(model=evaluator_llm)
        self.rubric = EVALUATION_RUBRIC
    
    def evaluate_screening(self, 
                          article: Dict,
                          criteria: Dict,
                          primary_decision: Dict) -> Dict:
        """
        Evaluate a screening decision using rubric
        
        Args:
            article: {title, abstract, ...}
            criteria: {population, concept, context}
            primary_decision: {decision, reasoning, confidence}
            
        Returns:
            {
                'evaluation_score': float (0-5),
                'agreement': bool,
                'quality_matrix': Dict[str, float],
                'recommendations': List[str],
                'final_confidence': float (0-1),
                'requires_manual_review': bool
            }
        """
        pass
    
    def calculate_final_confidence(self,
                                   primary_confidence: float,
                                   evaluation_score: float,
                                   agreement: bool,
                                   reasoning_quality: str) -> float:
        """
        Matrix-based confidence calculation
        
        Formula:
        confidence = 0.40 * primary_confidence +
                    0.35 * (evaluation_score / 5) +
                    0.15 * (1 if agreement else 0) +
                    0.10 * reasoning_quality_score
        """
        pass
```

**Phase 2.2: UI Enhancements (Month 4)**
```python
# frontend/pages/2_AI_Screening.py
def show_evaluation_results(result: Dict):
    """
    Display dual-LLM evaluation results
    
    UI Components:
    - Primary LLM decision card
    - Evaluation rubric matrix (interactive table)
    - Confidence breakdown (pie chart)
    - Agreement indicator (green check / red X)
    - Manual review flags
    """
    
    # Rubric Matrix Display
    st.subheader("📊 Evaluation Matrix")
    matrix_df = pd.DataFrame([
        {
            'Criterion': 'Relevance to PCC',
            'Score': result['quality_matrix']['relevance_to_criteria'],
            'Weight': 0.35,
            'Weighted Score': result['quality_matrix']['relevance_to_criteria'] * 0.35
        },
        # ... other criteria
    ])
    st.dataframe(matrix_df)
    
    # Confidence Breakdown
    st.subheader("🎯 Confidence Calculation")
    fig = go.Figure(data=[go.Pie(
        labels=['Primary LLM', 'Evaluation Score', 'Agreement', 'Reasoning Quality'],
        values=[0.40, 0.35, 0.15, 0.10],
        hole=0.4
    )])
    st.plotly_chart(fig)
```

**Phase 2.3: A/B Testing & Validation (Month 5)**
- Compare single-LLM vs dual-LLM decisions on 500 test articles
- Measure inter-rater reliability with human reviewers
- Calculate false positive/negative rates
- Optimize rubric weights based on validation data
- Cost-benefit analysis (dual-LLM doubles API costs - is it worth it?)

**Phase 2.4: Production Integration (Month 6)**
- Add dual-LLM evaluation as optional feature (toggle in UI)
- Batch evaluation mode for efficiency
- Export evaluation matrices to Excel
- Training materials for users on interpreting matrices

**Expected Benefits:**
- ✅ **Transparency**: Confidence scores now explainable via matrix
- ✅ **Quality Control**: Second layer catches screening errors
- ✅ **Reduced Manual Review**: Higher confidence in auto-decisions
- ✅ **Learning Tool**: Rubric helps researchers understand AI reasoning
- ✅ **Customizable**: Users can adjust rubric weights for their domain

**Cost Considerations:**
```
Single-LLM System:
  5,000 articles × $0.003/article = $15

Dual-LLM System (with evaluation):
  5,000 articles × $0.003 (screening) = $15
  5,000 articles × $0.004 (evaluation) = $20
  Total: $35 (2.3× more expensive)
  
Cost Mitigation:
  - Selective evaluation (only borderline cases 0.6-0.8 confidence)
  - Use cheaper model for evaluation (Grok-3-mini-fast)
  - Batch API calls for efficiency
```

---

#### 2. **Advanced AI Models:**
   - Support for Claude, GPT-4, Gemini
   - Model ensemble (vote-based decisions)
   - Fine-tuned models for specific domains (medical, engineering)

#### 3. **Collaboration Features:**
   - Multi-user screening (shared workspace)
   - Conflict resolution interface
   - Inter-rater reliability calculation

3. **Machine Learning:**
   - Active learning (prioritize uncertain articles)
   - Transfer learning from previous reviews
   - Predictive title screening

4. **Integrations:**
   - Direct import from PubMed API
   - Export to Covidence/Rayyan
   - Integration with reference managers (Zotero, Mendeley)

### Phase 3 Features (6-12 months)
1. **Full-text Screening:**
   - PDF upload and parsing
   - Full-text AI analysis
   - Citation network analysis

2. **Data Extraction:**
   - Automated data extraction from included articles
   - Table/figure extraction
   - Statistical data synthesis

3. **Reporting:**
   - Automated manuscript generation
   - Quality assessment (GRADE, etc.)
   - Meta-analysis preparation

---

## 📞 Contact & Support

### Development Team
- **Project Lead:** AI Scoping Review Team
- **Architecture:** AI Agent
- **Repository:** `Core function/` (standalone)

### Getting Help
- **Documentation:** See `docs/` folder
- **Issues:** GitHub Issues (if using version control)
- **Questions:** Include detailed context and error logs

---

## 📝 Appendix

### A. Field Mapping Reference
**Standard Fields (Required):**
- `title`: Article title
- `abstract`: Article abstract/summary
- `authors`: Author list
- `journal`: Journal/source name
- `year`: Publication year

**Standard Fields (Optional):**
- `doi`: Digital Object Identifier
- `keywords`: Author keywords
- `pmid`: PubMed ID (if available)
- `url`: Article URL
- `database`: Source database name

### B. Deduplication Algorithm Pseudocode
```
INPUT: DataFrame with articles
OUTPUT: DataFrame without duplicates

1. DOI Matching:
   - Group by DOI (non-empty)
   - Keep first occurrence of each DOI
   - Mark others as duplicates

2. Title Similarity:
   - Clean titles (lowercase, remove punctuation)
   - Vectorize with TF-IDF
   - Compute pairwise cosine similarity
   - If similarity >= 0.85, mark as duplicate

3. Metadata Matching:
   - Create combined key: first_50_chars(authors) + year + first_30_chars(journal)
   - Find duplicates on combined key
   - Mark as duplicates

4. Combine Results:
   - Union of all duplicate indices
   - Remove duplicates from DataFrame
   - Generate report with duplicate counts

RETURN: Cleaned DataFrame + Report
```

### C. Parallel Screening Workflow
```
INPUT: List of N articles, Screening criteria
OUTPUT: List of N screening results

1. Initialize:
   - Create ThreadPoolExecutor (8 workers)
   - Create results queue
   - Load checkpoint (if resuming)

2. Batch Submission:
   - For each article not yet screened:
       - Submit screening task to executor
       - Store future -> article mapping

3. Result Collection:
   - For each completed future:
       - Get result (with timeout=30s)
       - Append to results list
       - Update progress bar
       - Save checkpoint every 50 articles
       - Handle errors gracefully

4. Finalization:
   - Wait for all futures to complete
   - Aggregate results
   - Calculate statistics
   - Save final results

RETURN: Screening results + Statistics
```

### D. Technology Stack Summary

**Core Framework:**
- Streamlit (Web UI)
- Pandas (Data processing)
- NumPy (Numerical operations)

**AI/ML:**
- Grok API (Screening)
- Scikit-learn (TF-IDF, cosine similarity)
- (Future Phase 2: Dual-LLM Evaluation System with rubric-based quality matrix)
- (Future: HuggingFace, BioBERT)

**Visualization:**
- Plotly (Interactive charts)
- Matplotlib (Static diagrams)

**Utilities:**
- PyYAML (Configuration)
- python-dotenv (Environment variables)
- pathlib (File handling)

**Testing:**
- pytest (Unit tests)
- pytest-cov (Coverage)

**Deployment:**
- Docker (Containerization)
- (Future: Streamlit Cloud, AWS, Azure)

---

**End of Migration Plan Document**

*This document will be updated as the migration progresses and new insights emerge.*
