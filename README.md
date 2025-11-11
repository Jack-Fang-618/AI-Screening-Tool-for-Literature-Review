# 🔬 AI Scoping Review - Core Function Module

**Version:** 2.0.0
**Status:** ✅ Core Features Complete
**Architecture:** FastAPI Backend + Streamlit Frontend
**Purpose:** Production-ready toolkit for scoping review data management and AI-powered screening

---

## 📋 Overview

**AI Scoping Review Core Function** is a modern, hybrid-architecture application designed for researchers conducting systematic literature reviews. It combines:

- **FastAPI Backend**: High-performance async server for data processing and AI screening
- **Streamlit Frontend**: Simple, intuitive UI for researchers

### Why This Architecture?

✅ **Performance**: Async backend handles long-running tasks (screening 5,000 articles in ~30 min)
✅ **Scalability**: Support multiple concurrent users
✅ **Simplicity**: No complex authentication, just start and use
✅ **Real-time Updates**: Frontend polls backend for live progress tracking
✅ **Future-proof**: Easy to add features, switch frontends, or deploy to cloud

### Core Workflows (✅ All Complete)

1. **📊 Data Management**: Upload, merge, clean, and deduplicate articles from multiple database exports
2. **🤖 AI Screening**: Parallel AI-powered screening with up to 50 concurrent workers (processes 5,000 articles in ~10-30 min)
3. **📈 Results & Export**: Interactive results table, filtering, and multi-format export with full abstracts

### Key Features

✅ **Multi-format Support**: Excel (.xlsx, .xls), CSV, RIS files
✅ **Intelligent Field Mapping**: LLM-powered auto-detection and standardization of fields
✅ **Smart Deduplication**: 5-stage process with DOI matching, title similarity, and metadata validation
✅ **Parallel AI Screening**: Up to 50× faster processing with checkpoint/resume support
✅ **Cost Tracking**: Real-time token usage and cost estimation with HKD pricing
✅ **Database Persistence**: SQLite with SQLAlchemy ORM - all data persists across restarts
✅ **Interactive Results**: Filter by decision, confidence, search across title/abstract/reasoning
✅ **Full Export**: CSV/Excel export with complete abstracts and screening metadata
✅ **No Login Required**: Simple, focused workflow for researchers

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- XAI API key (for AI screening - get from https://console.x.ai/)
- 4GB+ RAM recommended for large datasets

### Installation

```bash
# 1. Clone repository
git clone https://github.com/Jack-Fang-618/AI-Screening-Tool-for-Literature-Review.git
cd AI-Screening-Tool-for-Literature-Review

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure environment
# Create .env file in project root and add your XAI API key
echo "XAI_API_KEY=your_api_key_here" > .env
```

**Important**: Use `XAI_API_KEY` (not `GROK_API_KEY`) in your `.env` file.

### Preparing Your Data

**Before using the toolkit, download your dataset files from literature databases with ALL necessary metadata:**

#### Required Fields

- **Title**: Article title (essential for screening and deduplication)
- **Abstract**: Full abstract text (required for AI screening)

#### Recommended Fields (for better deduplication and tracking)

- **Authors**: Author list
- **Journal**: Publication venue
- **Year**: Publication year
- **DOI**: Digital Object Identifier (best for exact duplicate matching)
- **Keywords**: Article keywords
- **Publication Type**: Article type (journal article, review, etc.)

#### Supported Database Export Formats

| Database                 | Recommended Export Format         | Fields to Include                                                                |
| ------------------------ | --------------------------------- | -------------------------------------------------------------------------------- |
| **PubMed**         | CSV or MEDLINE/PubMed XML (.nbib) | Title (TI), Abstract (AB), Authors (AU), Journal (TA), Year (DP), PMID, DOI      |
| **Scopus**         | CSV or RIS                        | Title, Abstract, Authors, Source title, Year, DOI, Keywords                      |
| **Web of Science** | Tab-delimited or Excel            | Title (TI), Abstract (AB), Authors (AU), Source (SO), Publication Year (PY), DOI |
| **Embase**         | CSV or RIS                        | Title, Abstract, Authors, Journal, Year, DOI                                     |
| **CINAHL**         | RIS or CSV                        | Title, Abstract, Authors, Journal, Year, Accession Number                        |

**Export Instructions by Database:**

**PubMed:**

1. After your search, click "Save" → Select "All results" or specific range
2. Format: Choose "PubMed" or "CSV"
3. **Important**: Ensure abstracts are included in export (default setting)

**Scopus:**

1. Select all results or filter → Click "Export"
2. Format: Choose "CSV Export" or "RIS Format"
3. **Fields**: Select "Citation information", "Bibliographical information", "Abstract & keywords"

**Web of Science:**

1. Select records → Click "Export" → "Excel" or "Tab-delimited"
2. Record Content: Choose "Full Record and Cited References"
3. **Ensure abstract field is included** in export settings

**General Tips:**

- ✅ **Always include abstracts** - critical for AI screening
- ✅ Include DOI when available - best for duplicate detection
- ✅ Export complete records, not just citations
- ⚠️ **Avoid exporting only titles** - insufficient for screening
- 💡 Most databases support CSV, Excel, or RIS - all compatible with this toolkit
- 💡 You can upload files from multiple databases and merge them in the Data Management page

### Running the Application

**Option 1: Launch Both Backend and Frontend (Recommended)**

```bash
# Activate virtual environment (if not already activated)
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Start both servers
python start_all.py
```

This will start:

- FastAPI Backend on `http://localhost:8000`
- Streamlit Frontend on `http://localhost:8501`

**Option 2: Launch Separately**

```bash
# Terminal 1: Start Backend
python start_backend.py

# Terminal 2: Start Frontend (in a new terminal)
python start_frontend.py
```

**Access Points:**

- 🌐 **Web Interface**: `http://localhost:8501` (Open this in your browser)
- 📚 **API Documentation**: `http://localhost:8000/docs` (Interactive Swagger UI)
- 🔍 **Alternative API Docs**: `http://localhost:8000/redoc` (ReDoc format)

---

## 📖 User Guide

### Workflow Overview

```
┌─────────────────┐
│ 1. Upload Files │ ← Excel/CSV/RIS from PubMed, Scopus, etc.
└────────┬────────┘
         ↓
┌─────────────────┐
│ 2. Map Fields   │ ← Auto-detect title, abstract, authors, etc.
└────────┬────────┘
         ↓
┌─────────────────┐
│ 3. Merge Data   │ ← Combine multiple sources
└────────┬────────┘
         ↓
┌─────────────────┐
│ 4. Deduplicate  │ ← Intelligent duplicate removal
└────────┬────────┘
         ↓
┌─────────────────┐
│ 5. AI Screening │ ← Parallel processing (8 workers)
└────────┬────────┘
         ↓
┌─────────────────┐
│ 6. View Results │ ← PRISMA diagram + Export
└─────────────────┘
```

### 1️⃣ Data Management

**Step 1: Upload Files**

- Navigate to "📊 Data Management" page
- Upload one or more files (Excel, CSV, or RIS format)
- System automatically detects file format and parses data
- Preview shows column names and first few rows

**Step 2: Map Fields (LLM-Powered)**

- Click "Auto-detect fields with LLM" for intelligent mapping
- Review and adjust auto-detected field mappings if needed
- Essential fields: Title, Abstract (other fields optional: Authors, Journal, Year, DOI)
- Unmapped columns are preserved in the dataset

**Step 3: Merge Data**

- Click "Merge Selected Datasets" to combine all sources
- System tracks data provenance (source file for each record)
- Automatic validation and cleaning
- Creates new merged dataset saved to database

**Step 4: Smart Deduplication**

- Click "Smart Deduplicate" button
- **5-Stage Intelligent Process**:
  1. Quality check (remove invalid records)
  2. DOI exact matching
  3. Title similarity (TF-IDF, default 0.85 threshold)
  4. Metadata validation (author/journal/year comparison)
  5. Return cleaned dataset + review dataset
- Review deduplication report with counts:
  - DOI duplicates removed
  - Title duplicates confirmed (same metadata)
  - Title duplicates flagged for manual review (different metadata)
- **Two output datasets**:
  - Cleaned dataset (duplicates removed)
  - Review dataset (potential duplicates for manual inspection)

### 2️⃣ AI Screening

**Step 1: Configure Screening**

- Navigate to "🤖 AI Screening" page
- Select cleaned/merged dataset from dropdown
- Choose AI model (grok-4-fast-reasoning recommended - cheapest + best performance)
- Review cost estimation before starting

**Step 2: Define Criteria** (PCC Framework)

- **Population**: Who/what is the study about? (e.g., "Type 1 diabetes patients aged 0-18")
- **Concept**: What is being studied? (e.g., "Continuous glucose monitoring devices")
- **Context**: Where/when is the study conducted? (e.g., "Any healthcare setting, any time period")

**Step 3: Run Screening**

- Click "Start AI Screening"
- Confirm estimated cost (shown in HKD)
- Real-time progress tracking:
  - Articles processed / total
  - Current cost / estimated total cost
  - Progress percentage
  - Estimated time remaining
- **Automatic checkpointing**: Saves every 100 articles to `data/checkpoints/`
- **Resume support**: Can restart interrupted tasks

**Performance:**

- 5,000 articles in ~30 minutes (8 parallel workers)
- Token tracking: input, output, reasoning, cached tokens
- Cost breakdown: Input, cached, output, reasoning costs

### 3️⃣ Results & Export

**View Results:**

- Navigate to "📈 Results" page
- Auto-detects completed screening tasks
- **Dropdown selector** shows:
  - Task completion time
  - Article counts
  - Task ID
- **Interactive data table** with:
  - Title, Abstract, Decision, Confidence, Reasoning
  - Horizontal scrolling for long text
  - Sortable columns

**Filter Results:**

- **By Decision**: Include / Exclude / Manual Review
- **By Confidence Range**: Slider to filter by confidence score
- **Text Search**: Search across title, abstract, and reasoning

**Summary Statistics:**

- Total articles screened
- Breakdown by decision (Include/Exclude/Manual Review)
- Average confidence score
- Total cost and processing time
- Token usage breakdown

**Export Options:**

- **Export Included Articles**: Download CSV with only included articles + full abstracts
- **Export All Results**: Download complete dataset with all fields
- Format: CSV with columns: Title, Abstract, Decision, Confidence, Reasoning, Cost, Processing Time
- **Full abstracts preserved** in export (not truncated)

---

## 🏗️ Architecture

### Project Structure

```
Core function/
├── docs/                          # Documentation
│   ├── REFACTORING_MIGRATION_PLAN.md
│   ├── API_REFERENCE.md
│   └── USER_GUIDE.md
│
├── backend/                       # FastAPI Backend
│   ├── main.py                   # FastAPI app entry point
│   ├── db/                       # Database layer
│   │   ├── config.py            # SQLAlchemy engine and session
│   │   └── __init__.py          # get_db() context manager
│   ├── models/                   # Database models
│   │   └── database.py          # SQLAlchemy ORM models
│   ├── api/                      # API endpoints
│   │   ├── data_management.py   # Upload, merge, deduplicate
│   │   ├── screening.py         # AI screening endpoints
│   │   └── results.py           # Results endpoints
│   ├── core/                     # Business logic
│   │   ├── data_processor.py    # Data processing
│   │   ├── smart_deduplicator.py # 5-stage deduplication
│   │   ├── llm_field_mapper.py  # LLM field mapping
│   │   └── screener.py          # AI screening engine
│   ├── services/                 # External services
│   │   └── grok_client.py       # XAI Grok API client
│   └── tasks/                    # Background tasks
│       └── task_manager.py      # Task state management
│
├── frontend/                      # Streamlit Frontend
│   ├── app.py                    # Main Streamlit app
│   ├── pages/                    # Streamlit pages
│   │   ├── 1_Data_Management.py # Upload, merge, dedup UI
│   │   ├── 2_AI_Screening.py    # Screening configuration UI
│   │   ├── 3_Results.py         # Results display and export
│   │   └── 4_Settings.py        # Application settings
│   ├── components/               # Reusable UI components
│   └── utils/
│       └── api_client.py        # Backend API client
│
├── data/                          # Data storage
│   ├── app.db                    # SQLite database
│   ├── checkpoints/              # Screening checkpoints
│   └── screening_results/        # Legacy JSON results
│
├── config/                        # Configuration
│   └── user_settings.json        # User preferences
│
├── start_backend.py               # Backend launcher
├── start_frontend.py              # Frontend launcher
├── start_all.py                   # Launch both
├── requirements.txt               # Dependencies
├── .env.example                   # Environment template
└── README.md                      # This file
```

### Technology Stack

- **Backend**: FastAPI 0.104+ (async web framework), uvicorn (ASGI server)
- **Frontend**: Streamlit 1.28+ (UI framework)
- **Database**: SQLite with SQLAlchemy 2.0 (ORM, persistent storage)
- **Data Processing**: Pandas, NumPy, Scikit-learn (TF-IDF for deduplication)
- **AI Integration**: XAI Grok API (grok-4-fast-reasoning model)
- **Visualization**: Plotly (interactive charts)
- **Data Models**: Pydantic v2 (validation and serialization)
- **HTTP Client**: requests (API communication)

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# API Configuration
XAI_API_KEY=your_api_key_here  # Get from https://console.x.ai/

# Optional: Database Configuration
# DATABASE_URL=sqlite:///./data/app.db  # Default, auto-created
```

**Note**: The system uses `XAI_API_KEY` (not `GROK_API_KEY`). All other settings (model, workers, thresholds) are configured through the UI.

---

## 📊 Performance

### Benchmarks (Actual Performance)

| Dataset Size | Upload | Merge | Deduplicate | Screen (30 workers) | Screen (50 workers) | Total (50 workers) |
| ------------ | ------ | ----- | ----------- | ------------------- | ------------------- | ------------------ |
| 100          | <1s    | 1s    | 2s          | 15s                 | 10s                 | ~15s               |
| 500          | 1s     | 2s    | 8s          | 1.5 min             | 1 min               | ~2 min             |
| 1,000        | 2s     | 3s    | 15s         | 2.5 min             | 2 min               | ~3 min             |
| 5,000        | 5s     | 10s   | 60s         | 12 min              | 10 min              | ~12 min            |

**Notes:**

- Screening time depends on abstract length, AI model, and worker count
- grok-4-fast-reasoning: ~0.06s per article (50 parallel workers)
- Checkpoint every 100 articles for resume support
- Cost: ~HKD 0.05 per article (varies by model)
- Worker count configurable: 8-50 workers (default 30)

### System Requirements

| Component | Minimum   | Recommended |
| --------- | --------- | ----------- |
| RAM       | 2GB       | 4GB+        |
| CPU       | Dual-core | Quad-core+  |
| Storage   | 500MB     | 2GB+        |
| Internet  | 5 Mbps    | 20 Mbps+    |

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📚 Documentation

### Available Guides

- **[REFACTORING_MIGRATION_PLAN.md](docs/REFACTORING_MIGRATION_PLAN.md)**: Detailed migration plan and architecture
- **[DUAL_LLM_EVALUATION_PLAN.md](docs/DUAL_LLM_EVALUATION_PLAN.md)**: Future dual-LLM evaluation system design
- **[SETUP.md](docs/SETUP.md)**: Detailed setup and configuration guide

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact & Support

### Getting Help

- **Documentation**: See `docs/` folder
- **Issues**: Report bugs or request features via GitHub Issues
- **Email**: wennbo@hku.hk

---

## 🗺️ Roadmap

### ✅ Version 2.0.0 (Current - November 2025)

- ✅ FastAPI backend with async processing
- ✅ Streamlit frontend with polling-based updates
- ✅ SQLite database with SQLAlchemy ORM (persistent storage)
- ✅ Multi-format file upload (Excel, CSV, RIS)
- ✅ LLM-powered field mapping
- ✅ Smart 5-stage deduplication with metadata validation
- ✅ Parallel AI screening (up to 50 workers, checkpoint/resume)
- ✅ Real-time cost tracking with HKD pricing
- ✅ Interactive results page with filtering and search
- ✅ Full export with abstracts (CSV format)
- ✅ Database persistence across restarts

### Future Versions

**v2.1.0** 

Dual-LLM evaluation system (see [DUAL_LLM_EVALUATION_PLAN.md](docs/DUAL_LLM_EVALUATION_PLAN.md))

- - Two LLMs discuss and reach consensus on screening decisions
  - Structured dialogue loop (max 3 rounds) for disagreements
  - Automatic flagging for human review when needed
- PRISMA-ScR flow diagram generation
- Excel export with multiple sheets
- Manual review interface for flagged articles
- Enhanced cost optimization (prompt caching)
- Bulk editing of screening decisions

**v2.2.0** 

- Support for additional AI models (Claude, GPT-4, Gemini)
- Full-text PDF screening
- Collaboration features (multi-user screening)
- Advanced analytics dashboard
- Integration with reference managers (Zotero, Mendeley)

**v3.0.0** 

- Machine learning-based screening optimization
- Automated data extraction from full-text
- Meta-analysis preparation tools
- RESTful API for external integrations
- Cloud deployment option

---

## 🙏 Acknowledgments

This project builds upon the original AI Scoping Review platform and incorporates best practices from:

- PRISMA-ScR guidelines
- Systematic review methodologies
- Open-source ML/AI tools

Special thanks to all researchers who provided feedback and testing.

---

**Last Updated**: November 11, 2025
**Version**: 2.0.0 - Core Features Complete ✅
