# 🚀 Quick Setup Guide for GitHub Users

This guide helps you get started after cloning the repository.

## Prerequisites

- Python 3.9 or higher
- Git
- XAI API key (get from https://console.x.ai/)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-scoping-review.git
cd ai-scoping-review
```

### 2. Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Copy the example environment file and add your API key:

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Edit `.env` and replace `your_xai_api_key_here` with your actual XAI API key:
```
XAI_API_KEY=your_actual_api_key_here
```

### 5. Set Up Configuration

```bash
# Windows
copy config\user_settings.example.json config\user_settings.json

# macOS/Linux
cp config/user_settings.example.json config/user_settings.json
```

### 6. Initialize Database

```bash
python tests/init_db.py
```

### 7. Run the Application

**Option 1: Run both servers (Recommended)**
```bash
python start_all.py
```

**Option 2: Run separately**
```bash
# Terminal 1: Backend
python start_backend.py

# Terminal 2: Frontend
python start_frontend.py
```

### 8. Access the Application

- **Web Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

## Troubleshooting

### Issue: "XAI_API_KEY not found"
**Solution**: Make sure you've created `.env` file and added your API key.

### Issue: "Database not found"
**Solution**: Run `python tests/init_db.py` to create the database.

### Issue: "Port already in use"
**Solution**: 
- Kill processes using ports 8000 or 8501
- Or change ports in `.env` file

### Issue: "Module not found"
**Solution**: Make sure you've activated the virtual environment and installed all requirements.

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Check [CONTRIBUTING.md](CONTRIBUTING.md) if you want to contribute
3. Run test data: Use files in `test_data/` folder for testing

## Support

- **Issues**: https://github.com/YOUR_USERNAME/ai-scoping-review/issues
- **Discussions**: https://github.com/YOUR_USERNAME/ai-scoping-review/discussions

Happy screening! 🔬
