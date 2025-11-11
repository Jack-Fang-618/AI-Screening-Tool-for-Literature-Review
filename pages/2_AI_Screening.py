"""
AI Screening Page - Streamlit Cloud Compatibility Wrapper
Redirects to the actual frontend page
"""
import sys
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "frontend"))

# Execute the actual page
exec(open(current_dir / "frontend" / "pages" / "2_AI_Screening.py", encoding='utf-8').read())
