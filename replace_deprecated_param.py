"""
Replace deprecated Streamlit parameter: use_container_width=True -> width="stretch"
"""
import re
from pathlib import Path

# Files to process
files = [
    "streamlit_app.py",
    "frontend/app.py",
    "frontend/pages/1_Data_Management.py",
    "frontend/pages/2_AI_Screening.py",
    "frontend/pages/3_Results.py",
    "frontend/pages/4_Settings.py"
]

print("🔄 Replacing deprecated Streamlit parameter...")

total_replacements = 0
root = Path(__file__).parent

for file_path in files:
    full_path = root / file_path
    
    if full_path.exists():
        print(f"\nProcessing: {file_path}")
        
        # Read content
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count matches
        match_count = len(re.findall(r'use_container_width=True', content))
        
        if match_count > 0:
            # Replace
            new_content = content.replace('use_container_width=True', 'width="stretch"')
            
            # Write back
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"  ✅ Replaced {match_count} occurrence(s)")
            total_replacements += match_count
        else:
            print(f"  ℹ️  No matches found")
    else:
        print(f"  ⚠️  File not found: {file_path}")

print(f"\n✨ Done! Total replacements: {total_replacements}")
print("📝 Please review the changes and commit if everything looks good.")
