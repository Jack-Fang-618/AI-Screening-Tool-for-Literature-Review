"""
Settings Page - Configure API Keys and Screening Criteria

Handles:
- XAI API key management
- Default inclusion/exclusion criteria templates
- User preferences
"""

import streamlit as st
import json
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Page configuration
st.set_page_config(
    page_title="Settings - PRISMA-ScR Toolkit",
    page_icon="⚙️",
    layout="wide"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border-left: 4px solid #17a2b8;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .section-box {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for settings
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'xai_api_key': '',
        'inclusion_criteria': '',
        'exclusion_criteria': '',
        'auto_save': True,
        'default_workers': 4,
        'default_model': 'grok-4-fast-reasoning'
    }

def load_settings():
    """Load settings from file"""
    settings_file = Path(__file__).parent.parent.parent / 'config' / 'user_settings.json'
    
    if settings_file.exists():
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                loaded_settings = json.load(f)
                st.session_state.settings.update(loaded_settings)
            return True
        except Exception as e:
            st.error(f"Failed to load settings: {str(e)}")
            return False
    return False

def save_settings():
    """Save settings to file"""
    settings_file = Path(__file__).parent.parent.parent / 'config' / 'user_settings.json'
    settings_file.parent.mkdir(exist_ok=True)
    
    try:
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.settings, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save settings: {str(e)}")
        return False

def check_env_api_key():
    """Check if API key exists in .env file"""
    # Path: frontend/pages/4_Settings.py -> pages/ -> frontend/ -> Core function/ -> .env
    env_file = Path(__file__).parent.parent.parent / '.env'
    
    if env_file.exists():
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('XAI_API_KEY='):
                        key = line.split('=', 1)[1].strip()
                        if key and key != 'your-xai-api-key-here':
                            return True, key[:20] + '...' + key[-4:] if len(key) > 24 else key
        except Exception:
            pass
    
    return False, None

def update_env_api_key(api_key):
    """Update .env file with new API key"""
    env_file = Path(__file__).parent.parent.parent / '.env'
    
    try:
        # Read existing .env content
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        else:
            lines = []
        
        # Update or add XAI_API_KEY
        updated = False
        for i, line in enumerate(lines):
            if line.startswith('XAI_API_KEY='):
                lines[i] = f'XAI_API_KEY={api_key}\n'
                updated = True
                break
        
        if not updated:
            lines.append(f'XAI_API_KEY={api_key}\n')
        
        # Write back to .env
        with open(env_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        return True
    except Exception as e:
        st.error(f"Failed to update .env file: {str(e)}")
        return False


def main():
    # Load settings on startup
    if 'settings_loaded' not in st.session_state:
        load_settings()
        st.session_state.settings_loaded = True
    
    # Header
    st.markdown('<h1 class="main-header">Settings</h1>', unsafe_allow_html=True)
    st.markdown("**Configure API keys, default criteria, and preferences**")
    
    st.markdown("---")
    
    # Tab layout for different settings sections
    tab1, tab2, tab3 = st.tabs(["🔑 API Configuration", "📋 Default Criteria", "⚙️ Preferences"])
    
    # ==================== API Configuration Tab ====================
    with tab1:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown("### XAI API Key Configuration")
        
        # Check current session API key
        has_session_key = 'xai_api_key' in st.session_state and st.session_state.xai_api_key
        
        st.markdown("""
        #### 🔑 API Key Management
        
        **Current System (Per-User):**
        - Each user provides their own API key
        - API key is stored in your browser session only
        - Your API usage is billed to your own X.AI account
        - Key is cleared when you close the browser
        
        **Security:**
        - Keys are never stored on the server
        - Keys are only used for your screening tasks
        - Each user pays for their own API usage
        """)
        
        if has_session_key:
            masked_key = st.session_state.xai_api_key[:8] + "..." + st.session_state.xai_api_key[-4:] if len(st.session_state.xai_api_key) > 12 else "***"
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown(f"**✓ API Key Active in Session:** `{masked_key}`")
            st.markdown("This key is used for all your screening tasks.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("**⚠️ No API Key Found in Session**")
            st.markdown("Please enter your API key below to use AI screening features.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # API Key Input
        st.markdown("#### Update Your Session API Key")
        st.caption("This key is only stored in your browser session (not on server)")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            current_key = st.session_state.xai_api_key if has_session_key else ""
            new_api_key = st.text_input(
                "API Key",
                value=current_key,
                type="password",
                placeholder="xai-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                help="Your X.AI API key (starts with 'xai-')",
                key="api_key_input"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Update API Key", type="primary", width="stretch"):
                if new_api_key and new_api_key.strip():
                    # Validate format
                    if new_api_key.startswith('xai-'):
                        st.session_state.xai_api_key = new_api_key.strip()
                        st.success("✓ API key updated successfully!")
                        st.info("💡 Your API key is active for this session only.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid API key format. Key should start with 'xai-'")
                else:
                    st.warning("⚠️ Please enter a valid API key")
        
        # Clear API key option
        if has_session_key:
            st.markdown("---")
            if st.button("🗑️ Clear API Key from Session", type="secondary", width="stretch"):
                if 'xai_api_key' in st.session_state:
                    del st.session_state.xai_api_key
                st.success("✓ API key cleared from session")
                time.sleep(1)
                st.rerun()
        
        st.markdown("---")
        
        # API Key Information
        with st.expander("ℹ️ How to Get Your XAI API Key"):
            st.markdown("""
            ### Steps to Get Your X.AI API Key:
            
            1. **Visit X.AI Console**: Go to [console.x.ai](https://console.x.ai)
            2. **Sign In**: Log in with your X.AI account
            3. **Navigate to API Keys**: Find the API Keys section in your dashboard
            4. **Create New Key**: Click "Create New API Key"
            5. **Copy Key**: Copy the generated key (starts with `xai-`)
            6. **Paste Here**: Paste the key in the field above and click "Update API Key"
            
            ### Important Notes:
            
            - Keep your API key **secure** and never share it publicly
            - The key is stored in the `.env` file on your local machine
            - You need to **restart the backend server** after saving a new key
            - If you lose your key, you can generate a new one from the X.AI console
            
            ### Pricing Information:
            
            - **Grok 4 Fast**: ~HKD $1.556 per 1M input tokens
            - Typical cost: HKD $0.01-0.05 per article screened
            - Check [x.ai/pricing](https://x.ai/pricing) for latest rates
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== Default Criteria Tab ====================
    with tab2:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown("### Default Screening Criteria Templates")
        
        st.markdown("""
        Define default **inclusion** and **exclusion** criteria that will be pre-filled 
        when you start a new screening task. This helps maintain consistency across reviews.
        """)
        
        st.markdown("---")
        
        # Inclusion Criteria
        st.markdown("#### Inclusion Criteria")
        st.markdown("*Articles must meet these criteria to be included in the review*")
        
        inclusion_criteria = st.text_area(
            "Default Inclusion Criteria",
            value=st.session_state.settings.get('inclusion_criteria', ''),
            height=150,
            placeholder="""Example:
- Published in peer-reviewed journals
- Written in English
- Focus on adult population (18+ years)
- Include empirical research (quantitative or qualitative)
- Published within the last 10 years (2015-2025)""",
            help="Enter each criterion on a new line. Use bullet points for clarity."
        )
        
        st.markdown("---")
        
        # Exclusion Criteria
        st.markdown("#### Exclusion Criteria")
        st.markdown("*Articles meeting any of these criteria will be excluded from the review*")
        
        exclusion_criteria = st.text_area(
            "Default Exclusion Criteria",
            value=st.session_state.settings.get('exclusion_criteria', ''),
            height=150,
            placeholder="""Example:
- Conference abstracts, posters, or presentations
- Grey literature (dissertations, reports, etc.)
- Non-English publications
- Case studies or single-subject designs
- Pediatric or adolescent populations (under 18 years)
- Published before 2015""",
            help="Enter each criterion on a new line. Use bullet points for clarity."
        )
        
        st.markdown("---")
        
        # Action buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("Save Default Criteria", type="primary", width="stretch"):
                st.session_state.settings['inclusion_criteria'] = inclusion_criteria
                st.session_state.settings['exclusion_criteria'] = exclusion_criteria
                
                if save_settings():
                    st.success("✓ Default criteria saved successfully!")
                else:
                    st.error("Failed to save criteria")
        
        with col2:
            if st.button("Clear All", width="stretch"):
                st.session_state.settings['inclusion_criteria'] = ''
                st.session_state.settings['exclusion_criteria'] = ''
                st.rerun()
        
        with col3:
            if st.button("Load Example", width="stretch"):
                st.session_state.settings['inclusion_criteria'] = """- Published in peer-reviewed journals
- Written in English
- Focus on adult population (18+ years)
- Include empirical research (quantitative or qualitative)
- Published within the last 10 years"""
                
                st.session_state.settings['exclusion_criteria'] = """- Conference abstracts or posters
- Grey literature (dissertations, reports)
- Non-English publications
- Case studies or single-subject designs
- Pediatric populations (under 18 years)"""
                st.rerun()
        
        st.markdown("---")
        
        # Tips
        with st.expander("💡 Tips for Writing Good Criteria"):
            st.markdown("""
            ### Best Practices for Screening Criteria:
            
            **Inclusion Criteria:**
            - Be specific and measurable
            - Focus on PICO/PCC elements (Population, Intervention/Concept, Comparison, Outcome/Context)
            - Consider study design requirements
            - Specify language and publication type
            - Define time frame if relevant
            
            **Exclusion Criteria:**
            - Mirror your inclusion criteria
            - Explicitly state what should be excluded
            - Include common edge cases
            - Consider quality thresholds
            - List publication types to exclude
            
            **General Tips:**
            - Use clear, unambiguous language
            - Avoid overlapping criteria
            - Test criteria on sample articles first
            - Review and refine based on pilot screening
            - Document rationale for each criterion
            
            ### Example Format:
            ```
            Study Design:
            - Include: RCTs, cohort studies, cross-sectional studies
            - Exclude: Case reports, editorials, reviews
            
            Population:
            - Include: Adults aged 18-65
            - Exclude: Children, adolescents, elderly (>65)
            
            Intervention/Concept:
            - Include: Mindfulness-based interventions
            - Exclude: Other psychological interventions
            ```
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== Preferences Tab ====================
    with tab3:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown("### General Preferences")
        
        # Screening defaults
        st.markdown("#### Default Screening Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_model = st.selectbox(
                "Default AI Model",
                options=[
                    "grok-4-fast-reasoning",
                    "grok-4-fast-non-reasoning"
                ],
                index=0 if st.session_state.settings.get('default_model') == 'grok-4-fast-reasoning' else 1,
                help="The default model to use for new screening tasks"
            )
        
        with col2:
            default_workers = st.slider(
                "Default Parallel Workers",
                min_value=1,
                max_value=8,
                value=st.session_state.settings.get('default_workers', 4),
                help="Number of concurrent API requests. Lower = more stable, Higher = faster but may hit rate limits"
            )
            
            # Add helpful guidance
            if default_workers > 4:
                st.warning(
                    f"⚠️ Using {default_workers} workers. If you experience network errors or failures, "
                    f"try reducing to 2-4 workers for better stability.",
                    icon="⚠️"
                )
            elif default_workers <= 4:
                st.success(
                    f"✅ {default_workers} workers is a stable configuration with good reliability.",
                    icon="✅"
                )
        
        st.markdown("---")
        
        # Application settings
        st.markdown("#### Application Settings")
        
        auto_save = st.checkbox(
            "Auto-save settings",
            value=st.session_state.settings.get('auto_save', True),
            help="Automatically save settings when changed"
        )
        
        st.markdown("---")
        
        # Action buttons
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("Save Preferences", type="primary", width="stretch"):
                st.session_state.settings['default_model'] = default_model
                st.session_state.settings['default_workers'] = default_workers
                st.session_state.settings['auto_save'] = auto_save
                
                if save_settings():
                    st.success("✓ Preferences saved successfully!")
                else:
                    st.error("Failed to save preferences")
        
        with col2:
            if st.button("Reset to Defaults", width="stretch"):
                st.session_state.settings = {
                    'xai_api_key': st.session_state.settings.get('xai_api_key', ''),
                    'inclusion_criteria': '',
                    'exclusion_criteria': '',
                    'auto_save': True,
                    'default_workers': 4,
                    'default_model': 'grok-4-fast-reasoning'
                }
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Settings file location
    with st.expander("📁 Settings File Location"):
        settings_file = Path(__file__).parent.parent.parent / 'config' / 'user_settings.json'
        env_file = Path(__file__).parent.parent.parent / '.env'
        
        st.markdown(f"""
        **User Settings:** `{settings_file}`
        
        **Environment File:** `{env_file}`
        
        These files store your API key and preferences locally on your machine.
        """)


if __name__ == "__main__":
    main()
