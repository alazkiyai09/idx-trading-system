"""
NextGen Design System for IDX Trading Dashboard

This module provides a centralized styling system matching the NextGen reference design.
- Colors: Zinc-based dark theme with Emerald accent
- Layout: Three-panel layout (Sidebar, Main, Right Panel)
- Typography: Monospace for prices, clean sans-serif for text
"""

# ============================================================================
# DESIGN TOKENS
# ============================================================================

COLORS = {
    # Base colors
    "background": "#09090b",      # Zinc 950
    "foreground": "#fafafa",      # Zinc 50
    "border": "#27272a",          # Zinc 800
    "border_light": "#3f3f46",    # Zinc 700

    # Accent colors
    "primary": "#10b981",         # Emerald 500
    "primary_light": "#34d399",   # Emerald 400
    "primary_dark": "#059669",    # Emerald 600
    "destructive": "#ef4444",     # Red 500
    "destructive_light": "#f87171",  # Red 400

    # Background variants
    "muted": "#18181b",           # Zinc 900
    "muted_foreground": "#a1a1aa",  # Zinc 400
    "card": "#18181b",            # Zinc 900
    "card_hover": "#1f1f23",      # Slightly lighter

    # Status colors
    "success": "#10b981",         # Emerald
    "warning": "#f59e0b",         # Amber
    "error": "#ef4444",           # Red
    "info": "#3b82f6",            # Blue

    # Chart colors
    "chart_up": "#10b981",        # Emerald
    "chart_down": "#ef4444",      # Red
    "chart_neutral": "#a1a1aa",   # Zinc 400
    "chart_volume_up": "rgba(16, 185, 129, 0.5)",
    "chart_volume_down": "rgba(239, 68, 68, 0.5)",

    # Agent colors (for multi-agent analysis)
    "agent_auditor": "#3b82f6",      # Blue
    "agent_growth": "#8b5cf6",       # Purple
    "agent_value": "#8b5cf6",        # Purple
    "agent_synthesizer": "#10b981",  # Emerald
}

FONTS = {
    "mono": "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
    "sans": "'Inter', 'Segoe UI', system-ui, sans-serif",
}


# ============================================================================
# CSS GENERATOR
# ============================================================================

def get_nextgen_css() -> str:
    """Return complete CSS for NextGen-style dashboard."""
    return f"""
    <style>
    /* ============================================
       NEXTGEN DESIGN SYSTEM - IDX TRADING DASHBOARD
       ============================================ */

    /* --- Base App Styling --- */
    .stApp {{
        background-color: {COLORS['background']} !important;
        color: {COLORS['foreground']} !important;
        font-family: {FONTS['sans']};
    }}

    /* Force light text on ALL elements */
    .stApp, .stApp *:not(input):not(textarea),
    .main, .main *,
    .block-container, .block-container *,
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] *,
    .element-container, .element-container * {{
        color: {COLORS['foreground']} !important;
    }}

    /* Global dropdown/popover styling - targets ALL popovers at document level */
    [data-baseweb="popover"],
    [data-baseweb="popover"] *,
    div[id^="popover"],
    div[id^="popover"] *,
    [data-baseweb="select"],
    [data-baseweb="select"] * {{
        background-color: {COLORS['muted']} !important;
        color: {COLORS['foreground']} !important;
    }}

    [data-baseweb="popover"] ul,
    [data-baseweb="popover"] ol,
    [role="listbox"],
    [role="listbox"] * {{
        background-color: {COLORS['muted']} !important;
        color: {COLORS['foreground']} !important;
    }}

    [data-baseweb="popover"] li,
    [role="option"],
    [role="option"] * {{
        background-color: {COLORS['muted']} !important;
        color: {COLORS['foreground']} !important;
        border-bottom: 1px solid {COLORS['border']} !important;
    }}

    [data-baseweb="popover"] li:hover,
    [role="option"]:hover {{
        background-color: {COLORS['card_hover']} !important;
    }}

    [data-baseweb="popover"] li[aria-selected="true"],
    [role="option"][aria-selected="true"] {{
        background-color: rgba(16, 185, 129, 0.2) !important;
    }}

    /* Target dropdown panels and their children */
    .stSelectbox [role="listbox"],
    .stMultiSelect [role="listbox"] {{
        background-color: {COLORS['muted']} !important;
    }}

    /* Streamlit's baseweb dropdown specific styling */
    [class*="st-"] [role="listbox"],
    [class*="st-"] [role="listbox"] * {{
        background-color: {COLORS['muted']} !important;
        color: {COLORS['foreground']} !important;
    }}

    /* Select / multiselect surfaces in idle state */
    [data-baseweb="select"] > div,
    [data-baseweb="select"] > div:hover,
    [data-baseweb="select"] > div:focus-within,
    .stSelectbox > div > div,
    .stMultiSelect > div > div {{
        background-color: {COLORS['muted']} !important;
        color: {COLORS['foreground']} !important;
        border-color: {COLORS['border']} !important;
    }}

    /* Dropdown container panel */
    div[data-testid="stSelectbox"] div[role="listbox"],
    div[data-testid="stMultiSelect"] div[role="listbox"] {{
        background-color: {COLORS['muted']} !important;
        border: 1px solid {COLORS['border']} !important;
    }}

    /* Keep input text visible */
    input, textarea, .stTextInput input, .stNumberInput input {{
        color: {COLORS['foreground']} !important;
        background-color: {COLORS['muted']} !important;
    }}

    /* Fix specific Streamlit elements */
    .stMarkdown, .stMarkdown *, p, span, div {{
        color: {COLORS['foreground']};
    }}

    /* Labels and captions */
    label, .stWidget label, .stMarkdown label {{
        color: {COLORS['muted_foreground']} !important;
    }}

    /* Code / preformatted blocks */
    .stCodeBlock,
    .stCodeBlock *,
    .stCode,
    .stCode *,
    pre,
    pre *,
    code {{
        background-color: {COLORS['muted']} !important;
        color: {COLORS['foreground']} !important;
    }}

    /* Expander body content */
    .streamlit-expanderContent,
    .streamlit-expanderContent *,
    details,
    details * {{
        color: {COLORS['foreground']} !important;
    }}

    [data-testid="stExpander"],
    [data-testid="stExpander"] *,
    [data-testid="stExpanderDetails"],
    [data-testid="stExpanderDetails"] *,
    [data-testid="stExpanderDetails"] > div,
    [data-testid="stExpanderDetails"] > div > div {{
        background: {COLORS['background']} !important;
        color: {COLORS['foreground']} !important;
    }}

    [data-testid="stExpanderDetails"] summary,
    [data-testid="stExpanderDetails"] summary * {{
        background: {COLORS['muted']} !important;
        color: {COLORS['foreground']} !important;
    }}

    /* File uploader text and dropzone */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] * {{
        color: {COLORS['foreground']} !important;
    }}

    [data-testid="stFileUploaderDropzone"] {{
        background: {COLORS['muted']} !important;
        border: 1px solid {COLORS['border']} !important;
    }}

    [data-testid="stFileUploaderDropzone"] section,
    [data-testid="stFileUploaderDropzone"] div {{
        background: {COLORS['muted']} !important;
    }}

    /* Remove default padding for full-width layout */
    .css-1d391kg {{
        padding-top: 0 !important;
    }}
    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }}

    /* Remove header spacing */
    header {{
        visibility: hidden;
        height: 0 !important;
    }}

    /* Hide Streamlit sidebar and its toggle entirely. */
    section[data-testid="stSidebar"],
    [data-testid="stSidebarNav"],
    [data-testid="stSidebarNavItems"],
    [data-testid="stSidebarNavSeparator"],
    button[data-testid="stSidebarCollapsedControl"],
    button[kind="header"],
    [data-testid="collapsedControl"] {{
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        min-width: 0 !important;
    }}

    /* Full width layout */
    .main .block-container {{
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }}

    /* Fix title area spacing */
    .stMarkdown h1 {{
        margin-top: 0 !important;
    }}

    /* --- Headers --- */
    h1, h2, h3, h4 {{
        color: {COLORS['foreground']};
        font-weight: 600;
        letter-spacing: -0.02em;
    }}

    h1 {{ font-size: 1.45rem; }}
    h2 {{ font-size: 1.25rem; color: {COLORS['muted_foreground']}; }}
    h3 {{ font-size: 1rem; color: {COLORS['muted_foreground']}; text-transform: uppercase; letter-spacing: 0.05em; }}

    /* --- Sidebar Styling --- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['background']} 0%, {COLORS['muted']} 100%);
        border-right: 1px solid {COLORS['border']};
    }}

    section[data-testid="stSidebar"] .element-container {{
        color: {COLORS['foreground']};
    }}

    section[data-testid="stSidebar"] h1 {{
        color: {COLORS['primary']};
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.1em;
    }}

    /* --- Metric Cards --- */
    .stMetric {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 16px;
        transition: all 0.2s ease;
    }}

    .stMetric:hover {{
        border-color: {COLORS['primary']};
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
    }}

    .stMetric label {{
        color: {COLORS['muted_foreground']} !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .stMetric > div > div:first-child {{
        color: {COLORS['foreground']} !important;
        font-family: {FONTS['mono']};
        font-size: 1.5rem;
        font-weight: 600;
    }}

    /* Positive delta */
    .stMetric [data-testid="stMetricDelta"] {{
        color: {COLORS['primary']} !important;
    }}

    /* --- Button Styling --- */
    .stButton,
    .stButton > div,
    .stButton [data-testid="stElementContainer"],
    .stButton [data-testid="stVerticalBlock"] {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}

    .stButton button {{
        background: {COLORS['muted']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 6px !important;
        color: {COLORS['foreground']} !important;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: none !important;
    }}

    .stButton button:hover {{
        background: {COLORS['card_hover']} !important;
        border-color: {COLORS['border_light']} !important;
        transform: translateY(-1px);
    }}

    .stDownloadButton,
    .stDownloadButton > div,
    .stDownloadButton [data-testid="stElementContainer"],
    .stDownloadButton [data-testid="stVerticalBlock"] {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}

    .stDownloadButton button {{
        background: {COLORS['muted']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 6px !important;
        color: {COLORS['foreground']} !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
    }}

    .stDownloadButton button:hover {{
        background: {COLORS['card_hover']} !important;
        border-color: {COLORS['border_light']} !important;
        color: {COLORS['foreground']} !important;
        transform: translateY(-1px);
    }}

    .stDownloadButton button:focus,
    .stDownloadButton button:focus-visible,
    .stDownloadButton button:active,
    .stDownloadButton button:visited {{
        background: {COLORS['muted']} !important;
        color: {COLORS['foreground']} !important;
        border-color: {COLORS['primary']} !important;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.18) !important;
        outline: none !important;
    }}

    .command-card {{
        border: 1px solid {COLORS['border']} !important;
        background: {COLORS['muted']} !important;
        border-radius: 10px !important;
        padding: 12px !important;
        margin-bottom: 12px !important;
    }}

    .command-card__header {{
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        gap: 12px !important;
        margin-bottom: 8px !important;
    }}

    .command-card__label {{
        font-size: 0.78rem !important;
        color: {COLORS['muted_foreground']} !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        font-weight: 600 !important;
    }}

    .command-card__copy {{
        background: transparent !important;
        color: {COLORS['primary']} !important;
        border: 1px solid {COLORS['border_light']} !important;
        border-radius: 999px !important;
        padding: 4px 10px !important;
        font-size: 0.75rem !important;
        cursor: pointer !important;
    }}

    .command-card__copy:hover {{
        background: rgba(16, 185, 129, 0.08) !important;
        border-color: {COLORS['primary']} !important;
    }}

    .command-card__copy:focus,
    .command-card__copy:focus-visible,
    .command-card__copy:active,
    .command-card__copy:visited {{
        background: rgba(16, 185, 129, 0.12) !important;
        color: {COLORS['foreground']} !important;
        border-color: {COLORS['primary']} !important;
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.18) !important;
    }}

    .command-card__body {{
        display: block !important;
        width: 100% !important;
        overflow-x: auto !important;
        white-space: nowrap !important;
        padding: 10px 12px !important;
        border-radius: 8px !important;
        background: {COLORS['background']} !important;
        border: 1px solid {COLORS['border']} !important;
        color: {COLORS['foreground']} !important;
        font-family: {FONTS['mono']} !important;
        font-size: 0.84rem !important;
        line-height: 1.5 !important;
    }}

    /* Primary button - Emerald */
    .stButton button[kind="primary"] {{
        background: {COLORS['primary']} !important;
        border-color: {COLORS['primary']} !important;
        color: {COLORS['background']} !important;
    }}

    .stButton button[kind="primary"]:hover {{
        background: {COLORS['primary_light']} !important;
        border-color: {COLORS['primary_light']} !important;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
    }}

    button[data-testid^="baseButton-"] {{
        background: transparent !important;
        box-shadow: none !important;
    }}

    div[data-testid="column"] {{
        background: transparent !important;
    }}

    /* --- Tab Styling --- */
    .stTabs button {{
        background: transparent;
        border-radius: 0;
        color: {COLORS['muted_foreground']};
        font-size: 0.75rem;
        font-weight: 500;
        padding: 12px 16px;
        transition: all 0.2s ease;
        border-bottom: 2px solid transparent;
    }}

    .stTabs button:hover {{
        background: rgba(16, 185, 129, 0.05);
        color: {COLORS['foreground']};
    }}

    .stTabs button[aria-selected="true"] {{
        background: rgba(16, 185, 129, 0.1);
        border-bottom: 2px solid {COLORS['primary']};
        color: {COLORS['primary']};
    }}

    /* --- Expander Styling --- */
    .streamlit-expanderHeader {{
        background-color: {COLORS['muted']};
        border-radius: 8px;
        border: 1px solid {COLORS['border']};
        color: {COLORS['foreground']};
        font-weight: 500;
        transition: all 0.2s ease;
    }}

    .streamlit-expanderHeader:hover {{
        background-color: {COLORS['card_hover']};
        border-color: {COLORS['border_light']};
    }}

    /* --- Input Styling --- */
    .stTextInput > div > div,
    .stNumberInput > div > div,
    .stSelectbox > div > div {{
        background-color: {COLORS['muted']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        color: {COLORS['foreground']};
    }}

    .stTextInput > div > div:focus-within,
    .stNumberInput > div > div:focus-within,
    .stSelectbox > div > div:focus-within {{
        border-color: {COLORS['primary']};
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
    }}

    /* ============================================
       COMPREHENSIVE DROPDOWN STYLING
       Targets all possible dropdown selectors
       ============================================ */

    /* Selectbox input text and container */
    .stSelectbox input,
    .stSelectbox .st-bq,
    .stSelectbox [data-baseweb="select"] {{
        color: {COLORS['foreground']} !important;
        background-color: {COLORS['muted']} !important;
    }}

    /* Selectbox selected value display */
    .stSelectbox div[data-baseweb="select"] > div,
    .stSelectbox div[data-testid="stSelectbox"] > div {{
        color: {COLORS['foreground']} !important;
        background-color: {COLORS['muted']} !important;
    }}

    /* Dropdown popover container - GLOBAL */
    [data-baseweb="popover"],
    div[data-baseweb="popover"],
    [id^="popover"],
    div[id^="popover"] {{
        background-color: {COLORS['muted']} !important;
        border: 1px solid {COLORS['border']} !important;
        color: {COLORS['foreground']} !important;
    }}

    /* All elements inside popover */
    [data-baseweb="popover"] *,
    div[data-baseweb="popover"] * {{
        color: {COLORS['foreground']} !important;
        background-color: {COLORS['muted']} !important;
    }}

    /* Dropdown listbox */
    [role="listbox"],
    div[role="listbox"],
    ul[role="listbox"] {{
        background-color: {COLORS['muted']} !important;
        color: {COLORS['foreground']} !important;
    }}

    /* Dropdown options/items */
    [role="option"],
    li[role="option"],
    div[role="listbox"] li,
    ul[role="listbox"] li {{
        background-color: {COLORS['muted']} !important;
        color: {COLORS['foreground']} !important;
    }}

    [role="option"]:hover,
    li[role="option"]:hover,
    div[role="listbox"] li:hover {{
        background-color: {COLORS['card_hover']} !important;
    }}

    [role="option"][aria-selected="true"],
    li[aria-selected="true"] {{
        background-color: rgba(16, 185, 129, 0.2) !important;
        color: {COLORS['primary']} !important;
    }}

    /* Text inside dropdown options */
    [role="option"] span,
    [role="option"] div,
    li[role="option"] span,
    li[role="option"] div,
    div[role="listbox"] li span,
    div[role="listbox"] li div {{
        color: {COLORS['foreground']} !important;
        background-color: transparent !important;
    }}

    /* Selectbox-specific dropdown */
    .stSelectbox div[data-baseweb="select"] div[role="listbox"],
    .stSelectbox div[role="listbox"] {{
        background-color: {COLORS['muted']} !important;
        border: 1px solid {COLORS['border']} !important;
    }}

    .stSelectbox div[data-baseweb="select"] ul li,
    .stSelectbox div[role="listbox"] li {{
        background-color: {COLORS['muted']} !important;
        color: {COLORS['foreground']} !important;
    }}

    .stSelectbox div[data-baseweb="select"] ul li:hover,
    .stSelectbox div[role="listbox"] li:hover {{
        background-color: {COLORS['card_hover']} !important;
    }}

    .stSelectbox div[data-baseweb="select"] ul li[aria-selected="true"] {{
        background-color: rgba(16, 185, 129, 0.2) !important;
        color: {COLORS['primary']} !important;
    }}

    /* Multiselect dropdown */
    .stMultiSelect div[data-baseweb="select"],
    .stMultiSelect div[role="listbox"] {{
        background-color: {COLORS['muted']} !important;
    }}

    .stMultiSelect div[role="listbox"] li {{
        background-color: {COLORS['muted']} !important;
        color: {COLORS['foreground']} !important;
    }}

    .stMultiSelect span {{
        color: {COLORS['foreground']} !important;
    }}

    /* Override white backgrounds anywhere */
    [style*="background: white"],
    [style*="background-color: white"],
    [style*="background: rgb(255"],
    [style*="background-color: rgb(255"],
    [style*="background:#fff"],
    [style*="background-color:#fff"] {{
        background-color: {COLORS['muted']} !important;
        background: {COLORS['muted']} !important;
    }}

    .stTextInput input,
    .stNumberInput input {{
        color: {COLORS['foreground']};
        font-family: {FONTS['mono']};
    }}

    /* --- Status Boxes --- */
    .stSuccess {{
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 3px solid {COLORS['success']};
        border-radius: 0 8px 8px 0;
        color: {COLORS['foreground']};
    }}

    .stInfo {{
        background-color: rgba(59, 130, 246, 0.1);
        border-left: 3px solid {COLORS['info']};
        border-radius: 0 8px 8px 0;
        color: {COLORS['foreground']};
    }}

    .stWarning {{
        background-color: rgba(245, 158, 11, 0.1);
        border-left: 3px solid {COLORS['warning']};
        border-radius: 0 8px 8px 0;
        color: {COLORS['foreground']};
    }}

    .stError {{
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 3px solid {COLORS['error']};
        border-radius: 0 8px 8px 0;
        color: {COLORS['foreground']};
    }}

    /* --- DataFrame / Table Styling --- */
    .stDataFrame {{
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        overflow: hidden;
        background: {COLORS['muted']} !important;
    }}

    .stDataFrame table {{
        background-color: {COLORS['muted']} !important;
    }}

    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] *,
    [data-testid="stDataEditor"],
    [data-testid="stDataEditor"] * {{
        background-color: {COLORS['muted']} !important;
        color: {COLORS['foreground']} !important;
    }}

    .stDataFrame thead th {{
        background-color: {COLORS['card']} !important;
        color: {COLORS['muted_foreground']} !important;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }}

    .stDataFrame tbody tr {{
        border-bottom: 1px solid {COLORS['border']} !important;
    }}

    .stDataFrame tbody tr:hover {{
        background-color: rgba(16, 185, 129, 0.05) !important;
    }}

    [data-testid="stDataFrame"] [role="gridcell"],
    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stDataEditor"] [role="gridcell"],
    [data-testid="stDataEditor"] [role="columnheader"] {{
        background: {COLORS['muted']} !important;
        color: {COLORS['foreground']} !important;
        border-color: {COLORS['border']} !important;
    }}

    /* --- Top Navigation Shell --- */
    .idx-top-nav-anchor {{
        display: block;
        height: 0;
    }}

    .idx-top-nav-anchor + div[data-testid="stHorizontalBlock"] {{
        position: sticky;
        top: 0;
        z-index: 30;
        padding: 8px 0 10px 0;
        background:
            linear-gradient(180deg, rgba(9, 9, 11, 0.98), rgba(9, 9, 11, 0.92));
        border-bottom: 1px solid rgba(63, 63, 70, 0.65);
        backdrop-filter: blur(14px);
    }}

    .idx-top-nav-shell {{
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: center;
        gap: 14px;
    }}

    .idx-top-nav-links {{
        display: grid;
        grid-template-columns: repeat(8, minmax(0, 1fr));
        gap: 10px;
        align-items: center;
    }}

    .idx-top-nav-link {{
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        padding: 8px 12px;
        border-radius: 999px;
        border: 1px solid transparent;
        background: transparent;
        color: {COLORS['muted_foreground']} !important;
        font-size: 0.84rem;
        font-weight: 500;
        line-height: 1;
        text-decoration: none !important;
        transition: all 0.18s ease;
    }}

    .idx-top-nav-link:hover {{
        color: {COLORS['foreground']} !important;
        background: rgba(255, 255, 255, 0.04);
        border-color: {COLORS['border']};
    }}

    .idx-top-nav-link-active {{
        color: {COLORS['primary']} !important;
        background: rgba(16, 185, 129, 0.10);
        border-color: rgba(16, 185, 129, 0.45);
        box-shadow: inset 0 0 0 1px rgba(16, 185, 129, 0.08);
    }}

    .idx-build-stamp {{
        display: flex;
        align-items: center;
        justify-content: flex-end;
        min-height: 38px;
        padding: 8px 12px;
        color: {COLORS['muted_foreground']} !important;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        white-space: nowrap;
    }}

    .idx-link-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
    }}

    .idx-link-pill,
    .idx-module-link {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 42px;
        padding: 10px 14px;
        border-radius: 12px;
        border: 1px solid {COLORS['border']};
        background: linear-gradient(135deg, rgba(24, 24, 27, 0.96), rgba(39, 39, 42, 0.92));
        color: {COLORS['foreground']} !important;
        text-decoration: none !important;
        font-weight: 500;
        transition: all 0.18s ease;
    }}

    .idx-link-pill:hover,
    .idx-module-link:hover {{
        border-color: {COLORS['border_light']};
        background: {COLORS['card_hover']};
        transform: translateY(-1px);
    }}

    .idx-module-link {{
        width: 100%;
        margin-top: 4px;
    }}

    @media (max-width: 1100px) {{
        .idx-top-nav-shell {{
            grid-template-columns: 1fr;
        }}

        .idx-top-nav-links {{
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }}

        .idx-build-stamp {{
            justify-content: flex-start;
            padding-left: 0;
        }}
    }}

    @media (max-width: 900px) {{
        .idx-link-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
    }}

    .idx-status-strip-anchor {{
        display: block;
        height: 0;
    }}

    .idx-status-strip-anchor + div[data-testid="stHorizontalBlock"] {{
        margin-top: 4px;
        margin-bottom: 8px;
    }}

    .idx-status-chip {{
        padding: 8px 10px;
        border-radius: 10px;
        border: 1px solid {COLORS['border']};
        background: linear-gradient(135deg, rgba(24, 24, 27, 0.96), rgba(39, 39, 42, 0.92));
        min-height: 58px;
    }}

    .idx-status-label {{
        font-size: 0.66rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {COLORS['muted_foreground']} !important;
        margin-bottom: 6px;
    }}

    .idx-status-value {{
        font-size: 0.92rem;
        font-weight: 600;
        color: {COLORS['foreground']} !important;
    }}

    /* --- Progress Bar --- */
    .stProgress > div > div {{
        background-color: {COLORS['muted']};
        border-radius: 4px;
    }}

    .stProgress > div > div > div {{
        background: linear-gradient(90deg, {COLORS['primary_dark']}, {COLORS['primary']});
        border-radius: 4px;
    }}

    /* --- Slider --- */
    .stSlider > div > div > div {{
        background-color: {COLORS['muted']};
    }}

    /* --- Checkbox & Radio --- */
    .stCheckbox label,
    .stRadio > label {{
        color: {COLORS['foreground']};
    }}

    /* --- Toast Notifications --- */
    .stToast {{
        background: {COLORS['muted']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        color: {COLORS['foreground']};
    }}

    /* --- Scrollbar --- */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}

    ::-webkit-scrollbar-track {{
        background: transparent;
    }}

    ::-webkit-scrollbar-thumb {{
        background: {COLORS['border']};
        border-radius: 3px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['border_light']};
    }}

    /* ============================================
       CUSTOM COMPONENTS
       ============================================ */

    /* --- NextGen Card --- */
    .nextgen-card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }}

    .nextgen-card:hover {{
        border-color: {COLORS['primary']};
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.05);
    }}

    /* --- Price Display (Monospace) --- */
    .price-mono {{
        font-family: {FONTS['mono']};
        font-weight: 500;
    }}

    /* --- Positive/Negative Values --- */
    .positive {{
        color: {COLORS['primary']};
    }}

    .negative {{
        color: {COLORS['destructive']};
    }}

    /* --- Live Indicator Badge --- */
    .live-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 4px 12px;
        border-radius: 9999px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: {COLORS['primary']};
    }}

    .live-badge .pulse {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: {COLORS['primary']};
        animation: pulse 2s infinite;
    }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}

    /* --- Stock List Item --- */
    .stock-item {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.15s ease;
    }}

    .stock-item:hover {{
        background: {COLORS['muted']};
    }}

    .stock-item.active {{
        background: {COLORS['card_hover']};
        border-left: 2px solid {COLORS['primary']};
    }}

    .stock-item .symbol {{
        font-weight: 600;
        color: {COLORS['foreground']};
    }}

    .stock-item.active .symbol {{
        color: {COLORS['primary']};
    }}

    .stock-item .price {{
        font-family: {FONTS['mono']};
        font-size: 0.875rem;
        color: {COLORS['muted_foreground']};
    }}

    /* --- Signal Badge --- */
    .signal-badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }}

    .signal-badge.bullish {{
        background: rgba(16, 185, 129, 0.2);
        color: {COLORS['primary']};
        border: 1px solid rgba(16, 185, 129, 0.3);
    }}

    .signal-badge.bearish {{
        background: rgba(239, 68, 68, 0.2);
        color: {COLORS['destructive']};
        border: 1px solid rgba(239, 68, 68, 0.3);
    }}

    .signal-badge.neutral {{
        background: rgba(161, 161, 170, 0.2);
        color: {COLORS['muted_foreground']};
        border: 1px solid rgba(161, 161, 170, 0.3);
    }}

    /* --- Agent Card --- */
    .agent-card {{
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
        background: rgba(24, 24, 27, 0.5);
        border-left: 3px solid;
    }}

    .agent-card.auditor {{ border-left-color: {COLORS['agent_auditor']}; }}
    .agent-card.growth {{ border-left-color: {COLORS['agent_growth']}; }}
    .agent-card.synthesizer {{
        border-left-color: {COLORS['agent_synthesizer']};
        background: rgba(16, 185, 129, 0.05);
        border-color: rgba(16, 185, 129, 0.3);
    }}

    /* --- Technical Indicator Grid --- */
    .indicator-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
    }}

    .indicator-card {{
        background: {COLORS['muted']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 12px;
        text-align: center;
    }}

    .indicator-card.bullish {{
        background: rgba(16, 185, 129, 0.1);
        border-color: rgba(16, 185, 129, 0.3);
    }}

    .indicator-card .label {{
        font-size: 0.75rem;
        color: {COLORS['muted_foreground']};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }}

    .indicator-card .value {{
        font-family: {FONTS['mono']};
        font-size: 1.1rem;
        font-weight: 600;
        color: {COLORS['foreground']};
    }}

    .indicator-card.bullish .value {{
        color: {COLORS['primary']};
    }}

    /* --- Conviction Score --- */
    .conviction-score {{
        font-size: 3rem;
        font-weight: 300;
        font-family: {FONTS['mono']};
    }}

    .conviction-bar {{
        height: 8px;
        background: {COLORS['muted']};
        border-radius: 4px;
        overflow: hidden;
        border: 1px solid {COLORS['border']};
    }}

    .conviction-bar .fill {{
        height: 100%;
        background: {COLORS['primary']};
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
        transition: width 0.3s ease;
    }}

    /* --- Buy/Sell Buttons --- */
    .buy-button {{
        background: rgba(16, 185, 129, 0.9);
        color: #fafafa;
        border: 1px solid rgba(16, 185, 129, 0.5);
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
    }}

    .buy-button:hover {{
        background: {COLORS['primary']};
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
    }}

    .sell-button {{
        background: rgba(239, 68, 68, 0.9);
        color: #fafafa;
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
    }}

    .sell-button:hover {{
        background: {COLORS['destructive']};
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.3);
    }}

    /* --- Section Headers --- */
    .section-header {{
        font-size: 0.75rem;
        font-weight: 600;
        color: {COLORS['muted_foreground']};
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    /* --- Foreign Flow Card --- */
    .flow-card {{
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }}

    .flow-card::after {{
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, {COLORS['primary']}, transparent);
    }}

    .flow-card .value {{
        font-family: {FONTS['mono']};
        font-size: 1.75rem;
        font-weight: 600;
        color: {COLORS['primary']};
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.3);
    }}

    /* --- Support/Resistance Levels --- */
    .sr-level {{
        display: flex;
        justify-content: space-between;
        padding: 6px 8px;
        font-size: 0.8rem;
        border-bottom: 1px solid {COLORS['border']};
    }}

    .sr-level.current {{
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 4px;
        color: {COLORS['primary']};
        font-weight: 500;
    }}
    </style>
    """


def get_chart_colors() -> dict:
    """Return color palette for Plotly charts."""
    return {
        "up": COLORS["chart_up"],
        "down": COLORS["chart_down"],
        "neutral": COLORS["chart_neutral"],
        "volume_up": COLORS["chart_volume_up"],
        "volume_down": COLORS["chart_volume_down"],
        "background": COLORS["background"],
        "grid": COLORS["border"],
        "text": COLORS["foreground"],
        "muted_text": COLORS["muted_foreground"],
        "primary": COLORS["primary"],
        "ma20": "#ff9800",
        "ma50": "#3b82f6",
        "rsi": "#a855f7",
        "macd": "#3b82f6",
        "signal": "#ff9800",
    }


def apply_chart_theme(fig) -> None:
    """Apply NextGen theme to a Plotly figure."""
    chart_colors = get_chart_colors()

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=chart_colors["background"],
        plot_bgcolor=chart_colors["background"],
        font=dict(
            family=FONTS["sans"],
            color=chart_colors["text"]
        ),
        xaxis=dict(
            gridcolor=chart_colors["grid"],
            linecolor=chart_colors["grid"],
        ),
        yaxis=dict(
            gridcolor=chart_colors["grid"],
            linecolor=chart_colors["grid"],
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=chart_colors["grid"],
        ),
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_price(price: float) -> str:
    """Format price with monospace styling."""
    return f'<span class="price-mono">{price:,.0f}</span>'


def format_change(change: float, as_percent: bool = True) -> str:
    """Format price change with color."""
    if as_percent:
        value = f"{change:+.2f}%"
    else:
        value = f"{change:+,.0f}"

    css_class = "positive" if change >= 0 else "negative"
    return f'<span class="{css_class}">{value}</span>'


def render_live_badge(text: str = "LIVE") -> str:
    """Render a live indicator badge."""
    return (
        '<div class="live-badge">'
        '<div class="pulse"></div>'
        f"{text}"
        "</div>"
    )


def render_signal_badge(signal_type: str) -> str:
    """Render a signal badge (bullish/bearish/neutral)."""
    signal_type = signal_type.lower()
    if signal_type in ["buy", "bullish", "strong buy"]:
        return f'<span class="signal-badge bullish">BULLISH</span>'
    elif signal_type in ["sell", "bearish", "strong sell"]:
        return f'<span class="signal-badge bearish">BEARISH</span>'
    else:
        return f'<span class="signal-badge neutral">NEUTRAL</span>'
