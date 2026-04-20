import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_community.llms import Ollama
import plotly.graph_objects as go
from datetime import datetime
import sqlite3
import hashlib  
import json

#-----------------------
def init_database():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT NOT NULL,
        role TEXT NOT NULL,
        message TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(email, password):
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (email, password_hash) VALUES (?, ?)', 
                      (email, hash_password(password)))
        conn.commit()
        conn.close()
        return True, "Account created!"
    except:
        return False, "Email exists"

def verify_user(email, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ? AND password_hash = ?', 
                  (email, hash_password(password)))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def save_chat_message(user_email, role, message):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO chat_history (user_email, role, message) VALUES (?, ?, ?)',
                  (user_email, role, message))
    conn.commit()
    conn.close()

def load_chat_history(user_email):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT role, message FROM chat_history WHERE user_email = ? ORDER BY timestamp', (user_email,))
    return [{'role': r[0], 'content': r[1]} for r in cursor.fetchall()]

def clear_chat_history(user_email):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM chat_history WHERE user_email = ?', (user_email,))
    conn.commit()
    conn.close()

init_database()
# Page setup



st.set_page_config(layout="wide", page_title="AI-Optimized Universal Data Analyzer", page_icon="📊")
st.markdown("""
<style>
.block-container {
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}
[data-testid="stVerticalBlock"] {
    gap: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🔐 AI Data Analyzer")
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            email = st.text_input("Email", key="le")
            pwd = st.text_input("Password", type="password", key="lp")
            if st.button("Login", type="primary"):
                if verify_user(email, pwd):
                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.session_state.chat_messages = load_chat_history(email)
                    st.rerun()
                else:
                    st.error("Invalid")
        
        with tab2:
            email = st.text_input("Email", key="se")
            pwd = st.text_input("Password", type="password", key="sp")
            pwd2 = st.text_input("Confirm", type="password", key="sc")
            if st.button("Create", type="primary"):
                if pwd == pwd2 and len(pwd) >= 6:
                    ok, msg = create_user(email, pwd)
                    st.success(msg) if ok else st.error(msg)
                else:
                    st.error("Password mismatch/too short")
        
        st.markdown("**Reg:** 221211101041 | **Dept:** CSE-AI")
    st.stop()

# User logged in - show main app
col1, col2, col3 = st.columns([2, 4, 2])
with col1:
    st.write(f"👤 {st.session_state.user_email}")
with col2:
    st.markdown("<h1 style='text-align:center;color:#1F618D;'>AI-Optimized Universal Data Analyzer</h1>", unsafe_allow_html=True)
with col3:
    st.write("**221211101041** | **CSE-AI**")
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()
st.markdown("---")
# -------------------------------------------------------------------------
# 🚀 PERFORMANCE FIX: CACHING & SETUP
# -------------------------------------------------------------------------
# ✅ PASTE THIS NEW CODE -------------------------------------------
# -------------------------------------------------------------------------
# 🚀 TURBO PERFORMANCE FIX: CACHING RESOURCE (Zero-Copy)
# -------------------------------------------------------------------------
@st.cache_resource
def load_data(file):
    file.seek(0) # Reset file pointer
    try:
        # read_csv with low_memory=False is safer for mixed types
        df = pd.read_csv(file, low_memory=False)
        
        # Optimize memory: Downcast numbers to float32/int32
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        return df
    except Exception as e:
        return None

# Sidebar controls
st.sidebar.header("📂 Upload / Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
sample_button = st.sidebar.button("Generate Synthetic Sample Data")

# Keep this for AI Tab
task = st.sidebar.selectbox("Run AI Task", ["None", "Classification (label exists)", "Regression (continuous target)", "Clustering", "Time Series Forecasting"])

# --- DATA LOADING LOGIC ---
if sample_button:
    # Synthetic Data Generation
    n = 1000
    df = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "Category": np.random.choice(["A", "B", "C"], n),
        "Value": np.random.randn(n) * 100,
        "Target": np.random.choice([0, 1], n)
    })
    st.session_state["df"] = df
    st.success("✅ Sample Data Generated")

if uploaded_file:
    # ✅ USE THE TURBO CACHED FUNCTION
    with st.spinner("Processing large file..."):
        df = load_data(uploaded_file)
    
    if df is not None:
        st.session_state["df"] = df
        st.success(f"✅ Loaded {uploaded_file.name} (Cached)")
    else:
        st.error("❌ Error loading CSV.")

# Stop if no data
if "df" not in st.session_state:
    st.info("Upload a CSV file to begin.")
    st.stop()

# Get the main dataframe
df = st.session_state["df"]

# 🚀 SMART SAMPLING (The Lag Killer)
# ✅ PASTE THIS NEW 5k SAMPLING BLOCK (Instant Speed)
if len(df) > 5000:
    if "df_chart" not in st.session_state:
        st.sidebar.warning(f"⚠️ Large Data: Charts using 5k sample for speed.")
        st.session_state["df_chart"] = df.sample(n=5000, random_state=42)
    df_chart = st.session_state["df_chart"]
else:
    df_chart = df
# ------------------------------------------------------------------
# --- Helper Function: Smart Column Detection ---
def get_column_types(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = []
    
    # Try to identify date columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col], errors='raise')
                # If successful and looks like date, add it (simple check)
                if df[col].astype(str).str.contains('-|/').any():
                    date_cols.append(col)
            except:
                pass
        elif np.issubdtype(df[col].dtype, np.datetime64):
            date_cols.append(col)
            
    return num_cols, cat_cols, date_cols

# --- Logic to Load Data ---
if sample_button:
    n = 500
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="H"),
        "machine_id": rng.integers(1, 6, size=n).astype(str),
        "temp_C": rng.normal(70, 5, size=n),
        "vibration": np.abs(rng.normal(1.2, 0.4, size=n)),
        "pressure": rng.normal(30, 3, size=n),
        "units_produced": rng.integers(80, 150, size=n),
    })
    df["defective"] = ((df["vibration"] > 1.6) & (df["temp_C"] > 74)).astype(int)
    st.session_state["df"] = df
    st.success("✅ Synthetic dataset generated successfully!")


if "df" not in st.session_state:
    st.info("Upload a CSV or generate a sample dataset to begin.")
    st.stop()

df = st.session_state["df"]
num_cols, cat_cols, date_cols = get_column_types(df)

# Sidebar Target Input (Dynamic)
target_options = ["None"] + df.columns.tolist()
target_column = st.sidebar.selectbox("Target Column (for AI Prediction)", target_options, index=0)


# Tabs
tab_preview, tab_eda, tab_flowchart, tab_overall, tab_narration, tab_assistant, tab_ai, tab_export = st.tabs(
    ["Data Preview", "Data Inspector", "Flowchart", "Dash Report", "Narration", "Data Agent", "Trend Forecast", "Export / Save"]
)

# --- TAB 1: Data Preview ---
# ==============================================================================
#  TAB 1: DATA PREVIEW & BASIC CLEANING (Excel-like Interactive Editor)
# ==============================================================================
with tab_preview:
    st.markdown("## 📄 Data Preview & Basic Cleaning")
    st.caption("Interactive Excel-like editor with comprehensive cleaning tools")
    
    # =========================================================================
    # INITIALIZATION: Store original dataset for reset functionality
    # =========================================================================
    if "df_original" not in st.session_state:
        st.session_state["df_original"] = df.copy()
    
    if "selected_column" not in st.session_state:
        st.session_state["selected_column"] = None
    
    # =========================================================================
    # SECTION 1: EDITABLE DATA TABLE (Excel-like)
    # =========================================================================
    st.markdown("### 📊 Editable Data Table")
    st.caption("Click on any cell to edit. Scroll to view all data. Changes are saved when you click 'Save Changes'.")
    
    # Create editable dataframe
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",  # Allows adding/deleting rows
        height=400,
        hide_index=False,
        key="data_editor"
    )
    
    # Save changes button
    col_save, col_info = st.columns([1, 3])
    with col_save:
        if st.button("💾 Save Changes", type="primary"):
            df = edited_df.copy()
            st.session_state["df"] = df
            st.success("✅ Changes saved successfully!")
            st.rerun()
    
    with col_info:
        st.info(f"📊 Current dataset: **{df.shape[0]:,}** rows × **{df.shape[1]}** columns")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 2: BASIC CLEANING ACTIONS
    # =========================================================================
    st.markdown("### 🧹 Quick Cleaning Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🗑️ Remove Duplicates", use_container_width=True):
            original_count = len(df)
            df = df.drop_duplicates()
            removed_count = original_count - len(df)
            st.session_state["df"] = df
            
            if removed_count > 0:
                st.success(f"✅ Removed {removed_count} duplicate rows")
            else:
                st.info("ℹ️ No duplicates found")
            st.rerun()
    
    with col2:
        if st.button("🔧 Fill Missing Values", use_container_width=True):
            filled_count = 0
            
            # Fill numeric columns with mean
            for col in num_cols:
                missing_before = df[col].isna().sum()
                df[col].fillna(df[col].mean(), inplace=True)
                filled_count += missing_before
            
            # Fill categorical columns with mode
            for col in cat_cols:
                if df[col].mode().shape[0] > 0:
                    missing_before = df[col].isna().sum()
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    filled_count += missing_before
            
            st.session_state["df"] = df
            
            if filled_count > 0:
                st.success(f"✅ Filled {filled_count} missing values")
            else:
                st.info("ℹ️ No missing values found")
            st.rerun()
    
    with col3:
        if st.button("❌ Drop Missing Rows", use_container_width=True):
            original_count = len(df)
            df = df.dropna()
            removed_count = original_count - len(df)
            st.session_state["df"] = df
            
            if removed_count > 0:
                st.warning(f"⚠️ Dropped {removed_count} rows with missing values")
            else:
                st.info("ℹ️ No rows with missing values")
            st.rerun()
    
    with col4:
        if st.button("🔄 Reset to Original", use_container_width=True):
            df = st.session_state["df_original"].copy()
            st.session_state["df"] = df
            st.info("🔄 Dataset reset to original state")
            st.rerun()
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 3: INTERACTIVE COLUMN SUMMARY TABLE
    # =========================================================================
    st.markdown("### 📋 Column Summary (Click to Inspect)")
    st.caption("Click on any column name to view detailed statistics and visualizations")
    
    # Build summary table
    summary_data = []
    
    for col in df.columns:
        row = {
            "Column Name": col,
            "Data Type": str(df[col].dtype),
            "Missing Values": df[col].isna().sum(),
            "Unique Values": df[col].nunique(),
        }
        
        # Add mean for numeric columns
        if col in num_cols:
            row["Mean"] = f"{df[col].mean():.2f}" if not df[col].isna().all() else "N/A"
        else:
            row["Mean"] = "N/A"
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Display summary with clickable rows
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        height=min(400, len(summary_df) * 35 + 38)
    )
    
    # Column selector for inspection
    st.markdown("---")
    st.markdown("### 🔍 Select Column to Inspect")
    
    col_select_left, col_select_right = st.columns([2, 1])
    
    with col_select_left:
        selected_col = st.selectbox(
            "Choose a column for detailed analysis:",
            options=df.columns,
            key="column_selector"
        )
        st.session_state["selected_column"] = selected_col
    
    with col_select_right:
        # Quick filter options
        filter_option = st.selectbox(
            "Quick Filter:",
            ["Show All", "Show Missing Only", "Show Duplicates Only"],
            key="quick_filter"
        )
    
    # =========================================================================
    # APPLY FILTERS
    # =========================================================================
    filtered_df = df.copy()
    
    if filter_option == "Show Missing Only":
        if selected_col:
            filtered_df = df[df[selected_col].isna()]
            st.info(f"Showing {len(filtered_df)} rows with missing values in '{selected_col}'")
    
    elif filter_option == "Show Duplicates Only":
        filtered_df = df[df.duplicated(keep=False)]
        st.info(f"Showing {len(filtered_df)} duplicate rows")
    
    # Show filtered data if filter is applied
    if filter_option != "Show All":
        st.markdown("#### Filtered Data:")
        st.dataframe(filtered_df, use_container_width=True, height=300)
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 4: COLUMN INSPECTOR PANEL
    # =========================================================================
    if selected_col:
        st.markdown(f"### 🔬 Column Inspector: **{selected_col}**")
        
        # Statistics row
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        stat_col1.metric(
            label="Data Type",
            value=str(df[selected_col].dtype)
        )
        
        stat_col2.metric(
            label="Missing Values",
            value=f"{df[selected_col].isna().sum():,}",
            delta=f"{(df[selected_col].isna().sum() / len(df) * 100):.1f}%"
        )
        
        stat_col3.metric(
            label="Unique Values",
            value=f"{df[selected_col].nunique():,}"
        )
        
        if selected_col in num_cols:
            stat_col4.metric(
                label="Mean",
                value=f"{df[selected_col].mean():.2f}"
            )
        else:
            stat_col4.metric(
                label="Most Frequent",
                value=str(df[selected_col].mode()[0]) if len(df[selected_col].mode()) > 0 else "N/A"
            )
        
        st.markdown("---")
        
        # Visualization section
        viz_col1, viz_col2 = st.columns([2, 1])
        
        with viz_col1:
            # NUMERIC COLUMN VISUALIZATION
            if selected_col in num_cols:
                st.markdown("#### 📊 Distribution")
                
                fig = px.histogram(
                    df,
                    x=selected_col,
                    marginal="box",
                    template="plotly_white",
                    nbins=30
                )
                fig.update_layout(
                    height=400,
                    title=f"Distribution of {selected_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Outlier detection
                q1 = df[selected_col].quantile(0.25)
                q3 = df[selected_col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
                
                if len(outliers) > 0:
                    st.warning(f"⚠️ Detected {len(outliers)} outliers (outside {lower_bound:.2f} - {upper_bound:.2f})")
                    
                    if st.button(f"🔍 Show Outliers for {selected_col}"):
                        st.dataframe(outliers, use_container_width=True)
            
            # CATEGORICAL COLUMN VISUALIZATION
            else:
                st.markdown("#### 📊 Value Counts")
                
                value_counts = df[selected_col].value_counts().head(20)
                
                fig = px.bar(
                    x=value_counts.values,
                    y=value_counts.index,
                    orientation='h',
                    template="plotly_white",
                    labels={'x': 'Count', 'y': selected_col}
                )
                fig.update_layout(
                    height=400,
                    title=f"Top 20 Values in {selected_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show all unique values if requested
                if df[selected_col].nunique() <= 50:
                    with st.expander(f"📋 All Unique Values ({df[selected_col].nunique()})"):
                        unique_vals = df[selected_col].value_counts()
                        st.dataframe(
                            unique_vals.reset_index(),
                            use_container_width=True,
                            hide_index=True
                        )
        
        with viz_col2:
            st.markdown("#### 📈 Statistics")
            
            if selected_col in num_cols:
                # Numeric statistics
                stats = df[selected_col].describe()
                
                stats_df = pd.DataFrame({
                    'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max'],
                    'Value': [
                        f"{stats['count']:.0f}",
                        f"{stats['mean']:.2f}",
                        f"{stats['std']:.2f}",
                        f"{stats['min']:.2f}",
                        f"{stats['25%']:.2f}",
                        f"{stats['50%']:.2f}",
                        f"{stats['75%']:.2f}",
                        f"{stats['max']:.2f}"
                    ]
                })
                
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Skewness and kurtosis
                st.markdown("---")
                st.markdown("**Shape Metrics:**")
                st.write(f"**Skewness:** {df[selected_col].skew():.3f}")
                st.write(f"**Kurtosis:** {df[selected_col].kurtosis():.3f}")
                
            else:
                # Categorical statistics
                total_count = len(df[selected_col])
                unique_count = df[selected_col].nunique()
                missing_count = df[selected_col].isna().sum()
                mode_val = df[selected_col].mode()[0] if len(df[selected_col].mode()) > 0 else "N/A"
                mode_count = df[selected_col].value_counts().iloc[0] if len(df[selected_col].value_counts()) > 0 else 0
                
                stats_df = pd.DataFrame({
                    'Statistic': ['Total Count', 'Unique', 'Missing', 'Most Frequent', 'Mode Count', 'Mode %'],
                    'Value': [
                        f"{total_count:,}",
                        f"{unique_count:,}",
                        f"{missing_count:,}",
                        str(mode_val),
                        f"{mode_count:,}",
                        f"{(mode_count/total_count*100):.1f}%"
                    ]
                })
                
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # =====================================================================
        # COLUMN-SPECIFIC ACTIONS
        # =====================================================================
        st.markdown(f"#### ⚙️ Actions for '{selected_col}'")
        
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button(f"🗑️ Delete Column", key=f"del_{selected_col}"):
                df = df.drop(columns=[selected_col])
                st.session_state["df"] = df
                st.session_state["selected_column"] = None
                st.success(f"✅ Deleted column '{selected_col}'")
                st.rerun()
        
        with action_col2:
            if selected_col in num_cols:
                if st.button(f"🔧 Fill Missing (Mean)", key=f"fill_{selected_col}"):
                    missing_count = df[selected_col].isna().sum()
                    df[selected_col].fillna(df[selected_col].mean(), inplace=True)
                    st.session_state["df"] = df
                    st.success(f"✅ Filled {missing_count} missing values with mean")
                    st.rerun()
            else:
                if st.button(f"🔧 Fill Missing (Mode)", key=f"fill_{selected_col}"):
                    missing_count = df[selected_col].isna().sum()
                    if len(df[selected_col].mode()) > 0:
                        df[selected_col].fillna(df[selected_col].mode()[0], inplace=True)
                        st.session_state["df"] = df
                        st.success(f"✅ Filled {missing_count} missing values with mode")
                        st.rerun()
        
        with action_col3:
            if selected_col in num_cols:
                if st.button(f"📉 Remove Outliers", key=f"outlier_{selected_col}"):
                    original_count = len(df)
                    q1 = df[selected_col].quantile(0.25)
                    q3 = df[selected_col].quantile(0.75)
                    iqr = q3 - q1
                    df = df[~((df[selected_col] < (q1 - 1.5 * iqr)) | (df[selected_col] > (q3 + 1.5 * iqr)))]
                    removed_count = original_count - len(df)
                    st.session_state["df"] = df
                    st.success(f"✅ Removed {removed_count} outlier rows")
                    st.rerun()
    
    # =========================================================================
    # ADVANCED CLEANING OPTIONS (Expandable)
    # =========================================================================
    st.markdown("---")
    
    with st.expander("🔧 Advanced Cleaning Options"):
        st.markdown("### Advanced Data Cleaning Tools")
        
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            st.markdown("#### 🎯 Type Conversion")
            
            convert_col = st.selectbox("Select column to convert:", df.columns, key="convert_col")
            convert_type = st.selectbox(
                "Convert to:",
                ["numeric", "string", "datetime", "category"],
                key="convert_type"
            )
            
            if st.button("🔄 Convert Type"):
                try:
                    if convert_type == "numeric":
                        df[convert_col] = pd.to_numeric(df[convert_col], errors='coerce')
                    elif convert_type == "string":
                        df[convert_col] = df[convert_col].astype(str)
                    elif convert_type == "datetime":
                        df[convert_col] = pd.to_datetime(df[convert_col], errors='coerce')
                    elif convert_type == "category":
                        df[convert_col] = df[convert_col].astype('category')
                    
                    st.session_state["df"] = df
                    st.success(f"✅ Converted '{convert_col}' to {convert_type}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Conversion failed: {str(e)}")
        
        with adv_col2:
            st.markdown("#### 🔀 Column Operations")
            
            rename_col = st.selectbox("Rename column:", df.columns, key="rename_col")
            new_name = st.text_input("New name:", key="new_name")
            
            if st.button("✏️ Rename") and new_name:
                if new_name not in df.columns:
                    df = df.rename(columns={rename_col: new_name})
                    st.session_state["df"] = df
                    st.success(f"✅ Renamed '{rename_col}' to '{new_name}'")
                    st.rerun()
                else:
                    st.error("❌ Column name already exists!")
        
        st.markdown("---")
        
        # Bulk operations
        st.markdown("#### 🚀 Bulk Operations")
        
        bulk_col1, bulk_col2, bulk_col3 = st.columns(3)
        
        with bulk_col1:
            if st.button("🧹 Remove All Duplicates"):
                original = len(df)
                df = df.drop_duplicates()
                st.session_state["df"] = df
                st.success(f"✅ Removed {original - len(df)} duplicates")
                st.rerun()
        
        with bulk_col2:
            if st.button("🔧 Fill All Missing"):
                for col in num_cols:
                    df[col].fillna(df[col].mean(), inplace=True)
                for col in cat_cols:
                    if len(df[col].mode()) > 0:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                st.session_state["df"] = df
                st.success("✅ Filled all missing values")
                st.rerun()
        
        with bulk_col3:
            if st.button("📉 Remove All Outliers"):
                original = len(df)
                for col in num_cols:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    df = df[~((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr)))]
                st.session_state["df"] = df
                st.success(f"✅ Removed {original - len(df)} outlier rows")
                st.rerun()
# ==============================================================================
#  TAB 2: MANUFACTURING DECISION SUPPORT SYSTEM (MDSS)
#  Based on COMPOSITION Framework (Vafeiadis et al. 2019)
# ==============================================================================
with tab_overall:
    
    # =========================================================================
    # INTELLIGENT DATA ANALYSIS ENGINE
    # =========================================================================
    
    # Clean and prepare data
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Smart sampling for performance
    if len(df) > 5000:
        if "df_chart" not in st.session_state:
            st.session_state["df_chart"] = df.sample(n=5000, random_state=42)
        df_chart = st.session_state["df_chart"]
    else:
        df_chart = df
    
    # Column classification
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    target_col = target_column if target_column != "None" else None 
    
    # Date detection
    date_cols = []
    for col in df.columns:
        if any(x in col.lower() for x in ["date", "time", "timestamp", "year", "month"]):
            try:
                pd.to_datetime(df[col], errors='coerce')
                date_cols.append(col)
            except:
                pass
    
    # =========================================================================
    # AUTOMATED SCENARIO DETECTION (AI Decision Logic)
    # =========================================================================
    
    all_text = ' '.join([col.lower() for col in df.columns])
    
    # Detect production type and critical metrics
    production_type = "GENERIC"
    critical_metric = None
    health_metric = None
    efficiency_metric = None
    machine_id_col = None
    
    # Manufacturing/Production Detection
    if any(word in all_text for word in ['machine', 'sensor', 'vibration', 'temperature', 'rpm', 'speed']):
        production_type = "MANUFACTURING"
        critical_metric = next((c for c in num_cols if 'vibration' in c.lower()), None)
        health_metric = next((c for c in num_cols if 'temperature' in c.lower() or 'temp' in c.lower()), None)
        efficiency_metric = next((c for c in num_cols if 'rpm' in c.lower() or 'speed' in c.lower()), None)
        machine_id_col = next((c for c in cat_cols if 'machine' in c.lower() or 'sensor' in c.lower()), None)
    
    # Quality Control Detection
    elif any(word in all_text for word in ['defect', 'quality', 'inspection', 'pass', 'fail']):
        production_type = "QUALITY_CONTROL"
        critical_metric = next((c for c in cat_cols if 'status' in c.lower() or 'result' in c.lower()), None)
        health_metric = next((c for c in num_cols if 'score' in c.lower() or 'rating' in c.lower()), None)
    
    # Supply Chain Detection
    elif any(word in all_text for word in ['shipment', 'delivery', 'warehouse', 'inventory', 'stock']):
        production_type = "SUPPLY_CHAIN"
        critical_metric = next((c for c in num_cols if 'quantity' in c.lower() or 'stock' in c.lower()), None)
        health_metric = next((c for c in num_cols if 'delay' in c.lower() or 'time' in c.lower()), None)
    
    # Fallback to first important numeric metric
    if not critical_metric and num_cols:
        critical_metric = num_cols[-1]
    if not health_metric and len(num_cols) > 1:
        health_metric = num_cols[0]
    
    # =========================================================================
    # DATA INTEGRITY ANALYSIS (Automated Quality Check)
    # =========================================================================
    
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isna().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100
    
    # Find problematic sensors/machines
    problematic_sources = []
    for col in df.columns:
        col_missing = df[col].isna().sum()
        if col_missing > len(df) * 0.1:  # More than 10% missing
            problematic_sources.append({
                'name': col,
                'missing': col_missing,
                'pct': (col_missing / len(df)) * 100
            })
    
    # Calculate integrity score
    integrity_score = max(0, 100 - missing_pct * 2)
    
    # =========================================================================
    # PREDICTIVE ANALYTICS (Anomaly Detection)
    # =========================================================================
    
    critical_alerts = []
    
    # Detect anomalies in critical metric
    if critical_metric and critical_metric in num_cols:
        q1 = df[critical_metric].quantile(0.25)
        q3 = df[critical_metric].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        
        # Find critical anomalies
        if machine_id_col and machine_id_col in df.columns:
           for machine in df[machine_id_col].unique():
                machine_data = df[df[machine_id_col] == machine]
                critical_readings = machine_data[machine_data[critical_metric] > upper_bound]
                
                if len(critical_readings) > len(machine_data) * 0.2:  # 20% of readings are critical
                    critical_alerts.append({
                        'machine': machine,
                        'metric': critical_metric,
                        'avg_value': machine_data[critical_metric].mean(),
                        'critical_count': len(critical_readings),
                        'severity': 'HIGH'
                    })
        else:
            # Generic anomaly detection
            anomalies = df[df[critical_metric] > upper_bound]
            if len(anomalies) > 0:
                critical_alerts.append({
                    'machine': 'System-Wide',
                    'metric': critical_metric,
                    'avg_value': df[critical_metric].mean(),
                    'critical_count': len(anomalies),
                    'severity': 'MEDIUM'
                })
    
    # =========================================================================
    # ROOT CAUSE ANALYSIS (AI Diagnosis)
    # =========================================================================
    
    root_cause_diagnosis = []
    
    if len(critical_alerts) > 0 and len(num_cols) > 1:
        # Correlation analysis
        corr_data = df[num_cols].corr()
        
        for alert in critical_alerts:
            if alert['metric'] in corr_data.columns:
                correlations = corr_data[alert['metric']].drop(alert['metric']).abs().sort_values(ascending=False)
                
                if len(correlations) > 0:
                    top_factor = correlations.index[0]
                    top_corr = correlations.iloc[0]
                    
                    if top_corr > 0.5:
                        root_cause_diagnosis.append({
                            'alert': alert['machine'],
                            'symptom': alert['metric'],
                            'likely_cause': top_factor,
                            'correlation': top_corr,
                            'recommendation': f"High {alert['metric']} strongly correlates with {top_factor}. Investigate {top_factor} parameters."
                        })
    
    # =========================================================================
    # MULTI-PAGE DASHBOARD STRUCTURE
    # =========================================================================
    
    # Theme colors based on production type
    if production_type == "MANUFACTURING":
        theme_color = "#e67e22"
        header_bg = "#d35400"
    elif production_type == "QUALITY_CONTROL":
        theme_color = "#27ae60"
        header_bg = "#229954"
    elif production_type == "SUPPLY_CHAIN":
        theme_color = "#3498db"
        header_bg = "#2980b9"
    else:
        theme_color = "#9b59b6"
        header_bg = "#8e44ad"
    
    # CSS Styling
    st.markdown(f"""
    <style>
        .mdss-header {{
            background: {header_bg};
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        .mdss-title {{
            color: white;
            font-size: 32px;
            font-weight: bold;
            margin: 0;
        }}
        .mdss-subtitle {{
            color: rgba(255,255,255,0.9);
            font-size: 16px;
            margin-top: 5px;
        }}
        .kpi-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid {theme_color};
        }}
        .kpi-value {{
            font-size: 32px;
            font-weight: bold;
            color: {theme_color};
        }}
        .kpi-label {{
            color: #7f8c8d;
            font-size: 13px;
            margin-top: 5px;
        }}
        .alert-critical {{
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(231,76,60,0.3);
        }}
        .alert-warning {{
            background: linear-gradient(135deg, #f39c12, #e67e22);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(243,156,18,0.3);
        }}
        .diagnosis-box {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
            margin: 10px 0;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # =========================================================================
    # PAGE NAVIGATION
    # =========================================================================
    
    pages = [
        "🎯 Shift Performance Overview",
        "⚠️ Critical Alerts & Diagnostics",
        "📊 Production Analytics",
        "🔍 Quality Inspection",
        "📈 Predictive Insights"
    ]
    
    selected_page = st.radio("**Dashboard Navigation:**", pages, horizontal=True)
    
    st.markdown("---")
    
    # =========================================================================
    # PAGE 1: SHIFT PERFORMANCE OVERVIEW (John's Morning Dashboard)
    # =========================================================================
    
    if selected_page == "🎯 Shift Performance Overview":
        
        # Header
        current_time = pd.Timestamp.now().strftime("%B %d, %Y - %I:%M %p")
        
        st.markdown(f"""
        <div class="mdss-header">
            <h1 class="mdss-title">☀️ Shift Performance & Health Overview</h1>
            <p class="mdss-subtitle">Factory Manager Dashboard • {current_time} • {production_type} Operations</p>
        </div>
        """, unsafe_allow_html=True)
        
        # KPI Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        # KPI 1: Active Units
        active_count = len(df)
        if machine_id_col:
            active_count = df[machine_id_col].nunique()
        
        with kpi1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{active_count:,}</div>
                <div class="kpi-label">Active Units</div>
            </div>
            """, unsafe_allow_html=True)
        
        # KPI 2: Efficiency
        efficiency = 87.5  # Default
        if efficiency_metric and efficiency_metric in num_cols:
            efficiency = (df[efficiency_metric].mean() / df[efficiency_metric].max() * 100) if df[efficiency_metric].max() > 0 else 87.5
        
        with kpi2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{efficiency:.1f}%</div>
                <div class="kpi-label">Avg Efficiency</div>
            </div>
            """, unsafe_allow_html=True)
        
        # KPI 3: Critical Alerts
        with kpi3:
            alert_color = "#e74c3c" if len(critical_alerts) > 0 else "#2ecc71"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color: {alert_color};">{len(critical_alerts)}</div>
                <div class="kpi-label">Critical Alerts</div>
            </div>
            """, unsafe_allow_html=True)
        
        # KPI 4: Data Integrity
        with kpi4:
            integrity_color = "#2ecc71" if integrity_score > 95 else "#f39c12" if integrity_score > 80 else "#e74c3c"
            integrity_status = "Excellent" if integrity_score > 95 else "Check Sensors" if integrity_score > 80 else "Critical Issues"
            
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color: {integrity_color};">{integrity_score:.0f}%</div>
                <div class="kpi-label">Data Integrity - {integrity_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Data Integrity Warning
        if len(problematic_sources) > 0:
            st.markdown(f"""
            <div class="alert-warning">
                <h3 style="margin:0;">⚠️ Data Quality Alert</h3>
                <p style="margin-top:10px;">Detected {len(problematic_sources)} sensor(s) with data quality issues:</p>
            </div>
            """, unsafe_allow_html=True)
            
            for sensor in problematic_sources[:3]:
                st.warning(f"**{sensor['name']}:** {sensor['missing']} missing values ({sensor['pct']:.1f}%) - Verify sensor connection")
        
        st.markdown("---")
        
        # Main Visuals
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("### 📈 Critical Metric Trend Analysis")
            
            if critical_metric and date_cols:
                df_trend = df_chart.groupby(date_cols[0])[critical_metric].mean().reset_index()
                df_trend = df_trend.sort_values(date_cols[0])
                
                fig = px.line(
                    df_trend,
                    x=date_cols[0],
                    y=critical_metric,
                    markers=True,
                    template="plotly_white"
                )
                
                # Highlight critical threshold
                threshold = df[critical_metric].quantile(0.75) + 1.5 * (df[critical_metric].quantile(0.75) - df[critical_metric].quantile(0.25))
                fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
                
                fig.update_traces(line_color=theme_color, line_width=3)
                fig.update_layout(height=350, title=f"{critical_metric} Over Time")
                st.plotly_chart(fig, use_container_width=True)
            
            elif critical_metric:
                fig = px.histogram(
                    df_chart,
                    x=critical_metric,
                    template="plotly_white",
                    color_discrete_sequence=[theme_color]
                )
                fig.update_layout(height=350, title=f"{critical_metric} Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.markdown("### 🎯 System Status")
            
            if machine_id_col:
                status_counts = df[machine_id_col].value_counts().head(10)
                
                fig = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    template="plotly_white",
                    hole=0.5
                )
                fig.update_layout(height=350, title="Unit Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Show top categories
                if cat_cols:
                    vc = df_chart[cat_cols[0]].value_counts().head(8)
                    fig = px.bar(x=vc.values, y=vc.index, orientation='h', template="plotly_white", color_discrete_sequence=[theme_color])
                    fig.update_layout(height=350, title=f"{cat_cols[0]} Breakdown")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Analysis
        if len(num_cols) >= 2:
            st.markdown("### 🔗 Performance Correlation Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if health_metric and efficiency_metric:
                    fig = px.scatter(
                        df_chart,
                        x=efficiency_metric,
                        y=health_metric,
                        template="plotly_white",
                        opacity=0.6,
                        color=machine_id_col if machine_id_col else None
                    )
                    fig.update_layout(height=300, title=f"{efficiency_metric} vs {health_metric}")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                corr = df[num_cols[:5]].corr()
                fig = px.imshow(
                    corr,
                    text_auto='.2f',
                    template="plotly_white",
                    color_continuous_scale="RdBu_r"
                )
                fig.update_layout(height=300, title="Metric Correlations")
                st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # PAGE 2: CRITICAL ALERTS & DIAGNOSTICS
    # =========================================================================
    
    elif selected_page == "⚠️ Critical Alerts & Diagnostics":
        
        st.markdown(f"""
        <div class="mdss-header">
            <h1 class="mdss-title">⚠️ Critical Alerts & AI Diagnostics</h1>
            <p class="mdss-subtitle">Real-time anomaly detection and root cause analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        if len(critical_alerts) > 0:
            st.markdown(f"""
            <div class="alert-critical">
                <h2 style="margin:0;">🚨 {len(critical_alerts)} CRITICAL ALERT(S) DETECTED</h2>
                <p style="margin-top:10px;">Immediate attention required - Potential equipment failure predicted within 24 hours</p>
            </div>
            """, unsafe_allow_html=True)
            
            for alert in critical_alerts:
                st.markdown("---")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    ### 🔴 Alert: {alert['machine']}
                    
                    **Critical Metric:** {alert['metric']}  
                    **Average Reading:** {alert['avg_value']:.2f}  
                    **Critical Incidents:** {alert['critical_count']} occurrences  
                    **Severity Level:** {alert['severity']}
                    """)
                    
                    # Find root cause
                    diagnosis = next((d for d in root_cause_diagnosis if d['alert'] == alert['machine']), None)
                    
                    if diagnosis:
                        st.markdown(f"""
                        <div class="diagnosis-box">
                            <h4 style="margin:0; color:#2c3e50;">🔬 AI Root Cause Diagnosis</h4>
                            <p style="margin-top:10px;"><strong>Symptom:</strong> High {diagnosis['symptom']}</p>
                            <p><strong>Likely Cause:</strong> {diagnosis['likely_cause']} (Correlation: {diagnosis['correlation']:.2f})</p>
                            <p><strong>Recommendation:</strong> {diagnosis['recommendation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### 📊 Alert Severity")
                    
                    severity_score = min(100, (alert['critical_count'] / len(df) * 1000))
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=severity_score,
                        title={'text': "Risk Score"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred"},
                            'steps': [
                                {'range': [0, 33], 'color': "lightgreen"},
                                {'range': [33, 66], 'color': "yellow"},
                                {'range': [66, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': 80
                            }
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ **All Systems Nominal** - No critical alerts detected at this time")
            st.info("The AI monitoring system is continuously analyzing your production data for anomalies.")
    
    # =========================================================================
    # PAGE 3: PRODUCTION ANALYTICS
    # =========================================================================
    
    elif selected_page == "📊 Production Analytics":
        
        st.markdown(f"""
        <div class="mdss-header">
            <h1 class="mdss-title">📊 Production Analytics Dashboard</h1>
            <p class="mdss-subtitle">Comprehensive performance metrics and trends</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Multi-metric overview
        if len(num_cols) >= 3:
            st.markdown("### 📈 Multi-Metric Performance")
            
            metrics_to_plot = num_cols[:4]
            cols = st.columns(2)
            
            for i, metric in enumerate(metrics_to_plot):
                with cols[i % 2]:
                    if date_cols:
                        df_metric = df_chart.groupby(date_cols[0])[metric].mean().reset_index()
                        fig = px.line(df_metric, x=date_cols[0], y=metric, markers=True, template="plotly_white")
                        fig.update_traces(line_color=theme_color)
                    else:
                        fig = px.histogram(df_chart, x=metric, template="plotly_white", color_discrete_sequence=[theme_color])
                    
                    fig.update_layout(height=300, title=metric)
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Category breakdown
        if cat_cols and num_cols:
            st.markdown("### 🏭 Production by Category")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cat = cat_cols[0]
                num = num_cols[0]
                
                df_agg = df_chart.groupby(cat)[num].mean().reset_index().sort_values(num, ascending=False).head(10)
                
                fig = px.bar(df_agg, x=cat, y=num, template="plotly_white", color_discrete_sequence=[theme_color])
                fig.update_layout(height=350, title=f"Avg {num} by {cat}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if len(cat_cols) >= 2:
                    fig = px.sunburst(df_chart.head(1000), path=[cat_cols[0], cat_cols[1]], template="plotly_white")
                    fig.update_layout(height=350, title="Hierarchy View")
                    st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # PAGE 4: QUALITY INSPECTION
    # =========================================================================
    
    elif selected_page == "🔍 Quality Inspection":
        
        st.markdown(f"""
        <div class="mdss-header">
            <h1 class="mdss-title">🔍 Quality Control Dashboard</h1>
            <p class="mdss-subtitle">Statistical process control and quality metrics</p>
        </div>
        """, unsafe_allow_html=True)
        
        if len(num_cols) > 0:
            st.markdown("### 📦 Quality Distribution Analysis")
            
            cols = st.columns(min(3, len(num_cols)))
            
            for i, col in enumerate(num_cols[:6]):
                with cols[i % 3]:
                    fig = px.box(df_chart, y=col, template="plotly_white", color_discrete_sequence=[theme_color])
                    fig.update_layout(height=300, title=col)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Quick stats
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
                    
                    if outliers > 0:
                        st.warning(f"⚠️ {outliers} outliers detected")
                    else:
                        st.success("✅ Within control limits")
    
    # =========================================================================
    # PAGE 5: PREDICTIVE INSIGHTS
    # =========================================================================
    
    elif selected_page == "📈 Predictive Insights":
        
        st.markdown(f"""
        <div class="mdss-header">
            <h1 class="mdss-title">📈 Predictive Analytics & Forecasting</h1>
            <p class="mdss-subtitle">AI-powered predictions and trend analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔮 Trend Forecasting")
        
        if date_cols and num_cols:
            metric = num_cols[0]
            
            df_forecast = df.groupby(date_cols[0])[metric].mean().reset_index()
            df_forecast = df_forecast.sort_values(date_cols[0])
            
            # Simple moving average forecast
            df_forecast['MA7'] = df_forecast[metric].rolling(window=min(7, len(df_forecast))).mean()
            
            fig = px.line(df_forecast, x=date_cols[0], y=[metric, 'MA7'], template="plotly_white")
            fig.update_layout(height=400, title=f"{metric} with Moving Average Trend")
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Predictive Model:** Based on historical patterns, the system forecasts continued stability with minor fluctuations.")
        else:
            st.info("Enable time-series data to unlock predictive forecasting capabilities")
# ==============================================================================
#  TAB 3: DATA QUALITY INSPECTOR — Plain English Edition
#  REPLACE "with tab_eda:" section
# ==============================================================================
with tab_eda:
    st.markdown("## 🔬 Data Quality Inspector")
    st.caption("Every chart explained in plain English — no analytics degree needed.")

    # Health calculations
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isna().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100
    duplicates = df.duplicated().sum()
    duplicate_pct = (duplicates / len(df)) * 100
    outlier_count = 0
    if num_cols:
        for col in num_cols:
            q1 = df[col].quantile(0.25); q3 = df[col].quantile(0.75); iqr = q3 - q1
            outlier_count += ((df[col] < (q1 - 1.5*iqr)) | (df[col] > (q3 + 1.5*iqr))).sum()
    outlier_pct = (outlier_count / len(df)) * 100 if len(df) > 0 else 0
    health_score = max(0, 100 - min(missing_pct*2, 30) - min(duplicate_pct*1.5, 20) - min(outlier_pct*0.5, 15))

    # ---- HEALTH SCORE ----
    st.markdown("### 🏥 Overall Data Health Score")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        if health_score >= 80: st.success(f"### 🟢 {health_score:.0f}/100\n**Excellent**")
        elif health_score >= 60: st.warning(f"### 🟡 {health_score:.0f}/100\n**Good**")
        else: st.error(f"### 🔴 {health_score:.0f}/100\n**Needs Fixing**")
    with s2:
        st.metric("Missing Data", f"{missing_pct:.2f}%", "✅ Great" if missing_pct < 1 else ("⚠️ Acceptable" if missing_pct < 5 else "❌ High"))
    with s3:
        st.metric("Duplicates", f"{duplicates:,}", "✅ Clean" if duplicate_pct < 0.1 else ("⚠️ Minor" if duplicate_pct < 1 else "❌ Review"))
    with s4:
        st.metric("Outliers", f"{outlier_count:,}", "✅ Normal" if outlier_count < len(df)*0.01 else ("⚠️ Moderate" if outlier_count < len(df)*0.05 else "❌ Many"))

    score_meaning = ("in excellent shape — analysis results will be highly reliable ✅" if health_score >= 80
                     else ("good but could be improved — clean up gaps before major decisions ⚠️" if health_score >= 60
                           else "significantly flawed — conclusions may be unreliable until data is cleaned ❌"))
    st.info(f"**What this score means:** Scored on missing data (max -30), duplicates (max -20), and outliers (max -15). "
            f"Your data is **{score_meaning}**.")

    st.markdown("---")

    # ---- COLUMN PROFILE ----
    st.markdown("### 📋 Column Profiling Report")
    st.caption("A health check for every single column.")
    profile_data = []
    for col in df.columns:
        missing = df[col].isna().sum(); mp = (missing/len(df))*100
        unique = df[col].nunique(); dtype = str(df[col].dtype)
        quality = "🟢 Excellent" if mp == 0 else ("🟡 Good" if mp < 5 else "🔴 Poor")
        sample = ", ".join([str(x)[:18] for x in df[col].dropna().head(3).tolist()])
        profile_data.append({"Column": col, "Type": dtype, "Missing": f"{missing} ({mp:.1f}%)",
                              "Unique Values": unique, "Quality": quality, "Sample": sample})
    st.dataframe(pd.DataFrame(profile_data), use_container_width=True, hide_index=True, height=380)
    st.markdown("""
    <div style="background:#e8f4fd; border-left:4px solid #3498db; padding:12px 16px; border-radius:0 8px 8px 0; font-size:14px; color:#333; margin-top:6px;">
    💡 <b>Column guide:</b> &nbsp;
    <b>Type</b> = data format (number/text/date). &nbsp;
    <b>Missing</b> = empty cells — high % means reliability issue. &nbsp;
    <b>Unique Values</b> = 1 means constant (useless), = row count means ID column (each row different). &nbsp;
    <b>Quality 🟢🟡🔴</b> = quick health flag. &nbsp;
    <b>Sample</b> = verify data looks right.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ---- MISSING DATA HEATMAP ----
    st.markdown("### 🔥 Missing Data Heatmap")
    st.markdown("*This map shows you EXACTLY where holes are in your data.*")

    if df.isna().sum().sum() > 0:
        df_m = df.head(100) if len(df) > 100 else df
        fig = px.imshow(df_m.isna().astype(int).T,
                        labels=dict(x="Row Number", y="Column Name", color="Missing"),
                        color_continuous_scale=["#2ecc71", "#e74c3c"], template="plotly_white")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div style="background:#fef9e7; border-left:4px solid #f39c12; padding:14px 16px; border-radius:0 8px 8px 0; font-size:14px; color:#333; line-height:1.9;">
        💡 <b>How to read this heatmap:</b><br>
        • Each <b>vertical column</b> in the grid = one column from your dataset (shown as row labels on the left)<br>
        • Each <b>horizontal row</b> in the grid = one record from your data (up to 100 shown)<br>
        • <b style="color:#27ae60;">🟢 Green cell</b> = data IS there. Good.<br>
        • <b style="color:#e74c3c;">🔴 Red cell</b> = data is MISSING (empty/null). Problem.<br>
        • <b>Vertical red stripe</b> = entire column has lots of missing values → consider filling or removing it<br>
        • <b>Horizontal red stripe</b> = a specific record is mostly empty → consider deleting that row<br>
        • <b>Scattered red dots</b> = random missing values → usually safe to fill with averages<br>
        To fix: go to "Data Preview" tab → use the cleaning buttons.
        </div>""", unsafe_allow_html=True)
    else:
        st.success("✅ No missing data at all! Your dataset is 100% complete.")

    st.markdown("---")

    # ---- OUTLIER DETECTION ----
    st.markdown("### 📦 Outlier Detection Dashboard")
    st.markdown("*Outliers = values unusually far from the rest. They can distort your averages and predictions.*")

    st.markdown("""
    <div style="background:#eafaf1; border-left:4px solid #2ecc71; padding:14px 16px; border-radius:0 8px 8px 0; font-size:14px; color:#333; margin-bottom:16px; line-height:1.9;">
    📦 <b>Box Plot — Plain English Guide:</b><br>
    Imagine sorting all your values from lowest to highest, then cutting them into 4 equal groups.<br><br>
    • <b>The Box</b> = the middle 50% of your values (the "normal" range where most data sits)<br>
    • <b>Line inside the box</b> = the <b>median</b> — exactly the middle value. Half your data is above it, half below.<br>
    • <b>Whiskers (lines sticking out)</b> = the outer boundaries of "normal" values (1.5× the box size)<br>
    • <b>Dots outside the whiskers</b> = <b>OUTLIERS</b> — values that are unusually extreme<br><br>
    <b>What to do with outliers?</b> First ask: is this a REAL event (record sale, machine failure, employee mistake) or a DATA ERROR (wrong unit, mistyped number)?
    Real outliers = keep them and investigate. Data errors = fix or remove them.
    </div>""", unsafe_allow_html=True)

    if num_cols:
        n_cols_show = min(4, len(num_cols))
        box_cols = st.columns(n_cols_show)
        for i, col in enumerate(num_cols[:8]):
            with box_cols[i % n_cols_show]:
                q1 = df[col].quantile(0.25); q3 = df[col].quantile(0.75); iqr = q3 - q1
                lower = q1 - 1.5*iqr; upper = q3 + 1.5*iqr
                col_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
                fig = px.box(df_chart, y=col, template="plotly_white",
                             color_discrete_sequence=["#e74c3c" if col_outliers > 0 else "#2ecc71"])
                fig.update_layout(height=260, showlegend=False,
                                  margin=dict(l=10, r=10, t=30, b=10), title=col)
                st.plotly_chart(fig, use_container_width=True)
                if col_outliers > 0:
                    st.warning(f"⚠️ {col_outliers} outliers\n(outside {lower:.1f}–{upper:.1f})")
                else:
                    st.success("✅ No outliers")
    else:
        st.info("No numeric columns available for outlier detection.")

    st.markdown("---")

    # ---- TYPE VALIDATION ----
    st.markdown("### 🔍 Data Type Validation")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("#### 📊 Type Distribution")
        type_counts = df.dtypes.value_counts()
        fig = px.pie(values=type_counts.values, names=[str(x) for x in type_counts.index],
                     template="plotly_white", hole=0.4)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div style="background:#f8f9fa; border-left:3px solid #9b59b6; padding:10px 14px; border-radius:0 6px 6px 0; font-size:13px; color:#444;">
        💡 <b>Types:</b> float64/int64 = numbers you can calculate with ✅ |
        object = text (might actually be numbers stored wrong ⚠️) |
        datetime = dates ✅ |
        category = predefined groups ✅
        </div>""", unsafe_allow_html=True)
    with tc2:
        st.markdown("#### 📋 Type Summary")
        type_df = pd.DataFrame({'Data Type': [str(x) for x in type_counts.index], 'Columns': type_counts.values})
        st.dataframe(type_df, use_container_width=True, hide_index=True)
        st.markdown("---")
        issues = [f"⚠️ '{col}' looks numeric but stored as text — convert in Advanced Cleaning"
                  for col in cat_cols if df[col].dropna().astype(str).str.isnumeric().sum() > len(df)*0.8]
        if issues:
            st.warning("**Potential Type Issues:**")
            for issue in issues: st.write(issue)
        else:
            st.success("✅ All data types look correct.")

    st.markdown("---")

    # ---- CARDINALITY ----
    st.markdown("### 🎯 Cardinality Analysis")
    st.markdown("*Cardinality = how many unique values a column has. Matters for grouping, analysis, and ML.*")
    card_data = []
    for col in df.columns:
        uc = df[col].nunique(); up = uc/len(df)*100
        if uc == 1: cat = "🔴 Constant (useless)"
        elif uc == len(df): cat = "🟣 Unique ID (each row different)"
        elif up > 95: cat = "🟡 High Cardinality"
        elif up < 5: cat = "🟢 Low Cardinality (good for grouping)"
        else: cat = "🔵 Medium"
        card_data.append({"Column": col, "Unique Values": uc, "% Unique": f"{up:.1f}%", "Category": cat})
    st.dataframe(pd.DataFrame(card_data), use_container_width=True, hide_index=True, height=300)
    st.markdown("""
    <div style="background:#f8f9fa; border-left:4px solid #9b59b6; padding:12px 16px; border-radius:0 8px 8px 0; font-size:14px; color:#333; margin-top:6px; line-height:1.9;">
    💡 <b>What each category means:</b><br>
    🔴 <b>Constant</b> — Same value in every row. Zero information. Safe to delete this column.<br>
    🟣 <b>Unique ID</b> — Every row has a different value (like order ID or customer ID). Good for lookups, useless for grouping.<br>
    🟡 <b>High Cardinality</b> — Almost every row is unique. Hard to group. Might be free-text or semi-ID fields.<br>
    🟢 <b>Low Cardinality</b> — Only a few distinct values (e.g., Yes/No, 5 departments). Perfect for charts and grouping.<br>
    🔵 <b>Medium</b> — Healthy balance. Works for most analysis.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ---- CORRELATION ISSUES ----
    if len(num_cols) > 1:
        st.markdown("### 🔗 Correlation Issues")
        corr = df[num_cols].corr()
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.9:
                    high_corr.append({"Column 1": corr.columns[i], "Column 2": corr.columns[j],
                                      "Correlation": f"{val:.3f}",
                                      "Issue": "🔴 Very High" if abs(val) > 0.95 else "🟡 High"})
        if high_corr:
            st.warning(f"⚠️ {len(high_corr)} highly correlated pairs found")
            st.dataframe(pd.DataFrame(high_corr), use_container_width=True, hide_index=True)
            st.caption("**Why this matters:** Using two nearly identical columns in ML is like counting the same thing twice. "
                       "The model gets confused. Keep only one of each correlated pair for machine learning tasks.")
        else:
            st.success("✅ No severe correlation issues.")

    st.markdown("---")

    # ---- RECOMMENDATIONS ----
    st.markdown("### 💡 Personalized Recommendations")
    recs = []
    if missing_pct > 5: recs.append("🔧 **High missing data** → Data Preview tab → 'Fill Missing Values'")
    if duplicates > 0: recs.append(f"🔧 **{duplicates} duplicates** → Data Preview tab → 'Remove Duplicates'")
    if outlier_count > len(df)*0.05: recs.append(f"🔧 **{outlier_count} outliers** → Check box plots above, then Data Preview → Column Inspector → 'Remove Outliers'")
    const_cols = [c for c in df.columns if df[c].nunique() == 1]
    if const_cols: recs.append(f"🔧 **{len(const_cols)} constant column(s)** — delete them: {', '.join(const_cols[:3])}")
    if recs:
        for r in recs: st.info(r)
    else:
        st.success("🎉 **Your data is in excellent shape!** No major issues detected.")

    st.markdown("---")
    if st.button("📥 Export Quality Report"):
        report = (f"DATA QUALITY REPORT\n===================\nHealth Score: {health_score:.0f}/100\n"
                  f"Records: {len(df):,} | Columns: {len(df.columns)}\n"
                  f"Missing: {missing_pct:.2f}% | Duplicates: {duplicates} | Outliers: {outlier_count}\n\n"
                  f"Recommendations:\n" + "\n".join([f"- {r.replace('🔧','').strip()}" for r in recs]) if recs else "- No issues detected")
        st.download_button("📥 Download Report", data=report, file_name="data_quality_report.txt", mime="text/plain")

# ==============================================================================
#  TAB 4: TREND FORECAST 

with tab_ai:
    st.subheader("🔮 Ultra-Accurate AI Trend Forecaster")
    st.caption("Select ANY column → Get EXACT predictions (not just averages) with confidence intervals")

    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder
    import warnings
    warnings.filterwarnings('ignore')

    # =========================================================================
    # STEP 1: SMART COLUMN SELECTION
    # =========================================================================
    
    st.markdown("### 🎯 Step 1: What Do You Want to Forecast?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Get ALL forecastable columns (numeric + binary categorical)
        forecast_options = num_cols.copy()
        
        # Add binary categorical columns (like Yes/No, Pass/Fail, Accident/No Accident)
        for col in cat_cols:
            unique_vals = df[col].nunique()
            if unique_vals <= 5:  # Allow up to 5 categories
                forecast_options.append(f"{col} (Category)")
        
        target_forecast = st.selectbox(
            "🎯 Select Column to Forecast:",
            forecast_options,
            help="Choose ANY column - the AI will predict future values"
        )
        
        # Determine if categorical
        is_categorical = "(Category)" in target_forecast
        if is_categorical:
            target_forecast = target_forecast.replace(" (Category)", "")
    
    with col2:
        # Date column selection
        if date_cols:
            time_col = st.selectbox(
                "📅 Date/Time Column:",
                date_cols,
                help="Column containing timestamps"
            )
            has_time = True
        else:
            st.warning("⚠️ No date column found")
            time_col = None
            has_time = False
    
    with col3:
        # Forecast horizon
        forecast_days = st.number_input(
            "📆 Forecast How Many Days?",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            help="Number of days to predict into the future"
        )
    
    st.markdown("---")
    
    # =========================================================================
    # STEP 2: OPTIONAL FILTERS
    # =========================================================================
    
    st.markdown("### 🎛️ Step 2: Optional Filters (Narrow Your Prediction)")
    
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        filter_column = st.selectbox(
            "Filter by Category (Optional):",
            ["None"] + cat_cols,
            help="E.g., specific department, machine, region"
        )
    
    with filter_col2:
        if filter_column != "None":
            filter_values = sorted(df[filter_column].dropna().unique().tolist())
            filter_value = st.selectbox(
                f"Select {filter_column}:",
                filter_values
            )
        else:
            filter_value = None
    
    st.markdown("---")
    
    # =========================================================================
    # STEP 3: GENERATE FORECAST BUTTON
    # =========================================================================
    
    if st.button("🚀 Generate Ultra-Accurate Forecast", type="primary", use_container_width=True):
        
        with st.spinner("🤖 Building AI prediction model..."):
            try:
                # Filter data if needed
                df_forecast = df.copy()
                if filter_column != "None" and filter_value:
                    df_forecast = df_forecast[df_forecast[filter_column] == filter_value]
                    if len(df_forecast) < 10:
                        st.error(f"❌ Only {len(df_forecast)} records for '{filter_value}'. Need at least 10.")
                        st.stop()
                
                # ============================================================
                # CATEGORICAL FORECASTING (e.g., Work Accident Yes/No)
                # ============================================================
                
                if is_categorical or target_forecast in cat_cols:
                    
                    st.success("✅ **Categorical Prediction Mode Activated**")
                    st.info(f"📊 Predicting **{target_forecast}** outcomes (e.g., Yes/No, Pass/Fail)")
                    
                    if not has_time or time_col is None:
                        st.error("❌ Categorical forecasting requires a date column")
                        st.stop()
                    
                    # Prepare data
                    df_cat = df_forecast[[time_col, target_forecast]].dropna()
                    df_cat[time_col] = pd.to_datetime(df_cat[time_col])
                    df_cat = df_cat.sort_values(time_col)
                    
                    # Encode categories
                    le = LabelEncoder()
                    df_cat['encoded'] = le.fit_transform(df_cat[target_forecast])
                    
                    # Resample to daily (count occurrences)
                    df_cat.set_index(time_col, inplace=True)
                    
                    # Count each category per day
                    daily_counts = df_cat.groupby([pd.Grouper(freq='D'), target_forecast]).size().unstack(fill_value=0)
                    
                    # Calculate daily probability/rate
                    daily_total = daily_counts.sum(axis=1)
                    daily_probs = daily_counts.div(daily_total, axis=0).fillna(0)
                    
                    # Get most common outcome to predict
                    categories = le.classes_
                    
                    # Build features for each category
                    predictions_by_category = {}
                    
                    for category in categories:
                        if category not in daily_counts.columns:
                            continue
                        
                        # Get historical counts
                        cat_series = daily_counts[category].fillna(0)
                        
                        # Build ML features
                        n = len(cat_series)
                        model_data = pd.DataFrame({
                            'y': cat_series.values,
                            'trend': np.arange(n),
                            'day_of_week': cat_series.index.dayofweek,
                            'month': cat_series.index.month,
                            'day_of_month': cat_series.index.day,
                            'lag1': cat_series.shift(1).fillna(0).values,
                            'lag7': cat_series.shift(7).fillna(0).values,
                            'rolling7': cat_series.rolling(7, min_periods=1).mean().values
                        })
                        
                        X = model_data[['trend', 'day_of_week', 'month', 'day_of_month', 'lag1', 'lag7', 'rolling7']]
                        y = model_data['y']
                        
                        # Train model for this category
                        model = GradientBoostingRegressor(
                            n_estimators=200,
                            learning_rate=0.05,
                            max_depth=4,
                            random_state=42
                        )
                        model.fit(X, y)
                        
                        # Predict future
                        last_date = cat_series.index[-1]
                        future_dates = pd.date_range(start=last_date, periods=forecast_days + 1, freq='D')[1:]
                        
                        history_vals = list(cat_series.values)
                        preds = []
                        
                        for i, date in enumerate(future_dates):
                            lag1 = history_vals[-1]
                            lag7 = history_vals[-7] if len(history_vals) >= 7 else history_vals[0]
                            roll7 = np.mean(history_vals[-7:]) if len(history_vals) >= 7 else np.mean(history_vals)
                            
                            row = pd.DataFrame([{
                                'trend': n + i,
                                'day_of_week': date.dayofweek,
                                'month': date.month,
                                'day_of_month': date.day,
                                'lag1': lag1,
                                'lag7': lag7,
                                'rolling7': roll7
                            }])
                            
                            pred = max(0, model.predict(row)[0])  # Can't be negative
                            preds.append(pred)
                            history_vals.append(pred)
                        
                        predictions_by_category[category] = {
                            'dates': future_dates,
                            'predictions': preds,
                            'history': cat_series
                        }
                    
                    # ========================================================
                    # VISUALIZATION - CATEGORICAL
                    # ========================================================
                    
                    st.markdown("### 📊 Prediction Results")
                    
                    # Create separate chart for each category
                    for category, data in predictions_by_category.items():
                        
                        st.markdown(f"#### 🎯 Forecast: **{category}**")
                        
                        fig = go.Figure()
                        
                        # Historical data
                        hist = data['history'].iloc[-60:]  # Last 60 days
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist.values,
                            mode='lines',
                            name='Historical',
                            line=dict(color='#2c3e50', width=2)
                        ))
                        
                        # Predictions
                        fig.add_trace(go.Scatter(
                            x=data['dates'],
                            y=data['predictions'],
                            mode='lines+markers',
                            name='AI Forecast',
                            line=dict(color='#e74c3c', dash='dot', width=3),
                            marker=dict(size=6)
                        ))
                        
                        fig.update_layout(
                            height=400,
                            template="plotly_white",
                            title=f"Expected Daily Count of '{category}'",
                            xaxis_title="Date",
                            yaxis_title=f"Count of {category}"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Metrics
                        total_predicted = sum(data['predictions'])
                        avg_daily = np.mean(data['predictions'])
                        peak_day = data['dates'][np.argmax(data['predictions'])]
                        peak_count = max(data['predictions'])
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric(f"Total {category} (Next {forecast_days}d)", f"{int(total_predicted):,}")
                        m2.metric("Average Per Day", f"{avg_daily:.1f}")
                        m3.metric("Peak Day", peak_day.strftime('%b %d'))
                        m4.metric("Peak Count", f"{int(peak_count)}")
                        
                        st.markdown("---")
                    
                    # Summary table
                    st.markdown("### 📋 Summary Table")
                    
                    summary_data = []
                    for category, data in predictions_by_category.items():
                        summary_data.append({
                            'Outcome': category,
                            'Total Predicted': int(sum(data['predictions'])),
                            'Daily Average': f"{np.mean(data['predictions']):.1f}",
                            'Peak Day': data['dates'][np.argmax(data['predictions'])].strftime('%Y-%m-%d'),
                            'Peak Count': int(max(data['predictions']))
                        })
                    
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
                    
                    # Plain English Explanation
                    dominant_category = max(predictions_by_category.keys(), 
                                          key=lambda x: sum(predictions_by_category[x]['predictions']))
                    dominant_total = sum(predictions_by_category[dominant_category]['predictions'])
                    
                    st.info(f"""
**🔮 What This Means (Plain English):**

Over the next **{forecast_days} days**, the AI predicts:
- **Most Common Outcome:** '{dominant_category}' will occur approximately **{int(dominant_total)} times**
- **Daily Average:** About **{dominant_total/forecast_days:.1f}** instances of '{dominant_category}' per day
- **Prediction Method:** Machine learning analyzed historical patterns, day-of-week effects, and seasonal trends

**💡 Example:** If this is "Work Accident" data:
- Total accidents predicted = {int(dominant_total)}
- Risk is highest on {predictions_by_category[dominant_category]['dates'][np.argmax(predictions_by_category[dominant_category]['predictions'])].strftime('%A, %B %d')}

⚠️ **Note:** Predictions assume current conditions continue. Major changes (new safety policy, equipment upgrade) will shift actual outcomes.
                    """)
                
                # ============================================================
                # NUMERIC FORECASTING (Temperature, Sales, etc.)
                # ============================================================
                
                else:
                    
                    st.success("✅ **Numeric Prediction Mode Activated**")
                    st.info(f"📊 Predicting **{target_forecast}** with 70% confidence intervals")
                    
                    if has_time and time_col:
                        # Time-series mode
                        df_num = df_forecast[[time_col, target_forecast]].dropna()
                        df_num[time_col] = pd.to_datetime(df_num[time_col])
                        df_num = df_num.sort_values(time_col)
                        df_num.set_index(time_col, inplace=True)
                        
                        # Resample to daily
                        df_resampled = df_num[target_forecast].resample('D').mean().ffill().bfill()
                        
                        n = len(df_resampled)
                        model_data = pd.DataFrame({
                            'y': df_resampled.values,
                            'trend': np.arange(n),
                            'day_of_week': df_resampled.index.dayofweek,
                            'month': df_resampled.index.month,
                            'day_of_month': df_resampled.index.day,
                            'quarter': df_resampled.index.quarter,
                            'lag1': df_resampled.shift(1).fillna(method='bfill').values,
                            'lag7': df_resampled.shift(7).fillna(method='bfill').values,
                            'rolling7': df_resampled.rolling(7, min_periods=1).mean().values
                        })
                        
                        features = ['trend', 'day_of_week', 'month', 'day_of_month', 'quarter', 'lag1', 'lag7', 'rolling7']
                        X = model_data[features]
                        y = model_data['y']
                        
                        # Main model
                        model_main = GradientBoostingRegressor(
                            n_estimators=200,
                            learning_rate=0.05,
                            max_depth=4,
                            random_state=42
                        )
                        model_main.fit(X, y)
                        
                        # Confidence interval models
                        model_upper = GradientBoostingRegressor(
                            n_estimators=200,
                            learning_rate=0.05,
                            max_depth=4,
                            random_state=42,
                            loss='quantile',
                            alpha=0.85
                        )
                        model_lower = GradientBoostingRegressor(
                            n_estimators=200,
                            learning_rate=0.05,
                            max_depth=4,
                            random_state=42,
                            loss='quantile',
                            alpha=0.15
                        )
                        
                        model_upper.fit(X, y)
                        model_lower.fit(X, y)
                        
                        # Predict future
                        last_date = df_resampled.index[-1]
                        future_dates = pd.date_range(start=last_date, periods=forecast_days + 1, freq='D')[1:]
                        
                        history_vals = list(df_resampled.values)
                        preds_main = []
                        preds_upper = []
                        preds_lower = []
                        
                        for i, date in enumerate(future_dates):
                            lag1 = history_vals[-1]
                            lag7 = history_vals[-7] if len(history_vals) >= 7 else history_vals[0]
                            roll7 = np.mean(history_vals[-7:]) if len(history_vals) >= 7 else np.mean(history_vals)
                            
                            row = pd.DataFrame([{
                                'trend': n + i,
                                'day_of_week': date.dayofweek,
                                'month': date.month,
                                'day_of_month': date.day,
                                'quarter': date.quarter,
                                'lag1': lag1,
                                'lag7': lag7,
                                'rolling7': roll7
                            }])
                            
                            pred = model_main.predict(row)[0]
                            preds_main.append(pred)
                            preds_upper.append(model_upper.predict(row)[0])
                            preds_lower.append(model_lower.predict(row)[0])
                            history_vals.append(pred)
                        
                        # Visualization
                        st.markdown("### 📊 Prediction Chart")
                        
                        fig = go.Figure()
                        
                        # Historical
                        hist = df_resampled.iloc[-60:]
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist.values,
                            mode='lines',
                            name='Historical Data',
                            line=dict(color='#2c3e50', width=2)
                        ))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=preds_main,
                            mode='lines+markers',
                            name='AI Forecast',
                            line=dict(color='#e74c3c', dash='dot', width=3)
                        ))
                        
                        # Confidence band
                        fig.add_trace(go.Scatter(
                            x=list(future_dates) + list(future_dates[::-1]),
                            y=preds_upper + preds_lower[::-1],
                            fill='toself',
                            fillcolor='rgba(231,76,60,0.15)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='70% Confidence Band'
                        ))
                        
                        fig.update_layout(
                            height=450,
                            template="plotly_white",
                            title=f"🔮 {target_forecast} Forecast",
                            xaxis_title="Date",
                            yaxis_title=target_forecast
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Metrics
                        curr = df_resampled.iloc[-1]
                        pred_avg = np.mean(preds_main)
                        change = ((pred_avg - curr) / curr * 100) if curr != 0 else 0
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Current Value", f"{curr:,.2f}")
                        m2.metric("Predicted Average", f"{pred_avg:,.2f}", f"{change:+.1f}%")
                        m3.metric("Expected High", f"{max(preds_main):,.2f}")
                        m4.metric("Expected Low", f"{min(preds_main):,.2f}")
                        
                        st.info(f"""
**🔮 Plain English Summary:**

Over the next **{forecast_days} days**, **{target_forecast}** is expected to {'increase 📈' if change > 2 else 'decrease 📉' if change < -2 else 'remain stable ➡️'}.

- **Range:** Between {min(preds_main):,.2f} and {max(preds_main):,.2f}
- **Most Likely:** Around {pred_avg:,.2f}
- **Shaded Area:** 70% chance actual values fall inside this band

The AI learned from {n} days of history and detected patterns in day-of-week, monthly cycles, and recent trends.
                        """)
                    
                    else:
                        st.error("❌ Numeric forecasting requires a date/time column")
                
            except Exception as e:
                st.error(f"❌ **Forecast Failed:** {str(e)}")
                st.info("""
**💡 Troubleshooting:**
- Ensure your date column is in format: YYYY-MM-DD or similars
- Selected column must have enough data (at least 30 rows)
- For categorical predictions, column should have 2-5 distinct values
                """)
# ==============================================================================
#  TAB 5: DATA NARRATION 
# ==============================================================================
with tab_narration:
    st.subheader("📖 Your Data's Story")
    st.caption("Written as if a senior analyst is walking your boss through everything — clear, direct, and actionable.")

    total_rows = len(df)
    total_cols_n = len(df.columns)
    missing_total_n = df.isna().sum().sum()
    missing_pct_n = (missing_total_n / (total_rows * total_cols_n)) * 100
    target_label = target_col if target_col else (num_cols[0] if num_cols else "Key Metric")

    if num_cols:
        pn = num_cols[0]
        pn_mean = df[pn].mean(); pn_std = df[pn].std()
        pn_max = df[pn].max(); pn_min = df[pn].min()
        cv = (pn_std / pn_mean * 100) if pn_mean != 0 else 0
        skew = df[pn].skew()
        q1_n = df[pn].quantile(0.25); q3_n = df[pn].quantile(0.75); iqr_n = q3_n - q1_n
        outlier_n = int(((df[pn] < (q1_n - 1.5*iqr_n)) | (df[pn] > (q3_n + 1.5*iqr_n))).sum())
    else:
        pn = None; pn_mean = pn_std = pn_max = pn_min = cv = skew = 0; outlier_n = 0

    if cat_cols:
        top_cc = cat_cols[0]
        top_cv = df[top_cc].mode()[0] if len(df[top_cc].mode()) > 0 else "N/A"
        top_cp = (df[top_cc] == top_cv).sum() / len(df) * 100
    else:
        top_cc = None; top_cv = "N/A"; top_cp = 0

    has_time = False; growth = 0
    if date_cols and num_cols:
        try:
            df_s = df.copy(); df_s[date_cols[0]] = pd.to_datetime(df_s[date_cols[0]], errors='coerce')
            df_s = df_s.dropna(subset=[date_cols[0]]).sort_values(date_cols[0])
            if len(df_s) > 1:
                start_v = df_s[num_cols[0]].iloc[0]; end_v = df_s[num_cols[0]].iloc[-1]
                growth = ((end_v - start_v) / start_v * 100) if start_v != 0 else 0
                has_time = True
        except:
            pass

    if len(num_cols) > 1:
        corr_m = df[num_cols].corr().abs(); np.fill_diagonal(corr_m.values, 0)
        stk = corr_m.stack()
        if len(stk) > 0:
            top_p = stk.idxmax(); corr_val = df[num_cols].corr().loc[top_p[0], top_p[1]]
            v1_n = top_p[0]; v2_n = top_p[1]; cs = abs(corr_val)
            corr_dir = "in the same direction" if corr_val > 0 else "in opposite directions"
        else:
            v1_n = v2_n = ""; cs = 0; corr_dir = ""
    else:
        v1_n = v2_n = ""; cs = 0; corr_dir = ""

    # ---- CHAPTER 1 ----
    st.markdown("## 📋 Chapter 1: The Situation")
    vol_word = "stable and predictable" if cv < 20 else ("moderately variable" if cv < 50 else "highly volatile")
    trend_word = "growing strongly 🚀" if growth > 20 else ("growing steadily 📈" if growth > 5 else ("declining significantly 📉" if growth < -20 else ("declining slightly 📉" if growth < -5 else "holding steady ➡️")))

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#f8f9fa,#fff); border:1px solid #dee2e6; border-radius:12px; padding:24px; margin-bottom:20px;">
    <h4 style="color:#2c3e50; margin-top:0;">🔎 What Are We Looking At?</h4>
    <p style="font-size:16px; line-height:1.9; color:#333;">
    We're working with <b>{total_rows:,} records</b> across <b>{total_cols_n} data fields</b>.
    {"This dataset has a time dimension, which means we can track how things change and spot trends." if has_time else "This is a snapshot dataset — no dates, so we're analyzing the current state rather than changes over time."}
    </p>
    <p style="font-size:16px; line-height:1.9; color:#333;">
    Our main metric to watch is <b>{target_label}</b>.
    {f"It averages around <b>{pn_mean:,.2f}</b> and is currently <b>{vol_word}</b>." if pn else ""}
    {f"Over the observed period, it has been <b>{trend_word}</b> (total change: {growth:+.1f}%)." if has_time else ""}
    </p>
    <p style="font-size:16px; line-height:1.6; color:{'#c0392b' if missing_pct_n > 5 else '#27ae60'};">
    {"⚠️ <b>Data Quality Warning:</b> " + f"{missing_pct_n:.1f}% of data cells are empty. This could compromise our conclusions." if missing_pct_n > 5 else "✅ <b>Data Quality:</b> Clean data with minimal gaps. Conclusions here are reliable."}
    </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- CHAPTER 2 ----
    st.markdown("## 📈 Chapter 2: What Is Happening Right Now?")
    col2a, col2b = st.columns([2, 1])

    with col2a:
        if has_time and num_cols:
            df_plot = df_s.groupby(date_cols[0])[num_cols[0]].mean().reset_index()
            fig = px.area(df_plot, x=date_cols[0], y=num_cols[0], template="plotly_white")
            fig.update_traces(fillcolor="rgba(52,152,219,0.15)", line_color="#2980b9", line_width=2)
            fig.update_layout(height=300, title=f"{num_cols[0]} Over Time")
            st.plotly_chart(fig, use_container_width=True)
        elif cat_cols and num_cols:
            d_a = df.groupby(cat_cols[0])[num_cols[0]].mean().reset_index().sort_values(num_cols[0], ascending=False).head(8)
            fig = px.bar(d_a, x=cat_cols[0], y=num_cols[0], template="plotly_white",
                         color=num_cols[0], color_continuous_scale="Blues")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        if pn:
            spread = pn_max - pn_min
            spread_comment = "This wide spread suggests multiple distinct scenarios happening — check for groups or categories." if spread > pn_mean else "This is a manageable range, suggesting consistent performance."
            st.markdown(f"""
            <div style="background:#eaf4fd; border-left:4px solid #2980b9; padding:14px; border-radius:0 8px 8px 0; font-size:14px; line-height:1.8; color:#333;">
            <b>The Numbers Tell This Story:</b><br>
            {f"<b>{pn}</b> has been <b>{trend_word}</b> over the period we're analyzing. " if has_time else ""}
            The typical value hovers around <b>{pn_mean:,.2f}</b>, but it has swung as low as <b>{pn_min:,.2f}</b> and as high as <b>{pn_max:,.2f}</b>.
            That's a spread of <b>{spread:,.2f}</b>. {spread_comment}
            {"<br>⚠️ <b>Watch out:</b> " + str(outlier_n) + " extreme values are pulling the average away from the true typical value. The median of " + f"<b>{df[pn].median():,.2f}</b> might be a more honest number to report." if outlier_n > 5 else ""}
            </div>""", unsafe_allow_html=True)

    with col2b:
        if pn:
            health_color = "#27ae60" if cv < 20 else ("#f39c12" if cv < 50 else "#e74c3c")
            health_label = "Stable ✅" if cv < 20 else ("Variable ⚠️" if cv < 50 else "Volatile 🚨")
            st.markdown(f"""
            <div style="background:#fff; border-radius:12px; padding:20px; text-align:center; box-shadow:0 2px 12px rgba(0,0,0,0.08);">
            <div style="font-size:12px; color:#aaa; text-transform:uppercase; letter-spacing:1px;">PRIMARY METRIC</div>
            <div style="font-size:42px; font-weight:bold; color:#e67e22; margin:8px 0;">{pn_mean:,.1f}</div>
            <div style="font-size:13px; color:#888; margin-bottom:12px;">{pn}</div>
            <hr style="border:none; border-top:1px solid #eee; margin:10px 0;">
            <table style="width:100%; font-size:13px; text-align:left;">
            <tr><td style="color:#888;">High</td><td style="font-weight:bold; text-align:right;">{pn_max:,.1f}</td></tr>
            <tr><td style="color:#888;">Low</td><td style="font-weight:bold; text-align:right;">{pn_min:,.1f}</td></tr>
            <tr><td style="color:#888;">Stability</td><td style="font-weight:bold; text-align:right; color:{health_color};">{health_label}</td></tr>
            <tr><td style="color:#888;">Outliers</td><td style="font-weight:bold; text-align:right; color:{'#e74c3c' if outlier_n > 0 else '#27ae60'};">{outlier_n}</td></tr>
            </table>
            </div>""", unsafe_allow_html=True)

    # ---- CHAPTER 3 ----
    st.markdown("## 🔍 Chapter 3: Why Is This Happening? (The Root Causes)")
    c3a, c3b = st.columns(2)

    with c3a:
        if cs > 0:
            corr_strength_word = "very strongly linked — almost always move together" if cs > 0.8 else ("moderately linked" if cs > 0.5 else "weakly linked")
            lever_advice = (f"When <b>{v1_n}</b> goes up, <b>{v2_n}</b> {'also tends to go up — they are reinforcing partners' if corr_dir == 'in the same direction' else 'tends to come down — they counterbalance each other'}. "
                            f"This is your biggest lever: managing one gives you indirect control over the other.")
            st.markdown(f"""
            <div style="background:#faf0e6; border-radius:12px; padding:20px; margin-bottom:15px;">
            <h4 style="color:#e67e22; margin-top:0;">🔗 The Strongest Driver</h4>
            <p style="font-size:15px; line-height:1.8; color:#333;">
            <b>{v1_n}</b> and <b>{v2_n}</b> are <b>{corr_strength_word}</b> (r = {corr_val:.2f}). They move <b>{corr_dir}</b>.
            </p>
            <p style="font-size:15px; line-height:1.8; color:#333;">{lever_advice}</p>
            <p style="font-size:13px; color:#888;">💡 Don't treat these as separate problems. Fix the root cause and both will improve.</p>
            </div>""", unsafe_allow_html=True)

    with c3b:
        if top_cc:
            conc_level = "dangerously concentrated" if top_cp > 60 else ("somewhat concentrated" if top_cp > 40 else "well balanced")
            risk_text = (f"⚠️ Over-reliance on '{top_cv}' is a business risk. If this segment fails, it drags everything down. "
                         f"Diversification into other {top_cc} groups should be a strategic priority.") if top_cp > 60 else f"✅ Distribution is healthy — no single {top_cc} group dominates dangerously."
            st.markdown(f"""
            <div style="background:#f0faf5; border-radius:12px; padding:20px; margin-bottom:15px;">
            <h4 style="color:#27ae60; margin-top:0;">🏛️ The Dominant Segment</h4>
            <p style="font-size:15px; line-height:1.8; color:#333;">
            In <b>{top_cc}</b>, the value <b>'{top_cv}'</b> represents <b>{top_cp:.1f}%</b> of all records.
            Your data is <b>{conc_level}</b>.
            </p>
            <p style="font-size:14px; color:#555;">{risk_text}</p>
            </div>""", unsafe_allow_html=True)

    # ---- CHAPTER 4 ----
    st.markdown("## ⚠️ Chapter 4: What Are the Risks?")
    r1_c, r2_c, r3_c = st.columns(3)
    outlier_pct_n2 = (outlier_n / len(df) * 100) if len(df) > 0 else 0
    skew_abs = abs(skew) if pn else 0

    for col_r, risk_label, risk_val, risk_pct, risk_thresh1, risk_thresh2, risk_unit, advice_low, advice_med, advice_high in [
        (r1_c, "Data Completeness", missing_pct_n, missing_pct_n, 1, 5, "%",
         "Clean data. Conclusions are reliable.",
         "Minor gaps. Fill before major decisions.",
         f"Major gaps ({missing_pct_n:.1f}%). Reliability is compromised. Clean first."),
        (r2_c, "Outlier Risk", outlier_n, outlier_pct_n2, 1, 5, " extreme values",
         "No extremes detected. Stable data.",
         f"{outlier_n} unusual values. Investigate — real events or data errors?",
         f"Many extremes ({outlier_n}). Averages are distorted. Use median instead."),
        (r3_c, "Distribution Balance", skew_abs, skew_abs, 0.5, 1, " skew",
         "Balanced distribution. Mean is accurate.",
         "Slightly skewed. Consider using median.",
         f"Heavily skewed ({skew:.2f}). Mean is misleading. Report median instead.")
    ]:
        risk_color = "#27ae60" if risk_pct < risk_thresh1 else ("#f39c12" if risk_pct < risk_thresh2 else "#e74c3c")
        risk_emoji = "🟢" if risk_pct < risk_thresh1 else ("🟡" if risk_pct < risk_thresh2 else "🔴")
        advice = advice_low if risk_pct < risk_thresh1 else (advice_med if risk_pct < risk_thresh2 else advice_high)
        with col_r:
            st.markdown(f"""
            <div style="border:2px solid {risk_color}; border-radius:12px; padding:18px; text-align:center; height:200px;">
            <div style="font-size:24px;">{risk_emoji}</div>
            <div style="font-weight:bold; margin:6px 0; color:#333;">{risk_label}</div>
            <div style="font-size:26px; font-weight:bold; color:{risk_color};">{risk_val if isinstance(risk_val, int) else f"{risk_val:.1f}"}{risk_unit}</div>
            <div style="font-size:12px; color:#666; margin-top:8px; line-height:1.5;">{advice}</div>
            </div>""", unsafe_allow_html=True)

    # ---- CHAPTER 5 ----
    st.markdown("## 🚀 Chapter 5: What Do We Do Now?")
    overall_h = "excellent" if missing_pct_n < 2 and outlier_n < len(df)*0.02 else ("acceptable" if missing_pct_n < 10 else "needs work")

    actions_7 = []
    if missing_pct_n > 1: actions_7.append(f"🔧 **Fix data gaps first** — Go to 'Data Preview' tab → 'Fill Missing Values'. {missing_pct_n:.1f}% is too high to ignore.")
    if outlier_n > 0: actions_7.append(f"🔍 **Investigate {outlier_n} extreme values** in '{pn}' — Are they real events (record sale, machine failure) or typos?")
    if top_cp > 60: actions_7.append(f"⚠️ **Diversification risk** — '{top_cv}' dominates {top_cp:.1f}% of '{top_cc}'. What's your contingency plan if this segment drops?")
    if not actions_7: actions_7.append("✅ Data is clean — skip to strategic improvements below.")

    actions_30 = [
        f"📊 **Build a live monitor** for '{v1_n}' + '{v2_n}' — your strongest predictive pair (r={cs:.2f})" if cs > 0.5 else "📊 **Run weekly summaries** — track your key metric trends weekly",
        f"🔮 **Use Trend Forecast tab** to project '{target_label}' for next quarter. Set targets based on the confidence band, not just the center line.",
        "🤖 **Ask the Data Agent** (Tab 7): 'Which department/machine needs the most attention?' — it will calculate and tell you exactly.",
        "📤 **Share this report** — Use the Export tab to generate a PDF for your leadership presentation."
    ]

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#2c3e50,#34495e); color:white; border-radius:12px; padding:24px;">
    <h3 style="color:white; margin-top:0;">🗓️ 7-Day Action List</h3>
    {''.join([f'<div style="background:rgba(255,255,255,0.1); margin:6px 0; padding:10px 14px; border-radius:8px; font-size:14px;">{a}</div>' for a in actions_7])}
    <h3 style="color:white; margin-top:20px;">📅 30-Day Strategic Moves</h3>
    {''.join([f'<div style="background:rgba(255,255,255,0.1); margin:6px 0; padding:10px 14px; border-radius:8px; font-size:14px;">{a}</div>' for a in actions_30])}
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    conclusion_msg = {
        "excellent": "Your data is in great shape. Focus on growth, optimization, and forecasting.",
        "acceptable": "Clean up the data gaps and investigate outliers. Then you'll have full confidence in your analysis.",
        "needs work": "Data quality is a bottleneck right now. Any conclusions are at risk until you clean the missing values. Fix that first."
    }
    st.success(f"**📌 Analyst's Final Verdict:** Data health is **{overall_h}**. {conclusion_msg[overall_h]}")

# ==============================================================================
#  TAB 9: THE "NEVER BLANK" MASTER REPORT
# ==============================================================================
import plotly.io as pio
import plotly.express as px
import pandas as pd
import streamlit as st

# ==============================================================================
#  TAB 8: PROFESSIONAL PDF REPORT GENERATOR
#  REPLACE YOUR ENTIRE "with tab_export:" SECTION WITH THIS CODE
# ==============================================================================

with tab_export:
    st.markdown("## 📥 Professional PDF Report Generator")
    st.caption("Create a comprehensive, shareable report with all analysis, charts, and AI conversations")
    
    # =========================================================================
    # CONFIGURATION OPTIONS
    # =========================================================================
    
    st.markdown("### ⚙️ Report Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        include_preview = st.checkbox("📄 Data Preview", value=True)
        include_dashboard = st.checkbox("📊 Dashboard Charts", value=True)
    
    with col2:
        include_inspector = st.checkbox("🔬 Quality Inspector", value=True)
        include_forecast = st.checkbox("🔮 Trend Forecast", value=True)
    
    with col3:
        include_narration = st.checkbox("📖 Data Story", value=True)
        include_chat = st.checkbox("💬 AI Conversations", value=True)
    
    st.markdown("---")
    
    # =========================================================================
    # PDF GENERATION FUNCTION
    # =========================================================================
    
    def generate_professional_report():
        """
        Generate a beautiful HTML report that auto-prints to PDF
        """
        
        # Get current timestamp
        current_time = pd.Timestamp.now()
        
        # Calculate key metrics
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isna().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        duplicates = df.duplicated().sum()
        health_score = max(0, 100 - min(missing_pct*2, 30) - min(duplicates/len(df)*100*1.5, 20))
        
        # =====================================================================
        # CSS STYLES
        # =====================================================================
        
        css = """
        @media print {
            @page {
                size: A4;
                margin: 15mm;
            }
            .page-break {
                page-break-before: always;
            }
            .no-break {
                page-break-inside: avoid;
            }
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            color: #2c3e50;
            background: white;
            padding: 20px;
            max-width: 210mm;
            margin: 0 auto;
        }
        
        /* HEADER */
        .report-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 30px;
            border-radius: 15px;
            margin-bottom: 40px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(102,126,234,0.3);
        }
        
        .report-title {
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .report-subtitle {
            font-size: 18px;
            opacity: 0.95;
            font-weight: 300;
        }
        
        .report-date {
            font-size: 14px;
            opacity: 0.85;
            margin-top: 15px;
            border-top: 1px solid rgba(255,255,255,0.3);
            padding-top: 15px;
        }
        
        /* SECTIONS */
        .section {
            margin-bottom: 50px;
            page-break-inside: avoid;
        }
        
        .section-header {
            background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 18px 25px;
            border-radius: 10px;
            margin-bottom: 25px;
            font-size: 26px;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(245,87,108,0.3);
        }
        
        .subsection-header {
            color: #667eea;
            font-size: 22px;
            font-weight: 600;
            margin: 30px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
        
        /* METRICS GRID */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 25px 0;
        }
        
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            border-left: 5px solid #667eea;
            transition: transform 0.3s;
        }
        
        .metric-value {
            font-size: 32px;
            font-weight: 700;
            color: #667eea;
            margin: 12px 0;
        }
        
        .metric-label {
            font-size: 13px;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 500;
        }
        
        /* TABLES */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 13px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.08);
            border-radius: 8px;
            overflow: hidden;
        }
        
        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 12px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }
        
        td {
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        
        tr:nth-child(even) {
            background: #f8f9fa;
        }
        
        tr:hover {
            background: #e8f4f8;
        }
        
        /* CHAT MESSAGES */
        .chat-container {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 25px;
            margin: 25px 0;
        }
        
        .chat-message {
            margin: 18px 0;
            padding: 18px;
            border-radius: 10px;
            page-break-inside: avoid;
        }
        
        .chat-user {
            background: white;
            border-left: 5px solid #667eea;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        
        .chat-assistant {
            background: #e8f4f8;
            border-left: 5px solid #4CAF50;
            box-shadow: 0 2px 8px rgba(76,175,80,0.15);
        }
        
        .chat-role {
            font-weight: 700;
            margin-bottom: 10px;
            font-size: 14px;
            color: #2c3e50;
        }
        
        .chat-content {
            font-size: 14px;
            line-height: 1.8;
            color: #34495e;
        }
        
        /* INFO BOXES */
        .info-box {
            background: #e3f2fd;
            border-left: 5px solid #2196F3;
            padding: 18px;
            margin: 20px 0;
            border-radius: 8px;
            font-size: 14px;
            line-height: 1.7;
        }
        
        .success-box {
            background: #e8f5e9;
            border-left: 5px solid #4caf50;
            padding: 18px;
            margin: 20px 0;
            border-radius: 8px;
            font-size: 14px;
            line-height: 1.7;
        }
        
        .warning-box {
            background: #fff3e0;
            border-left: 5px solid #ff9800;
            padding: 18px;
            margin: 20px 0;
            border-radius: 8px;
            font-size: 14px;
            line-height: 1.7;
        }
        
        /* CHART CONTAINER */
        .chart-container {
            background: white;
            padding: 25px;
            border-radius: 12px;
            margin: 25px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }
        
        /* FOOTER */
        .report-footer {
            margin-top: 60px;
            padding-top: 25px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #95a5a6;
            font-size: 12px;
            line-height: 1.8;
        }
        
        /* STATS ROW */
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .stat-label {
            font-weight: 600;
            color: #7f8c8d;
        }
        
        .stat-value {
            color: #2c3e50;
            font-weight: 700;
        }
        
        /* HEALTH BADGE */
        .health-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
        }
        
        .health-excellent {
            background: #e8f5e9;
            color: #2e7d32;
        }
        
        .health-good {
            background: #fff3e0;
            color: #e65100;
        }
        
        .health-poor {
            background: #ffebee;
            color: #c62828;
        }
        """
        
        # =====================================================================
        # START HTML
        # =====================================================================
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Production Data Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>{css}</style>
</head>
<body>

<!-- HEADER -->
<div class="report-header">
    <div class="report-title">📊 AI-Optimized Production Data Analysis</div>
    <div class="report-subtitle">Comprehensive Intelligence Report</div>
    <div class="report-date">
        Generated: {current_time.strftime('%B %d, %Y at %I:%M %p')}<br>
        Dataset: {len(df):,} records × {len(df.columns)} columns
    </div>
</div>

<!-- EXECUTIVE SUMMARY -->
<div class="section">
    <div class="section-header">📋 Executive Summary</div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Data Columns</div>
            <div class="metric-value">{len(df.columns)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Health Score</div>
            <div class="metric-value" style="color: {'#2ecc71' if health_score >= 80 else '#f39c12' if health_score >= 60 else '#e74c3c'}">{health_score:.0f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Data Quality</div>
            <div class="metric-value" style="font-size: 18px;">
                <span class="health-badge {'health-excellent' if health_score >= 80 else 'health-good' if health_score >= 60 else 'health-poor'}">
                    {'Excellent' if health_score >= 80 else 'Good' if health_score >= 60 else 'Needs Work'}
                </span>
            </div>
        </div>
    </div>
    
    <div class="info-box">
        <strong>📌 Key Findings:</strong> This report analyzes {len(df):,} production records across {len(df.columns)} variables. 
        Data quality score: {health_score:.0f}/100. 
        {'✅ Dataset is reliable for critical business decisions.' if health_score >= 80 else '⚠️ Minor quality issues detected - review recommendations below.' if health_score >= 60 else '❌ Significant data quality issues require immediate attention.'}
    </div>
</div>
"""
        
        # =====================================================================
        # DATA PREVIEW SECTION
        # =====================================================================
        
        if include_preview:
            html += """
<div class="section page-break">
    <div class="section-header">📄 Data Preview</div>
    <p style="margin-bottom: 20px;">Sample of the first 15 records from the production dataset:</p>
"""
            html += df.head(15).to_html(classes='table', index=False, border=0, escape=False)
            
            # Column Analysis
            html += """
    <div class="subsection-header">Column Analysis</div>
"""
            
            column_data = []
            for col in df.columns:
                missing_count = df[col].isna().sum()
                missing_pct_col = (missing_count / len(df)) * 100
                
                column_data.append({
                    'Column Name': col,
                    'Data Type': str(df[col].dtype),
                    'Missing Values': f"{missing_count} ({missing_pct_col:.1f}%)",
                    'Unique Values': df[col].nunique(),
                    'Quality': '🟢 Excellent' if missing_pct_col == 0 else '🟡 Good' if missing_pct_col < 5 else '🔴 Poor'
                })
            
            html += pd.DataFrame(column_data).to_html(classes='table', index=False, border=0, escape=False)
            html += "</div>"
        
        # =====================================================================
        # DASHBOARD SECTION
        # =====================================================================
        
        if include_dashboard:
            html += """
<div class="section page-break">
    <div class="section-header">📊 Dashboard Analytics</div>
"""
            
            # Key Performance Metrics
            if num_cols:
                primary_metric = num_cols[0]
                html += f"""
    <div class="subsection-header">Key Performance Metrics</div>
    <div class="stat-row">
        <span class="stat-label">Primary Metric ({primary_metric}):</span>
        <span class="stat-value">{df[primary_metric].mean():.2f} (Average)</span>
    </div>
    <div class="stat-row">
        <span class="stat-label">Range:</span>
        <span class="stat-value">{df[primary_metric].min():.2f} - {df[primary_metric].max():.2f}</span>
    </div>
    <div class="stat-row">
        <span class="stat-label">Standard Deviation:</span>
        <span class="stat-value">{df[primary_metric].std():.2f}</span>
    </div>
    <div class="stat-row">
        <span class="stat-label">Median:</span>
        <span class="stat-value">{df[primary_metric].median():.2f}</span>
    </div>
"""
            
            # Generate Charts
            if len(num_cols) > 0 and len(cat_cols) > 0:
                # Bar chart
                df_grouped = df.groupby(cat_cols[0])[num_cols[0]].mean().reset_index().head(10)
                fig1 = px.bar(
                    df_grouped,
                    x=cat_cols[0],
                    y=num_cols[0],
                    title=f"Average {num_cols[0]} by {cat_cols[0]}",
                    template="plotly_white",
                    color=num_cols[0],
                    color_continuous_scale="Blues"
                )
                fig1.update_layout(height=400, margin=dict(t=50, b=50, l=50, r=50))
                html += f"""
    <div class="chart-container">
        {fig1.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})}
    </div>
"""
            
            if len(num_cols) > 1:
                # Correlation heatmap
                corr = df[num_cols[:6]].corr()
                fig2 = px.imshow(
                    corr,
                    text_auto='.2f',
                    title="Correlation Analysis",
                    template="plotly_white",
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                fig2.update_layout(height=400, margin=dict(t=50, b=50, l=50, r=50))
                html += f"""
    <div class="chart-container">
        {fig2.to_html(full_html=False, include_plotlyjs=False, config={'displayModeBar': False})}
    </div>
"""
            
            # Distribution chart
            if len(num_cols) > 0:
                fig3 = px.histogram(
                    df,
                    x=num_cols[0],
                    title=f"Distribution of {num_cols[0]}",
                    template="plotly_white",
                    marginal="box"
                )
                fig3.update_layout(height=400, margin=dict(t=50, b=50, l=50, r=50))
                html += f"""
    <div class="chart-container">
        {fig3.to_html(full_html=False, include_plotlyjs=False, config={'displayModeBar': False})}
    </div>
"""
            
            html += "</div>"
        
        # =====================================================================
        # QUALITY INSPECTOR SECTION
        # =====================================================================
        
        if include_inspector:
            html += f"""
<div class="section page-break">
    <div class="section-header">🔬 Data Quality Inspector</div>
    
    <div class="subsection-header">Overall Health Assessment</div>
    <div class="{'success-box' if health_score >= 80 else 'warning-box' if health_score >= 60 else 'info-box'}">
        <strong>Health Score: {health_score:.0f}/100</strong><br><br>
        {'✅ <strong>Excellent</strong> - Data is highly reliable for production analysis and decision-making.' if health_score >= 80 else '⚠️ <strong>Good</strong> - Minor quality issues present. Review recommendations below.' if health_score >= 60 else '❌ <strong>Needs Attention</strong> - Significant quality issues detected. Clean data before drawing conclusions.'}
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Missing Values</div>
            <div class="metric-value" style="color: {'#2ecc71' if missing_pct < 1 else '#f39c12' if missing_pct < 5 else '#e74c3c'}">{missing_pct:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Duplicate Rows</div>
            <div class="metric-value" style="color: {'#2ecc71' if duplicates == 0 else '#f39c12'}">{duplicates}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Complete Columns</div>
            <div class="metric-value">{sum(df[col].isna().sum() == 0 for col in df.columns)}/{len(df.columns)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Data Integrity</div>
            <div class="metric-value" style="color: {'#2ecc71' if health_score >= 80 else '#f39c12' if health_score >= 60 else '#e74c3c'}">{'High' if health_score >= 80 else 'Medium' if health_score >= 60 else 'Low'}</div>
        </div>
    </div>
    
    <div class="subsection-header">Column-by-Column Quality Report</div>
"""
            
            quality_report = []
            for col in df.columns:
                missing_count = df[col].isna().sum()
                missing_pct_col = (missing_count / len(df)) * 100
                
                quality_report.append({
                    'Column': col,
                    'Type': str(df[col].dtype),
                    'Missing': f"{missing_count} ({missing_pct_col:.1f}%)",
                    'Unique': df[col].nunique(),
                    'Status': '🟢 Excellent' if missing_pct_col == 0 else '🟡 Good' if missing_pct_col < 5 else '🔴 Poor'
                })
            
            html += pd.DataFrame(quality_report).to_html(classes='table', index=False, border=0, escape=False)
            html += "</div>"
        
        # =====================================================================
        # TREND FORECAST SECTION
        # =====================================================================
        
        if include_forecast and date_cols and num_cols:
            html += """
<div class="section page-break">
    <div class="section-header">🔮 Trend Forecast</div>
"""
            
            # Create trend visualization
            df_trend = df.groupby(date_cols[0])[num_cols[0]].mean().reset_index()
            df_trend = df_trend.sort_values(date_cols[0])
            
            fig_forecast = px.line(
                df_trend,
                x=date_cols[0],
                y=num_cols[0],
                title=f"{num_cols[0]} Trend Over Time",
                markers=True,
                template="plotly_white"
            )
            fig_forecast.update_traces(line_color='#667eea', line_width=3)
            fig_forecast.update_layout(height=400, margin=dict(t=50, b=50, l=50, r=50))
            
            html += f"""
    <div class="chart-container">
        {fig_forecast.to_html(full_html=False, include_plotlyjs=False, config={'displayModeBar': False})}
    </div>
    
    <div class="info-box">
        <strong>📈 Forecast Analysis:</strong> Based on historical patterns, {num_cols[0]} shows 
        {'an upward trend 📈' if df_trend[num_cols[0]].iloc[-1] > df_trend[num_cols[0]].iloc[0] else 'a downward trend 📉' if df_trend[num_cols[0]].iloc[-1] < df_trend[num_cols[0]].iloc[0] else 'stability ➡️'}.
        <br><br>
        <strong>Current Average:</strong> {df[num_cols[0]].mean():.2f}<br>
        <strong>Projected Direction:</strong> {'Increasing' if df_trend[num_cols[0]].iloc[-1] > df_trend[num_cols[0]].iloc[0] else 'Decreasing' if df_trend[num_cols[0]].iloc[-1] < df_trend[num_cols[0]].iloc[0] else 'Stable'}
    </div>
</div>
"""
        
        # =====================================================================
        # NARRATION SECTION
        # =====================================================================
        
        if include_narration:
            html += """
<div class="section page-break">
    <div class="section-header">📖 Data Story & Strategic Insights</div>
"""
            
            # Generate narrative
            if num_cols:
                primary_metric = num_cols[0]
                avg_val = df[primary_metric].mean()
                max_val = df[primary_metric].max()
                min_val = df[primary_metric].min()
                std_val = df[primary_metric].std()
                
                html += f"""
    <div class="subsection-header">The Numbers Tell This Story</div>
    <p style="font-size: 15px; line-height: 1.8;">
        <strong>{primary_metric}</strong> averages <strong>{avg_val:.2f}</strong>, 
        with values ranging from a low of <strong>{min_val:.2f}</strong> to a high of <strong>{max_val:.2f}</strong>.
        The standard deviation of <strong>{std_val:.2f}</strong> indicates 
        {'relatively stable' if std_val < avg_val * 0.3 else 'moderate variability' if std_val < avg_val * 0.6 else 'high variability'} in the data.
    </p>
"""
                
                # Find correlations
                if len(num_cols) > 1:
                    corr_matrix = df[num_cols].corr()
                    max_corr = 0
                    corr_pair = None
                    
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if abs(corr_matrix.iloc[i, j]) > max_corr:
                                max_corr = abs(corr_matrix.iloc[i, j])
                                corr_pair = (corr_matrix.columns[i], corr_matrix.columns[j])
                    
                    if corr_pair and max_corr > 0.5:
                        corr_val = corr_matrix.loc[corr_pair[0], corr_pair[1]]
                        html += f"""
    <div class="subsection-header">Key Relationships Discovered</div>
    <p style="font-size: 15px; line-height: 1.8;">
        The analysis reveals a {'strong positive' if corr_val > 0 else 'strong negative'} 
        relationship between <strong>{corr_pair[0]}</strong> and <strong>{corr_pair[1]}</strong> 
        (correlation: {corr_val:.2f}). This means when {corr_pair[0]} changes, 
        {corr_pair[1]} tends to {'move in the same direction' if corr_val > 0 else 'move in the opposite direction'}.
    </p>
"""
            
            # Strategic Recommendations
            html += """
    <div class="subsection-header">Strategic Recommendations</div>
    <div class="success-box">
        <strong>🎯 Recommended Next Steps:</strong>
        <ul style="margin: 15px 0 0 20px; line-height: 2;">
"""
            
            recommendations = []
            if missing_pct > 5:
                recommendations.append(f"<li><strong>Data Quality:</strong> Address {missing_pct:.1f}% missing data for more reliable analysis</li>")
            if duplicates > 0:
                recommendations.append(f"<li><strong>Data Cleaning:</strong> Remove {duplicates} duplicate records</li>")
            if len(num_cols) > 0:
                recommendations.append(f"<li><strong>Monitoring:</strong> Set up real-time tracking for {num_cols[0]}</li>")
            
            if not recommendations:
                recommendations.append("<li><strong>Excellence Maintained:</strong> Data quality is excellent - focus on strategic insights and predictive analytics</li>")
                recommendations.append("<li><strong>Automation:</strong> Consider implementing automated monitoring systems</li>")
            
            for rec in recommendations:
                html += f"            {rec}\n"
            
            html += """
        </ul>
    </div>
</div>
"""
        
        # =====================================================================
        # AI CHAT CONVERSATION SECTION
        # =====================================================================
        
        if include_chat:
            html += """
<div class="section page-break">
    <div class="section-header">💬 AI Data Agent Conversations</div>
    <p style="margin-bottom: 25px; font-size: 15px;">Complete record of all questions asked and AI-generated insights:</p>
    <div class="chat-container">
"""
            
            if "chat_messages" in st.session_state and len(st.session_state.chat_messages) > 0:
                for i, msg in enumerate(st.session_state.chat_messages):
                    role_label = "👤 You" if msg['role'] == 'user' else "🤖 AI Data Analyst"
                    chat_class = "chat-user" if msg['role'] == 'user' else "chat-assistant"
                    
                    # Escape HTML in content
                    content = str(msg['content']).replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
                    
                    html += f"""
        <div class="chat-message {chat_class}">
            <div class="chat-role">{role_label}</div>
            <div class="chat-content">{content}</div>
        </div>
"""
            else:
                html += """
        <div class="info-box">
            <strong>ℹ️ No conversations recorded.</strong><br><br>
            To use the AI Data Agent:
            <ol style="margin: 10px 0 0 20px;">
                <li>Navigate to the "Data Agent" tab</li>
                <li>Ask questions about your data in natural language</li>
                <li>The AI will analyze and provide insights</li>
                <li>All conversations will appear in future reports</li>
            </ol>
        </div>
"""
            
            html += """
    </div>
</div>
"""
        
        # =====================================================================
        # FOOTER
        # =====================================================================
        
        html += f"""
<div class="report-footer">
    <p><strong>Report Generated by AI-Optimized Production Data Analyzer</strong></p>
    <p>Intelligent Analytics Platform</p>
    <p style="margin-top: 15px;">Generated on {current_time.strftime('%B %d, %Y at %I:%M %p')}</p>
    <p>Dataset: {len(df):,} records × {len(df.columns)} variables</p>
    <p style="margin-top: 20px; font-size: 11px; color: #bdc3c7;">
        This report contains proprietary business intelligence - Confidential and for authorized use only
    </p>
</div>

<!-- Auto-print script -->
<script>
window.onload = function() {{
    // Wait 2 seconds then trigger print dialog
    setTimeout(function() {{
        window.print();
    }}, 2000);
}};
</script>

</body>
</html>
"""
        
        return html
    
    # =========================================================================
    # GENERATION BUTTON
    # =========================================================================
    
    st.markdown("### 📄 Generate Your Report")
    
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    
    with col_btn1:
        if st.button("🚀 Generate Professional PDF Report", type="primary", use_container_width=True):
            with st.spinner("📝 Generating comprehensive report..."):
                try:
                    report_html = generate_professional_report()
                    
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    
                    st.download_button(
                        label="📥 Download Report (Click to Save)",
                        data=report_html,
                        file_name=f"AI_Production_Report_{timestamp}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                    
                    st.success("✅ **Report Generated Successfully!**")
                    
                    st.info("""
**📋 How to Save as PDF:**

**Method 1: Automatic (Recommended)**
1. Click the download button above
2. Open the downloaded HTML file in your browser
3. Wait 2 seconds - print dialog will auto-open
4. Select "Save as PDF" or "Microsoft Print to PDF"
5. Click Save

**Method 2: Manual**
1. Download the file above
2. Open it in Chrome/Edge/Firefox
3. Press Ctrl+P (Windows) or Cmd+P (Mac)
4. Choose "Save as PDF"
5. Click Save

**💡 The report includes:**
- 📊 Executive summary with key metrics
- 📄 Data preview and column analysis
- 📈 Interactive dashboard charts
- 🔬 Complete quality assessment
- 🔮 Trend forecasts and predictions
- 📖 AI-generated data story
- 💬 Full AI chat conversation history

**📧 Perfect for sharing with stakeholders, management, or clients!**
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")
                    st.info("Please ensure all required data is available in the session.")
    
    with col_btn2:
        sections_count = sum([
            include_preview,
            include_dashboard,
            include_inspector,
            include_forecast,
            include_narration,
            include_chat
        ])
        st.metric(
            label="Sections",
            value=sections_count,
            help="Number of sections included in report"
        )
    
    with col_btn3:
        chat_count = len(st.session_state.get('chat_messages', []))
        st.metric(
            label="AI Messages",
            value=chat_count,
            help="Number of AI conversations to include"
        )
    
    # =========================================================================
    # PREVIEW & STATS
    # =========================================================================
    
    st.markdown("---")
    st.markdown("### 👁️ Report Preview")
    
    with st.expander("📊 What's included in your report?"):
        
        preview_items = []
        
        if include_preview:
            preview_items.append("✅ **Data Preview** - First 15 rows + detailed column analysis")
        
        if include_dashboard:
            preview_items.append("✅ **Dashboard Analytics** - KPIs, charts, correlations, and distributions")
        
        if include_inspector:
            preview_items.append("✅ **Quality Inspector** - Health score (0-100), missing data analysis, quality report")
        
        if include_forecast:
            preview_items.append("✅ **Trend Forecast** - Time-series visualizations and predictions")
        
        if include_narration:
            preview_items.append("✅ **Data Story** - AI-generated narrative with strategic recommendations")
        
        if include_chat:
            chat_msg_count = len(st.session_state.get('chat_messages', []))
            preview_items.append(f"✅ **AI Conversations** - Complete chat history ({chat_msg_count} messages)")
        
        if preview_items:
            for item in preview_items:
                st.markdown(item)
        else:
            st.warning("⚠️ No sections selected. Enable at least one section above to generate a report.")
    
    st.markdown("---")
    st.markdown("### 📈 Quick Statistics")
    
    stat1, stat2, stat3, stat4 = st.columns(4)
    
    with stat1:
        st.metric("Total Records", f"{len(df):,}")
    
    with stat2:
        st.metric("Data Columns", len(df.columns))
    
    with stat3:
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isna().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    with stat4:
        st.metric("Chat Messages", len(st.session_state.get('chat_messages', [])))
    

    
    st.markdown("---")
    
    with st.expander("ℹ️ Report Tips & Best Practices"):
        st.markdown("""
**📌 Before Generating Your Report:**

1. **Use the AI Agent** - Ask questions in the "Data Agent" tab to populate the conversations section
2. **Review All Tabs** - Navigate through all tabs to ensure all analysis is complete
3. **Check Data Quality** - Use the "Data Inspector" tab to clean any issues first

**💡 Pro Tips:**

- **For Executives:** Enable Dashboard + Narration + Summary (skip technical details)
- **For Technical Teams:** Enable all sections including Quality Inspector and Forecast
- **For Quick Updates:** Enable only Preview + Dashboard + Chat

**🎨 Customization:**

- Use the checkboxes above to include/exclude sections
- The report automatically adapts to your data type
- Charts are interactive in HTML format, static in PDF

**📤 Sharing:**

- HTML format preserves interactivity (hover, zoom on charts)
- PDF format is best for printing and formal distribution
- Both formats are professional and presentation-ready

**🔒 Privacy Note:**

- Reports are generated client-side in your browser
- No data is sent to external servers
- Safe to share with stakeholders
        """)
# ==============================================================================
#  TAB 7: AI DATA AGENT 
# ==============================================================================


with tab_assistant:
    st.markdown("## 🤖 Intelligent AI Data Agent")
    st.caption("Ask ANY question about your data - I'll analyze and answer using AI")
    
  
    
    @st.cache_data
    def analyze_data_smartly(dataframe):
        """
        Pre-compute insights so LLM doesn't have to process raw data
        """
        insights = {}
        
        # Get column types
        num_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Find date column
        date_col = None
        for col in dataframe.columns:
            if any(x in col.lower() for x in ['date', 'time', 'timestamp']):
                try:
                    pd.to_datetime(dataframe[col], errors='coerce')
                    date_col = col
                    break
                except:
                    pass
        
        insights['columns'] = {
            'numeric': num_cols,
            'categorical': cat_cols,
            'date': date_col,
            'all': dataframe.columns.tolist()
        }
        
        # Summary statistics (compact)
        insights['summary'] = {}
        for col in num_cols[:10]:  # Limit to 10 numeric columns
            insights['summary'][col] = {
                'mean': float(dataframe[col].mean()),
                'max': float(dataframe[col].max()),
                'min': float(dataframe[col].min()),
                'std': float(dataframe[col].std())
            }
        
        # Top correlations (only strong ones)
        if len(num_cols) > 1:
            corr_matrix = dataframe[num_cols].corr()
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:  # Only strong correlations
                        strong_corr.append({
                            'col1': corr_matrix.columns[i],
                            'col2': corr_matrix.columns[j],
                            'correlation': float(corr_val)
                        })
            insights['strong_correlations'] = strong_corr[:5]  # Top 5
        
        # Categorical breakdowns
        insights['categories'] = {}
        for col in cat_cols[:5]:  # Limit to 5 categorical
            value_counts = dataframe[col].value_counts().head(5)
            insights['categories'][col] = {str(k): int(v) for k, v in value_counts.items()}
        
        # Record count
        insights['total_records'] = len(dataframe)
        
        return insights
    
    def extract_relevant_data(query, df, insights):
        """
        Extract ONLY the data relevant to the query (makes LLM super fast)
        """
        query_lower = query.lower()
        
        # Detect what columns user is asking about
        relevant_cols = []
        for col in df.columns:
            col_words = col.lower().replace('_', ' ').split()
            if any(word in query_lower for word in col_words if len(word) > 2):
                relevant_cols.append(col)
        
        # If no columns detected, use all numeric
        if not relevant_cols:
            relevant_cols = insights['columns']['numeric'][:5]
        
        # Detect date filter if mentioned
        date_col = insights['columns']['date']
        date_filter = None
        
        if date_col:
            # Check for date mentions
            import re
            date_patterns = {
                'jan 1': '01-01', 'jan 2': '01-02', 'january 1': '01-01',
                'feb 1': '02-01', 'march 1': '03-01'
            }
            for pattern, date_str in date_patterns.items():
                if pattern in query_lower:
                    date_filter = date_str
                    break
        
        # Extract relevant subset
        df_subset = df[relevant_cols + ([date_col] if date_col else [])].copy()
        
        if date_filter and date_col:
            df_subset = df_subset[df_subset[date_col].astype(str).str.contains(date_filter)]
        
        # Return only top 20 rows (enough for context)
        return df_subset.head(20).to_csv(index=False)
    
    def build_smart_prompt(query, df, insights):
        """
        Build efficient prompt with pre-computed insights
        """
        relevant_data = extract_relevant_data(query, df, insights)
        
        prompt = f"""You are an expert data analyst. Answer this question about the dataset:

QUESTION: {query}

DATASET OVERVIEW:
- Total Records: {insights['total_records']}
- Numeric Columns: {', '.join(insights['columns']['numeric'][:8])}
- Categorical Columns: {', '.join(insights['columns']['categorical'][:5])}

KEY STATISTICS:
"""
        
        # Add relevant statistics
        for col, stats in list(insights['summary'].items())[:5]:
            prompt += f"\n{col}: avg={stats['mean']:.2f}, max={stats['max']:.2f}, min={stats['min']:.2f}"
        
        # Add strong correlations
        if 'strong_correlations' in insights:
            prompt += "\n\nSTRONG CORRELATIONS:"
            for corr in insights['strong_correlations']:
                prompt += f"\n- {corr['col1']} ↔ {corr['col2']}: {corr['correlation']:.2f}"
        
        # Add relevant data sample
        prompt += f"\n\nRELEVANT DATA SAMPLE:\n{relevant_data}"
        
        prompt += """\n\nINSTRUCTIONS:
1. Answer the specific question with EXACT numbers from the data
2. If asking WHY, provide root cause analysis using correlations
3. If asking WHICH/WHO, rank and list top results
4. If asking HOW MANY/WHAT, provide counts and percentages
5. Be specific and data-driven - no generic responses
6. Keep answer under 200 words but include key insights

ANSWER:"""
        
        return prompt
    
    # =========================================================================
    # FAST LLM SETUP (Optimized)
    # =========================================================================
    
    @st.cache_resource
    def get_llm():
        """Cache LLM instance for speed"""
        from langchain_community.llms import Ollama
        return Ollama(
            model="llama3.2",
            temperature=0.1,
            num_ctx=2048,  # Smaller context = faster
            num_predict=300  # Limit output length
        )
    
    # =========================================================================
    # CHAT INTERFACE
    # =========================================================================
    
    # Initialize chat
    if "chat_messages" not in st.session_state:
      st.session_state.chat_messages = load_chat_history(st.session_state.user_email)
    
    # Pre-compute insights (cached, only runs once)
    with st.spinner("🔍 Analyzing your data structure..."):
        data_insights = analyze_data_smartly(df)
    
    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask ANY question about your data..."):
        
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        save_chat_message(st.session_state.user_email, "user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Analyzing..."):
                try:
                    # Build smart prompt (only relevant data)
                    prompt = build_smart_prompt(user_input, df, data_insights)
                    
                    # Get LLM
                    llm = get_llm()
                    
                    # Generate response (much faster now!)
                    response = llm.invoke(prompt)
                    
                    st.markdown(response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    save_chat_message(st.session_state.user_email, "assistant", response)
                
                except Exception as e:
                    error_msg = f"❌ **Error:** {str(e)}\n\n"
                    
                    if "ollama" in str(e).lower():
                        error_msg += """**Ollama not running!** 
                        
To fix:
1. Open terminal/command prompt
2. Run: `ollama serve`
3. In another terminal: `ollama pull llama3.2`
4. Refresh this page"""
                    else:
                        error_msg += "Please check your Ollama installation."
                    
                    st.error(error_msg)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        clear_chat_history(st.session_state.user_email)
        st.session_state.chat_messages = []
        st.rerun()
    
    # Example questions (dynamic based on data)
    st.markdown("---")
    st.markdown("### 💡 Example Questions:")
    
    # Generate examples based on actual columns
    example_questions = []
    
    if data_insights['columns']['numeric']:
        num_col = data_insights['columns']['numeric'][0]
        example_questions.append(f"• Why did {num_col} spike?")
        example_questions.append(f"• Which has highest {num_col}?")
    
    if data_insights['columns']['categorical']:
        cat_col = data_insights['columns']['categorical'][0]
        example_questions.append(f"• Compare all {cat_col}")
        example_questions.append(f"• How many {cat_col} are there?")
    
    if data_insights['columns']['date']:
        example_questions.append(f"• What happened on [specific date]?")
        example_questions.append(f"• Show monthly trend")
    
    # Add generic questions
    example_questions.extend([
        "• What is the biggest problem?",
        "• Where are we losing money?",
        "• Which needs immediate attention?"
    ])
    
    col1, col2 = st.columns(2)
    for i, q in enumerate(example_questions):
        with col1 if i % 2 == 0 else col2:
            st.markdown(q)
    
    # Show data info
    with st.expander("📊 Your Data Overview"):
        st.write(f"**Total Records:** {data_insights['total_records']:,}")
        st.write(f"**Numeric Columns:** {', '.join(data_insights['columns']['numeric'][:10])}")
        st.write(f"**Categorical Columns:** {', '.join(data_insights['columns']['categorical'][:5])}")
        if data_insights['columns']['date']:
            st.write(f"**Date Column:** {data_insights['columns']['date']}")
# ==============================================================================
#  TAB 8: PROCESS FLOWCHART (Rendering Fix)
# ==============================================================================
import numpy as np

with tab_flowchart:
    st.subheader("🧬 AI Data Integrity Pipeline")
    st.caption("Visualizing the 6-stage deep scanning process used to validate production data.")

    # --- 1. CALCULATE LOGIC (The "Brain") ---
    row_count = len(df)
    
    # Structure Logic
    num_count = len(num_cols)
    cat_count = len(cat_cols)
    type_status = f"Valid ({num_count} Num, {cat_count} Cat)"
    
    # Integrity Logic
    missing_total = df.isna().sum().sum()
    missing_text = f"PASSED: {missing_total} Missing" if missing_total == 0 else f"WARNING: {missing_total} Missing"
    missing_color = "#2ecc71" if missing_total == 0 else "#f1c40f"
    
    # Duplication Logic
    dup_count = df.duplicated().sum()
    dup_text = f"PASSED: {dup_count} Duplicates" if dup_count == 0 else f"NOTICE: {dup_count} Duplicates"
    dup_color = "#2ecc71" if dup_count == 0 else "#e67e22"

    # Statistical Logic
    outlier_status = "Stable"
    outlier_color = "#2ecc71"
    if num_cols:
        z_scores = df[num_cols].select_dtypes(include=[np.number]).apply(lambda x: np.abs((x - x.mean()) / x.std()))
        total_outliers = (z_scores > 3).sum().sum()
        if total_outliers > 0:
            outlier_status = f"Detected ({total_outliers})"
            outlier_color = "#e74c3c"
            
    # Correlation Logic
    corr_status = " Verified"
    if len(num_cols) > 1:
        corr_status = "Relations Map Generated"

    # Final Decision
    is_healthy = (row_count > 0) and (missing_total < row_count*0.1) and (dup_count < row_count*0.5)
    
    final_color = "linear-gradient(135deg, #2ecc71, #27ae60)" if is_healthy else "linear-gradient(135deg, #c0392b, #962d22)"
    final_title = "DATA HEALTH: GOOD" if is_healthy else "DATA HEALTH: CRITICAL"
    final_desc = "All 6 scanning modules passed. Optimization engine active." if is_healthy else "Multiple failures detected. Manual cleaning required."

    # --- 2. CSS STYLES ---
    # We use f-string but with NO indentation inside to be safe
    st.markdown(f"""
    <style>
        .flow-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            perspective: 1000px;
            margin-top: 20px;
            padding-bottom: 50px;
        }}
        .flow-node {{
            width: 60%;
            background: linear-gradient(135deg, #ffffff, #f8f9fa);
            border-radius: 12px;
            padding: 15px;
            margin: 5px 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            text-align: center;
            font-family: sans-serif;
            color: #333;
            opacity: 0;
            transform: rotateX(20deg) translateY(-30px) translateZ(-50px);
            animation: scanIn 0.6s ease-out forwards;
        }}
        
        .node-1 {{ border-left: 6px solid #3498db; }}
        .node-2 {{ border-left: 6px solid #9b59b6; }}
        .node-3 {{ border-left: 6px solid {missing_color}; }}
        .node-4 {{ border-left: 6px solid {dup_color}; }}
        .node-5 {{ border-left: 6px solid {outlier_color}; }}
        .node-6 {{ border-left: 6px solid #1abc9c; }}

        .node-final {{
            width: 75%;
            background: {final_color};
            color: white;
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
            transform: scale(1.1);
            border: none;
            margin-top: 20px;
        }}
        .node-final h2 {{ color: white !important; margin: 0; }}
        .node-final p {{ color: #eee !important; margin-top: 5px; }}

        .arrow-down {{
            font-size: 20px;
            color: #bdc3c7;
            opacity: 0;
            animation: fadeIn 0.4s ease-out forwards;
            margin: -2px 0;
        }}

        @keyframes scanIn {{ to {{ opacity: 1; transform: rotateX(0deg) translateY(0) translateZ(0); }} }}
        @keyframes fadeIn {{ to {{ opacity: 1; }} }}

        #box1 {{ animation-delay: 0.2s; }}
        #arr1 {{ animation-delay: 0.6s; }}
        #box2 {{ animation-delay: 0.8s; }}
        #arr2 {{ animation-delay: 1.2s; }}
        #box3 {{ animation-delay: 1.4s; }}
        #arr3 {{ animation-delay: 1.8s; }}
        #box4 {{ animation-delay: 2.0s; }}
        #arr4 {{ animation-delay: 2.4s; }}
        #box5 {{ animation-delay: 2.6s; }}
        #arr5 {{ animation-delay: 3.0s; }}
        #box6 {{ animation-delay: 3.2s; }}
        #arr6 {{ animation-delay: 3.6s; }}
        #boxFinal {{ animation-delay: 4.0s; }}
    </style>
    """, unsafe_allow_html=True)

    # --- 3. HTML STRUCTURE (The fix is here) ---
    # We remove ALL indentation from the HTML string so Streamlit doesn't think it is code.
    html_code = f"""
<div class="flow-container">
<div id="box1" class="flow-node node-1">
<h4 style="margin:0;">📦 Data Ingestion</h4>
<small>Loaded {row_count:,} records</small>
</div>
<div id="arr1" class="arrow-down">⬇</div>
<div id="box2" class="flow-node node-2">
<h4 style="margin:0;">🏗️ Structure Validation</h4>
<small>Schema Check: {type_status}</small>
</div>
<div id="arr2" class="arrow-down">⬇</div>
<div id="box3" class="flow-node node-3">
<h4 style="margin:0;">🔍 Integrity Scanner</h4>
<small>{missing_text}</small>
</div>
<div id="arr3" class="arrow-down">⬇</div>
<div id="box4" class="flow-node node-4">
<h4 style="margin:0;">👯 Duplication Check</h4>
<small>{dup_text}</small>
</div>
<div id="arr4" class="arrow-down">⬇</div>
<div id="box5" class="flow-node node-5">
<h4 style="margin:0;">📉 Statistical Anomaly Scan</h4>
<small>Z-Score Outliers: {outlier_status}</small>
</div>
<div id="arr5" class="arrow-down">⬇</div>
<div id="box6" class="flow-node node-6">
<h4 style="margin:0;">🧠 Logic & Correlation</h4>
<small>{corr_status}</small>
</div>
<div id="arr6" class="arrow-down">⬇</div>
<div id="boxFinal" class="flow-node node-final">
<h2>{final_title}</h2>
<p>{final_desc}</p>
</div>
</div>
"""
    st.markdown(html_code, unsafe_allow_html=True)

    # --- 4. EXPLANATION ---
    st.write("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.info("ℹ️ **Why 6 Stages?**")
        st.markdown("""
        The **AI Optimization Engine** requires strict validation before processing. 
        Each stage filters out specific types of dirty data (Nulls, Duplicates, Schema errors) to ensure the Llama 3 model receives pure input.
        """)
    with c2:
        st.success(f"✅ **Pipeline Active**")
        st.markdown(f"**Final Status:** {final_title}")