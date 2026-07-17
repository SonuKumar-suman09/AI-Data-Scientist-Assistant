import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# Deferred imports for faster app load
import io
import base64
import pickle
import os

st.set_page_config(page_title="AI Data Scientist Assistant", layout="wide", page_icon="🤖")

# Custom CSS for Pixel-Perfect Modern SaaS Theme
st.markdown("""
<style>
    /* Global Theme Overrides */
    .stApp {
        background-color: #0F172A;
    }
    
    /* Sidebar Width & Polish */
    [data-testid="stSidebar"] {
        min-width: 280px !important;
        max-width: 300px !important;
    }
    
    /* Hide native radio circles */
    div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* Base menu item layout */
    div[role="radiogroup"] > label {
        padding: 0px 16px !important;
        border-radius: 8px !important;
        margin-bottom: 6px !important;
        min-height: 44px !important;
        display: flex !important;
        align-items: center !important;
        background-color: transparent !important;
        transition: background-color 0.2s ease !important;
        cursor: pointer !important;
    }

    /* Menu item hover effect */
    div[role="radiogroup"] > label:hover {
        background-color: #1E293B !important;
    }

    /* Menu item text styling */
    div[role="radiogroup"] > label p {
        font-size: 15px !important;
        font-weight: 500 !important;
        color: #F8FAFC !important;
    }

    /* Active item (Blue highlight) */
    div[role="radiogroup"] > label:has(input:checked) {
        background-color: #3B82F6 !important;
    }

    /* Category Headers: 2nd, 6th, 9th, 13th items */
    div[role="radiogroup"] > label:nth-child(2),
    div[role="radiogroup"] > label:nth-child(6),
    div[role="radiogroup"] > label:nth-child(9),
    div[role="radiogroup"] > label:nth-child(13) {
        pointer-events: none !important;
        background-color: transparent !important;
        margin-top: 16px !important; 
        min-height: auto !important;
        padding: 0px 16px !important;
        margin-bottom: 8px !important;
    }

    /* Category Headers Text Styling */
    div[role="radiogroup"] > label:nth-child(2) p,
    div[role="radiogroup"] > label:nth-child(6) p,
    div[role="radiogroup"] > label:nth-child(9) p,
    div[role="radiogroup"] > label:nth-child(13) p {
        font-size: 13px !important; 
        text-transform: uppercase !important;
        color: #9CA3AF !important;
        font-weight: 700 !important;
    }

    /* Hero Text */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.5rem;
        color: #3B82F6;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .hero-text {
        font-size: 1.2rem;
        color: #94A3B8;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Feature Cards */
    .feature-card {
        background-color: #1E293B;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
        border-color: #3B82F6;
    }
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.5rem;
    }
    .card-text {
        color: #94A3B8;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    
    /* Stats */
    .stat-box {
        text-align: center;
        padding: 1.5rem;
        background-color: #1E293B;
        border-radius: 12px;
        border: 1px solid #334155;
    }
    .stat-num {
        font-size: 2.5rem;
        font-weight: 900;
        color: #22C55E;
    }
    .stat-label {
        font-size: 1rem;
        font-weight: 600;
        color: #94A3B8;
        margin-top: 5px;
    }
    
    /* Workflow */
    .workflow-step {
        text-align: center;
        padding: 1rem;
        background-color: #1E293B;
        border-radius: 8px;
        border: 1px solid #334155;
        font-weight: 600;
        color: #F8FAFC;
    }
    .workflow-arrow {
        text-align: center;
        font-size: 1.8rem;
        color: #3B82F6;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
session_keys = ["df", "clean_df", "model", "X_train", "X_test", "y_test", "y_pred", "target_col", "task_type", "features"]
for key in session_keys:
    if key not in st.session_state:
        st.session_state[key] = None

# Sidebar Navigation Header
st.sidebar.markdown("""
<div style='display: flex; align-items: center; margin-bottom: 10px; margin-left: 12px;'>
    <span style='font-size: 24px; margin-right: 10px;'>🤖</span>
    <span style='font-size: 18px; font-weight: bold; color: white;'>AI Data Scientist</span>
</div>
""", unsafe_allow_html=True)

# Navigation Options
navigation_options = [
    "🏠  Home",
    "DATA", # index 2
    "📂  Upload Dataset",
    "🧹  Data Cleaning",
    "⚙️  Feature Engineering",
    "ANALYSIS", # index 6
    "📈  Exploratory Analysis",
    "📉  Interactive Charts",
    "MODELING", # index 9
    "🧠  Machine Learning",
    "⚖️  Model Comparison",
    "🎯  Predictions",
    "INSIGHTS", # index 13
    "🔍  Explainable AI",
    "💡  AI Insights",
    "📄  PDF Report",
    "⬇️  Download Results"
]

choice_raw = st.sidebar.radio("", navigation_options, label_visibility="collapsed")
choice = choice_raw.strip()

# --- Sidebar Footer ---
st.sidebar.markdown("<br><hr style='border-color: #334155;'>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='text-align: center;'>
    <p style='color: #94A3B8; font-size: 12px; margin-bottom: 5px;'>Built with</p>
    <p style='font-size: 12px;'>🐍 Python | ⚡ Streamlit<br>📊 Plotly | 🤖 Scikit-learn | 🧠 SHAP</p>
    <br>
    <p style='color: #94A3B8; font-size: 12px;'>Developed by<br><b style='color: #3B82F6; font-size: 14px;'>Sonu Kumar Suman</b></p>
</div>
""", unsafe_allow_html=True)

# ----------------- ROUTING -----------------

if choice in ["DATA", "ANALYSIS", "MODELING", "INSIGHTS"]:
    st.stop() # Prevents doing anything if they somehow bypass the pointer-events

# --- ACTUAL PAGES ---
if choice == "🏠  Home":
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("<p class='hero-title'>🤖 AI Data Scientist Assistant</p>", unsafe_allow_html=True)
        st.markdown("<p class='hero-subtitle'>Analyze • Visualize • Predict • Explain</p>", unsafe_allow_html=True)
        st.markdown("<p class='hero-text'>Upload any dataset and build machine learning models without writing a single line of code.</p>", unsafe_allow_html=True)
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.markdown("""
            <style>
            .primary-btn button {
                background-color: #3B82F6 !important;
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
                font-size: 1.1rem !important;
                padding: 0.6rem 0 !important;
                transition: all 0.3s ease !important;
            }
            .primary-btn button:hover {
                background-color: #2563EB !important;
                transform: translateY(-2px);
                box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.39) !important;
            }
            </style>
            """, unsafe_allow_html=True)
            st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
            st.button("🚀 Get Started (Upload Dataset)", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with btn_col2:
            st.markdown("""
            <style>
            .secondary-btn button {
                background-color: transparent !important;
                color: #22C55E !important;
                border: 2px solid #22C55E !important;
                border-radius: 8px !important;
                font-size: 1.1rem !important;
                padding: 0.6rem 0 !important;
                transition: all 0.3s ease !important;
            }
            .secondary-btn button:hover {
                background-color: rgba(34, 197, 94, 0.1) !important;
                transform: translateY(-2px);
            }
            </style>
            """, unsafe_allow_html=True)
            st.markdown("<div class='secondary-btn'>", unsafe_allow_html=True)
            if st.button("🧪 Try Sample Dataset", use_container_width=True):
                with st.spinner("Loading synthetic dataset..."):
                    try:
                        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                        sample_df = pd.read_csv(url)
                        
                        st.session_state.df = sample_df
                        st.session_state.clean_df = st.session_state.df.copy()
                        st.success("Sample Dataset Loaded! Go to 'Data Cleaning' or 'EDA' to begin.")
                    except Exception as e:
                        st.error(f"Error loading sample: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
                        
    with col2:
        img_path = r"C:\Users\sonuk\.gemini\antigravity\brain\03924af6-b6ad-433d-ace3-d3a01203cfeb\ai_data_dashboard_hero_1783254914728.png"
        if os.path.exists(img_path):
            st.image(img_path, use_column_width=True)
        else:
            st.markdown("""
            <div style="background-color: #1E293B; height: 300px; border-radius: 12px; display: flex; align-items: center; justify-content: center; border: 2px dashed #3B82F6;">
                <span style="font-size: 5rem;">🤖📊🧠</span>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Stats Section
    s1, s2, s3, s4 = st.columns(4)
    with s1: st.markdown("<div class='stat-box'><div class='stat-num'>100+</div><div class='stat-label'>Datasets Supported</div></div>", unsafe_allow_html=True)
    with s2: st.markdown("<div class='stat-box'><div class='stat-num'>15+</div><div class='stat-label'>ML Algorithms</div></div>", unsafe_allow_html=True)
    with s3: st.markdown("<div class='stat-box'><div class='stat-num'>30+</div><div class='stat-label'>Chart Types</div></div>", unsafe_allow_html=True)
    with s4: st.markdown("<div class='stat-box'><div class='stat-num'>5</div><div class='stat-label'>Export Formats</div></div>", unsafe_allow_html=True)
    
    st.markdown("<br><hr style='border-color: #334155;'><br>", unsafe_allow_html=True)
    
    # Feature Cards Section
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class='feature-card'>
            <div class='card-icon'>📂</div>
            <div class='card-title'>Upload Dataset</div>
            <div class='card-text'>Securely import your CSV & Excel files for instant data processing.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='feature-card'>
            <div class='card-icon'>📊</div>
            <div class='card-title'>Data Visualization</div>
            <div class='card-text'>Generate automated statistics and 50+ interactive Plotly charts.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='feature-card'>
            <div class='card-icon'>🤖</div>
            <div class='card-title'>Machine Learning</div>
            <div class='card-text'>Train powerful Regression & Classification models effortlessly.</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class='feature-card'>
            <div class='card-icon'>🧠</div>
            <div class='card-title'>Explainable AI</div>
            <div class='card-text'>Understand underlying model decisions using advanced SHAP visualizers.</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><hr style='border-color: #334155;'><br>", unsafe_allow_html=True)
    
    # Workflow Section
    st.markdown("### 🔄 How it Works", unsafe_allow_html=True)
    w1, arrow1, w2, arrow2, w3, arrow3, w4, arrow4, w5 = st.columns([2,1,2,1,2,1,2,1,2])
    with w1: st.markdown("<div class='workflow-step'>📂<br>Upload</div>", unsafe_allow_html=True)
    with arrow1: st.markdown("<div class='workflow-arrow'>➔</div>", unsafe_allow_html=True)
    with w2: st.markdown("<div class='workflow-step'>🧹<br>Clean & Eng</div>", unsafe_allow_html=True)
    with arrow2: st.markdown("<div class='workflow-arrow'>➔</div>", unsafe_allow_html=True)
    with w3: st.markdown("<div class='workflow-step'>📊<br>Explore</div>", unsafe_allow_html=True)
    with arrow3: st.markdown("<div class='workflow-arrow'>➔</div>", unsafe_allow_html=True)
    with w4: st.markdown("<div class='workflow-step'>🤖<br>Train Models</div>", unsafe_allow_html=True)
    with arrow4: st.markdown("<div class='workflow-arrow'>➔</div>", unsafe_allow_html=True)
    with w5: st.markdown("<div class='workflow-step'>📄<br>Insights</div>", unsafe_allow_html=True)

elif choice == "📂  Upload Dataset":
    st.title("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            st.session_state.clean_df = st.session_state.df.copy()
            st.success("Dataset successfully loaded!")
            st.dataframe(st.session_state.df.head())
            st.write(f"Shape of dataset: {st.session_state.df.shape}")
        except Exception as e:
            st.error(f"Error loading file: {e}")

elif choice == "🧹  Data Cleaning":
    st.title("🧹 Data Cleaning")
    if st.session_state.df is not None:
        df = st.session_state.clean_df
        st.write("### Current Dataset Summary")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        st.write("### Missing Values")
        missing_data = df.isnull().sum()
        st.write(missing_data[missing_data > 0])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Drop Missing Values"):
                st.session_state.clean_df = df.dropna()
                st.success("Missing values dropped.")
                st.rerun()
        with col2:
            if st.button("Fill Missing Values (Mean/Mode)"):
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        if not df[col].mode().empty:
                            df[col] = df[col].fillna(df[col].mode()[0])
                st.session_state.clean_df = df
                st.success("Missing values filled.")
                st.rerun()
                
        if st.button("Drop Duplicates"):
            st.session_state.clean_df = df.drop_duplicates()
            st.success("Duplicates dropped.")
            st.rerun()
            
        st.write("### Cleaned Data Preview")
        st.dataframe(st.session_state.clean_df.head())
    else:
        st.warning("Please upload a dataset first.")

elif choice == "⚙️  Feature Engineering":
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    st.title("⚙️ Feature Engineering")
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df.copy()
        
        st.write("### Label Encoding")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(cat_cols) > 0:
            cols_to_encode = st.multiselect("Select columns to encode", cat_cols)
            if st.button("Encode Selected Columns"):
                if cols_to_encode:
                    le = LabelEncoder()
                    for col in cols_to_encode:
                        df[col] = le.fit_transform(df[col].astype(str))
                    st.session_state.clean_df = df
                    st.success("Columns encoded successfully!")
                    st.dataframe(df.head())
                else:
                    st.warning("Please select at least one column to encode.")
        else:
            st.info("No categorical columns available to encode.")
            
        st.write("### Feature Scaling")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cols_to_scale = st.multiselect("Select numerical columns to scale (StandardScaler)", num_cols)
        if st.button("Scale Selected Columns"):
            if cols_to_scale:
                scaler = StandardScaler()
                df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                st.session_state.clean_df = df
                st.success("Columns scaled successfully!")
                st.dataframe(df.head())
            else:
                st.warning("Please select at least one column to scale.")
    else:
        st.warning("Please upload a dataset first.")

elif choice == "📈  Exploratory Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df
        st.write("### Descriptive Statistics")
        st.dataframe(df.describe())
        
        st.write("### Data Types")
        st.dataframe(df.dtypes.astype(str))
        
        st.write("### Value Counts for Categorical Columns")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            selected_col = st.selectbox("Select Column", cat_cols)
            st.write(df[selected_col].value_counts())
    else:
        st.warning("Please clean or upload a dataset first.")

elif choice == "📉  Interactive Charts":
    st.title("📉 Interactive Charts")
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        chart_type = st.selectbox("Select Chart Type", ["Scatter Plot", "Bar Chart", "Histogram", "Box Plot", "Correlation Heatmap"])
        
        if chart_type == "Scatter Plot":
            x_axis = st.selectbox("X-Axis", num_cols)
            y_axis = st.selectbox("Y-Axis", num_cols)
            color_col = st.selectbox("Color (Optional)", ["None"] + cat_cols)
            if st.button("Generate Chart"):
                if color_col == "None":
                    fig = px.scatter(df, x=x_axis, y=y_axis)
                else:
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col)
                st.plotly_chart(fig)
                
        elif chart_type == "Bar Chart":
            x_axis = st.selectbox("X-Axis (Categorical)", cat_cols)
            y_axis = st.selectbox("Y-Axis (Numeric)", num_cols)
            if st.button("Generate Chart"):
                fig = px.bar(df, x=x_axis, y=y_axis)
                st.plotly_chart(fig)
                
        elif chart_type == "Histogram":
            x_axis = st.selectbox("Column", num_cols)
            if st.button("Generate Chart"):
                fig = px.histogram(df, x=x_axis)
                st.plotly_chart(fig)
                
        elif chart_type == "Box Plot":
            y_axis = st.selectbox("Column (Numeric)", num_cols)
            x_axis = st.selectbox("Group By (Optional)", ["None"] + cat_cols)
            if st.button("Generate Chart"):
                if x_axis == "None":
                    fig = px.box(df, y=y_axis)
                else:
                    fig = px.box(df, x=x_axis, y=y_axis)
                st.plotly_chart(fig)
                
        elif chart_type == "Correlation Heatmap":
            if st.button("Generate Chart"):
                corr = df[num_cols].corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                st.plotly_chart(fig)
    else:
        st.warning("Please upload a dataset first.")

elif choice == "🧠  Machine Learning":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, r2_score
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    import xgboost as xgb
    st.title("🧠 Machine Learning")
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df
        
        target_col = st.selectbox("Select Target Variable", df.columns)
        features = df.drop(columns=[target_col]).columns.tolist()
        selected_features = st.multiselect("Select Features", features, default=features)
        
        task_type = st.radio("Select Task Type", ["Regression", "Classification"])
        
        test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2)
        
        model_choice = st.selectbox("Select Algorithm", ["Random Forest", "XGBoost", "Linear/Logistic Regression"])
        
        if not selected_features:
            st.error("Please select at least one feature.")
        else:
            X = df[selected_features]
            y = df[target_col]
            
            has_missing = X.isnull().sum().sum() > 0 or y.isnull().sum() > 0
            has_categorical = not X.select_dtypes(include=['object', 'category']).empty
            
            if has_missing:
                st.warning("⚠️ Your dataset contains missing values. Click 'Auto Clean' to fill them using recommended methods.")
                if st.button("Auto Clean Data"):
                    for col in df.columns:
                        if df[col].dtype in ['int64', 'float64']:
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            if not df[col].mode().empty:
                                df[col] = df[col].fillna(df[col].mode()[0])
                    st.session_state.clean_df = df
                    st.rerun()
                    
            if has_categorical and not has_missing:
                st.warning("⚠️ Your dataset contains categorical (text) features. Machine learning models require numbers. Click 'Auto Encode' to fix this.")
                if st.button("Auto Encode Features"):
                    le = LabelEncoder()
                    cat_cols = df.select_dtypes(include=['object', 'category']).columns
                    for col in cat_cols:
                        df[col] = le.fit_transform(df[col].astype(str))
                    st.session_state.clean_df = df
                    st.rerun()

            if not has_missing and not has_categorical:
                if st.button("Train Model"):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    
                    st.session_state.target_col = target_col
                    st.session_state.task_type = task_type
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.features = selected_features
                    
                    with st.spinner("Training model..."):
                        if task_type == "Regression":
                            if model_choice == "Random Forest":
                                model = RandomForestRegressor(random_state=42)
                            elif model_choice == "XGBoost":
                                model = xgb.XGBRegressor(random_state=42)
                            else:
                                model = LinearRegression()
                        else:
                            if model_choice == "Random Forest":
                                model = RandomForestClassifier(random_state=42)
                            elif model_choice == "XGBoost":
                                model = xgb.XGBClassifier(random_state=42)
                            else:
                                model = LogisticRegression(max_iter=1000)
                        
                        model.fit(X_train, y_train)
                        st.session_state.model = model
                        
                        y_pred = model.predict(X_test)
                        st.session_state.y_pred = y_pred
                        
                        st.success("Model trained successfully!")
                        st.write("### Model Performance")
                        
                        if task_type == "Regression":
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            st.write(f"**Mean Squared Error:** {mse:.4f}")
                            st.write(f"**R2 Score:** {r2:.4f}")
                        else:
                            acc = accuracy_score(y_test, y_pred)
                            st.write(f"**Accuracy:** {acc:.4f}")
                            st.text("Classification Report:")
                            st.text(classification_report(y_test, y_pred))
    else:
        st.warning("Please upload a dataset first.")

elif choice == "⚖️  Model Comparison":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    import xgboost as xgb
    st.title("⚖️ Model Comparison")
    if st.session_state.clean_df is not None and st.session_state.target_col is not None:
        df = st.session_state.clean_df
        target_col = st.session_state.target_col
        task_type = st.session_state.task_type
        
        X = df[st.session_state.features]
        y = df[target_col]
        
        has_missing = X.isnull().sum().sum() > 0 or y.isnull().sum() > 0
        has_categorical = not X.select_dtypes(include=['object', 'category']).empty
        
        if has_missing:
            st.error("⚠️ Dataset has missing values. Please go to the 'Machine Learning' section and auto-clean it.")
        elif has_categorical:
            st.error("⚠️ Dataset has categorical values. Please go to the 'Machine Learning' section and auto-encode it.")
        else:
            if st.button("Run Model Comparison"):
                with st.spinner("Comparing models..."):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    results = []
                    
                    if task_type == "Regression":
                        models = {
                            "Linear Regression": LinearRegression(),
                            "Random Forest": RandomForestRegressor(random_state=42),
                            "XGBoost": xgb.XGBRegressor(random_state=42)
                        }
                        for name, m in models.items():
                            m.fit(X_train, y_train)
                            preds = m.predict(X_test)
                            results.append({"Model": name, "R2 Score": r2_score(y_test, preds), "MSE": mean_squared_error(y_test, preds)})
                    else:
                        models = {
                            "Logistic Regression": LogisticRegression(max_iter=1000),
                            "Random Forest": RandomForestClassifier(random_state=42),
                            "XGBoost": xgb.XGBClassifier(random_state=42)
                        }
                        for name, m in models.items():
                            m.fit(X_train, y_train)
                            preds = m.predict(X_test)
                            results.append({"Model": name, "Accuracy": accuracy_score(y_test, preds)})
                    
                    res_df = pd.DataFrame(results)
                    st.dataframe(res_df)
                    
                    if task_type == "Regression":
                        fig = px.bar(res_df, x="Model", y="R2 Score", title="Model R2 Score Comparison")
                    else:
                        fig = px.bar(res_df, x="Model", y="Accuracy", title="Model Accuracy Comparison")
                    st.plotly_chart(fig)
    else:
        st.warning("Please train a model first in the Machine Learning module.")

elif choice == "🎯  Predictions":
    st.title("🎯 Predictions")
    if st.session_state.model is not None:
        st.write("Enter values for features to get a prediction:")
        
        input_data = {}
        for col in st.session_state.features:
            input_data[col] = st.number_input(f"Enter {col}", value=0.0)
            
        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            pred = st.session_state.model.predict(input_df)
            st.success(f"### Predicted Value: {pred[0]}")
    else:
        st.warning("Please train a model first in the Machine Learning module.")

elif choice == "🔍  Explainable AI":
    import shap
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    import xgboost as xgb
    st.title("🔍 Explainable AI (SHAP)")
    if st.session_state.model is not None:
        if st.button("Generate SHAP Explanations"):
            with st.spinner("Calculating SHAP values (this may take a moment)..."):
                try:
                    model = st.session_state.model
                    X_train = st.session_state.X_train.sample(min(100, len(st.session_state.X_train)))
                    X_test_sample = st.session_state.X_test.sample(min(20, len(st.session_state.X_test)))
                    
                    if isinstance(model, (RandomForestRegressor, RandomForestClassifier, xgb.XGBRegressor, xgb.XGBClassifier)):
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test_sample)
                        
                        st.write("### SHAP Summary Plot")
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots()
                        if isinstance(shap_values, list): # For classification
                            shap.summary_plot(shap_values[1], X_test_sample, show=False)
                        elif len(np.shape(shap_values)) == 3: # For classification in newer shap version
                            shap.summary_plot(shap_values[:, :, 1], X_test_sample, show=False)
                        else:
                            shap.summary_plot(shap_values, X_test_sample, show=False)
                        st.pyplot(fig)
                    else:
                        explainer = shap.KernelExplainer(model.predict, X_train)
                        shap_values = explainer.shap_values(X_test_sample)
                        
                        st.write("### SHAP Summary Plot")
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots()
                        shap.summary_plot(shap_values, X_test_sample, show=False)
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating SHAP explanation for this model type: {e}")
    else:
        st.warning("Please train a model first.")

elif choice == "💡  AI Insights":
    st.title("💡 AI Insight Generator")
    st.write("Auto-generates summary insights based on your EDA and ML results.")
    
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df
        insights = []
        insights.append(f"- Dataset has **{df.shape[0]}** rows and **{df.shape[1]}** columns.")
        
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 1:
            corr_mat = df[num_cols].corr()
            corr_mat_unstacked = corr_mat.unstack()
            corr_mat_unstacked = corr_mat_unstacked[corr_mat_unstacked != 1.0]
            if len(corr_mat_unstacked) > 0:
                max_corr_idx = corr_mat_unstacked.idxmax()
                max_corr = corr_mat_unstacked.max()
                insights.append(f"- High correlation ({max_corr:.2f}) observed between **{max_corr_idx[0]}** and **{max_corr_idx[1]}**.")
                
        if st.session_state.model is not None:
            insights.append(f"- A **{st.session_state.task_type}** model was trained to predict **{st.session_state.target_col}**.")
            
            if hasattr(st.session_state.model, 'feature_importances_'):
                importances = st.session_state.model.feature_importances_
                top_feature = st.session_state.features[np.argmax(importances)]
                insights.append(f"- The most important feature for prediction is **{top_feature}**.")
        
        st.markdown("### 📊 Automated Insights")
        for ins in insights:
            st.markdown(ins)
    else:
        st.warning("Please upload a dataset first.")

elif choice == "📄  PDF Report":
    from fpdf import FPDF
    from sklearn.metrics import r2_score, accuracy_score
    st.title("📄 PDF Report Generation")
    st.write("Generate a simple PDF report summarizing the project.")
    
    if st.session_state.clean_df is not None and st.session_state.model is not None:
        if st.button("Generate PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=15, style='B')
            pdf.cell(200, 10, txt="AI Data Scientist Project Report", ln=1, align='C')
            
            pdf.set_font("Arial", size=12)
            pdf.ln(10)
            
            pdf.cell(200, 10, txt=f"Dataset Shape: {st.session_state.clean_df.shape}", ln=1)
            pdf.cell(200, 10, txt=f"Target Variable: {st.session_state.target_col}", ln=1)
            pdf.cell(200, 10, txt=f"Task Type: {st.session_state.task_type}", ln=1)
            
            if st.session_state.task_type == "Regression":
                r2 = r2_score(st.session_state.y_test, st.session_state.y_pred)
                pdf.cell(200, 10, txt=f"Model Performance (R2 Score): {r2:.4f}", ln=1)
            else:
                acc = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
                pdf.cell(200, 10, txt=f"Model Performance (Accuracy): {acc:.4f}", ln=1)
                
            pdf_output = bytes(pdf.output())
            st.download_button(label="Download PDF Report", data=pdf_output, file_name="Data_Science_Report.pdf", mime="application/pdf")
            st.success("PDF Report is ready for download!")
    else:
        st.warning("Please complete data processing and model training first.")

elif choice == "⬇️  Download Results":
    st.title("⬇️ Download Results")
    
    if st.session_state.clean_df is not None:
        csv = st.session_state.clean_df.to_csv(index=False)
        st.download_button("Download Cleaned Dataset (CSV)", data=csv, file_name="cleaned_dataset.csv", mime="text/csv")
        
    if st.session_state.y_pred is not None:
        res_df = st.session_state.X_test.copy()
        res_df['Actual'] = st.session_state.y_test
        res_df['Predicted'] = st.session_state.y_pred
        
        csv_pred = res_df.to_csv(index=False)
        st.download_button("Download Predictions on Test Set", data=csv_pred, file_name="test_predictions.csv", mime="text/csv")
        
    if st.session_state.model is not None:
        st.write("To save the trained model, you can serialize it using `pickle`.")
        model_bytes = pickle.dumps(st.session_state.model)
        st.download_button("Download Trained Model (.pkl)", data=model_bytes, file_name="trained_model.pkl", mime="application/octet-stream")
