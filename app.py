import streamlit as st
import pandas as pd
from modules import visualizer as viz

st.set_page_config(page_title="UIDAI Ecosystem Intelligence", layout="wide")

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv('data/df_master.csv')
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# SIDEBAR FILTERS
st.sidebar.title("🛡️ UIDAI Control Room")
state_filter = st.sidebar.selectbox("Select State", ["National Overview"] + list(df['state'].unique()))

# DYNAMIC DATA FILTERING
main_df = df if state_filter == "National Overview" else df[df['state'] == state_filter]

# TABS FOR YOUR 20+ VISUALS
tab1, tab2, tab3, tab4 = st.tabs(["📊 Growth", "👥 Demographics", "🌊 Operational Stress", "🛡️ Strategy"])

st.sidebar.info(f"Currently viewing: {state_filter}")


# --- RAW DATA PREVIEW SECTION ---
with st.sidebar.expander("📂 View Raw/Unprocessed Data"):
    st.write("Current files found in project directory:")
    
    # Call the visualizer logic
    raw_info = viz.get_raw_data_summary()
    
    for item in raw_info:
        st.markdown(f"**File:** `{item['name']}`")
        if st.checkbox(f"Show details for {item['name']}", key=item['name']):
            st.write("Sample Rows:")
            st.dataframe(item['sample'])
            
            st.write("Column Types:")
            st.table(item['dtypes'])
            
            st.write("Missing Values Count:")
            st.table(item['missing'])
        st.divider()

# --- THE REST OF YOUR TABS START HERE ---
# tab1, tab2, tab3, tab4 = st.tabs(...)
with tab1:
    st.header("Growth & Adoption Dynamics")
    
    # --- VISUAL 1 ---
    st.subheader("1. State-wise Enrolment Leaderboard")
    fig1 = viz.get_state_leaderboard(main_df)
    st.pyplot(fig1)
    
    st.divider() # This adds a clean horizontal line
    
    # --- VISUAL 2 ---
    st.subheader("2. Localized District-Level Demand")
    with st.spinner("Generating granular district report..."):
        fig2 = viz.get_localized_demand(main_df)
        st.pyplot(fig2)
    
    st.info("💡 Insight: These hotspots represent areas where UIDAI should prioritize setting up 'Mega Centers' to handle high-volume localized demand.")