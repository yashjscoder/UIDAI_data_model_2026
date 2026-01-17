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
    
    # --- ROW 1: Leaders & Split ---
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("1. State-wise Enrolment Leaderboard")
        st.pyplot(viz.get_state_leaderboard(main_df))
    with col2:
        st.subheader("2. Service Mix")
        st.pyplot(viz.get_service_split(main_df, state_filter))

    st.divider()

    # --- ROW 2: Localized & Regional Split ---
    st.subheader("3. Localized District-Level Demand")
    st.pyplot(viz.get_localized_demand(main_df))
    
    st.subheader("4. State-wise Operation Breakdown")
    st.pyplot(viz.get_state_service_split(main_df))

    st.divider()

    # --- ROW 3: Strategy ---
    st.subheader("🎯 Strategic Intervention Zones")
    st.pyplot(viz.get_strategic_districts(main_df))
    st.success("Target these districts for specialized resource deployment.")

    # New Time Series Visual
    st.subheader("1. System Heartbeat (Temporal Trends)")
    st.pyplot(viz.get_service_timeseries(main_df, state_filter))
    st.info("💡 Note: The Y-axis is on a Logarithmic scale to compare New Enrolments and Updates effectively.")

    st.divider()

    
    # --- NEW VISUAL: Saturation Curve ---
    st.subheader("2. Market Saturation (Cumulative Delivery)")
    st.pyplot(viz.get_saturation_curve(main_df, state_filter))
    st.success(f"Successfully processed {main_df['total_enrolment'].sum():,.0f} new enrolments in this view.")

    st.divider()


with tab2:
    st.header("Demographic Intelligence")
    
    # Call the new diagnostic function
    fig_age, age_totals, hidden_adults = viz.get_demographic_mix(main_df)
    
    # Create Columns for Top Stats (The Integrity Check)
    m1, m2, m3 = st.columns(3)
    m1.metric("Child (0-5)", f"{age_totals['age_0_5']:,.0f}")
    m2.metric("Youth (5-17)", f"{age_totals['age_5_17']:,.0f}")
    m3.metric("Adult (18+)", f"{age_totals['age_18_greater']:,.0f}")
    
    st.divider()
    
    # Display the Donut Chart and the "Hidden" Insight
    col_chart, col_text = st.columns([2, 1])
    
    with col_chart:
        st.pyplot(fig_age)
        
    with col_text:
        st.subheader("🔍 Data Integrity Signal")
        st.write("Beyond new enrolments, we detected significant adult activity in maintenance logs:")
        st.info(f"**Total Adult Updates:** {hidden_adults:,.0f}")
        st.write("""
            **Analysis:** While new enrolments are dominated by children/youth, 
            the 'Adult' segment is primarily engaged through biometric and 
            demographic updates (Re-verification).
        """)

        st.divider()
    
    st.subheader("📍 Compliance Risk Distribution")
    st.write("""
        This violin plot reveals the **nature of the enrolment gap** in the critical 5-17 age group. 
        It helps decide if we need a state-wide policy or targeted local camps.
    """)
    
    fig_violin = viz.get_compliance_violin(main_df, state_filter)
    st.pyplot(fig_violin)
    
    with st.expander("📝 How to read this chart"):
        st.write("""
            - **Horizontal Red Line:** The goal (Zero missing enrolments).
            - **The 'Fat' part of the violin:** Where most districts in that state sit.
            - **The 'Tail':** Outlier districts that are either doing very well or very poorly.
        """)




        st.divider()
    
    st.subheader("🧬 Update DNA: Service Type Demand")
    st.write("""
        This grouped analysis identifies the **Mode of Engagement**. 
        - **High Biometric %**: Requires physical center presence (Hardware heavy).
        - **High Demographic %**: Potential for digital/online self-service adoption.
    """)
    
    fig_dna = viz.get_update_dna(main_df, state_filter)
    st.pyplot(fig_dna)
    
    st.warning("⚡ **Insight:** High Biometric demand areas need more physical machines and iris scanners.")



