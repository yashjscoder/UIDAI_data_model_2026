import streamlit as st
import pandas as pd
from modules import visualizer as viz

st.set_page_config(page_title="UIDAI Ecosystem Intelligence", layout="wide")

# --- CUSTOM CSS FOR PREMIUM TABS ---
st.markdown("""
    <style>
    /* 1. Target the Tab Container */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    /* 2. Style each individual Tab Button */
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #f0f2f6; /* Light gray background for unselected */
        border-radius: 8px 8px 0px 0px; /* Boxed shape */
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        border: 1px solid #dfe1e6;
    }

    /* 3. Style the Active/Selected Tab */
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        border-bottom: 3px solid #ff4b4b !important; /* UIDAI Red accent */
        font-weight: bold !important;
    }

    /* 4. Bold and Capitalize Tab Text */
    .stTabs [data-baseweb="tab"] p {
        font-size: 16px;
        font-weight: 800; /* Extra Bold */
        text-transform: uppercase; /* Capital Letters */
        color: #31333f;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA LOADING SECTION ---
@st.cache_data
def load_data():
    data = pd.read_csv('data/df_master.csv', low_memory=False)
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
    return data

df = load_data()

# SIDEBAR FILTERS
st.sidebar.title("🛡️ UIDAI Control Room by UIDAI_10362")
state_filter = st.sidebar.selectbox("Select State", ["National Overview"] + list(df['state'].unique()))

# DYNAMIC DATA FILTERING
main_df = df if state_filter == "National Overview" else df[df['state'] == state_filter]

# TABS FOR YOUR 20+ VISUALS
# Updated Naming Convention
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 GROWTH BASED VISUAL ANALYSIS", 
    "👥 DEMOGRAPHIC INTELLIGENCE & GAP", 
    "🌊 OPERATIONAL STRESS & RESILIENCE", 
    "🛡️ STRATEGIC INTERVENTION & RISK"
])

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
# tabs with enhanced names and AI smart guides (smart layer).
with tab1:
    st.header("GROWTH BASED VISUAL ANALYSIS") 
    
    # --- 1. Leaderboard ---
    st.subheader("1. State-wise Enrolment Concentration Leaderboard")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("leaderboard")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")
    st.pyplot(viz.get_state_leaderboard(main_df))
    st.divider() 

    # --- 2. Service Mix ---
    st.subheader("2. National Service Mix: Onboarding vs Lifecycle Maintenance")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("service_mix")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")
    st.pyplot(viz.get_service_split(main_df, state_filter))
    st.divider()

    # --- 3. Localized Demand ---
    st.subheader("3. State-to-District Demand Hotspots")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("localized_demand")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")
    st.pyplot(viz.get_localized_demand(main_df))
    st.divider()
    
    # --- 4. Operation Breakdown ---
    st.subheader("4. State-wise Operational Load Split: Enrolment vs Updates")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("state_service_split")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")
    st.pyplot(viz.get_state_service_split(main_df))
    st.divider()

    # --- 5. Strategic Zones ---
    st.subheader("5. Strategic Intervention Districts: Risk/Attention Bands")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("strategic_districts")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")
    st.pyplot(viz.get_strategic_districts(main_df))
    st.success("Target these districts for specialized resource deployment.")
    st.divider()

    # --- 6. Trends ---
    st.subheader("6. National System Heartbeat: Daily Service Volume Trend (Log Scale)")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("timeseries")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")
    st.pyplot(viz.get_service_timeseries(main_df, state_filter))
    st.divider()

    # --- 7. Saturation ---
    st.subheader("7. Enrolment Saturation Trajectory (Cumulative Adoption Curve)")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("saturation")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")
    st.pyplot(viz.get_saturation_curve(main_df, state_filter))

with tab2:
    st.header("Demographic Intelligence")
    
    # --- 1. Demographic Mix + AI Popover ---
    st.subheader("1. Aadhaar Lifecycle Composition: Age-Group Distribution")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("demographic_mix")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")

    fig_age, age_totals, hidden_adults = viz.get_demographic_mix(main_df)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Child (0-5)", f"{age_totals['age_0_5']:,.0f}")
    m2.metric("Youth (5-17)", f"{age_totals['age_5_17']:,.0f}")
    m3.metric("Adult (18+)", f"{age_totals['age_18_greater']:,.0f}")
    
    st.divider()
    
    col_chart, col_text = st.columns([2, 1])
    with col_chart:
        st.pyplot(fig_age)
        
    with col_text:
        # Keeping your important Integrity Signal as requested
        st.subheader("🔍 Data Integrity Signal")
        st.write("Beyond new enrolments, we detected significant adult activity in maintenance logs:")
        st.info(f"**Total Adult Updates:** {hidden_adults:,.0f}")
        st.write("""
            **Analysis:** While new enrolments are dominated by children/youth, 
            the 'Adult' segment is primarily engaged through biometric and 
            demographic updates (Re-verification).
        """)

    st.divider()
    
    # --- 2. Compliance Risk + AI Popover ---
    st.subheader("2. Regional Compliance & Enrolment Gap Distribution")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("compliance_violin")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")

    st.write("This violin plot reveals the **nature of the enrolment gap** in the critical 5-17 age group.")
    st.pyplot(viz.get_compliance_violin(main_df, state_filter))
    
    with st.expander("📝 How to read this chart"):
        st.write("""
            - **Horizontal Red Line:** The goal (Zero missing enrolments).
            - **The 'Fat' part of the violin:** Where most districts in that state sit.
            - **The 'Tail':** Outlier districts that are either doing very well or very poorly.
        """)

    st.divider()
    
    # --- 3. Update DNA + AI Popover ---
    st.subheader("3. Update DNA: Biometric vs. Demographic Engagement")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("update_dna")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")

    st.write("This grouped analysis identifies the **Mode of Engagement**.")
    st.pyplot(viz.get_update_dna(main_df, state_filter))
    st.warning("⚡ **Insight:** High Biometric demand areas need more physical machines and iris scanners.")

    st.divider()
    
    # --- 4. Digital Migration Signal + AI Popover ---
    st.subheader("4. Socio-Economic Mobility & Digital Migration Signal")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("migration_signal")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")

    st.write("This dual-axis chart compares **Address/Demographic** updates with **Biometric** updates.")
    st.plotly_chart(viz.get_migration_signal(main_df), use_container_width=True)
    st.info("💡 **Strategy:** High 'Blue Line' areas are candidates for enhanced digital self-service kiosks.")

with tab3:
    st.header("🌊 Operational Stress & Infrastructure Health")
    
    st.write("This tab analyzes the **resilience** of the UIDAI ecosystem.")
    
    # --- 1. Pincode Stability ---
    st.subheader("1. Center Persistence Analysis")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("pincode_stability")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")

    st.pyplot(viz.get_pincode_stability(main_df))
    st.warning("💡 **Operational Insight:** Regions with a high count of low-persistence pincodes (left side) are likely being served by mobile vans or temporary camps.")
    
    st.divider()
    
    # --- 2. Heatmap ---
    st.subheader("2. Temporal Stress: Peak Demand Windows")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("operational_heatmap")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")

    fig_heatmap = viz.get_operational_heatmap(main_df)
    st.pyplot(fig_heatmap)
    st.info("🎯 **Recommendation:** Use this data to schedule system maintenance during light-colored windows (low load) to minimize citizen impact.")

    st.divider()
    
    # --- 3. Efficiency Boxplot ---
    st.subheader("3. Workflow Consistency: Efficiency Benchmarking")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("efficiency_boxplot")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")

    st.pyplot(viz.get_efficiency_boxplot(main_df))
    with st.expander("💡 Strategic Analysis"):
        st.write("""
            States appearing **above the red line** are your 'Efficiency Leaders'. 
            States with **long vertical lines (whiskers)** have districts that are 
            severely over-leveraged.
        """)

    st.divider()
    
    # --- 4. Stability Matrix ---
    st.subheader("4. Infrastructure Stability Matrix")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("stability_matrix")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")

    st.pyplot(viz.get_stability_matrix(main_df))
    st.error("🚨 **Strategic Risk:** Points in the top-left quadrant represent millions of citizens relying on infrastructure that could vanish tomorrow.")

    st.divider()
    
    # --- 5. Ridgeline ---
    st.subheader("5. Seasonal Workload 'Fingerprints'")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("ridgeline_load")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")

    with st.spinner("Calculating density distributions..."):
        st.pyplot(viz.get_ridgeline_load(main_df))
    
    st.success("""
        **Strategic Takeaway:** - **Sharp Peaks:** Predictable demand. 
        - **Wide Bases:** Volatile demand. 
        - **Red Zones:** Black Swan events that overwhelm infrastructure.
    """)


with tab4:
    st.header("🎯 STRATEGIC INTERVENTION & RISK")

    # --- 1. Priority Treemap ---
    st.subheader("1. Strategic Priority Matrix (Treemap)")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("priority_treemap")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")

    fig_tree = viz.get_priority_treemap(main_df)
    st.plotly_chart(fig_tree, use_container_width=True)
    st.info("💡 **How to use:** Large Red blocks represent districts that need immediate budget for permanent centers.")

    st.divider()

    # --- 2. Risk Profiling ---
    st.subheader("2. Strategic Risk Profiling")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("risk_profiling")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")

    fig_risk = viz.get_district_risk_scatter(main_df)
    st.plotly_chart(fig_risk, use_container_width=True, key="risk_bubble_tab4")
    st.warning("**Executive Summary:** Bubbles in the Red Zone (Upper-Right) are high-impact areas where demand exceeds supply.")

    st.divider()

    # --- 3. Performance Radar ---
    st.subheader("3. Ecosystem Balance: Multi-Factor Analysis")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("performance_radar")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")

    fig_radar = viz.get_performance_radar(main_df, state_filter)
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # --- 4. Goldilocks Frontier ---
    st.subheader("4. The Goldilocks Frontier (Operational Efficiency)")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("goldilocks_frontier")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")

    fig_frontier = viz.get_frontier_density_map(main_df)
    st.pyplot(fig_frontier)
    st.success("🎯 **Strategic Mission:** Move resources from 'Dormant' centers to reinforce 'Fragile Hotspots'.")

    st.divider()

    # --- 5. System Flow (Sankey) ---
    st.subheader("5. The Aadhaar Lifecycle: System Flow Analysis")
    with st.popover("✨ Ask AI: About this Visual"):
        guide = viz.get_ai_guide("system_flow")
        st.markdown(f"### 🤖 {guide['title']}")
        st.write(f"**What it is:** {guide['what_it_is']}")
        st.write(f"**How to Read:** {guide['how_to_read']}")
        st.info(f"**Strategic Impact:** {guide['impact']}")

    fig_sankey = viz.get_system_flow_sankey(main_df)
    st.plotly_chart(fig_sankey, use_container_width=True)
    st.success("**Final Strategic Insight:** The future of UIDAI is about managing flows of 'Mandatory Cycles'.")
