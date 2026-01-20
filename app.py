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




    st.divider()
    
    st.subheader("📈 Digital Migration Signal")
    st.write("""
        This dual-axis chart compares **Address/Demographic** updates (blue) with **Biometric** updates (orange).
        - **Divergence:** If Address updates rise while Biometrics stay flat, it indicates high regional mobility (people moving for jobs/education).
        - **Convergence:** Shows standard identity maintenance cycles.
    """)
    
    # Use st.plotly_chart for the interactive Plotly figure
    fig_migration = viz.get_migration_signal(main_df)
    st.plotly_chart(fig_migration, use_container_width=True)
    
    st.info("💡 **Strategy:** High 'Blue Line' areas are candidates for enhanced digital self-service kiosks.")



with tab3:
    st.header("🌊 Operational Stress & Infrastructure Health")
    
    st.write("""
        This tab analyzes the **resilience** of the UIDAI ecosystem. 
        We look at how stable the centers are and identify regions where the infrastructure 
        is under 'Stress' due to high demand but low persistence.
    """)
    
    # Visual: Pincode Stability
    st.subheader("1. Center Persistence Analysis")
    fig_stability = viz.get_pincode_stability(main_df)
    st.pyplot(fig_stability)
    
    st.warning("💡 **Operational Insight:** Regions with a high count of low-persistence pincodes (left side) are likely being served by mobile vans or temporary camps.")
#############################
    
    st.divider()
    
    st.subheader("2. Temporal Stress: Peak Demand Windows")
    st.write("""
        This heatmap identifies **when** the system is under the most pressure. 
        Darker cells indicate periods where server bandwidth and staff availability 
        should be at their maximum.
    """)
    
    fig_heatmap = viz.get_operational_heatmap(main_df)
    st.pyplot(fig_heatmap)
    
    st.info("🎯 **Recommendation:** Use this data to schedule system maintenance during light-colored windows (low load) to minimize citizen impact.")



    # ... (Keep Heatmap and Stability visuals) ...
    
    st.divider()
    
    st.subheader("3. Workflow Consistency: Efficiency Benchmarking")
    st.write("""
        This boxplot measures the **quality of operations**. 
        - **Short boxes** indicate that all districts in that state perform similarly (Stable).
        - **Tall boxes** indicate a massive gap between the best and worst-performing districts (Inefficient).
    """)
    
    fig_efficiency = viz.get_efficiency_boxplot(main_df)
    st.pyplot(fig_efficiency)
    
    with st.expander("💡 Strategic Analysis"):
        st.write("""
            States appearing **above the red line** are your 'Efficiency Leaders'. 
            States with **long vertical lines (whiskers)** have districts that are 
            severely over-leveraged and at risk of hardware failure or staff burnout.
        """)





    # ... (Keep Heatmap and Boxplot visuals) ...
    
    st.divider()
    
    st.subheader("4. Infrastructure Stability Matrix")
    st.write("""
        This scatter plot maps every unique pincode into four strategic quadrants. 
        It identifies where we need to convert temporary camps into permanent 'Aadhaar Seva Kendras'.
    """)
    
    fig_matrix = viz.get_stability_matrix(main_df)
    st.pyplot(fig_matrix)
    
    st.error("🚨 **Strategic Risk:** Points in the top-left quadrant represent millions of citizens relying on infrastructure that could vanish tomorrow.")



   
    
    # Visual: Ridgeline Load
    st.subheader("1. Seasonal Workload 'Fingerprints'")
    with st.spinner("Calculating density distributions..."):
        fig_ridge = viz.get_ridgeline_load(main_df)
        st.pyplot(fig_ridge)
    
    st.success("""
        **Strategic Takeaway:** - **Sharp Peaks:** Highly predictable demand. Efficient for standard staffing.
        - **Wide Bases:** Volatile demand. Requires 'Buffer' capacity or mobile kits on standby.
        - **Red Zones:** These days represent 'Black Swan' events that overwhelm standard infrastructure.
    """)


with tab4:
    # ... (Keep the Ridgeline Load Plot here) ...

    st.divider()

    st.subheader("2. Strategic Priority Matrix (Treemap)")
    st.write("""
        This interactive treemap helps prioritize **budget allocation**.
        - **Box Size:** Total Enrolment (The 'Impact'—how many people live there).
        - **Box Color (Redder):** Higher Infrastructure Stress (The 'Urgency').
    """)
    
    # Use plotly_chart for interactive treemaps
    fig_tree = viz.get_priority_treemap(main_df)
    st.plotly_chart(fig_tree, use_container_width=True)
    
    st.info("""
        💡 **How to use:** Click on a State to drill down into its Districts. 
        Large Red blocks represent districts that need immediate budget for permanent Aadhaar Seva Kendras.
    """)




    # ... (Keep previous visuals like Ridgeline and Treemap) ...

    st.divider()

    st.subheader("3. Strategic Risk Profiling")
    st.write("""
        This Risk Matrix correlates **Operational Stress** with the **Enrolment Gap**. 
        Districts in the **Red Zone** are high-impact areas where demand is exceeding supply 
        and the citizen gap is widest.
    """)
    
    fig_risk = viz.get_district_risk_scatter(main_df)
    st.plotly_chart(fig_risk, use_container_width=True)
    
    st.warning("""
        **Executive Summary:** - **X-Axis (Stress):** High values mean centers are overworked.
        - **Y-Axis (Gap):** High values mean large numbers of 5-17 year olds are missing Aadhaar.
        - **Bubble Size:** Represents the total workload of that district.
    """)



    
    # ... (Previous Strategy visuals) ...

    st.divider()

    st.subheader("4. Ecosystem Balance: Multi-Factor Analysis")
    st.write("""
        The **Radar Chart** provides a balanced scorecard of the ecosystem. 
        A larger area indicates a more mature and stable Aadhaar infrastructure.
    """)
    
    # Use the filtered main_df so the radar reflects the sidebar selection
    fig_radar = viz.get_performance_radar(main_df, state_filter)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.info("""
        **Metric Breakdown:**
        - **Maturity:** High volume of historical enrolments.
        - **Digital Adoption:** High ratio of demographic updates over biometric.
        - **Compliance:** Success in closing the 5-17 age group gap.
    """)


    
    # ... (previous visuals) ...
    
    st.divider()
    
    st.subheader("3. District-Level Risk Analysis")
    st.write("""
        This matrix identifies districts where the system is failing both in **Efficiency** (High Stress) 
        and **Inclusion** (High Compliance Gap). 
    """)
    
    fig_risk = viz.get_district_risk_scatter(main_df)
    st.plotly_chart(fig_risk, use_container_width=True, key="risk_bubble_chart_unique")
    
    st.info("""
        **How to prioritize:** - **Upper-Right (Red):** Critical need for permanent centers and massive enrolment camps.
        - **Upper-Left:** High missing population, but infrastructure is underutilized (Needs awareness camps).
        - **Lower-Right:** System is very busy with updates, but enrolment is complete (Needs more update kiosks).
    """)



    
    # ... (previous visuals) ...

    st.divider()

    st.subheader("5. The Goldilocks Frontier (Operational Efficiency)")
    st.write("""
        This 2D Density Map identifies the **Efficiency Frontier**. 
        We are looking for centers that move out of 'Fragile Hotspots' and 'Dormant' zones 
        into the green dashed 'Goldilocks' zone.
    """)
    
    fig_frontier = viz.get_frontier_density_map(main_df)
    st.pyplot(fig_frontier)
    
    st.success("""
        **Strategic Mission:** Move resources from 'Dormant' centers (bottom right) 
        to reinforce 'Fragile Hotspots' (top left) to achieve a balanced national ecosystem.
    """)



    # ... (all previous Strategy visuals) ...

    st.divider()

    st.subheader("6. The Aadhaar Lifecycle: System Flow Analysis")
    st.write("""
        This Sankey diagram visualizes the **transition of the ecosystem**. 
        As the nation reaches saturation, the flow shifts from 'New Enrolments' (Left) 
        to 'Maintenance & Updates' (Right).
    """)
    
    fig_sankey = viz.get_system_flow_sankey(main_df)
    st.plotly_chart(fig_sankey, use_container_width=True)
    
    st.success("""
        **Final Strategic Insight:** The future of UIDAI is no longer about 
        onboarding—it's about managing the flow of 'Mandatory Cycles' and 
        streamlining 'Voluntary Corrections' through digital channels.
    """)