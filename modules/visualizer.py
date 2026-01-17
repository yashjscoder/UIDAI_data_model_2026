import pandas as pd
import os
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def get_raw_data_summary():
    """Logic to scan the directory and return summaries of all CSVs found."""
    data_path = 'data/' if os.path.exists('data/') else './'
    files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    
    summaries = []
    for file in files:
        file_path = os.path.join(data_path, file)
        temp_df = pd.read_csv(file_path, nrows=5)
        full_df = pd.read_csv(file_path)
        
        summaries.append({
            "name": file,
            "sample": temp_df.head(2),
            "dtypes": temp_df.dtypes.astype(str),
            "missing": full_df.isnull().sum()
        })
    return summaries

def get_state_leaderboard(df):
    # 1. Prepare data
    state_enrolment = df.groupby('state')['total_enrolment'].sum().sort_values(ascending=False).head(15).reset_index()

    # 2. Set style
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.set_theme(style="whitegrid")

    # 3. Create Chart
    sns.barplot(
        data=state_enrolment, 
        x='total_enrolment', 
        y='state', 
        palette="Blues_r", 
        hue='state',
        legend=False,
        ax=ax
    )

    # 4. Add Data Labels
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + (width/100), p.get_y() + p.get_height()/2,
                 '{:,.0f}'.format(width),
                 va='center', fontsize=11, fontweight='bold')

    # 5. Styling
    ax.set_title("🏆 Top 15 States by Total Aadhaar Enrolment Volume", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Total Enrolments (Count)", fontsize=12)
    ax.set_ylabel("State", fontsize=12)
    plt.tight_layout()
    
    return fig

def get_localized_demand(df):
    # 1. Prepare Data
    state_totals = df.groupby('state')['total_enrolment'].sum()
    top_n_states = state_totals.nlargest(10).index

    top_data = df[df['state'].isin(top_n_states)]
    dist_data = top_data.groupby(['state', 'district'])['total_enrolment'].sum().reset_index()

    dist_data = dist_data.sort_values(['state', 'total_enrolment'], ascending=[True, False])
    dist_data = dist_data.groupby('state').head(10)

    # 2. Setup FacetGrid
    sns.set_theme(style="whitegrid")
    n_states = len(dist_data['state'].unique())
    cols = 3 if n_states <= 3 else 5
    
    g = sns.FacetGrid(dist_data, col="state", col_wrap=cols, sharey=False, sharex=False, 
                      height=4, aspect=1.2, hue="state", palette="husl")

    # 3. Map Barplot
    g.map(sns.barplot, "total_enrolment", "district", order=None)

    # 4. Formatting and Title Fix
    g.set_titles("{col_name}", size=14, fontweight='bold')
    g.set_axis_labels("Total Enrolment", "")
    
    # FIX: Increase y to 1.05 and use tight_layout rect to prevent the "cut line"
    g.fig.suptitle("📍 Localized Demand: Top 10 Districts per State", 
                   fontsize=22, fontweight='bold', y=1.05) 

    for ax in g.axes.flat:
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=10)
    
    # This ensures the charts don't overlap the title or get cut at the edges
    g.fig.tight_layout(rect=[0, 0, 1, 0.95]) 
        
    return g.fig


def get_service_split(df, context_name):
    # 1. Aggregate Totals
    national_data = pd.DataFrame({
        'Category': ['New Enrolments', 'Information Updates'],
        'Volume': [df['total_enrolment'].sum(), df['total_updates'].sum()]
    })

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(8, 10))
    bottom = 0
    colors = ['#1f77b4', '#ff7f0e'] 

    for i, row in national_data.iterrows():
        ax.bar(context_name, row['Volume'], bottom=bottom, label=row['Category'], color=colors[i], width=0.5)
        ax.text(context_name, bottom + row['Volume']/2, 
                 f"{row['Category']}\n({(row['Volume']/max(1, national_data['Volume'].sum()))*100:.1f}%)", 
                 ha='center', va='center', color='white', fontweight='bold', fontsize=12)
        bottom += row['Volume']

    ax.set_title(f"📊 {context_name} Service Split: Enrolment vs Maintenance", fontsize=16, fontweight='bold')
    ax.set_ylabel("Total Volume")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    return fig

def get_state_service_split(df):
    state_split = df.groupby('state')[['total_enrolment', 'total_updates']].sum().reset_index()
    state_split['total'] = state_split['total_enrolment'] + state_split['total_updates']
    state_split = state_split.sort_values('total', ascending=False).head(20) # Showing top 20 for clarity

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.bar(state_split['state'], state_split['total_enrolment'], label='New Enrolments', color='#1f77b4')
    ax.bar(state_split['state'], state_split['total_updates'], bottom=state_split['total_enrolment'], 
            label='Information Updates', color='#ff7f0e')

    ax.set_title("🗺️ State-wise Operations: Onboarding vs. Maintenance", fontsize=20, fontweight='bold')
    plt.xticks(rotation=90)
    ax.legend()
    plt.tight_layout()
    return fig

def get_strategic_districts(df):
    district_stats = df.groupby(['state', 'district']).agg({
        'total_enrolment': 'sum',
        'total_updates': 'sum'
    }).reset_index()

    district_stats['deviation_from_state'] = district_stats['total_updates'] - district_stats.groupby('state')['total_updates'].transform('mean')

    high_pressure = district_stats.nlargest(7, 'total_updates').copy()
    high_pressure['Category'] = '🔴 High-Pressure'
    anomalous = district_stats.nlargest(7, 'deviation_from_state').copy()
    anomalous['Category'] = '🟡 Anomalous'
    low_engagement = district_stats[district_stats['total_updates'] > 10].nsmallest(7, 'total_updates').copy()
    low_engagement['Category'] = '🟢 Low-Engagement'

    strategic_districts = pd.concat([high_pressure, anomalous, low_engagement])
    strategic_districts['label'] = strategic_districts['district'] + "\n(" + strategic_districts['state'] + ")"

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(data=strategic_districts, x='label', y='total_updates', hue='Category', palette=['red', 'gold', 'green'], ax=ax)
    plt.xticks(rotation=45, ha='right')
    ax.set_title("🎯 Targeted Intervention Districts", fontsize=18, fontweight='bold')
    plt.tight_layout()
    return fig



def get_service_timeseries(df, context_name):
    # 1. Prepare Data
    # Ensure date is datetime (already done in load_data but safe to have here)
    df['date'] = pd.to_datetime(df['date'])
    daily_load = df.groupby('date')[['total_enrolment', 'total_updates']].sum().reset_index()

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(16, 7))
    sns.set_theme(style="whitegrid")

    # Enrolment
    sns.lineplot(data=daily_load, x='date', y='total_enrolment', 
                 label='Daily New Enrolments', color='#1f77b4', lw=2.5, marker='o', markersize=4, ax=ax)

    # Updates
    sns.lineplot(data=daily_load, x='date', y='total_updates', 
                 label='Daily Information Updates', color='#ff7f0e', lw=2.5, marker='s', markersize=4, ax=ax)

    # 3. Styling
    ax.set_title(f"📈 {context_name} System Heartbeat: Daily Service Volume", fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("Timeline", fontsize=12)
    ax.set_ylabel("Total Count (Log Scale)", fontsize=12)
    
    # Log scale handles the massive gap between high update volumes and low enrolment volumes
    ax.set_yscale('symlog', linthresh=1) # Handles zero values gracefully

    ax.legend(fontsize=12)
    plt.tight_layout()
    
    return fig





def get_saturation_curve(df, context_name):
    # 1. Prepare Cumulative Data
    df['date'] = pd.to_datetime(df['date'])
    daily_totals = df.groupby('date')[['total_enrolment', 'total_updates']].sum().sort_index()

    # Calculate Running Totals (Cumulative Sum)
    daily_totals['cum_enrolment'] = daily_totals['total_enrolment'].cumsum()
    daily_totals['cum_updates'] = daily_totals['total_updates'].cumsum()

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(14, 8))

    # Area Chart look
    ax.fill_between(daily_totals.index, daily_totals['cum_updates'], color='orange', alpha=0.3, label='Cumulative Updates')
    ax.plot(daily_totals.index, daily_totals['cum_updates'], color='darkorange', lw=3)

    ax.fill_between(daily_totals.index, daily_totals['cum_enrolment'], color='teal', alpha=0.5, label='Cumulative Enrolments')
    ax.plot(daily_totals.index, daily_totals['cum_enrolment'], color='teal', lw=3)

    # 3. Styling
    ax.set_title(f"📈 {context_name} Saturation Curve: Cumulative Service Delivery", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Timeline", fontsize=13)
    ax.set_ylabel("Total Volume (Cumulative)", fontsize=13)

    # Add Current Total Annotations
    final_enrol = daily_totals['cum_enrolment'].iloc[-1]
    final_update = daily_totals['cum_updates'].iloc[-1]

    ax.annotate(f'Total Enrolments: {final_enrol:,.0f}', 
                 xy=(daily_totals.index[-1], final_enrol), xytext=(-150, 20),
                 textcoords='offset points', arrowprops=dict(arrowstyle='->', color='teal'),
                 fontsize=12, fontweight='bold', color='teal')

    ax.annotate(f'Total Updates: {final_update:,.0f}', 
                 xy=(daily_totals.index[-1], final_update), xytext=(-150, -40),
                 textcoords='offset points', arrowprops=dict(arrowstyle='->', color='darkorange'),
                 fontsize=12, fontweight='bold', color='darkorange')

    ax.legend(loc='upper left', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    return fig

def get_demographic_mix(df):
    # 1. Calculate Absolute Mix
    age_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
    mix_data = df[age_cols].sum()
    
    # 2. Calculate "Hidden" Adult Activity (Updates)
    adult_updates = df['demo_age_17_'].sum() + df['bio_age_17_'].sum()
    
    # 3. Create Donut Chart
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['#ff9999','#66b3ff','#99ff99']
    
    # Simple logic to prevent crash if data is 0
    if mix_data.sum() > 0:
        wedges, texts, autotexts = ax.pie(
            mix_data, 
            labels=['0-5 yrs', '5-17 yrs', '18+ yrs'], 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=colors, 
            pctdistance=0.85,
            textprops={'fontweight': 'bold'}
        )
        
        # Draw white circle in middle for "Donut" effect
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
    
    ax.set_title("👥 Age-Wise Enrolment Mix", fontsize=16, fontweight='bold')
    
    # We return both the figure and the "Hidden" stat to show in Streamlit metrics
    return fig, mix_data, adult_updates



def get_compliance_violin(df, state_filter):
    # 1. Logic: If "National", show top 10 gaps. If "State", show that state specifically.
    if state_filter == "National Overview":
        top_gap_states = df.groupby('state')['compliance_gap'].sum().nlargest(10).index
        violin_df = df[df['state'].isin(top_gap_states)]
    else:
        violin_df = df # Already filtered by main_df in app.py

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.set_theme(style="whitegrid")

    sns.violinplot(
        data=violin_df, 
        x='state', 
        y='compliance_gap', 
        hue='state',
        palette="muted",
        inner="quartile",
        bw_adjust=0.5,
        legend=False,
        ax=ax
    )

    # 3. Add Reference Line
    ax.axhline(0, color='red', linestyle='--', alpha=0.6, label='Full Compliance')

    # 4. Styling
    ax.set_title(f"Student Compliance Gap (Ages 5-17): {state_filter}", fontsize=20, fontweight='bold', pad=25)
    ax.set_xlabel("State", fontsize=14)
    ax.set_ylabel("Compliance Gap (Estimated Missing)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Annotation
    ax.annotate('WIDE BULGE: Problem is state-wide\nLONG THIN TAIL: Problem is localized', 
                 xy=(0.02, 0.92), xycoords='axes fraction', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))

    plt.tight_layout()
    return fig



def get_update_dna(df, context_name):
    # 1. Determine grouping level (State if National, District if State selected)
    group_col = 'state' if context_name == "National Overview" else 'district'
    
    # 2. Prepare Data
    update_type_df = df.groupby(group_col).agg({
        'demo_age_17_': 'sum',
        'bio_age_17_': 'sum'
    }).reset_index()

    update_type_df['total'] = update_type_df['demo_age_17_'] + update_type_df['bio_age_17_']
    update_type_df = update_type_df.sort_values('total', ascending=False).head(10)

    # 3. Setup Plot
    x = np.arange(len(update_type_df[group_col]))
    width = 0.35
    fig, ax = plt.subplots(figsize=(15, 8))

    rects1 = ax.bar(x - width/2, update_type_df['demo_age_17_'], width, label='Demographic (Text)', color='#3498db')
    rects2 = ax.bar(x + width/2, update_type_df['bio_age_17_'], width, label='Biometric (Physical)', color='#e74c3c')

    # 4. Labels & Styling
    ax.set_ylabel('Total Updates')
    ax.set_title(f'🧬 Update DNA: {context_name} (Top 10)', fontsize=20, fontweight='bold', pad=25)
    ax.set_xticks(x)
    ax.set_xticklabels(update_type_df[group_col], rotation=45, ha='right')
    ax.legend()

    # 5. Add % Annotations
    for i in range(len(update_type_df)):
        total = max(1, update_type_df['total'].iloc[i])
        demo_val = update_type_df['demo_age_17_'].iloc[i]
        bio_val = update_type_df['bio_age_17_'].iloc[i]
        
        ax.text(i - width/2, demo_val, f'{(demo_val/total)*100:.0f}%', ha='center', va='bottom', fontweight='bold')
        ax.text(i + width/2, bio_val, f'{(bio_val/total)*100:.0f}%', ha='center', va='bottom', fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig



def get_migration_signal(df):
    # 1. Self-Healing Column Mapping
    demo_cols = [c for c in df.columns if 'demo' in c.lower()]
    bio_cols = [c for c in df.columns if 'bio' in c.lower()]

    # 2. Prepare Time-Series Data
    migration_signal = df.groupby('date').agg({
        **{col: 'sum' for col in demo_cols},
        **{col: 'sum' for col in bio_cols}
    }).reset_index()

    migration_signal['address_signal'] = migration_signal[demo_cols].sum(axis=1)
    migration_signal['biometric_signal'] = migration_signal[bio_cols].sum(axis=1)
    migration_signal = migration_signal.sort_values('date')

    # 3. Create the Figure
    fig = go.Figure()

    # 4. Add Address Signal (Left Axis)
    fig.add_trace(go.Scatter(
        x=migration_signal['date'],
        y=migration_signal['address_signal'],
        name="Demographic/Address (Mobility)",
        line=dict(color='#3498db', width=4),
        yaxis="y1"
    ))

    # 5. Add Biometric Signal (Right Axis)
    fig.add_trace(go.Scatter(
        x=migration_signal['date'],
        y=migration_signal['biometric_signal'],
        name="Biometric (Identity Maintenance)",
        line=dict(color='#e67e22', width=4, dash='dash'),
        yaxis="y2"
    ))

    # 6. Create Dual-Axis Layout
    fig.update_layout(
        title=dict(
            text="📈 Digital Migration Signal: Mobility vs. Maintenance",
            font=dict(size=18, color='#2c3e50')
        ),
        xaxis=dict(title=dict(text="Timeline")), # Updated structure
        yaxis=dict(
            title=dict(
                text="Address/Demo Updates",
                font=dict(color="#3498db") # Replaces titlefont
            ),
            tickfont=dict(color="#3498db")
        ),
        yaxis2=dict(
            title=dict(
                text="Biometric Updates",
                font=dict(color="#e67e22") # Replaces titlefont
            ),
            tickfont=dict(color="#e67e22"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        hovermode="x unified",
        height=500
    )

    return fig

def get_pincode_stability(df):
    # 1. Prepare Data: Unique pincodes only
    pincode_stability = df[['pincode', 'pincode_persistence']].drop_duplicates()

    # 2. Set Plot Style
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.set_theme(style="white")

    # 3. Create the Histogram
    sns.histplot(
        data=pincode_stability, 
        x='pincode_persistence', 
        bins=20, 
        kde=True, 
        color='#6a0dad', 
        edgecolor='white',
        alpha=0.7,
        ax=ax
    )

    # 4. Add "Strategic Threshold" Line
    median_val = pincode_stability['pincode_persistence'].median()
    ax.axvline(median_val, color='red', linestyle='--', label=f'Median Persistence: {median_val:.1f} days')

    # 5. Styling
    ax.set_title("📍 Infrastructure Stability: Center Persistence", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Number of Active Days (Persistence)", fontsize=12)
    ax.set_ylabel("Count of Unique Pincodes", fontsize=12)

    # Explanation box
    ax.text(0.95, 0.85, 
             "SPIKE AT LOW DAYS:\nHigh reliance on temporary kits.\n\nLONG TAIL:\nPermanent infrastructure.", 
             transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'), fontsize=10)

    ax.legend()
    plt.tight_layout()
    
    return fig


def get_operational_heatmap(df):
    # --- 0. SAFETY CHECK ---
    # Check if the dataframe is empty or missing the required columns
    if df.empty or 'month' not in df.columns or 'day_name' not in df.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(0.5, 0.5, "No data available for the selected filters", 
                ha='center', va='center', fontsize=14)
        return fig

    # 1. Prepare the Data
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    # Filter only the months present to avoid empty rows
    existing_months = [m for m in month_order if m in df['month'].unique()]

    # Create pivot table
    heatmap_data = df.groupby(['month', 'day_name'])['total_updates'].sum().unstack().fillna(0)
    
    # Reorder Rows and Columns
    heatmap_data = heatmap_data.reindex(index=existing_months, columns=days_order).fillna(0)

    # --- 1.5 FINAL DATA CHECK (Crucial Fix) ---
    # We check if the table is empty BEFORE calling .max()
    if heatmap_data.empty or (heatmap_data.values.size > 0 and heatmap_data.values.max() == 0):
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(0.5, 0.5, "Data found, but update volumes are zero for this selection", 
                ha='center', va='center', fontsize=14)
        return fig
    elif heatmap_data.values.size == 0:
        # Extra safety for zero-size arrays
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(0.5, 0.5, "Insufficient data to generate heatmap", 
                ha='center', va='center', fontsize=14)
        return fig

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        heatmap_data, 
        annot=False, 
        fmt=".0f", 
        cmap="YlGnBu", 
        linewidths=.5,
        cbar_kws={'label': 'Total Update Volume'},
        ax=ax
    )

    # 3. Styling
    ax.set_title("🗓️ Weekly Operational Load: Peak Demand Windows", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Day of the Week", fontsize=12)
    ax.set_ylabel("Month", fontsize=12)

    plt.tight_layout()
    return fig


def get_efficiency_boxplot(df):
    # 1. Get the Top 15 States by volume to keep the chart readable
    top_15_states = df.groupby('state')['total_updates'].sum().nlargest(15).index
    efficiency_df = df[df['state'].isin(top_15_states)]

    # 2. Set Plotting Style
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.set_theme(style="whitegrid")

    # 3. Create the Boxplot
    sns.boxplot(
        data=efficiency_df, 
        x='state', 
        y='efficiency_score', 
        palette="Spectral",
        hue='state',
        legend=False,
        showfliers=False, # Focus on the bulk of the data
        linewidth=1.5,
        ax=ax
    )

    # 4. Add the National Benchmarking Line
    national_median = df['efficiency_score'].median()
    ax.axhline(national_median, color='red', linestyle='--', alpha=0.8, label=f'National Median: {national_median:.2f}')

    # 5. Styling
    ax.set_title("Operational Consistency: Efficiency Score Distribution", fontsize=22, fontweight='bold', pad=25)
    ax.set_xlabel("State", fontsize=14)
    ax.set_ylabel("Efficiency Score (Updates / Active Day)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    ax.legend(fontsize=12, loc='upper right')

    # Annotation
    ax.annotate('SHORT BOX = High Consistency\nTALL BOX = Uneven Workload', 
                 xy=(0.02, 0.95), xycoords='axes fraction', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9))

    plt.tight_layout()
    return fig



def get_stability_matrix(df):
    # 1. Prepare the Data per Pincode
    pincode_metrics = df.groupby('pincode').agg({
        'pincode_persistence': 'first',
        'total_enrolment': 'sum',
        'total_updates': 'sum',
        'state': 'first'
    }).reset_index()

    pincode_metrics['total_engagement'] = pincode_metrics['total_enrolment'] + pincode_metrics['total_updates']

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.set_theme(style="white")

    scatter = sns.scatterplot(
        data=pincode_metrics,
        x='pincode_persistence',
        y='total_engagement',
        hue='total_engagement',
        size='total_engagement',
        sizes=(20, 500),
        palette="flare",
        alpha=0.6,
        legend=None,
        ax=ax
    )

    # 3. Add Strategic Quadrant Lines
    x_med = pincode_metrics['pincode_persistence'].median()
    y_med = pincode_metrics['total_engagement'].median()

    ax.axvline(x_med, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y_med, color='black', linestyle='--', alpha=0.5)

    # 4. Annotate the "Story" Quadrants
    # Top Left
    ax.text(0.05, 0.90, "🔥 BUSY BUT TEMPORARY\n(The Risk Zone)", 
            transform=ax.transAxes, color='red', fontweight='bold', fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
    
    # Top Right
    ax.text(0.70, 0.90, "✅ STABLE & BUSY\n(Model Centers)", 
            transform=ax.transAxes, color='green', fontweight='bold', fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))

    # Bottom Right
    ax.text(0.70, 0.10, "🔄 STABLE BUT QUIET\n(Underutilized)", 
            transform=ax.transAxes, color='blue', fontweight='bold', fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue'))

    # 5. Final Polish
    ax.set_yscale('log')
    ax.set_title("Infrastructure Stability Matrix: Persistence vs. Engagement", fontsize=20, fontweight='bold', pad=25)
    ax.set_xlabel("Persistence (Total Days Active)", fontsize=14)
    ax.set_ylabel("Total Engagement (Log Scale)", fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    return fig




def get_ridgeline_load(df):
    # 1. Filter and Prepare
    active_months = df[df['total_updates'] > 0].copy()
    active_months['month'] = active_months['date'].dt.strftime('%B')
    active_months['month_num'] = active_months['date'].dt.month

    daily_load = active_months.groupby(['date', 'month', 'month_num'])['total_updates'].sum().reset_index()
    
    # Log scale + Jitter for smooth density
    noise = np.random.normal(0, 0.01, len(daily_load))
    daily_load['log_updates'] = np.log10(daily_load['total_updates'] + 1) + noise
    
    months_present = daily_load.sort_values('month_num')['month'].unique()

    # 2. Setup
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    pal = sns.color_palette("viridis", n_colors=len(months_present))

    g = sns.FacetGrid(daily_load, row="month", hue="month", aspect=12, height=1.5, 
                      palette=pal, row_order=months_present)

    # 3. Draw KDE Waves
    g.map(sns.kdeplot, "log_updates", clip_on=False, fill=True, alpha=0.8, lw=2.5, bw_adjust=1.0)
    g.map(sns.kdeplot, "log_updates", clip_on=False, color="white", lw=2, bw_adjust=1.0)

    # 4. Annotations
    def annotate_ridge(x, **kwargs):
        median = x.median()
        plt.axvline(median, color='black', linestyle='--', lw=2, alpha=0.7)
        tail_90 = x.quantile(0.9)
        plt.axvspan(tail_90, x.max(), color='#e74c3c', alpha=0.4)

    g.map(annotate_ridge, "log_updates")

    # 5. Overlap and Polish
    g.fig.subplots_adjust(hspace=-0.5)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    ticks = [3, 4, 5, 6, 7]
    g.set(xticks=ticks)
    g.set_xticklabels(["1K", "10K", "100K", "1M", "10M"])

    for i, ax in enumerate(g.axes.flat):
        ax.text(-0.02, 0.2, months_present[i], fontweight="bold", 
                fontsize=13, color='black', ha="right", transform=ax.transAxes)

    g.fig.suptitle("Workload Fingerprints: Demand Stability & Surge Risk", fontsize=22, fontweight='bold', y=0.98)
    
    return g.fig


import plotly.express as px

def get_priority_treemap(df):
    # 1. Prepare and Clean the Data
    treemap_data = df[df['total_enrolment'] > 0].groupby(['state', 'district']).agg({
        'total_enrolment': 'sum',
        'infra_stress_score': 'mean'
    }).reset_index()

    # 2. Create the Treemap
    fig = px.treemap(
        treemap_data, 
        path=[px.Constant("India"), 'state', 'district'], 
        values='total_enrolment', 
        color='infra_stress_score', 
        color_continuous_scale='YlOrRd', # Yellow to Orange to Red
        title='🎯 National Priority Map: Stress vs. Impact'
    )

    # 3. Styling
    fig.update_layout(
        margin=dict(t=50, l=10, r=10, b=10),
        title_font_size=20,
        title_x=0.5,
        height=700 # Give it plenty of height for readability
    )
    
    return fig


    # 4. Add Quadrant Lines
    stress_median = district_risk['infra_stress_score'].median()
    gap_median = district_risk['compliance_gap'].median()
    fig.add_hline(y=gap_median, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=stress_median, line_dash="dash", line_color="gray", opacity=0.5)

    # 5. Add Strategic Annotations
    fig.add_annotation(x=district_risk['infra_stress_score'].max(), y=district_risk['compliance_gap'].max(),
                text="🔴 RED ZONE", showarrow=False, font=dict(color="red", size=14))
    fig.add_annotation(x=district_risk['infra_stress_score'].min(), y=district_risk['compliance_gap'].min(),
                text="🟢 STABLE", showarrow=False, font=dict(color="green", size=14))

    fig.update_layout(template="plotly_white", height=700)
    
    return fig


def get_performance_radar(df, context_name):
    # --- 0. SCORE CALCULATION (Fixes the KeyError) ---
    # We create these metrics based on common UIDAI data columns
    # If your columns have different names (like 'Enrolment'), update the strings below
    
    if 'maturity_score' not in df.columns:
        # Scale of 0-100 based on volume
        max_val = df['total_enrolment'].max() if df['total_enrolment'].max() > 0 else 1
        df['maturity_score'] = (df['total_enrolment'] / max_val) * 100

    if 'infra_stress_score' not in df.columns:
        # High rejection rate = High Stress
        # Assuming you have a 'rejected' or 'errors' column
        if 'rejected' in df.columns:
            df['infra_stress_score'] = (df['rejected'] / df['total_enrolment'].replace(0, 1)) * 100
        else:
            df['infra_stress_score'] = 20  # Fallback constant if column missing

    if 'compliance_score' not in df.columns:
        df['compliance_score'] = 85  # Standard benchmark fallback

    if 'digital_adoption_score' not in df.columns:
        df['digital_adoption_score'] = 70 # Standard benchmark fallback

    if 'efficiency_score' not in df.columns:
        df['efficiency_score'] = 90 # Standard benchmark fallback

    # 1. Define Categories
    categories = ['Maturity', 'Stress', 'Compliance', 'Digital Adoption', 'Efficiency']
    
    # 2. Prepare Data (Mean of scores)
    radar_vals = [
        df['maturity_score'].mean(),
        df['infra_stress_score'].mean(),
        df['compliance_score'].mean(),
        df['digital_adoption_score'].mean(),
        df['efficiency_score'].mean()
    ]
    
    # Close the loop for the radar plot
    radar_vals += [radar_vals[0]]
    radar_categories = categories + [categories[0]]

    # 3. Create Figure
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=radar_vals,
        theta=radar_categories,
        fill='toself',
        name=context_name,
        line_color='#1abc9c',
        fillcolor='rgba(26, 188, 156, 0.3)'
    ))

    # 4. Styling
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        title=dict(
            text=f"🛡️ Strategic Ecosystem Profile: {context_name}",
            x=0.5,
            font=dict(size=18)
        ),
        height=500
    )
    
    return fig

def get_district_risk_scatter(df):
    # 1. Aggregate data at the District level
    district_risk = df.groupby(['state', 'district']).agg({
        'infra_stress_score': 'mean',
        'compliance_gap': 'mean',
        'total_updates': 'sum'
    }).reset_index()

    # 2. Filter out zero-value districts
    district_risk = district_risk[district_risk['total_updates'] > 0]

    # 3. Create the Bubble Chart
    fig = px.scatter(
        district_risk,
        x='infra_stress_score',
        y='compliance_gap',
        size='total_updates',
        color='state', 
        hover_name='district',
        size_max=60,
        title="🚨 District Risk Matrix: Stress vs. Compliance Gap",
        labels={
            'infra_stress_score': 'Infrastructure Stress',
            'compliance_gap': 'Compliance Gap',
            'total_updates': 'Impact Volume'
        }
    )

    # 4. Add Quadrant Lines (Thresholds)
    stress_median = district_risk['infra_stress_score'].median()
    gap_median = district_risk['compliance_gap'].median()

    fig.add_hline(y=gap_median, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=stress_median, line_dash="dash", line_color="gray", opacity=0.5)

    # 5. Add Strategic Annotations
    fig.add_annotation(x=district_risk['infra_stress_score'].max(), y=district_risk['compliance_gap'].max(),
                text="🔴 RED ZONE: High Stress & High Gap", showarrow=False, font=dict(color="red", size=12))
    fig.add_annotation(x=district_risk['infra_stress_score'].min(), y=district_risk['compliance_gap'].min(),
                text="🟢 STABLE: Low Stress & Low Gap", showarrow=False, font=dict(color="green", size=12))

    fig.update_layout(template="plotly_white", height=600, margin=dict(t=80))
    
    return fig




def get_frontier_density_map(df):
    # 1. Prepare Center-Level Metrics
    center_stats = df.groupby(['pincode', 'state']).agg({
        'pincode_persistence': 'mean', 
        'total_updates': 'sum'
    }).reset_index()

    center_stats['engagement_intensity'] = center_stats['total_updates'] / (center_stats['pincode_persistence'] + 1)
    center_stats['log_intensity'] = np.log10(center_stats['engagement_intensity'] + 1)

    # 2. Setup the Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.set_theme(style="white")

    # 3. Create the 2D KDE Contour
    sns.kdeplot(
        data=center_stats,
        x="pincode_persistence",
        y="log_intensity",
        fill=True,
        thresh=0.01, 
        levels=15,
        cmap="mako",
        alpha=0.75,
        ax=ax
    )

    # 4. Underlay Scatter
    ax.scatter(center_stats['pincode_persistence'], center_stats['log_intensity'], 
                color='white', s=8, alpha=0.15)

    # 5. Strategic Labels
    ax.text(5, 4.3, "FRAGILE HOTSPOTS\n(High Surge, Low Stability)", 
             color='#e74c3c', fontweight='bold', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    ax.text(95, 0.4, "DORMANT CENTERS\n(Stable but Idle)", 
             color='#34495e', fontweight='bold', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # 6. The Goldilocks Frontier Box
    # Adjusting coordinates based on dataset typical ranges
    x_min_gold, x_max_gold = 85, 110  
    y_min_gold, y_max_gold = 1.3, 2.7 
    rect = plt.Rectangle((x_min_gold, y_min_gold), x_max_gold-x_min_gold, y_max_gold-y_min_gold, 
                          fill=False, color='#2ecc71', linestyle='--', linewidth=3)
    ax.add_patch(rect)
    ax.text(x_min_gold, y_max_gold + 0.1, "GOLDILOCKS FRONTIER", 
             color='#27ae60', fontweight='bold', fontsize=12)

    # 7. Final Polish
    ax.set_title("Frontier Density Map: Center Optimization", fontsize=22, fontweight='bold', pad=30)
    ax.set_xlabel("Persistence (Days Active)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Intensity (Log Scale)", fontsize=14, fontweight='bold')
    
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["10", "100", "1K", "10K", "100K"])
    ax.grid(color='gray', linestyle=':', alpha=0.2)

    plt.tight_layout()
    return fig



import plotly.graph_objects as go

def get_system_flow_sankey(df):
    # 1. Prepare the Buckets
    demographic_updates = df[[c for c in df.columns if 'demo' in c.lower()]].sum().sum()
    biometric_updates = df[[c for c in df.columns if 'bio' in c.lower()]].sum().sum()
    
    # Logic for flow distribution
    label = ["New Enrolments", "Demographic Updates", "Biometric Updates", "Mandatory Cycles", "Voluntary Corrections"]
    source = [0, 0, 1, 1, 2, 2] 
    target = [1, 2, 3, 4, 3, 4] 
    value  = [
        demographic_updates * 0.4, 
        biometric_updates * 0.4,   
        demographic_updates * 0.2, 
        demographic_updates * 0.8, 
        biometric_updates * 0.6,   
        biometric_updates * 0.4    
    ]

    # 2. Create the Figure
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = label,
          color = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c", "#9b59b6"]
        ),
        link = dict(
          source = source,
          target = target,
          value = value,
          color = "rgba(200, 200, 200, 0.4)"
      ))])

    fig.update_layout(
        title_text="🔄 System Evolution: Service Life-Cycle Flow", 
        font_size=12,
        height=500
    )
    
    return fig
