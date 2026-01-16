import pandas as pd
import os
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

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
# Line 60
def get_localized_demand(df):
    # Everything below this must be shifted right!
    state_totals = df.groupby('state')['total_enrolment'].sum()
    top_n_states = state_totals.nlargest(10).index

    top_data = df[df['state'].isin(top_n_states)]
    dist_data = top_data.groupby(['state', 'district'])['total_enrolment'].sum().reset_index()

    dist_data = dist_data.sort_values(['state', 'total_enrolment'], ascending=[True, False])
    dist_data = dist_data.groupby('state').head(10)

    sns.set_theme(style="whitegrid")
    
    n_states = len(dist_data['state'].unique())
    cols = 3 if n_states <= 3 else 5
    
    g = sns.FacetGrid(dist_data, col="state", col_wrap=cols, sharey=False, sharex=False, 
                      height=4, aspect=1.2, hue="state", palette="husl")

    g.map(sns.barplot, "total_enrolment", "district", order=None)

    g.set_titles("{col_name}", size=14, fontweight='bold')
    g.set_axis_labels("Total Enrolment", "")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("📍 Localized Demand: Top 10 Districts per State", fontsize=22, fontweight='bold')

    for ax in g.axes.flat:
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=10)
        
    return g.fig # Ensure this is also indented!