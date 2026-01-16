import plotly.express as px

def get_risk_bubble_chart(df):
    fig = px.scatter(df, x='stress', y='gap', size='total_updates')
    # ... your styling code ...
    return fig # Return the figure object