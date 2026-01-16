from modules.visualizer import get_risk_bubble_chart

# To display it:
fig = get_risk_bubble_chart(df)
st.plotly_chart(fig, use_container_width=True)