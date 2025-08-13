import streamlit as st

analytics_page = st.Page(page="pages/AnalyticsPage.py", title="Dataset")
app = st.Page(page="pages/App.py", title="Model")

pg = st.navigation([ analytics_page, app])

pg.run()