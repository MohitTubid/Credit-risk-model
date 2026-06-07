import streamlit as st

# set_page_config MUST be the first Streamlit command, and only goes in the entry file.
st.set_page_config(page_title="Credit Risk Model Study", page_icon=":bar_chart:", layout="wide")

# st.navigation gives full control over each page's sidebar LABEL (title=) and order,
# independent of the filename. This is why the home page can be labelled "App" and the
# charts page "Insights". The first page with default=True is the landing page.
pages = [
    st.Page("home.py",                 title="App", default=True),
    st.Page("pages/1_Leaderboards.py", title="Leaderboards"),
    st.Page("pages/2_Insights.py",     title="Insights"),
    st.Page("pages/3_Predict.py",      title="Predict"),
    st.Page("pages/4_Train.py",        title="Train"),
]

st.navigation(pages).run()
