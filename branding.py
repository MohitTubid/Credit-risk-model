import streamlit as st

ROYAL_BLUE = "#003399"
PREMIUM_GREY = "#DBDBDB"


def header(title, subtitle=""):
    """Render a royal-blue brand banner. Used at the top of every page so the brand
    color is visible everywhere (the theme's primaryColor only tints interactive widgets)."""
    sub = (f"<p style='color:{PREMIUM_GREY}; margin:4px 0 0 0; font-size:0.95rem;'>{subtitle}</p>"
           if subtitle else "")
    st.markdown(
        f"<div style='background-color:{ROYAL_BLUE}; padding:14px 20px; border-radius:8px; "
        f"margin-bottom:14px;'>"
        f"<h1 style='color:#FFFFFF; margin:0; font-size:1.7rem;'>{title}</h1>{sub}</div>",
        unsafe_allow_html=True,
    )
