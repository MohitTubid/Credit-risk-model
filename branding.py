import streamlit as st

ROYAL_BLUE = "#003399"
PREMIUM_GREY = "#DBDBDB"


def header(title, subtitle=""):
    """Premium-grey brand banner with a royal-blue title + left accent. The accent and a
    soft shadow keep it distinct from the (also grey) page background."""
    sub = (f"<p style='color:#333333; margin:4px 0 0 0; font-size:0.95rem;'>{subtitle}</p>"
           if subtitle else "")
    st.markdown(
        f"<div style='background-color:{PREMIUM_GREY}; border-left:6px solid {ROYAL_BLUE}; "
        f"padding:14px 20px; border-radius:8px; margin-bottom:14px; "
        f"box-shadow:0 1px 5px rgba(0,0,0,0.18);'>"
        f"<h1 style='color:{ROYAL_BLUE}; margin:0; font-size:1.7rem;'>{title}</h1>{sub}</div>",
        unsafe_allow_html=True,
    )
