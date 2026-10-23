import streamlit as st
st.set_page_config(page_title="AI Application Platform", layout="wide")

detection_page = st.Page(
    page="views/detection_page.py", 
    title="Object Detection", 
    icon="", 
    default=True
)

dashboard_page = st.Page(
    page="views/dashboard_page.py", 
    title="Analytics Dashboard", 
    icon=""
)

pg = st.navigation({
    "ระบบหลัก": [detection_page],
    "รายงานและสถิติ": [dashboard_page]
})

pg.run()
