import streamlit as st
import pandas as pd

st.set_page_config(page_title="Rice Leaf Health Dashboard", layout="wide")
st.title("Rice Leaf Health Dashboard")

field = st.selectbox("Field ID", ["field_A", "field_B"])
date_range = st.date_input("Date range", [])
st.write("Demo chart: % diseased area over time (placeholder)")
df = pd.DataFrame({"date": ["2025-10-01","2025-10-05","2025-10-10"], "percent_area": [5.2, 9.8, 7.1]})
st.line_chart(df.set_index("date"))
st.dataframe(df)
