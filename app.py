#app.py

import dataset
import visual
import streamlit as st

PAGES = {
    "Visualization": visual,
    "Dataset & Code": dataset

}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()