import streamlit as st

st.set_page_config(
    page_title="Multipage App",
    page_icon="👋",
)


# giving a title
st.title('Counsello')
st.write("An AI career counselling assistant for students")


st.sidebar.success("Go to Counsello")

st.sidebar.success("Select a page above.")