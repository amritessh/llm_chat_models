import langchain_helper as lch
import streamlit as st

st.title("Pet Name Generator")

animal_type = st.text_input("Enter an animal type: ")

if st.button("Generate Pet Name"):
    response = lch.generate_response(animal_type)
    st.write(response)