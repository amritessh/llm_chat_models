# import langchain_helper as lch
# import streamlit as st

# st.title("Pet Name Generator")

# # animal_type = st.text_input("Enter an animal type: ")

# animal_type = st.sidebar.selectbox("Select an animal type", ["Dog", "Cat", "Bird", "Fish", "Other"])


# if animal_type == "Cat":
#     pet_color = st.sidebar.text_area("What color is your cat?", height=100)
# elif animal_type == "Dog":
#     pet_color = st.sidebar.text_area("What color is your dog?", height=100)
# elif animal_type == "Bird":
#     pet_color = st.sidebar.text_area("What color is your bird?", height=100)
# elif animal_type == "Fish":
#     pet_color = st.sidebar.text_area("What color is your fish?", height=100)
# else:
#     pet_color = st.sidebar.text_area("What color is your pet?", height=100)

# st.write(f"You selected {animal_type} and your pet's color is {pet_color}")                       

# if st.button("Generate Pet Name"):
#     response = lch.generate_response(animal_type, pet_color)
#     st.write(response)

import streamlit as st
import langchain_doc as ld
import textwrap

st.title("Youtube Video Chatbot")

with st.sidebar:
    with st.form(key="my_form"):
        youtube_url = st.text_input("Enter a Youtube URL")
        query = st.text_input("Enter a query")
        submit_button = st.form_submit_button("Submit")

if submit_button and youtube_url and query:
    with st.spinner("Processing..."):
        db = ld.create_vector_db_from_youtube_url(youtube_url)
        response = ld.get_response_from_query(query, db)
        
        st.subheader("Answer")
        st.write(textwrap.fill(response, width=80))

