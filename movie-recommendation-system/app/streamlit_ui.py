import streamlit as st
from recommend import recommend_movies

st.title("🎬 Movie Recommendation System")
st.write("Enter a movie name to get recommendations!")

# User input for movie title
movie_name = st.text_input("Enter a movie title:")

if st.button("Get Recommendations"):
    recommendations = recommend_movies(movie_name)
    st.write("### Recommended Movies:")
    for movie in recommendations:
        st.write(f"- {movie}")
