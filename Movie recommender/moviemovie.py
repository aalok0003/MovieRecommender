# import some necessary libraries for performing some NLP and machine learning tasks
import streamlit as st
import pickle
import difflib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# This code reads a CSV file containing the movie data and stores it in a pandas DataFrame called movies_data.
movies_data=pd.read_csv("movies (1).csv")

# This code fills any missing values in the selected features (genres, keywords, tagline, cast, and director) with an empty string.
selected_features=['genres','keywords','tagline','cast','director']
for feature in selected_features:
    movies_data[feature]= movies_data[feature].fillna('')

# This code concatenates the values of the selected features into a single column called combined_features.
combined_features= movies_data['genres']+''+movies_data['keywords']+''+movies_data['tagline']+''+movies_data['cast']+''+movies_data['director']

# This code uses the TfidfVectorizer function from scikit-learn to convert the textual data in combined_features to numerical data, and then computes the cosine similarity between all pairs of movies based on their feature vectors.
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)
similarity= cosine_similarity(feature_vector)

# This code creates a list of all movie titles.
lsit_of_all_titles= movies_data['title'].tolist()

# This code defines a function called recommend that takes a movie name as input and returns a list of 30 recommended movies based on their similarity score to the input movie.
def recommend(movie_name):
    i=1
    recommendations=[]
    for movie in sorted_similar_movie_list:
        index=movie[0]
        title_from_index = movies_data[movies_data.index==index]['title'].values[0]
        if (i<30):
            recommendations.append(title_from_index)
            i+=1
    return recommendations 

# This code loads three pickle files containing precomputed data required for the recommendation system: a list of all movie titles (movielist.pkl), a sorted list of similar movies for each movie in the dataset (sorted_similar_movie.pkl), and the original movie data (movies_data.pkl).
movies_list= pickle.load(open("movielist.pkl",'rb'))
sorted_similar_movie_list= pickle.load(open("sorted_similar_movie.pkl",'rb'))
movies_data= pickle.load(open("movies_data.pkl",'rb'))

# This code sets up the Streamlit app and allows the user to select a movie from a dropdown list. It then finds the closest match to the selected movie in the dataset and retrieves the similarity scores between the selected movie and all other movies.
st.title("Movie Recommender System")

selected_movie = st.selectbox("Chose a movie here", movies_list)

close_match_list = difflib.get_close_matches(selected_movie,lsit_of_all_titles)
closest_match = close_match_list[0]
input_movie_index= movies_data[movies_data.title == closest_match]['index'].values[0]

similarity_score= list(enumerate(similarity[input_movie_index ]))
sorted_similar_movie_list=sorted(similarity_score, key= lambda x:x[1], reverse=True)

# This code displays a "Recommend" button in the Streamlit app
if st.button('Recommend'):
    recommended_movies= recommend(selected_movie)
    for j in recommended_movies:
        st.write(j)
