# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 11:14:43 2023

@author: alok
"""
import streamlit as st
import pickle
import difflib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


movies_data=pd.read_csv("movies (1).csv")

selected_features=['genres','keywords','tagline','cast','director'] # imdb, adult rating,etc

for feature in selected_features:
    movies_data[feature]= movies_data[feature].fillna('')

combined_features= movies_data['genres']+''+movies_data['keywords']+''+movies_data['tagline']+''+movies_data['cast']+''+movies_data['director']

vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)

similarity= cosine_similarity(feature_vector)

lsit_of_all_titles= movies_data['title'].tolist()












def recommend(movie_name):
    #print('Other Recommendations: \n')

    
    i=1
    recommendations=[]
    for movie in sorted_similar_movie_list:
        index=movie[0]
        title_from_index = movies_data[movies_data.index==index]['title'].values[0]

    ## suggest 30  movies
        if (i<30):
            recommendations.append(title_from_index)
            i+=1
    return recommendations 

movies_list= pickle.load(open("movielist.pkl",'rb'))
sorted_similar_movie_list= pickle.load(open("sorted_similar_movie.pkl",'rb'))
title_from_index= pickle.load(open("title_from_index.pkl",'rb'))
similarity_score= pickle.load(open("similarity_score.pkl",'rb'))
similarity= pickle.load(open("similarity.pkl",'rb'))
movies_data= pickle.load(open("movies_data.pkl",'rb'))

input_movie_index= pickle.load(open("input_movie_index.pkl",'rb'))
close_match_list= pickle.load(open("close_match_list.pkl",'rb'))
closest_match= pickle.load(open("closest_match.pkl",'rb'))
combined_features= pickle.load(open("combined_features.pkl",'rb'))
vectorizer= pickle.load(open("vectorizer.pkl",'rb'))
combined_features= pickle.load(open("combined_features.pkl",'rb'))
feature_vector= pickle.load(open("feature_vector.pkl",'rb'))

st.title("Movie Recommender System")

selected_movie = st.selectbox("Chose a movie here", movies_list)

close_match_list = difflib.get_close_matches(selected_movie,lsit_of_all_titles)

closest_match = close_match_list[0]

input_movie_index= movies_data[movies_data.title == closest_match]['index'].values[0]

similarity_score= list(enumerate(similarity[input_movie_index ]))

## sorting the movies based on similarity score
sorted_similar_movie_list=sorted(similarity_score, key= lambda x:x[1], reverse=True)

## x[0]==index, x[1]= similarity score








if st.button('Recommend'):
    
    recommended_movies= recommend(selected_movie)
    for j in recommended_movies:
        st.write(j)