# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 11:14:43 2023

@author: alok
"""
import streamlit as st
import pickle


def recommend(movie):
    #print('Other Recommendations: \n')
    i=1
    recommended=[]
    for movie in sorted_similar_movie_list:
        index=movie[0]
        title_from_index = movies_data[movies_data.index==index]['title'].values[0]

    ## suggest 30  movies
        if (i<30):
            recommended.append(title_from_index)
            i+=1
    return recommended

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

if st.button('Recommend'):
    recommendations=recommend(selected_movie)
    for i in recommendations:
        st.write(i)
