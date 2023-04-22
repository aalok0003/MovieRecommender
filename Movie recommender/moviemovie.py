# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 11:14:43 2023

@author: alok
"""

import streamlit as st
import pickle

movies_list= pickle.load(open("movielist.pkl",'rb'))
st.title("Movie Recommender System")
st.selectbox("Chose a movie here", movies_list)