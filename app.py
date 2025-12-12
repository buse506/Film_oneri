import streamlit as st
import pickle
import pandas as pd

# --- VERİLERİ YÜKLE ---
# Colab'de dosyalar direkt ana dizinde oluştuğu için yol belirtmeye gerek yok
try:
    movies_list = pickle.load(open('movies.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model dosyaları (pkl) bulunamadı. Lütfen önce model oluşturma kodlarını çalıştırın.")
    st.stop()

def recommend(movie):
    movie_index = movies_list[movies_list['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_sorted = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_sorted:
        recommended_movies.append(movies_list.iloc[i[0]].title)
    return recommended_movies

st.title('🎬 Film Öneri Sistemi (Colab Versiyonu)')

selected_movie_name = st.selectbox(
    'Film Seçin:',
    movies_list['title'].values
)

if st.button('Öneri Getir'):
    recommendations = recommend(selected_movie_name)
    for i, movie in enumerate(recommendations, 1):
        st.write(f"{i}. {movie}")
