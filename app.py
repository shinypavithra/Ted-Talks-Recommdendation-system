import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from nltk.corpus import stopwords
import string
import nltk

nltk.download('stopwords')

# Load the vectorizer from the pickle file
with open('ted_talks_recommendation.pkl', 'rb') as file:
    vectorizer = pickle.load(file)[0]

# Load your data
df = pd.read_csv('ted_main.csv')

# Ensure your data has a 'url' column
df['name'] = df['title'] + ' ' + df['main_speaker']
df = df[['main_speaker', 'name', 'url']]
df.dropna(inplace=True)
data = df.copy()

def remove_stopwords(text):
    stop_words = stopwords.words('english')
    imp_words = [word.lower() for word in str(text).split() if word.lower() not in stop_words]
    return " ".join(imp_words)

def cleaning_punctuations(text):
    signal = str.maketrans('', '', string.punctuation)
    return text.translate(signal)

df['name'] = df['name'].apply(remove_stopwords)
df['name'] = df['name'].apply(cleaning_punctuations)

def get_similarities(talk_content, data=df):
    talk_array1 = vectorizer.transform(talk_content).toarray()
    sim = []
    pea = []
    for idx, row in data.iterrows():
        details = row['name']
        talk_array2 = vectorizer.transform(data[data['name'] == details]['name']).toarray()
        cos_sim = cosine_similarity(talk_array1, talk_array2)[0][0]
        pea_sim = pearsonr(talk_array1.squeeze(), talk_array2.squeeze())[0]
        sim.append(cos_sim)
        pea.append(pea_sim)
    return sim, pea

def recommend_talks(talk_content, data=data):
    data['cos_sim'], data['pea_sim'] = get_similarities(talk_content)
    data.sort_values(by=['cos_sim', 'pea_sim'], ascending=[False, False], inplace=True)
    return data[['main_speaker', 'name', 'url']].head()

# Streamlit app layout
st.title('TED Talks Recommendation System')
st.write("Enter the content of a TED Talk you are interested in, and get recommendations for similar talks.")

user_input = st.text_area("Enter TED Talk content:")

if st.button('Recommend'):
    if user_input:
        talk_content = [user_input]
        recommendations = recommend_talks(talk_content)
        st.write("Here are some recommendations for you:")

        for index, row in recommendations.iterrows():
            st.write(f"**{row['main_speaker']}**: {row['name']}")
            st.write(f"[Watch on TED]({row['url']})")
    else:
        st.write("Please enter some text to get recommendations.")
