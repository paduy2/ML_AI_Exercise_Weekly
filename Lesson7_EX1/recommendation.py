import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movie_data/movies.csv", encoding="latin-1", sep='\t', usecols=["title", "genres"])
movies["genres"] = movies["genres"].apply(lambda s: s.replace("|", " ").replace("-",""))
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies["genres"])
cosine_sm = cosine_similarity(tfidf_matrix)
cosine_sm_df = pd.DataFrame(cosine_sm, index=movies["title"], columns=movies["title"])

top_k = 20
user_input = "Batman Forever (1995)"
data = cosine_sm_df.loc[user_input, :]
result = data.sort_values(ascending=False)[:top_k]
print (result.shape)