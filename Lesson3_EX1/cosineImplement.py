import gensim.downloader as api
import numpy as np
import math

# 25, 50, 100 or 200. Số càng lớn thì càng chính xác, nhưng chạy càng lâu các bạn nhé
#load model
model = api.load("glove-twitter-25")

# Implement cosine similarity with numpy
def cosine_similarity_numpy(word1, word2, model):
    # Get the vectors for the two words
    vec1 = model[word1]
    vec2 = model[word2]
    print (model[word1])
    
    # Compute the dot product of the vectors
    dot_product = np.dot(vec1, vec2)
    
    # Compute the L2 norms (magnitudes) of the vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Compute the cosine similarity
    similarity = dot_product / (norm1 * norm2)
    
    return similarity
# Implement cosine similarity without numpy 
def dot_product(vec1, vec2):
    return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))

def l2_norm(vec):
    return math.sqrt(sum(v ** 2 for v in vec))

def cosine_similarity_non_numpy(word1, word2, model):
    # Get the vectors for the two words
    vec1 = model[word1]
    vec2 = model[word2]
    
    # Compute the dot product of the vectors
    dot_prod = dot_product(vec1, vec2)
    
    # Compute the L2 norms (magnitudes) of the vectors
    norm1 = l2_norm(vec1)
    norm2 = l2_norm(vec2)
    
    # Compute the cosine similarity
    similarity = dot_prod / (norm1 * norm2)
    
    return similarity

# Test the function with examples
word1 = "man"
word2 = "woman"
similarity = cosine_similarity_numpy(word1, word2, model)
print(f"Cosine similarity between '{word1}' and '{word2}': {similarity}")

word1 = "marriage"
word2 = "happiness"
similarity = cosine_similarity_non_numpy(word1, word2, model)
print(f"Cosine similarity non numpy between '{word1}' and '{word2}': {similarity}")

#Verify
print("3----------verify")
result = model.similarity("man", "woman")
print(result)

print("6----------verify for non numpy")
result = model.similarity("marriage", "happiness")
print(result)
