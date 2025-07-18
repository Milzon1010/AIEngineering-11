"""
Step 1: Pendahuluan dan Manual TF-IDF
"""

import math

# Documents
docs = [
    "This movie is absolutely fantastic and amazing",
    "The movie was terrible and boring",
    "Amazing acting but terrible plot",
    "Fantastic movie with great acting"
]

# Preprocessing
documents = [doc.lower().split() for doc in docs]
N = len(documents)

# Count TF and DF
from collections import Counter, defaultdict

# Term Frequencies (TF)
TF = []
DF = defaultdict(int)

for doc in documents:
    tf = Counter(doc)
    TF.append(tf)
    for word in set(doc):
        DF[word] += 1

# Calculate TF-IDF for a few examples
def tfidf(word, doc_index):
    tf = TF[doc_index][word] / len(documents[doc_index])
    idf = math.log(N / (1 + DF[word]))
    return tf * idf

print("TF-IDF('amazing', Doc 1):", tfidf('amazing', 0))
print("TF-IDF('terrible', Doc 2):", tfidf('terrible', 1))
