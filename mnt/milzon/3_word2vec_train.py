"""
Step 3: Train Word2Vec and Explore Analogies
"""

from gensim.models import Word2Vec

# Define corpus
sentences = [
    "king queen prince princess royal palace".split(),
    "man woman boy girl gender equality".split(),
    "dog cat pet animal tail fur".split(),
    "apple banana fruit mango grape tropical".split(),
    "computer internet technology network digital cloud".split(),
    "car truck vehicle transport road tire".split(),
    "happy joy smile laugh cheerful positive".split(),
    "sad unhappy cry tears lonely depressed".split(),
    "fast quick speed run sprint hurry".split(),
    "slow delay wait late lazy sluggish".split()
] * 10  # duplicate to simulate larger corpus

# Train Word2Vec model
model = Word2Vec(
    sentences,
    vector_size=50,
    window=3,
    min_count=1,
    workers=2,
    seed=42
)

# Save model for future use (optional)
model.save("milzon_word2vec.model")

# Test analogy
print("\nAnalogy: king - man + woman = ?")
print(model.wv.most_similar(positive=["king", "woman"], negative=["man"]))

# Try different analogies
print("\nAnalogy: car - road + cloud = ?")
print(model.wv.most_similar(positive=["car", "cloud"], negative=["road"]))

print("\nAnalogy: happy - sad + fast = ?")
print(model.wv.most_similar(positive=["happy", "fast"], negative=["sad"]))

# Similarity clusters
for word in ["happy", "computer", "fast"]:
    print(f"\nWords similar to '{word}':")
    print(model.wv.most_similar(word))
