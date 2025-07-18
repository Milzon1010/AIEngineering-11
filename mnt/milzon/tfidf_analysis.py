
import math
from collections import Counter

class TFIDFCalculator:
    def __init__(self, corpus):
        self.corpus = corpus
        self.documents = [doc.lower().split() for doc in corpus]
        self.vocab = set(word for doc in self.documents for word in doc)
        self.idf_scores = self._calculate_idf()

    def _calculate_idf(self):
        idf = {}
        total_docs = len(self.documents)
        for word in self.vocab:
            doc_count = sum(1 for doc in self.documents if word in doc)
            idf[word] = math.log(total_docs / (1 + doc_count))
        return idf

    def _calculate_tf(self, doc):
        tf = Counter(doc)
        total_terms = len(doc)
        return {word: count / total_terms for word, count in tf.items()}

    def calculate_tfidf(self, doc_index):
        doc = self.documents[doc_index]
        tf = self._calculate_tf(doc)
        return {word: tf[word] * self.idf_scores[word] for word in doc}

    def find_most_important_words(self, doc_index, top_n=5):
        tfidf_scores = self.calculate_tfidf(doc_index)
        sorted_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_n]

    def compare_documents(self, doc1_index, doc2_index):
        tfidf1 = self.calculate_tfidf(doc1_index)
        tfidf2 = self.calculate_tfidf(doc2_index)
        common_words = set(tfidf1.keys()) & set(tfidf2.keys())
        return {word: (tfidf1[word], tfidf2[word]) for word in common_words}

if __name__ == "__main__":
    corpus = [
        "This movie is absolutely fantastic and amazing",
        "The movie was terrible and boring",
        "Amazing acting but terrible plot",
        "Fantastic movie with great acting",
        "The plot was boring and predictable"
    ]

    tfidf = TFIDFCalculator(corpus)

    print("Top 5 words in Document 0:")
    print(tfidf.find_most_important_words(0))

    print("\nCommon important words between Document 0 and 2:")
    print(tfidf.compare_documents(0, 2))
