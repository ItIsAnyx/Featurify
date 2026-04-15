from sentence_transformers import SentenceTransformer
import numpy as np
from .retriever_docs import documents, evaluation_data

model = SentenceTransformer('all-MiniLM-L6-v2')

class SemanticRetriever:
    def __init__(self, documents):
        self.model = model
        self.documents = documents
        self.embeddings = self.model.encode(documents)

    def retrieve_with_indices(self, query, top_k=3):
        query_emb = self.model.encode([query])[0]

        similarities = np.dot(self.embeddings, query_emb) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-10
        )

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return top_indices

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)

    hits = sum([1 for doc_id in retrieved_k if doc_id in relevant_set])
    return hits / k

def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)

    hits = sum([1 for doc_id in retrieved_k if doc_id in relevant_set])
    return hits / len(relevant)

def top_k_accuracy(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    return int(any(doc_id in relevant for doc_id in retrieved_k))

if __name__ == "__main__":
    retriever = SemanticRetriever(documents)

    k = 3

    precisions = []
    recalls = []
    accuracies = []

    for item in evaluation_data:
        retrieved = retriever.retrieve_with_indices(item["query"], top_k=k)
        print(f"\nQuery: {item['query']}")
        print("Retrieved:", retrieved)
        print("Relevant:", item["relevant_docs"])

        p = precision_at_k(retrieved, item["relevant_docs"], k)
        r = recall_at_k(retrieved, item["relevant_docs"], k)
        a = top_k_accuracy(retrieved, item["relevant_docs"], k)

        precisions.append(p)
        recalls.append(r)
        accuracies.append(a)

    print("Precision@k:", sum(precisions) / len(precisions))
    print("Recall@k:", sum(recalls) / len(recalls))
    print("Top-k Accuracy:", sum(accuracies) / len(accuracies))