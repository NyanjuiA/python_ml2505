# Python scripto to demonstrate calculating tf-idf (Term Frequency-Inverse Document Frequency)

# Import the required module
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data from documents
documents = [
   "Text mining transforms unstructured text in structured data.",
   "TF-IDF is a technique used in text mining",
   "Text analysis often involves TF-IDF"
]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Compute TF-IDF matrix
tfid_matrix = vectorizer.fit_transform(documents)

# Get the feature names (terms) and their corresponding TF-IDF scores
feature_names = vectorizer.fit_transform(documents)
tfidf_scores = tfid_matrix.toarray()

# Display results
print("Feature names:", feature_names)
print("TF-IDF scores:", tfidf_scores)