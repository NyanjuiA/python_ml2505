# Python script to demonstrate word embeddings using spacy
# NB: Ensure the spacy module is installed (pip install spacy)

# -----------------------------------------------------------------------------
# 0. Import the required modules
# -----------------------------------------------------------------------------
import numpy as np
import spacy
import sys
from sklearn.preprocessing import normalize

# Download the medium-sized English model for spacy
# spacy.cli.download('en_core_web_md')

# Load Spacy's Medium model with GloVe embeddings
nlp = spacy.load('en_core_web_md')

# Function to get sentence vector
def get_sentence_vector(doc):
   """Returns the normalised vector of a sentence."""
   vector = doc.vector
   return normalize([vector])[0]

# Function display/print the word vectors
def print_token_vectors(doc):
   """Prints/displays word vectors for content words. """
   print("\nToken vectors (First 5 Dimensions):")
   for token in doc:
      if token.is_stop or token.is_punct:
         continue
      print(f"Word: {token.text}, Vector: {token.vector[:5]}")

def save_vectors(vector, filename="sentence_vectors.npy"):
   "Saves word vectors to a .npy file."
   np.save(filename, vector)
   print(f"Saved word vectors to {filename}")

def compute_similarity(text1, text2):
   """Computes the semantic similarity between two sentences."""
   doc1 = nlp(text1)
   doc2 = nlp(text2)
   similarity = doc1.similarity(doc2)
   print(f"\nSimilarity between:\n- {text1}\n- '{text2}\n= {similarity:.3f}'")

# Get in put from the command line or use defaults
texts = sys.argv[1:] or [
   "Machine learning and artificial intelligence are evolving exponentially.",
   "Natural language processing is a key component of AI.",
]

for idx, text in enumerate(texts):
   doc = nlp(text)
   print(f"\n==== Text {idx+1} ====")
   print(f"Input: {text}")

   print_token_vectors(doc)

   sentence_vector = get_sentence_vector(doc)
   print("Normalised Sentence Embedding (First 5 Dimensions):", sentence_vector[:5])

   # optional: save vector
   save_vectors(sentence_vector, filename=f"sentence_vectors {idx+1}.npy")

# Also optional: compare similarity between first two text
if(len(texts) >= 2):
   compute_similarity(texts[0], texts[1])

