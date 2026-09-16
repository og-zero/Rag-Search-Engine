import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt_tab')

# Define sample text
text = "This is an example showing how to filter out stop words."

# Load English stop words
stop_words = set(stopwords.words('english'))
print(stop_words)
# Tokenize the sentence into individual words
words = word_tokenize(text)
print(words)
# Filter out the stop words (ignoring case)
filtered_text = [w for w in words if w.lower() not in stop_words]


print("Original Tokens:", words)
print("After Stop Word Removal:", filtered_text)
