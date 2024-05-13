import pandas as pd
from keybert import KeyBERT
import spacy
from sentence_transformers import SentenceTransformer

# Load the English model from spaCy
nlp = spacy.load('en_core_web_md')

# Initialize the KeyBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=model)


def lemmatize_nouns(keywords, nlp):
    """Lemmatize nouns in the extracted keywords to ensure they are in singular form."""
    lemmatized_keywords = []
    for keyword in keywords:
        doc = nlp(keyword)
        lemmatized_phrase = ' '.join(token.lemma_ if token.pos_ == 'NOUN' else token.text for token in doc)
        lemmatized_keywords.append(lemmatized_phrase.lower())  # Convert to lowercase
    return lemmatized_keywords

def extract_keywords_keybert(text, kw_model, nlp, top_n=20):
    """Use KeyBERT to extract keywords and lemmatize nouns."""
    # Extract keywords with KeyBERT
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', use_maxsum=True, top_n=top_n)
    # Get the text of the keywords
    keywords_text = [keyword[0].replace('_', ' ') for keyword in keywords]  # Replace underscores with spaces
    # Lemmatize nouns and convert to lowercase
    lemmatized_keywords = lemmatize_nouns(keywords_text, nlp)
    # Convert the list of processed keywords to a comma-separated string
    keywords_str = ','.join(lemmatized_keywords)
    return keywords_str

# Read the CSV file
df = pd.read_csv('C:/Users/ytl/Desktop/LLM.csv')

# Combine title and abstract, and extract keywords
df['combined'] = df['title'] + '.' + df['abstract']  # Use a period to separate title and abstract
df['keywords'] = df['combined'].apply(lambda text: extract_keywords_keybert(text, kw_model, nlp))

# Optionally: Save the results to a new CSV file
df.to_csv('C:/Users/ytl/Desktop/llm/keybert_keywords2.csv', index=False)
