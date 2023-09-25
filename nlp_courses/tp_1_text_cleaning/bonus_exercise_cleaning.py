import re  # noqa: F401
import string  # noqa: F401

import nltk  # noqa: F401
import pandas as pd
from nltk.corpus import stopwords, wordnet  # noqa: F401
from nltk.stem import WordNetLemmatizer  # noqa: F401
from sklearn.pipeline import Pipeline  # noqa: F401
from sklearn.preprocessing import FunctionTransformer  # noqa: F401
from utils import emojis_unicode, emoticons, slang_words  # noqa: F401
from collections import Counter
from spellchecker import SpellChecker

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Declare your cleaning functions here
# Chain those functions together inside the preprocessing pipeline
# You can use (or not) Sklearn pipelines and functionTransformer for readability
# and modularity
# --- Documentation ---
# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html

# Global variables

PUNCT_TO_REMOVE = string.punctuation
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Lowercase
def Lowercase(text: str) -> str:
    return text.lower()

# removal of Punctuation
def remove_punctuation(text: str) -> str:
    
    translation_table = str.maketrans('', '', PUNCT_TO_REMOVE)
    return text.translate(translation_table)

# Stopwords removal
def remove_stopwords(text: str) -> str:
    
    words = text.split()
    filtered_words = [word for word in words if word not in STOPWORDS]
    return ' '.join(filtered_words)

# frequent words removal
#def remove_freqwords(text: str, freq_words: list=FREQWORDS) -> str: 
    
    words = text.split()
    wo_freq_words = [word for word in words if word not in freq_words]
    return ' '.join(wo_freq_words)

# rare words removal
#def remove_rarewords(text: str, rare_words: list=RAREWORDS) -> str:
    
    words = text.split()
    freq_words = [word for word in words if word not in rare_words]     
    return ' '.join(freq_words)

# Stopwords + Frequent + Rare words removal in one function
#def filter_text(text: str, words_to_remove :list=TO_REMOVE) -> str:

    words = text.split()
    filter_words = [word for word in words if word not in words_to_remove]
    return ' '.join(filter_words)


# lemmatization
def lemmatize_words(text: str) -> str:

    wordnet_map = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
        'J': wordnet.ADJ
    }
    pos_tagged_text = nltk.pos_tag(nltk.word_tokenize(text))
    lemmatized_pos_tagged_text = [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text]
    return ' '.join(lemmatized_pos_tagged_text)

# emoticons conversion
def convert_emoticons(text :str) -> str:

    EMOTICONS = emoticons()
    for emoticon, description in EMOTICONS.items():
        cleaned_description = re.sub(",", "", description)
        joined_description = "_".join(cleaned_description.split())
        pattern = u'('+re.escape(emoticon)+')'
        text = re.sub(pattern, joined_description, text)
    return text

# emoji conversion
def convert_emojis(text :str) -> str:

    EMO_UNICODE = emojis_unicode()
    for emoji_code, emoji in EMO_UNICODE.items():
        description = emoji_code.strip(":")  
        no_commas = re.sub(",", "", description)
        joined_description = "_".join(no_commas.split())
        pattern = u'('+re.escape(emoji)+')'
        text = re.sub(pattern, joined_description, text)
    return text

# urls removal
def remove_urls(text :str) -> str:

    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

# html tags removal
def remove_html(text :str) -> str:
    
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

# chat words conversion
def chat_words_conversion(text: str) -> str:
    slang_words_list = slang_words()
    chat_words_list = list(slang_words_list.keys())
    new_text = []
    
    for word in text.split():
        if word.upper() in chat_words_list:
            new_text.append(slang_words_list[word.upper()])
        else:
            new_text.append(word)

    return ' '.join(new_text)


#spelling correction
def correct_spellings(text: str) -> str:
    
    spell = SpellChecker()
    corrected_text = []
    
    misspelled = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled:
            corrected_word = spell.correction(word)
            corrected_text.append(corrected_word if corrected_word is not None else word)
        else:
            corrected_text.append(word)


    return ' '.join(corrected_text)





def preprocessing_pipeline(text: str) -> str:
    """
    Add a short description of your preprocessing pipeline here
    (see TP_1 for references)
    """
    # Your code here:
    text = remove_urls(text)
    text = remove_html(text)
    text = Lowercase(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    #text = remove_freqwords(text)
    #text = remove_rarewords(text)
    text = lemmatize_words(text)
    text = convert_emoticons(text)
    text = convert_emojis(text)
    text = chat_words_conversion(text)
    text = correct_spellings(text)
    
    return text



# TRANSFORMER SKLEARN PIPELINE

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Transformer for each function
lowercase_transformer = FunctionTransformer(np.vectorize(Lowercase))
punctuation_transformer = FunctionTransformer(np.vectorize(remove_punctuation))
stopwords_transformer = FunctionTransformer(np.vectorize(remove_stopwords))
#freqwords_transformer = FunctionTransformer(np.vectorize(remove_freqwords))
#rarewords_transformer = FunctionTransformer(np.vectorize(remove_rarewords))
lemmatize_transformer = FunctionTransformer(np.vectorize(lemmatize_words))
emoticons_transformer = FunctionTransformer(np.vectorize(convert_emoticons))
emojis_transformer = FunctionTransformer(np.vectorize(convert_emojis))
urls_transformer = FunctionTransformer(np.vectorize(remove_urls))
html_transformer = FunctionTransformer(np.vectorize(remove_html))
chat_words_transformer = FunctionTransformer(np.vectorize(chat_words_conversion))
spellings_transformer = FunctionTransformer(np.vectorize(correct_spellings))

# Combining transformers into a sklearn pipeline
pipeline = Pipeline([
    ('lowercase', lowercase_transformer),
    ('remove_html', html_transformer),
    ('remove_urls', urls_transformer),
    ('remove_punctuation', punctuation_transformer),
    ('convert_emoticons', emoticons_transformer),
    ('convert_emojis', emojis_transformer),
    ('chat_words_conversion', chat_words_transformer),
    ('remove_stopwords', stopwords_transformer),
    #('remove_freqwords', freqwords_transformer),
    #('remove_rarewords', rarewords_transformer),
    ('correct_spellings', spellings_transformer),
    ('lemmatize_words', lemmatize_transformer)
    
])
        
        
if __name__ == "__main__":
    df = pd.read_csv("nlp_courses/tp_1_text_cleaning/to_clean.csv", index_col=0)
#     df["cleaned_text"] = df.text.apply(lambda x: preprocessing_pipeline(x))
    df["cleaned_text"] = pipeline.transform(df["text"].values) 
    for idx, row in df.iterrows():
        print(f"\nBase text: {row.text}")
        print(f"Cleaned text: {row.cleaned_text}\n")

