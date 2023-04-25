import re
from typing import Dict, List, Optional, Set, Text, Tuple, Union

import contractions
from utils.constants import CHAT_WORDS_STR


def md_links(text: Text) -> Text:
    markdown_link=re.compile(r'\[.*?\]\(.*?\)')
    return markdown_link.sub(r'',text)

def scrape_links(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html_tags(text: Text) -> Text:
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def chat_words_conversion(text: Text) -> Text:
    # First, we're going to convert this long string into set of words and its shortcut
    chat_words_map_dict = {}
    chat_shortcut_list = set()
    for line in CHAT_WORDS_STR.split("\n"):
        if line != '':
            shortcut = line.split('=')[0] # split the line from `=` sign and select shortcut
            chat_words = line.split('=')[1]
            chat_shortcut_list.add(shortcut) # add the chat  shortcut to the set
            chat_words_map_dict[shortcut] = chat_words # add each chat_words corresponding to its shortcut

    chat_words_map_dict

    new_text = []
    for word in text.split():
        if word.upper() in chat_words_map_dict:
            new_text.append(chat_words_map_dict[word.upper()])
        else:
            new_text.append(word)
    return " ".join(new_text)

def en_contractions(text: Text) -> Text:
    return ' '.join([contractions.fix(word)
                     if word in contractions.contractions_dict else word
                     for word in text.split()])
                     
def handle_data_preprocessing(dataset):
    def preprocess_column(col):
        col = md_links(col)
        col = scrape_links(col)
        col = remove_html_tags(col)
        return col

    dataset['review'] = dataset['review'].apply(preprocess_column)
    dataset['summary'] = dataset['summary'].apply(preprocess_column)
    return dataset





