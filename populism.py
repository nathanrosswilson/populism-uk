# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:59:57 2020

@author: natha
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy 
from textstat.textstat import textstatistics, legacy_round 

WL = ["elit",
      "consensus",
      "undemocratic",
      "referend",
      "corrupt",
      "propagand",
      "politici",
      "deceit",
      "deveiv",
      "betray",
      "shame",
      "scandal",
      "truth",
      "dishonest",
      "establishm",
      "ruling",
      "citizen",
      "consumer",
      "taxpayer",
      "voter",
      "people"]

def pop(text):
    try:
        text_tolk = word_tokenize(text)
    except:
        return 0, 0
    stop_words = set(stopwords.words("english"))
    
    new_words = [word for word in text_tolk if word.isalnum() or word not in stop_words]
    
    count = 0
    for word in new_words:
        for W in WL:
            if W in word.lower():
                count += 1
    
    return count/len(new_words) * 100, len(new_words)

def length(text):
    try:
        text_tolk = word_tokenize(text)
    except:
        print(text)
        return 0
    stop_words = set(stopwords.words("english"))
    
    new_words = [word for word in text_tolk if word.isalnum() or word not in stop_words]
    
    return len(new_words)


def break_sentences(text): 
    nlp = spacy.load('en_core_web_sm') 
    doc = nlp(str(text)) 
    return list(doc.sents) 
  
# Returns Number of Words in the text 
def word_count(text): 
    sentences = break_sentences(text) 
    words = 0
    for sentence in sentences: 
        words += len([token for token in sentence]) 
    return words 
  
# Returns the number of sentences in the text 
def sentence_count(text): 
    sentences = break_sentences(text) 
    return len(sentences) 

def syllables_count(word): 
    return textstatistics().syllable_count(word) 

def avg_sentence_length(text): 
    words = word_count(text) 
    sentences = sentence_count(text) 
    average_sentence_length = float(words / sentences) 
    return average_sentence_length 

def avg_syllables_per_word(text): 
    syllable = syllables_count(text) 
    words = word_count(text) 
    ASPW = float(syllable) / float(words) 
    return legacy_round(ASPW, 1) 

def flesch(text):
    text = str(text)
    FRE = 206.835 - float(1.015 * avg_sentence_length(text)) - \
        float(84.6 * avg_syllables_per_word(text))
    return legacy_round(FRE, 2)
tqdm.pandas()

tqdm.pandas()

df = pd.read_csv("data_cut.csv")

df.replace('', np.nan, inplace=True)
df.dropna(subset=['name'], inplace=True)

df["score"], df["length"] = zip(*df["text"].progress_apply(pop))
df["complex"] = df["text"].progress_apply(flesch)


df.to_csv("data_with_pop_scores.csv", index=False, encoding='utf-8-sig')