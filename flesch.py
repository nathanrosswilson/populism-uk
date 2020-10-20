# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:12:21 2020

@author: natha
"""

import pandas as pd
import spacy 
from tqdm import tqdm
from textstat.textstat import textstatistics, legacy_round 


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

df = pd.read_csv("data_cut.csv")

df["score"] = df["text"].progress_apply(flesch)

df.to_csv("data_with_scores.csv", index=False, encoding='utf-8-sig')
