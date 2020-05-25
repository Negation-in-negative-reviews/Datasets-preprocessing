import os
import spacy
import numpy as np
import random

def write_file(sents, out_file):
    with open(out_file, "w") as fout:
        for s in sents:
            fout.write(s.strip("\n")+"\n")


def read_file(filename):
    reviews = []    
    with open(filename, "r") as fin:    
        for line in fin:
            line = line.strip("\n")
            reviews.append(line)
        return reviews


nlp = spacy.load("en_core_web_md")
tokenizer = nlp.Defaults.create_tokenizer(nlp)

def get_sents(reviews, min_no_of_tokens: int = 5):
    all_sents = []    
    for review in reviews:   
        doc = nlp(review)        
        for sent in doc.sents:
            tokens = tokenizer(sent.string.strip())
            if len(tokens) >= min_no_of_tokens:
                all_sents.append(sent.string.strip().strip("\n")) 

    return all_sents