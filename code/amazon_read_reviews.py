import numpy as np
import spacy
import json
import gzip
from pathlib import Path
import os
import random
import pprint
import re
import imdb_read_dataset

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

def get_samples(data, n_samples):
    indices = np.random.choice(np.arange(len(data)), size=n_samples)
    sampled_data = [data[idx] for idx in indices]
    return sampled_data

def write_data_to_file(data, filename):
    with open(filename, "w") as fout:
        for d in data:
            fout.write(d.strip("\n")+"\n")

def get_clean_review(review: str):
    review = review.replace("\n", " ")
    review = re.sub(' +', ' ', review)
    review = review.strip("\n")
    return review

def read_json_gz(args: dict()):
    filename = args["data_filepath"]
    output_dir = args["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fout = open(os.path.join(output_dir, "reviews"), "w")    
    
    pos_reviews = []
    neg_reviews = []

    with gzip.open(filename, "rb") as f:
        jl_data = f.read().decode('utf-8')
        jl_data = list(jl_data.split("\n"))
        for line in jl_data:
            if line:
                json_content = json.loads(line)
                if "reviewText" in json_content:
                    if json_content["reviewText"].strip() != "":
                        rev = get_clean_review(json_content["reviewText"])                        
                        rating = float(json_content["overall"])
                        fout.write(rev+"\n")
                        if rating >= 4:                    
                            pos_reviews.append(rev)
                        elif rating <= 2:                    
                            neg_reviews.append(rev)
        return pos_reviews, neg_reviews

if __name__ == "__main__":
    seed_val = 23
    np.random.seed(seed_val)

    data = [
        {
            "data_filepath": "/data/madhu/amazon-reviews-2018/Cell_Phones_and_Accessories_5.json.gz",
            "name": "Amazon Cellphones reviews",
            "output_dir": "/data/madhu/amazon-reviews-2018/processed_data/cellphones_and_accessories/"                        
        },
        # {
        #     "data_filepath": "/data/madhu/amazon-reviews-2018/Digital_Music_5.json.gz",
        #     "name": "Amazon digital music reviews",
        #     "output_dir": "/data/madhu/amazon-reviews-2018/processed_data/digital_music/"                        
        # },
        # {
        #     "data_filepath": "/data/madhu/amazon-reviews-2018/Grocery_and_Gourmet_Food_5.json.gz",
        #     "name": "Amazon grocery reviews",
        #     "output_dir": "/data/madhu/amazon-reviews-2018/processed_data/grocery_and_gourmet_food/"                        
        # },
        # {
        #     "data_filepath": "/data/madhu/amazon-reviews-2018/Musical_Instruments_5.json.gz",
        #     "name": "Amazon musical instruments reviews",
        #     "output_dir": "/data/madhu/amazon-reviews-2018/processed_data/musical_instruments/"                        
        # },
        # {
        #     "data_filepath": "/data/madhu/amazon-reviews-2018/Electronics_5.json.gz",
        #     "name": "Amazon electronics reviews",
        #     "output_dir": "/data/madhu/amazon-reviews-2018/processed_data/electronics/"                        
        # },
        {
            "data_filepath": "/data/madhu/amazon-reviews-2018/Automotive_5.json.gz",
            "name": "Automotive reviews",
            "output_dir": "/data/madhu/amazon-reviews-2018/processed_data/automotive/"                        
        },
        {
            "data_filepath": "/data/madhu/amazon-reviews-2018/Luxury_Beauty_5.json.gz",
            "name": "Luxury Beauty reviews",
            "output_dir": "/data/madhu/amazon-reviews-2018/processed_data/luxury_beauty/"                        
        },
        {
            "data_filepath": "/data/madhu/amazon-reviews-2018/Pet_Supplies_5.json.gz",
            "name": "Pet supplies reviews",
            "output_dir": "/data/madhu/amazon-reviews-2018/processed_data/pet_supplies/"                        
        },
        {
            "data_filepath": "/data/madhu/amazon-reviews-2018/Sports_and_Outdoors_5.json.gz",
            "name": "Sports and Outdoors reviews",
            "output_dir": "/data/madhu/amazon-reviews-2018/processed_data/sports_and_outdoors/"
        }
    ]    
    n_samples = int(2*1e4)

    for d in data:
        myprint(d)
        pos_reviews, neg_reviews = read_json_gz(d)
        random.shuffle(pos_reviews)
        random.shuffle(neg_reviews)

        train_size_pos = int(len(pos_reviews)*3/4)
        train_size_neg = int(len(neg_reviews)*3/4)

        # Train data - reviews and sentences        
        train_pos_reviews = pos_reviews[:train_size_pos]
        train_neg_reviews = neg_reviews[:train_size_neg]
        write_data_to_file(train_pos_reviews, d["output_dir"]+"/pos_reviews_train")
        write_data_to_file(train_neg_reviews, d["output_dir"]+"/neg_reviews_train")

        train_pos_reviews_samples = get_samples(train_pos_reviews, n_samples)
        train_neg_reviews_samples = get_samples(train_neg_reviews, n_samples)

        write_data_to_file(train_pos_reviews_samples, d["output_dir"]+"/pos_reviews_train_"+str(n_samples))
        write_data_to_file(train_neg_reviews_samples, d["output_dir"]+"/neg_reviews_train_"+str(n_samples))
        
        train_pos_sents = imdb_read_dataset.get_sents(train_pos_reviews_samples)
        train_neg_sents = imdb_read_dataset.get_sents(train_neg_reviews_samples)

        write_data_to_file(train_pos_sents, d["output_dir"]+"/pos_sents_train")
        write_data_to_file(train_neg_sents, d["output_dir"]+"/neg_sents_train")


        # Test data - reviews and sentences (sentences in 20K sampled reviews)
        test_pos_reviews = pos_reviews[train_size_pos+1:]
        test_neg_reviews = neg_reviews[train_size_neg+1:]
        write_data_to_file(test_pos_reviews, d["output_dir"]+"/pos_reviews_test")
        write_data_to_file(test_neg_reviews, d["output_dir"]+"/neg_reviews_test")

        test_pos_reviews_samples = get_samples(test_pos_reviews, n_samples)
        test_neg_reviews_samples = get_samples(test_neg_reviews, n_samples)
        
        test_pos_sents = imdb_read_dataset.get_sents(test_pos_reviews_samples)
        test_neg_sents = imdb_read_dataset.get_sents(test_neg_reviews_samples)

        write_data_to_file(test_pos_sents, d["output_dir"]+"/pos_sents_test")
        write_data_to_file(test_neg_sents, d["output_dir"]+"/neg_sents_test")
