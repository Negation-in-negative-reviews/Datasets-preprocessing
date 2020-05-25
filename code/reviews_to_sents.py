
import spacy
import os
from pathlib import Path
import utils
import pprint
import argparse

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint


if __name__ == "__main__":
    datasets = [
        # {
        #     "positive": {
        #         "data_filepath": "/data/madhu/stanford-sentiment-treebank/matched_data/pos_reviews_train",
        #         # "n_samples": 5000
        #     },
        #     "negative": {
        #         "data_filepath": "/data/madhu/stanford-sentiment-treebank/matched_data/neg_reviews_train",
        #         # "n_samples": 5000
        #     },
        #     "name": "SST",
        #     "saves_dir": "sst"
        # },
        
        # {
        #     "positive": {
        #         "data_filepath": "/data/madhu/yelp/yelp_processed_data/pos_reviews_train",         
        #         "n_samples": 50000
        #     },
        #     "negative": {
        #         "data_filepath": "/data/madhu/yelp/yelp_processed_data/neg_reviews_train",            
        #         "n_samples": 50000
        #     }, 
        #     "name": "Yelp",
        #     "saves_dir": "yelp"
        # },      
        
        # {
        #     "positive": {
        #         "data_filepath": "/data/madhu/imdb_dataset/processed_data/pos_reviews_train",      
        #         "n_samples": 50000
        #     },
        #     "negative": {
        #         "data_filepath": "/data/madhu/imdb_dataset/processed_data/neg_reviews_train",            
        #         "n_samples": 50000
        #     },
        #     "name": "IMDB",
        #     "saves_dir": "imdb"
        # },

        # {
        #     "positive": {
        #         "data_filepath": "/data/madhu/tripadvisor/processed_data/pos_reviews_train",
        #         "n_samples": 50000
        #     },
        #     "negative": {
        #         "data_filepath": "/data/madhu/tripadvisor/processed_data/neg_reviews_train",                
        #         "n_samples": 50000
        #     },
        #     "name": "Tripadvisor",
        #     "saves_dir": "tripadvisor"
        # },

        # {
        #     "positive": {
        #         "data_filepath": "/data/madhu/amazon-reviews-2018/processed_data/cellphones_and_accessories/pos_reviews_train",
        #         "n_samples": 50000
        #     },
        #     "negative": {
        #         "data_filepath": "/data/madhu/amazon-reviews-2018/processed_data/cellphones_and_accessories/neg_reviews_train",
        #         "n_samples": 50000
        #     },
        #     "name": "Cellphones",
        #     "saves_dir": "cellphones"
        # },

        # {
        #     "positive": {
        #         "data_filepath": "/data/madhu/amazon-reviews-2018/processed_data/pet_supplies/pos_reviews_train",
        #         "n_samples": 50000
        #     },
        #     "negative": {
        #         "data_filepath": "/data/madhu/amazon-reviews-2018/processed_data/pet_supplies/neg_reviews_train",
        #         "n_samples": 50000
        #     },
        #     "name": "Pet Supplies",
        #     "saves_dir": "pet_supplies"        
        # },
        
        # {
        #     "positive": {
        #         "data_filepath": "/data/madhu/amazon-reviews-2018/processed_data/automotive/pos_reviews_train",
        #         "n_samples": 50000
        #     },
        #     "negative": {
        #         "data_filepath": "/data/madhu/amazon-reviews-2018/processed_data/automotive/neg_reviews_train",
        #         "n_samples": 50000
        #     },
        #     "name": "Automotive",
        #     "saves_dir": "automotive"
        # },

        # {
        #     "positive": {
        #         "data_filepath": "/data/madhu/amazon-reviews-2018/processed_data/luxury_beauty/pos_reviews_train",
        #         "n_samples": 50000
        #     },
        #     "negative": {
        #         "data_filepath": "/data/madhu/amazon-reviews-2018/processed_data/luxury_beauty/neg_reviews_train",
        #         "n_samples": 50000
        #     },
        #     "name": "Luxury Beauty",
        #     "saves_dir": "luxury_beauty"
        # },

        {
            "positive": {
                "data_filepath": "/data/madhu/amazon-reviews-2018/processed_data/sports_and_outdoors/pos_reviews_train",
                "n_samples": 50000
            },
            "negative": {
                "data_filepath": "/data/madhu/amazon-reviews-2018/processed_data/sports_and_outdoors/neg_reviews_train",
                "n_samples": 50000
            },
            "name": "Sports",
            "saves_dir": "sports_and_outdoors"
        }
    ]    
    # parser = argparse.ArgumentParser()

    # ## Required parameters
    # parser.add_argument("--pos_data_path",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     )
    # parser.add_argument("--neg_data_path",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     )
    # parser.add_argument("--name",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     )
    # parser.add_argument("--n_samples",
    #                     default=None,
    #                     type=int)
    
    # args = parser.parse_args()

    for data in datasets:
        myprint(data)
        for key in ["positive", "negative"]:
            myprint(key)
            reviews = utils.read_file(data[key]["data_filepath"])
            parent_dir = os.path.dirname(data[key]["data_filepath"])
            sents = utils.get_sents(reviews)
            train_size = int(0.9*len(sents))
            split_data_dir = os.path.join(os.path.dirname(data[key]["data_filepath"]), "split_data")
            Path(split_data_dir).mkdir(parents=True, exist_ok=True)
            sents_filename_train = os.path.join(split_data_dir, key+"_reviews_train_sents")
            sents_filename_dev = os.path.join(split_data_dir, key+"_reviews_dev_sents")
            # sents_filename = data[key]["data_filepath"]+"_sents"
            utils.write_file(sents[:train_size], sents_filename_train)
            utils.write_file(sents[train_size:], sents_filename_dev)
