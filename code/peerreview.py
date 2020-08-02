import os
import spacy
import numpy as np
import random
import json
import utils
from pathlib import Path
DATASET_DIR = "/data/madhu/PeerRead/data/acl_2017/train/reviews"
# data_category = "train"
# IMDB_DATASET_PATH = "/data/madhu/imdb_dataset/aclImdb/test/"

def read_files(DATASET_DIR):
    # files_dir = os.path.join(DATASET_DIR)
    
    onlyfiles = [f for f in os.listdir(DATASET_DIR) if os.path.isfile(os.path.join(DATASET_DIR, f))]
    # f_out = open(os.path.join(out_dir, catgeory+"_reviews_"+data_category), "w")

    pos_reviews = []
    neg_reviews = []
    for file in onlyfiles:
        print(file)
        file_path = os.path.join(DATASET_DIR, file)
        with open(file_path, "r") as f:
            line = f.readline().strip("\n")
            # line = line.strip("\n")
            # print(line)
            json_content = json.loads(line)
            reviews = json_content["reviews"]
            for rev in reviews:
                score = float(rev["RECOMMENDATION"])
                if score <=3:
                    neg_reviews.append(rev["comments"].replace("\n", " ").strip("\n"))
                elif score >= 4:
                    pos_reviews.append(rev["comments"].replace("\n", " ").strip("\n"))

    return pos_reviews, neg_reviews


nlp = spacy.load("en_core_web_md")
tokenizer = nlp.Defaults.create_tokenizer(nlp)
   

if __name__ == "__main__":
    seed_val = 23
    np.random.seed(seed_val)
    random.seed(seed_val)

    pos_reviews, neg_reviews = read_files(DATASET_DIR)

    out_dir = os.path.join(DATASET_DIR, "processed_data")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    utils.write_file(pos_reviews, os.path.join(out_dir, 'pos_reviews'))
    utils.write_file(neg_reviews, os.path.join(out_dir, 'neg_reviews'))
    
    print("Execution finished")

