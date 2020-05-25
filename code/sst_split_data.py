
import os
import spacy
import random

SST_DATA_DIR = "/data/madhu/stanford-sentiment-treebank/matched_data/"

def write_file(reviews, out_file):
    with open(out_file, "w") as fout:
        for rev in reviews:
            fout.write(rev.strip("\n")+"\n")

def read_file(filename):
    reviews = []    
    with open(filename, "r") as fin:    
        for line in fin:
            line = line.strip("\n")
            reviews.append(line)
        return reviews

def split_data(reviews, size, filename_train, filename_test):
    random.shuffle(reviews)
    size = int(size)
    train_reviews = reviews[:size]
    test_reviews = reviews[size:]

    with open(filename_train, "w") as fout:
        for rev in train_reviews:
            fout.write(rev+"\n")

    with open(filename_test, "w") as fout:
        for rev in test_reviews:
            fout.write(rev+"\n")

if __name__ == "__main__":

    seed_val = 23
    random.seed(seed_val)

    pos_reviews = read_file(os.path.join(SST_DATA_DIR, "pos_reviews"))
    neg_reviews = read_file(os.path.join(SST_DATA_DIR, "neg_reviews"))    

    split_data(pos_reviews, len(pos_reviews)*3.0/4, os.path.join(SST_DATA_DIR, "pos_reviews_train"), os.path.join(SST_DATA_DIR, "pos_reviews_test"))
    split_data(neg_reviews, len(neg_reviews)*3.0/4, os.path.join(SST_DATA_DIR, "neg_reviews_train"), os.path.join(SST_DATA_DIR, "neg_reviews_test"))

    print("execution finished")  
    