
import spacy
import os
from pathlib import Path
import utils
import pprint
import argparse

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        )    
    parser.add_argument("--name",
                        default=None,
                        type=str,
                        required=True,
                        )
    parser.add_argument("--n_samples",
                        default=None,
                        type=int
                        )
    
    args = parser.parse_args()

    for data_file in ["pos_reviews_test", "neg_reviews_test"]:
        myprint(data_file)
        reviews = utils.read_file(os.path.join(args.data_dir, data_file))
        sents = utils.get_sents(reviews)
        split_data_dir = os.path.join(args.data_dir, "split_data")
        Path(split_data_dir).mkdir(parents=True, exist_ok=True)
        sents_filename_test = os.path.join(split_data_dir, data_file+"_sents")
        utils.write_file(sents, sents_filename_test)
