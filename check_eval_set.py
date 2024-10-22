import random
import numpy as np
import hashlib
import os
from pathlib import Path
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--train_file_path', type=str, default='data/train')
parser.add_argument('--test_file_path', type=str, default='data/test')

args = parser.parse_args()

train_file_path = args.train_file_path
test_file_path = args.test_file_path

train_hash_set = set()

num_test = 0
num_duplicate = 0
# Load training hashes
for path, _, files in os.walk(Path(train_file_path)):
    for name in files:
        if 'expert' in name:
            p = os.path.join(path, name)
            with open(p, 'r') as f:
                trajectory = json.load(f)
            grid_train_str = str(trajectory['in_grid'])
            train_hash_set.add(hashlib.sha256(grid_train_str.encode()).hexdigest())

# Check evaluation grids
for path, _, files in os.walk(Path(test_file_path)):
    for name in files:
        if 'expert' in name:
            num_test += 1
            p = os.path.join(path, name)
            with open(p, 'r') as f:
                trajectory = json.load(f)
            grid_eval_str = str(trajectory['in_grid'])
            eval_hash = hashlib.sha256(grid_eval_str.encode()).hexdigest()
            if eval_hash in train_hash_set:
                num_duplicate += 1

print(f"Number of Duplicate: {num_duplicate} of {num_test}")
