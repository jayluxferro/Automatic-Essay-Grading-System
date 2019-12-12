#!/bin/bash
# all prompts with lstm
cd data && python3 preprocess_asap.py -i training_set_rel3.tsv
cd ..
python train.py -tr data/fold_0/train.tsv -tu data/fold_0/dev.tsv -ts data/fold_0/test.tsv -p 1 --emb embeddings.w2v.txt -o output_dir -u lstm

