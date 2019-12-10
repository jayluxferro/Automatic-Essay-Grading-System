# all prompts with lstm
python train.py -tr data/fold_0/train.tsv -tu data/fold_0/dev.tsv -ts data/fold_0/test.tsv -p 0 --emb embeddings.w2v.txt -o output_dir
zip -r scn1.zip output_dir/

# all prompts with gru
python train.py -tr data/fold_0/train.tsv -tu data/fold_0/dev.tsv -ts data/fold_0/test.tsv -p 0 --emb embeddings.w2v.txt -o output_dir -u gru
zip -r scn2.zip output_dir/
