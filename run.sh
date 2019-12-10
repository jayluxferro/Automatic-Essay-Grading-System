# all prompts with lstm
for x in $(seq 1 8);
do
  python train.py -tr data/fold_0/train.tsv -tu data/fold_0/dev.tsv -ts data/fold_0/test.tsv -p $x --emb embeddings.w2v.txt -o output_dir -u lstm
  zip -r scn-lstm-$x.zip output_dir/
done

# all prompts with gru
for x in $(seq 1 8);
do
  python train.py -tr data/fold_0/train.tsv -tu data/fold_0/dev.tsv -ts data/fold_0/test.tsv -p $x --emb embeddings.w2v.txt -o output_dir -u gru
  zip -r scn-gru-$x.zip output_dir/
done
