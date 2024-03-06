python ../tools/preprocess_data.py \
       --input /scratch/user/siweicui/Megatron-LM/experiments/codeparrot_data.json \
       --output-prefix /scratch/user/siweicui/Megatron-LM/experiments/data/codeparrot \
       --vocab-file /scratch/user/siweicui/Megatron-LM/vocab.json \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file /scratch/user/siweicui/Megatron-LM/merges.txt \
       --json-keys content \
       --workers 64 \
       --partitions 4 \
       --append-eod

