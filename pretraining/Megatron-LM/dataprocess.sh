python tools/preprocess_data.py \
       --input /scratch/user/siweicui/Megatron-LM/experiments/codeparrot_data.json \
       --output-prefix /scratch/user/siweicui/Megatron-LM/experiments/codeparrot \
       --vocab /scratch/user/siweicui/Megatron-LM/vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file /scratch/user/siweicui/Megatron-LM/merges.txt \
       --json-keys content \
       --workers 32 \
       --chunk-size 25 \
       --append-eod