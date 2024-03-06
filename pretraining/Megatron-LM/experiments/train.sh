GPUS_PER_NODE=2
MASTER_ADDR=localhost
export CUDA_DEVICE_MAX_CONNECTIONS=1
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
CHECKPOINT_PATH=/scratch/user/siweicui/Megatron-LM/experiments/codeparrot-small
VOCAB_FILE=/scratch/user/siweicui/Megatron-LM/vocab.json
MERGE_FILE=/scratch/user/siweicui/Megatron-LM/merges.txt
DATA_PATH=/scratch/user/siweicui/Megatron-LM/experiments/codeparrot_0_content_document
GPT_ARGS="--num-layers 12
--hidden-size 768
--num-attention-heads 12
--seq-length 1024
--max-position-embeddings 1024
--micro-batch-size 12
--global-batch-size 192
--lr 0.0005
--train-iters 1500
--lr-decay-iters 1500
--lr-decay-style cosine
--lr-warmup-iters 200
--weight-decay .1
--adam-beta2 .999
--fp16
--log-interval 10
--save-interval 200
--eval-interval 200
--eval-iters 10
"
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"
torchrun $DISTRIBUTED_ARGS \
        ../pretrain_gpt.py \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        $GPT_ARGS \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        $TENSORBOARD_ARGS