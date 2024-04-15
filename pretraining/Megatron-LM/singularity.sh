export SINGULARITY_CACHEDIR=$SCRATCH/.singularity

module load WebProxy 

export SINGULARITY_BINDPATH="/scratch,$TMPDIR"
cd ./Megatron-LM/experiments
singularity run --nv ./Megatron-2310.sif 