# Profiling

```
nvprof ./main -m ./models/Mixtral-8x7B-Instruct-v0.1-q4_0.gguf -n 128 -p "[INST] Explain Deep Learning. [/INST]"  --n-gpu-layers 33 > ./gpuprofile_mixtral_inst_8x7b_33_layers.log 2>&1

```