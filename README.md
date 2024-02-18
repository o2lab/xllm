# XLLM

Toolkit work in progress for Large Language Model (LLM) profiling and inference acceleration.


### RoadMap

- [X] llama.cpp profiling
- [X] GPU profiling metrics 



### LLAMA.CPP Profiling

> Clone the [llama.cpp](https://github.com/ggerganov/llama.cpp).
  We tested with llama.cpp version b1752, you may checkout the same version via `git checkout b1752`
  Copy files in [xllm profiler](https://github.com/o2lab/xllm/tree/master/kineto/llama.cpp) to your local clone of llama.cpp.
  Change the CUDA to point to your local install in [Makefile](https://github.com/o2lab/xllm/blob/f4a769363899bec71904d575d29e0b51a42b5018/kineto/llama.cpp/Makefile#L600).
  Compile and install the [libkineto](https://github.com/pytorch/kineto/tree/main/libkineto) dependency. Please feel free to use our [customized kineto](https://github.com/o2lab/xllm/tree/master/kineto/kineto) if you are running on TAMU HPRC.
  Update the [Makefile](https://github.com/o2lab/xllm/blob/f4a769363899bec71904d575d29e0b51a42b5018/kineto/llama.cpp/Makefile#L611).


To start profiling, simply run `kineto_profiler` under `llama.cpp/profiler`, with the same argument as running `./main` in [llama.cpp](https://github.com/ggerganov/llama.cpp).


#### Trace Analysis

We leverage the [Holistic Trace Analysis (HTA)](https://hta.readthedocs.io/en/latest/) to provide insights on llama.cpp on LLM inference. We provide [jupyter notebooks](https://github.com/o2lab/xllm/tree/master/kineto/tracing) for initial tracing result analyses to play with.



### GPU profiling Metrics 

Based on the CUDA API and CUPTI API, we can collect available GPU profiling metrics with [tensor_usage_collector](https://github.com/o2lab/xllm/blob/master/profiling/profiler/tensor_usage_collector.cpp). 

After running `make` under `xllm/profiling/profiler` (Please update the [Makefile](https://github.com/o2lab/xllm/blob/master/profiling/profiler/Makefile#L11) to your desired CUDA version matching the CUDA displayed in `nvidia-smi`, especially when you have multiple CUDA installations), run `tensor_usage_collector `. All available metrics will be collected into `tensor_usage_results.csv` in the same directory.

Results can be analyzed with jupyter notebooks under [this folder](https://github.com/o2lab/xllm/tree/master/profiling).


### Models
Mixtral-8x7B-Instruct: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1


