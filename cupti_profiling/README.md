# Cupti profiler

Module based off of the sample CUPTI programs.


### Data structures and version
This profiler is based off of the callback_profiler sample found in CUPTI sample director. This implementation uses CUDA 12.3 and makes use of the premade data structures
found in the smaples directory. Make sure to double check the version of cuda you are using and if the data structures are supported.


### Configuration
In the constructor in profiler.cc, you can find the metrics being surveyed for. It being stored in the pProfiling data structure. The profiler makes use of Autorange profiling but can be switched to Userrange.

### LLAMA.CPP Profiling
While using the llama.cpp, the collection of metrics is quite slow. This is due to the multiple passes being done on the gpu by the CUPTI API.

### Usage
In order to use the profiler, simply reate the CuptiProfiler object. Then use the start() function at the beginning of te area you wish to profile and the stop() function at the end.
