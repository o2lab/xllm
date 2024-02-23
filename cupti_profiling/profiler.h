#ifndef PROFILER_H
#define PROFILER_H
// System headers
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

// CUPTI headers
#include "helper_cupti.h"
#include <cupti_target.h>
#include <cupti_callbacks.h>
#include <cupti_driver_cbid.h>
#include <cupti_profiler_target.h>

// Perfworks headers
#include <nvperf_host.h>

// Make use of example code wrappers for NVPW calls.
#include "extensions/include/profilerhost_util/Eval.h"
#include "extensions/include/profilerhost_util/Metric.h"
//#include "extensions/include/c_util/FileOp.h"

// Structures
typedef struct ProfilingData_t
{
    int numRanges = 2;
    bool bProfiling = false;
    std::string chipName;
    std::vector<std::string> metricNames;
    std::string counterDataFileName = "SimpleCupti.counterdata";
    std::string counterDataSBFileName = "SimpleCupti.counterdataSB";
    CUpti_ProfilerRange profilerRange = CUPTI_AutoRange;
    CUpti_ProfilerReplayMode profilerReplayMode = CUPTI_KernelReplay;
    bool allPassesSubmitted = true;
    std::vector<uint8_t> counterDataImagePrefix;
    std::vector<uint8_t> configImage;
    std::vector<uint8_t> counterDataImage;
    std::vector<uint8_t> counterDataScratchBuffer;
  std::string kernelName;
} ProfilingData;

class CuptiProfiler{

  
  ProfilingData* pProfilingData;
  
  
 public:
  CuptiProfiler();
  ~CuptiProfiler();
  
  static void EnableProfiling(ProfilingData *pProfilingData);
  static void DisableProfiling(ProfilingData *pProfilingData);
  static void BeginSession(ProfilingData *pProfilingData);
  static void SetConfig(ProfilingData *pProfilingData);
  static void CreateCounterDataImage(int numRanges,
			      std::vector<uint8_t>& counterDataImagePrefix,
			      std::vector<uint8_t>& counterDataScratchBuffer,
			      std::vector<uint8_t>& counterDataImage);

  static void SetupProfiling(ProfilingData *pProfilingData);
  void StopProfiling();

  static void ProfilingCallbackHandler(
    void *pUserData,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callbackId,
    void *pCallbackData);
  void start();
  void stop();
  
};
 
#endif
