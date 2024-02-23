#include "profiler.h"

#include <fstream>
#include <iostream>

bool WriteBinaryFile(const char* pFileName, const std::vector<uint8_t>& data)
{
    FILE* fp = fopen(pFileName, "wb");
    if (fp)
    {
        if (data.size())
        {
            fwrite(&data[0], 1, data.size(), fp);
        }
        fclose(fp);
    }
    else
    {
        std::cout << "ERROR!! Failed to open " << pFileName << "\n";
        std::cout << "Make sure the file or directory has write access\n";
        return false;
    }
    return true;
}

bool ReadBinaryFile(const char* pFileName, std::vector<uint8_t>& image)
{
    FILE* fp = fopen(pFileName, "rb");
    if (!fp)
    {
        std::cout << "ERROR!! Failed to open " << pFileName << "\n";
        std::cout << "Make sure the file or directory has read access\n";
        return false;
    }

    fseek(fp, 0, SEEK_END);
    const long fileLength = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (!fileLength)
    {
        std::cout << pFileName << " has zero length\n";
        fclose(fp);
        return false;
    }

    image.resize((size_t)fileLength);
    fread(&image[0], 1, image.size(), fp);
    fclose(fp);
    return true;
}


void CuptiProfiler::EnableProfiling(ProfilingData *pProfilingData){
    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = { CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE };
    CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
}



void CuptiProfiler::DisableProfiling(ProfilingData *pProfilingData){
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = { CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE };
    CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
    pProfilingData->allPassesSubmitted = true;
    CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = { CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE };
    CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams));
}

void CuptiProfiler::BeginSession(ProfilingData *pProfilingData){
    CUpti_Profiler_BeginSession_Params beginSessionParams = { CUpti_Profiler_BeginSession_Params_STRUCT_SIZE };
    beginSessionParams.ctx = NULL;
    beginSessionParams.counterDataImageSize = pProfilingData->counterDataImage.size();
    beginSessionParams.pCounterDataImage = &pProfilingData->counterDataImage[0];
    beginSessionParams.counterDataScratchBufferSize = pProfilingData->counterDataScratchBuffer.size();
    beginSessionParams.pCounterDataScratchBuffer = &pProfilingData->counterDataScratchBuffer[0];
    beginSessionParams.range = pProfilingData->profilerRange;
    beginSessionParams.replayMode = pProfilingData->profilerReplayMode;
    beginSessionParams.maxRangesPerPass = pProfilingData->numRanges;
    beginSessionParams.maxLaunchesPerPass = pProfilingData->numRanges;
    CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));
}

void CuptiProfiler::SetConfig(ProfilingData *pProfilingData){
    CUpti_Profiler_SetConfig_Params setConfigParams = { CUpti_Profiler_SetConfig_Params_STRUCT_SIZE };
    setConfigParams.pConfig = &pProfilingData->configImage[0];
    setConfigParams.configSize = pProfilingData->configImage.size();
    setConfigParams.passIndex = 0;
    CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));
}

void CuptiProfiler::CreateCounterDataImage(
    int numRanges,
    std::vector<uint8_t>& counterDataImagePrefix,
    std::vector<uint8_t>& counterDataScratchBuffer,
    std::vector<uint8_t>& counterDataImage)
{
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
    counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
    counterDataImageOptions.maxNumRanges = numRanges;
    counterDataImageOptions.maxNumRangeTreeNodes = numRanges;
    counterDataImageOptions.maxRangeNameLength = 64;

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = { CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE };
    calculateSizeParams.pOptions = &counterDataImageOptions;
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = { CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE };
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.pOptions = &counterDataImageOptions;
    initializeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    counterDataImage.resize(calculateSizeParams.counterDataImageSize);
    initializeParams.pCounterDataImage = &counterDataImage[0];
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = { CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE };
    scratchBufferSizeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    scratchBufferSizeParams.pCounterDataImage = initializeParams.pCounterDataImage;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));
    counterDataScratchBuffer.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = { CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE };
    initScratchBufferParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    initScratchBufferParams.pCounterDataImage = initializeParams.pCounterDataImage;
    initScratchBufferParams.counterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize;
    initScratchBufferParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));
}

void CuptiProfiler::SetupProfiling(ProfilingData *pProfilingData){
    // Generate configuration for metrics, this can also be done offline.
    NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
    NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));

    if (pProfilingData->metricNames.size())
    {
        if (!NV::Metric::Config::GetConfigImage(pProfilingData->chipName, pProfilingData->metricNames, pProfilingData->configImage))
        {
            std::cout << "Failed to create configImage" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (!NV::Metric::Config::GetCounterDataPrefixImage(pProfilingData->chipName, pProfilingData->metricNames, pProfilingData->counterDataImagePrefix))
        {
            std::cout << "Failed to create counterDataImagePrefix" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        std::cout << "No metrics provided to profile" << std::endl;
        exit(EXIT_FAILURE);
    }

    CreateCounterDataImage(pProfilingData->numRanges, pProfilingData->counterDataImagePrefix,
                           pProfilingData->counterDataScratchBuffer, pProfilingData->counterDataImage);

    BeginSession(pProfilingData);
    SetConfig(pProfilingData);
}

void CuptiProfiler::StopProfiling()
{
    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = { CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE };
    CUpti_Profiler_EndSession_Params endSessionParams = { CUpti_Profiler_EndSession_Params_STRUCT_SIZE };
    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};

    CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
    CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));
    CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));

    // Dump counterDataImage and counterDataScratchBuffer in file.
    WriteBinaryFile(pProfilingData->counterDataFileName.c_str(), pProfilingData->counterDataImage);
    WriteBinaryFile(pProfilingData->counterDataSBFileName.c_str(), pProfilingData->counterDataScratchBuffer);
}

void CuptiProfiler::ProfilingCallbackHandler(
    void *pUserData,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callbackId,
    void *pCallbackData)
{
    ProfilingData* pProfilingData = (ProfilingData *)(pUserData);
    const CUpti_CallbackData *pCallbackInfo = (CUpti_CallbackData *)pCallbackData;
    //pProfilingData->kernelName = pCallbackInfo->symbolName;

    switch (domain)
    {
        case CUPTI_CB_DOMAIN_DRIVER_API:
        {
            switch (callbackId)
            {
                case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
                {
                    if (pCallbackInfo->callbackSite == CUPTI_API_ENTER)
                    {

		      if (callbackId == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
                        callbackId == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz)
                        {
			    pProfilingData->kernelName = pCallbackInfo->symbolName;
			}
			else
			{
			    pProfilingData->kernelName = pCallbackInfo->functionName;
			}
		      
                        EnableProfiling(pProfilingData);
                    }
                    else
                    {
                        DisableProfiling(pProfilingData);
                    }
                }
                break;
                default:
                    break;
            }
            break;
        }
        case CUPTI_CB_DOMAIN_RESOURCE:
        {
            switch (callbackId)
            {
                case CUPTI_CBID_RESOURCE_CONTEXT_CREATED:
                {
                    SetupProfiling(pProfilingData);
                    pProfilingData->bProfiling = true;
                }
                break;
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }

}

CuptiProfiler::CuptiProfiler(){
  CUdevice cuDevice = 0;
  int deviceCount, deviceNum = 0;
  int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
  DRIVER_API_CALL(cuInit(0));
  DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0)
    {
        printf("Warning: There is no device supporting CUDA.\nWaiving test.\n");
        exit(EXIT_WAIVED);
    }

    DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));

    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
    printf("Compute Capability of Device: %d.%d\n", computeCapabilityMajor, computeCapabilityMinor);

    // Initialize profiler API support and test device compatibility.
    CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));
    CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
    params.cuDevice = deviceNum;
    params.api = CUPTI_PROFILER_RANGE_PROFILING;
    CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

    if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
    {
        ::std::cerr << "Unable to profile on device " << deviceNum << ::std::endl;

        if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice architecture is not supported" << ::std::endl;
        }

        if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice sli configuration is not supported" << ::std::endl;
        }

        if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice vgpu configuration is not supported" << ::std::endl;
        }
        else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
        {
            ::std::cerr << "\tdevice vgpu configuration disabled profiling support" << ::std::endl;
        }

        if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice confidential compute configuration is not supported" << ::std::endl;
        }

        if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << ::std::endl;
        }

        if (params.wsl == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tWSL is not supported" << ::std::endl;
        }
        exit(EXIT_WAIVED);
    }

    pProfilingData = new ProfilingData();
    pProfilingData->metricNames = {"sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__sass_thread_inst_executed_op_fp16_pred_on.sum",  
    "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fp64_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_integer_pred_on.sum"};
    pProfilingData->profilerReplayMode = CUPTI_KernelReplay;

    CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    getChipNameParams.deviceIndex = deviceNum;
    CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
    pProfilingData->chipName = getChipNameParams.pChipName;
}

CuptiProfiler::~CuptiProfiler(){
  if(pProfilingData != nullptr)
    delete pProfilingData;
}

void CuptiProfiler::start(){
  CUpti_SubscriberHandle subscriber;
    CUPTI_API_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)(ProfilingCallbackHandler), pProfilingData));
    CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED));
    CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
}

void CuptiProfiler::stop(){
    StopProfiling();
    pProfilingData->bProfiling = false;
    NV::Metric::Eval::PrintMetricValues(pProfilingData->chipName, pProfilingData->counterDataImage, pProfilingData->metricNames);
   
    delete pProfilingData;
    pProfilingData = nullptr;
}
