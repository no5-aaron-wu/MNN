#include <MNN/MNNDefine.h>
#include <math.h>
#include <MNN/Interpreter.hpp>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <sys/time.h>
#endif

template <typename T>
bool checkVectorByRelativeError(const T* result, const T* rightData, int size, float rtol) {
    MNN_ASSERT(result != nullptr);
    MNN_ASSERT(rightData != nullptr);
    MNN_ASSERT(size >= 0);

    float maxValue = 0.0f;
    for (int i = 0; i < size; ++i) {
        maxValue = fmax(fabs(rightData[i]), maxValue);
    }
    for (int i = 0; i < size; ++i) {
        if (fabs(result[i] - rightData[i]) > maxValue * rtol) {
            std::cout << i << ": right: " << rightData[i] << ", compute: " << result[i] << std::endl;
            return false;
        }
    }
    return true;
}

void getRandomData(float* data, int n, int lb, int hb, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; ++i) {
        data[i] = static_cast<float>(rand() % (hb - lb + 1) + lb);
    }
}

static inline int64_t getTimeInUs() {
    uint64_t time;
#if defined(_MSC_VER)
    LARGE_INTEGER now, freq;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&freq);
    uint64_t sec  = now.QuadPart / freq.QuadPart;
    uint64_t usec = (now.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart;
    time          = sec * 1000000 + usec;
#else
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    time = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
#endif
    return time;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: ./roiAlignDemo.out model.mnn testMode(0[default]: sync, 1: profiler) loopCnt(default=100)"
                  << std::endl;
    }

    const auto model = argv[1];

    auto testMode = 0;
    if (argc > 2) {
        testMode = ::atoi(argv[2]);
    }

    if (testMode == 0) {
        // create net and session
        auto mnnNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model));
        MNN::ScheduleConfig netConfig;
        netConfig.type      = MNN_FORWARD_CPU;
        netConfig.numThread = 4;
        auto session        = mnnNet->createSession(netConfig);

        // get input/output tensor
        auto feats  = mnnNet->getSessionInput(session, "feats");
        auto rois   = mnnNet->getSessionInput(session, "rois");
        auto output = mnnNet->getSessionOutput(session, nullptr);

        const float spatialScale = 1.f / 16;

        const std::vector<float> inputData  = {// [0, 0, :, :]
                                              -0.2280, 1.0844, -0.5641, 1.0726, -1.3492, 1.7201, 0.9563, -0.0467,
                                              0.6715, 1.9948, -0.8688, 0.7443, 1.2673, -0.8053, -0.5623, -1.0905,
                                              // [0, 1, :, :]
                                              1.3876, 0.0851, -0.2216, 0.9205, -0.2518, 0.3693, -0.8745, -0.8473,
                                              -1.2684, -0.5659, 1.7629, -1.1017, -0.6808, -0.3398, 0.3905, -1.4468,
                                              // [0, 2, :, :]
                                              0.3544, -1.3111, -0.4820, -0.7897, -0.6373, 0.2016, 0.6290, -0.4031,
                                              -1.1968, 1.2157, 0.7829, 0.0650, 1.2640, -0.1800, 0.5405, 0.5089,
                                              // [0, 3, :, :]
                                              -1.0524, -0.3499, -1.0782, 2.3311, 0.3991, -1.0441, 1.4817, -0.1174,
                                              -1.2182, 1.6637, 0.0859, 0.2630, 0.1632, -0.7833, 1.9403, 2.7318,
                                              // [0, 4, :, :]
                                              1.9020, -0.6179, -1.9794, -0.5269, 1.7679, -1.0336, 0.6070, 0.9047,
                                              0.5333, 0.0791, -1.3838, -2.0784, 0.9168, -0.2555, 1.6408, 0.7012,
                                              // [1, 0, :, :]
                                              -1.3412, 0.1940, 1.1810, -0.1982, -0.6593, 1.3539, -1.0911, 0.7103,
                                              0.0096, 0.4430, -0.0726, 0.3056, -0.5561, -0.0727, 1.3280, -0.1100,
                                              // [1, 1, :, :]
                                              2.1148, -1.5199, -1.7310, -1.2742, -0.5361, -0.1515, 0.3356, -0.4218,
                                              -0.0041, -0.1434, 1.1420, 0.5182, -0.0388, 1.4477, -0.0358, -1.3596,
                                              // [1, 2, :, :]
                                              -0.5037, -1.2696, 0.1072, 0.4771, 1.0347, -0.8084, 0.0574, -0.2152,
                                              -0.4784, -0.1899, -0.4062, -1.2771, -0.1168, 0.9634, 0.8064, 1.3298,
                                              // [1, 3, :, :]
                                              0.9451, 0.5655, 1.3067, -2.3284, -1.0857, 0.0211, 0.2268, -0.4762, 0.4864,
                                              -0.0473, 0.1158, 0.1223, -0.2063, -0.6610, -1.3957, 0.5784,
                                              // [1, 4, :, :]
                                              -0.6520, 1.7823, 1.6848, -1.3009, 0.8962, 0.7669, -0.1907, 0.0051,
                                              -0.2227, 0.0253, -0.3897, -1.2509, 1.0388, 1.2419, 1.3504, 0.9589,
                                              // [2, 0, :, :]
                                              -0.6821, -1.3088, 0.6674, 0.3037, -0.6739, 0.7570, 1.2381, 0.9869, 0.6168,
                                              0.3319, 0.0818, -1.4019, 0.4963, -0.6783, -0.3395, 0.1034,
                                              // [2, 1, :, :]
                                              -1.6103, 0.1561, -1.8178, -0.6959, 0.2309, -2.1099, 0.2700, 1.1527,
                                              -0.1562, -0.8010, -1.7923, 1.9186, 0.0420, 1.0442, 0.4630, -1.7146,
                                              // [2, 2, :, :]
                                              2.0714, 0.6615, 0.4553, -0.2865, -0.5504, 1.7192, 1.1452, -0.1363, 0.8048,
                                              0.9660, 1.3715, -1.0151, -1.2480, -0.3135, -0.3928, -0.2055,
                                              // [2, 3, :, :]
                                              0.9049, -2.9842, 0.1725, 0.6841, -0.7629, 0.0941, 0.1685, 0.2651, 0.2957,
                                              0.7492, 1.0405, -1.3762, -0.2437, 1.3722, 0.0890, 0.2810,
                                              // [2, 4, :, :]
                                              0.4548, -0.4797, 0.3163, -0.3530, 1.0514, -0.9240, 1.1464, 0.1866,
                                              -0.1598, 0.1525, 0.9954, 2.0155, -0.0096, -0.3440, 1.0122, 0.5473};
        const std::vector<float> roiData    = {//
                                            0, 0 / spatialScale, 1 / spatialScale, 2 / spatialScale, 3 / spatialScale,
                                            //
                                            2, 0.5f / spatialScale, 1 / spatialScale, 1.5f / spatialScale,
                                            2 / spatialScale};
        const std::vector<float> outputData = {
            // [0, 0, :, :]
            -1.1623, 0.2259, 1.4623, -0.3388, 0.7593, 1.5552, 0.7708, 1.1495, 1.1371,
            // [0, 1, :, :]
            0.0214, 0.1717, 0.1407, -0.7601, -0.4292, -0.0079, -1.1705, -0.8493, -0.1845,
            // [0, 2, :, :]
            -0.4720, -0.2613, 0.0319, -0.9171, -0.1042, 0.7082, -0.7867, 0.0982, 0.9430,
            // [0, 3, :, :]
            0.1572, -0.3856, -0.5978, -0.4096, -0.0499, 0.3888, -0.9880, 0.1340, 1.1124,
            // [0, 4, :, :]
            1.7902, 0.4130, -0.7743, 1.1506, 0.3367, -0.4624, 0.5972, 0.3103, -0.1272,
            // [1, 0, :, :]
            -0.5525, -0.3041, -0.0558, -0.4082, 0.0164, 0.4409, -0.1005, 0.1858, 0.4721,
            // [1, 1, :, :]
            -0.5448, -0.8687, -1.1926, -0.2118, -0.9114, -1.6111, -0.1940, -0.7859, -1.3777,
            // [1, 2, :, :]
            0.4974, 0.8451, 1.1928, -0.0466, 0.6295, 1.3057, 0.1625, 0.6847, 1.2070,
            // [1, 3, :, :]
            -0.3278, -0.5695, -0.8112, -0.5422, -0.3281, -0.1139, -0.2896, -0.0488, 0.1921,
            // [1, 4, :, :]
            0.5811, 0.0383, -0.5045, 0.6700, 0.0577, -0.5545, 0.4455, 0.0412, -0.3630};

        MNN::Tensor* featsHostTensor =
            MNN::Tensor::create<float>(feats->shape(), (void*)inputData.data(), MNN::Tensor::CAFFE);
        MNN::Tensor* roisHostTensor =
            MNN::Tensor::create<float>(rois->shape(), (void*)roiData.data(), MNN::Tensor::CAFFE);
        MNN::Tensor* outputHostTensor = MNN::Tensor::create<float>(output->shape(), nullptr, MNN::Tensor::CAFFE);

        // copy data from host tensor
        feats->copyFromHostTensor(featsHostTensor);
        rois->copyFromHostTensor(roisHostTensor);

        // run
        mnnNet->runSession(session);

        // get output data
        output->copyToHostTensor(outputHostTensor);
        if (!checkVectorByRelativeError<float>(outputHostTensor->host<float>(), outputData.data(), outputData.size(),
                                               0.001)) {
            printf("the output data not same!\n");
        }
        printf("all done!\n");
    } else {
        int loopCnt = 100;
        if (argc > 3) {
            loopCnt = ::atoi(argv[3]);
        }

        // create net and session
        auto mnnNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model));
        mnnNet->setSessionMode(MNN::Interpreter::Session_Debug);
        MNN::ScheduleConfig netConfig;
        netConfig.type      = MNN_FORWARD_CPU;
        netConfig.numThread = 4;
        auto session        = mnnNet->createSession(netConfig);

        // get input/output tensor
        auto feats  = mnnNet->getSessionInput(session, "feats");
        auto rois   = mnnNet->getSessionInput(session, "rois");
        auto output = mnnNet->getSessionOutput(session, nullptr);

        // release model
        mnnNet->releaseModel();

        auto batchSize = feats->shape()[0];
        auto channel   = feats->shape()[1];
        auto inputH    = feats->shape()[2];
        auto inputW    = feats->shape()[3];
        auto roiNum    = rois->shape()[0];
        auto roiLen    = rois->shape()[1];

        // create host tensor
        int lb = 0, hb = 100, seed = 0;
        std::vector<float> inputData(batchSize * channel * inputH * inputW);
        std::vector<float> roiData(roiNum * roiLen);
        getRandomData(inputData.data(), batchSize * channel * inputH * inputW, lb, hb, seed);
        getRandomData(roiData.data(), roiNum * roiLen, lb, hb, seed);
        for (int i = 0; i < roiNum; ++i) {
            roiData.data()[5 * i] = floorf(roiData.data()[roiLen * i] / (hb - lb) * batchSize);
        }
        MNN::Tensor* featsHostTensor =
            MNN::Tensor::create<float>(feats->shape(), (void*)inputData.data(), MNN::Tensor::CAFFE);
        MNN::Tensor* roisHostTensor =
            MNN::Tensor::create<float>(rois->shape(), (void*)roiData.data(), MNN::Tensor::CAFFE);
        MNN::Tensor* outputHostTensor = MNN::Tensor::create<float>(output->shape(), nullptr, MNN::Tensor::CAFFE);
        {
            MNN_PRINT("N=%d, C=%d, H=%d, W=%d, roiNum=%d\n", batchSize, channel, inputH, inputW, roiNum);

            int t = loopCnt;
            std::map<std::string, std::pair<float, float>> opTimes;
            std::map<std::string, std::string> opTypes;
            uint64_t opBegin = 0;

            MNN::TensorCallBackWithInfo beforeCallBack = [&](const std::vector<MNN::Tensor*>& ntensors,
                                                             const MNN::OperatorInfo* info) {
                if (opTypes.find(info->name()) == opTypes.end()) {
                    opTypes.insert(std::make_pair(info->name(), info->type()));
                }
                opBegin = getTimeInUs();
                if (opTimes.find(info->name()) == opTimes.end()) {
                    opTimes.insert(std::make_pair(info->name(), std::make_pair(0.0f, info->flops())));
                }
                return true;
            };
            MNN::TensorCallBackWithInfo afterCallBack = [&](const std::vector<MNN::Tensor*>& ntensors,
                                                            const MNN::OperatorInfo* info) {
                auto opEnd = getTimeInUs();
                float cost = (float)(opEnd - opBegin) / 1000.0f;

                opTimes[info->name()].first += cost;
                return true;
            };

            if (t > 0) {
                std::vector<float> times(t, 0.0f);
                for (int i = 0; i < t; ++i) {
                    auto begin = getTimeInUs();
                    feats->copyFromHostTensor(featsHostTensor);
                    rois->copyFromHostTensor(roisHostTensor);
                    mnnNet->runSessionWithCallBackInfo(session, beforeCallBack, afterCallBack, false);
                    output->copyToHostTensor(outputHostTensor);
                    auto end = getTimeInUs();
                    times[i] = (end - begin) / 1000.0f;
                }

                auto minTime = std::min_element(times.begin(), times.end());
                auto maxTime = std::max_element(times.begin(), times.end());
                float sum    = 0.0f;
                for (auto time : times) {
                    sum += time;
                }
                std::vector<std::pair<float, std::pair<std::string, float>>> allOpsTimes;
                float sumFlops = 0.0f;
                for (auto& iter : opTimes) {
                    allOpsTimes.push_back(
                        std::make_pair(iter.second.first, std::make_pair(iter.first, iter.second.second)));
                    sumFlops += iter.second.second;
                }

                std::sort(allOpsTimes.begin(), allOpsTimes.end());
                for (auto& iter : allOpsTimes) {
                    MNN_PRINT("%*s \t[%s] run %d average cost %f ms, %.3f %%, FlopsRate: %.3f %%\n", 4,
                              iter.second.first.c_str(), opTypes[iter.second.first].c_str(), loopCnt,
                              iter.first / (float)loopCnt, iter.first / sum * 100.0f,
                              iter.second.second / sumFlops * 100.0f);
                }
                MNN_PRINT("Avg= %f ms, min= %f ms, max= %f ms\n", sum / (float)t, *minTime, *maxTime);
            }
        }
    }
    return 0;
}