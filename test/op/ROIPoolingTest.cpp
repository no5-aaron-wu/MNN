﻿//
//  ROIPoolingTest.cpp
//  MNNTests
//
//  Created by MNN on 2021/10/27.
//  Copyright ? 2018, Alibaba Group Holding Limited
//
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
// #include <MNN/expr/Optimizer.hpp>
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN;
using namespace MNN::Express;

static VARP _ROIPooling(VARP input, VARP roi, int pooledHeight, int pooledWidth, float spatialScale) {
    std::unique_ptr<RoiPoolingT> roiPooling(new RoiPoolingT);
    roiPooling->pooledHeight = pooledHeight;
    roiPooling->pooledWidth  = pooledWidth;
    roiPooling->spatialScale = spatialScale;

    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_ROIPooling;
    op->main.type  = OpParameter_RoiPooling;
    op->main.value = roiPooling.release();

    return (Variable::create(Expr::create(op.get(), {input, roi})));
}

class ROIPoolingTest : public MNNTestCase {
public:
    virtual ~ROIPoolingTest() = default;
    virtual bool run(int precision) override { return testOnBackend(MNN_FORWARD_CPU, "CPU", "ROIPooling"); }

protected:
    static bool testOnBackend(MNNForwardType type, const std::string& deviceName, const std::string& testOpName) {
        // case1
        {
            const int n = 1, c = 1, h = 16, w = 16;
            const int pooledHeight = 7, pooledWidth = 7;
            const float spatialScale = 1.f / 16;

            const std::vector<float> inputData = {// h = 0
                                                  -0.4504, -1.6300, -1.1528, 2.0047, -0.7722, 1.2869, 0.9607, -0.1543,
                                                  -1.7898, -0.4389, 1.0762, -0.5312, -0.3816, 0.5593, 2.1539, -0.8473,
                                                  // h = 1
                                                  1.2878, -0.3931, -0.5860, -2.2671, 0.1664, -0.1624, 0.7083, -0.9036,
                                                  -1.8571, -0.9804, 0.4889, -0.7063, -0.3265, -0.3187, -0.4380, 0.6685,
                                                  // h = 2
                                                  -1.0542, 0.2055, 0.9351, -0.2695, 1.0169, 0.9110, -0.3597, 0.9373,
                                                  -0.6850, 0.4412, -0.7418, 0.2520, -0.6617, -1.2510, -2.0578, 1.5503,
                                                  // h = 3
                                                  -0.0070, -0.6194, 1.1525, -0.1175, -0.5980, 0.6628, -1.5139, 0.5271,
                                                  -1.7624, -0.8540, 2.1995, 0.0201, 0.1946, 0.9929, 0.3413, -1.4626,
                                                  // h = 4
                                                  2.4488, 0.1626, 0.3751, 0.7000, -0.1860, -1.0407, -1.0444, 0.0756,
                                                  -1.4499, 0.2524, 0.3682, 1.2193, -1.3560, 2.3694, 0.5913, -1.1003,
                                                  // h = 5
                                                  -0.7432, -2.1867, -0.9452, -1.4011, 0.2582, 0.4201, 0.1170, 3.1787,
                                                  -0.4540, -1.9947, -1.9697, 1.9943, 1.2668, 0.4033, -0.1934, 1.4952,
                                                  // h = 6
                                                  -1.1622, -0.3598, 0.1791, -0.5496, 0.2055, -0.9481, -0.6539, -1.3166,
                                                  -0.2553, 1.1040, -1.1132, 0.6486, 1.3773, 0.4321, -0.6301, -0.0220,
                                                  // h = 7
                                                  0.7045, -1.3188, 0.9659, 0.3345, 0.1435, 1.4948, -1.3958, 0.8596,
                                                  -0.2846, -1.6227, 3.0450, 0.6862, -1.2075, 0.6156, -0.2682, -0.4627,
                                                  // h = 8
                                                  0.4168, -0.9499, 0.2084, 2.2114, -1.1819, -0.8128, -1.0761, -0.0629,
                                                  1.4855, -0.0506, 0.7821, -2.1390, -0.0286, 0.2027, 0.7717, -1.3940,
                                                  // h = 9
                                                  0.2336, -0.2081, 0.4518, 0.5892, 1.6259, 1.4382, 1.3699, -0.3971,
                                                  -1.0778, 0.3523, 1.3481, 0.0274, 0.8596, -1.3746, -1.5045, -0.0377,
                                                  // h = 10
                                                  0.6351, -0.8386, -0.7822, -0.2523, -0.3953, 0.0625, -0.9319, -0.4681,
                                                  -1.0337, -0.4972, -2.3686, -0.0097, -0.4136, 1.6763, 0.2910, -1.6629,
                                                  // h = 11
                                                  -1.4581, 0.6477, -0.9243, -0.7744, -1.4067, -0.4087, -0.3171, 1.6140,
                                                  -0.1184, -1.4282, -0.1889, -1.5489, 0.9621, 0.0987, 0.0585, 0.5535,
                                                  // h = 12
                                                  0.1638, 1.4905, -0.7721, -0.6452, 1.3665, -2.0732, -0.0865, 1.2407,
                                                  -1.0586, 0.5204, 1.2189, -0.5717, -0.3990, 0.7323, -0.5211, 0.4576,
                                                  // h = 13
                                                  -0.6822, -0.0130, 0.6325, 1.7409, -0.4098, -0.1671, 1.3580, -1.3922,
                                                  -1.1549, -0.5770, 0.0470, 1.8368, 0.4054, -1.2064, 1.1032, -0.4081,
                                                  // h = 14
                                                  -1.6945, -0.3223, -0.5065, -0.4902, 0.3292, 0.7854, -0.7723, -0.4000,
                                                  0.8361, -2.2352, 0.8832, -0.6669, 0.8367, 0.2200, 0.6050, -0.8180,
                                                  // h = 15
                                                  1.2200, 1.3076, -0.8782, 1.5257, -0.7750, 0.0775, -1.5619, 0.6683,
                                                  -0.3300, 1.3241, -0.0514, 0.3862, 1.1214, 0.0751, 0.0594, -0.4008};
            const std::vector<float> roiData   = {0, 5 / spatialScale, 10 / spatialScale, 10 / spatialScale,
                                                15 / spatialScale};
            // the output data calculated by torchvision.ops.roi_pool function using same input data
            const std::vector<float> outputData = {//
                                                   0.0625, 0.0625, -0.4681, -0.4681, -0.4972, -0.4972, -2.3686,
                                                   //
                                                   0.0625, 0.0625, 1.6140, 1.6140, -0.1184, -0.1889, -0.1889,
                                                   //
                                                   -0.4087, -0.0865, 1.6140, 1.6140, 0.5204, 1.2189, 1.2189,
                                                   //
                                                   -0.1671, 1.3580, 1.3580, 1.2407, 0.5204, 1.2189, 1.2189,
                                                   //
                                                   0.7854, 1.3580, 1.3580, 0.8361, 0.8361, 0.8832, 0.8832,
                                                   //
                                                   0.7854, 0.7854, 0.6683, 0.8361, 1.3241, 1.3241, 0.8832,
                                                   //
                                                   0.0775, 0.0775, 0.6683, 0.6683, 1.3241, 1.3241, -0.0514};

            auto input = _Input({n, c, h, w}, NCHW, halide_type_of<float>());
            auto roi   = _Input({1, 1, 1, 5}, NCHW, halide_type_of<float>());
            auto output =
                _ROIPooling(_Convert(input, NC4HW4), _Convert(roi, NC4HW4), pooledHeight, pooledWidth, spatialScale);
            output = _Convert(output, NCHW);
            ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
            ::memcpy(roi->writeMap<float>(), roiData.data(), roiData.size() * sizeof(float));
            if (!checkVectorByRelativeError<float>(output->readMap<float>(), outputData.data(), outputData.size(),
                                                   0.001)) {
                MNN_ERROR("%s(%s) test failed!\n", testOpName.c_str(), deviceName.c_str());
                return false;
            }
        }
        // case2
        {
            const int n = 3, c = 8, h = 4, w = 4;
            const int pooledHeight = 3, pooledWidth = 3;
            const float spatialScale = 1.f / 16;

            const std::vector<float> inputData  = {// [0, 0, :, :]
                                                  0.8920, -0.3827, 0.3091, -0.4437, 1.7567, -0.1442, -0.2077, 2.5444,
                                                  -1.0456, -1.4245, 0.3565, -0.7401, 1.6603, -0.6706, 0.9318, 0.5754,
                                                  // [0, 1, :, :]
                                                  -1.6634, -0.6708, -0.6057, -1.4094, -1.3711, -1.6219, 0.9869, 1.3267,
                                                  0.4860, 1.1062, 0.5772, 0.7650, 1.3054, -1.4838, -0.5682, -1.5017,
                                                  // [0, 2, :, :]
                                                  -0.1707, -0.7036, -0.8317, 1.1863, -0.5500, 0.7535, 0.2407, -1.8227,
                                                  0.5328, 0.0397, -0.3823, 0.1057, -1.7117, 0.7832, -0.4010, 2.2293,
                                                  // [0, 3, :, :]
                                                  -0.7436, 0.6582, -0.2531, 1.7429, -0.8094, 1.8887, -1.1704, -0.3731,
                                                  0.2225, -0.7276, -0.2258, -0.1122, 1.4591, -1.1133, -1.7965, 0.2713,
                                                  // [0, 4, :, :]
                                                  0.4072, 0.4878, -0.6191, -0.3366, -1.2363, -1.1270, -0.2438, 0.4199,
                                                  1.8140, -1.1963, 1.4354, 0.0711, -3.0981, 0.5496, 0.2768, -0.3147,
                                                  // [0, 5, :, :]
                                                  -0.7117, 0.5946, -2.2895, -1.4817, 0.2315, 0.0660, 0.6461, -0.1594,
                                                  0.2676, 1.3061, -0.0610, -0.5869, 0.6408, 0.0247, -0.0717, 2.1988,
                                                  // [0, 6, :, :]
                                                  -0.0542, -0.4836, 0.5969, -1.1248, 0.7609, -0.5489, -0.4592, 0.2301,
                                                  -0.1900, 0.2825, 1.2588, 1.0059, -0.8721, 0.9606, 0.9456, 0.4636,
                                                  // [0, 7, :, :]
                                                  1.7054, -0.8216, 0.1668, 0.2457, -0.0889, -0.4620, 0.6117, -2.0759,
                                                  0.1798, -1.1737, 0.5726, -0.5008, -1.4096, -0.6150, -0.7288, 1.2607,
                                                  // [1, 0, :, :]
                                                  0.6050, -0.3634, -0.7518, 1.5528, 0.3748, -1.7550, 1.7403, -0.2321,
                                                  1.4193, 1.4177, -1.3958, -1.6701, 0.3498, -0.7555, -0.6739, 0.9091,
                                                  // [1, 1, :, :]
                                                  1.4023, -1.9502, -0.6308, -1.6386, -0.3561, 0.5153, 0.7248, -0.5405,
                                                  -0.9245, 0.8007, -0.3660, 0.8325, -0.0450, 0.4732, -0.8307, 0.0350,
                                                  // [1, 2, :, :]
                                                  0.3842, -0.1005, 2.2814, 1.2281, -1.1018, -0.0818, -0.3758, 0.3081,
                                                  -1.0054, -0.4555, -0.0503, -0.1661, -1.7964, 1.7836, -0.0562, 0.3733,
                                                  // [1, 3, :, :]
                                                  -1.3536, -0.6981, -0.3674, -0.4937, -0.5134, 0.1983, -0.1889, -1.0043,
                                                  -2.5630, -0.4750, 0.2788, 0.5176, -1.3869, -1.5367, -0.4633, -0.3842,
                                                  // [1, 4, :, :]
                                                  -0.2090, 1.4044, 0.7152, 0.1843, -2.0927, 1.1251, -0.6426, -0.2249,
                                                  0.9406, 0.2157, -2.1107, 0.5089, 0.3984, 0.8583, -0.1455, -1.0221,
                                                  // [1, 5, :, :]
                                                  -0.2359, 0.3414, -1.2478, -2.9151, 0.5235, 0.9570, 0.3158, -0.2351,
                                                  1.0386, 0.6343, 0.8563, -1.9042, -0.3495, -1.7429, 0.3704, -0.3070,
                                                  // [1, 6, :, :]
                                                  0.7652, -0.9303, 1.2019, 0.1853, -1.0821, -0.1062, 0.9089, 1.8699,
                                                  -0.7525, 0.6330, 0.7048, -0.6174, 1.1180, -0.7515, -0.5902, 0.5961,
                                                  // [1, 7, :, :]
                                                  1.0532, 0.5334, -0.8172, -0.2425, 2.1320, -0.4429, 0.8101, -0.6770,
                                                  -0.2732, -0.8624, -1.0899, -1.5551, -2.8009, -0.1296, 0.3150, 0.0271,
                                                  // [2, 0, :, :]
                                                  -0.8222, 0.7508, -0.6819, 1.5907, 0.9431, -1.5707, -0.2142, 1.4580,
                                                  -0.9117, 1.6879, -1.2356, 0.4487, 0.6762, 0.0263, -2.1004, -0.5938,
                                                  // [2, 1, :, :]
                                                  -1.1185, 0.4820, -0.5047, -0.2872, 0.0815, -0.9545, -0.6277, 1.6142,
                                                  -0.1267, -1.2646, 0.0303, -1.1049, -0.4473, -1.6977, -0.7080, -0.5386,
                                                  // [2, 2, :, :]
                                                  0.2484, 1.0132, -0.0780, 1.2668, -0.1218, 0.1850, 0.0552, -0.2980,
                                                  -0.5786, -1.3575, -1.0561, -0.2138, 0.6092, 0.0781, -0.2969, -1.3248,
                                                  // [2, 3, :, :]
                                                  0.2388, -1.8207, 0.6686, 1.5191, 0.7969, 0.0628, -0.6797, 2.5548,
                                                  -0.0785, -0.7654, -0.8120, 0.8116, -0.7039, 1.1555, -2.4323, 0.1931,
                                                  // [2, 4, :, :]
                                                  -1.0638, 2.2248, 0.1709, -0.0818, -0.1393, 0.4214, 0.7031, 0.7916,
                                                  0.2808, 1.5516, 0.3634, 1.7315, -1.2339, 0.4970, 0.1331, 0.6771,
                                                  // [2, 5, :, :]
                                                  -2.0071, -0.6452, -0.3309, -1.4164, 1.1681, -1.0288, 1.6746, -1.3841,
                                                  -0.6872, 0.5344, 0.6790, 1.8175, 0.3682, 0.6544, 0.8827, -1.1641,
                                                  // [2, 6, :, :]
                                                  -0.6468, 0.1078, 1.0659, -2.0334, -0.2771, -0.6038, 0.2742, -1.2950,
                                                  -1.3631, 0.0617, 0.0035, 0.7741, 0.1598, 0.1088, -1.1016, 0.3756,
                                                  // [2, 7, :, :]
                                                  0.5522, -0.5326, 0.2930, 1.6583, 0.6128, 1.1916, -0.6481, 0.1856,
                                                  1.5111, 0.6582, -0.6021, 0.7241, 0.3866, 1.1469, 0.0162, 0.1743};
            const std::vector<float> roiData    = {2, 1 / spatialScale, 2 / spatialScale, 3 / spatialScale,
                                                3 / spatialScale};
            const std::vector<float> outputData = {
                // [0, 0, :, :]
                1.6879, -1.2356, 0.4487, 1.6879, -1.2356, 0.4487, 0.0263, -2.1004, -0.5938,
                // [0, 1, :, :]
                -1.2646, 0.0303, -1.1049, -1.2646, 0.0303, -0.5386, -1.6977, -0.7080, -0.5386,
                // [0, 2, :, :]
                -1.3575, -1.0561, -0.2138, 0.0781, -0.2969, -0.2138, 0.0781, -0.2969, -1.3248,
                // [0, 3, :, :]
                -0.7654, -0.8120, 0.8116, 1.1555, -0.8120, 0.8116, 1.1555, -2.4323, 0.1931,
                // [0, 4, :, :]
                1.5516, 0.3634, 1.7315, 1.5516, 0.3634, 1.7315, 0.4970, 0.1331, 0.6771,
                // [0, 5, :, :]
                0.5344, 0.6790, 1.8175, 0.6544, 0.8827, 1.8175, 0.6544, 0.8827, -1.1641,
                // [0, 6, :, :]
                0.0617, 0.0035, 0.7741, 0.1088, 0.0035, 0.7741, 0.1088, -1.1016, 0.3756,
                // [0, 7, :, :]
                0.6582, -0.6021, 0.7241, 1.1469, 0.0162, 0.7241, 1.1469, 0.0162, 0.1743};

            auto input  = _Input({n, c, h, w}, NCHW, halide_type_of<float>());
            auto roi    = _Input({1, 5}, NCHW, halide_type_of<float>());
            auto output = _ROIPooling(_Convert(input, NC4HW4), roi, pooledHeight, pooledWidth, spatialScale);
            output      = _Convert(output, NCHW);
            ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
            ::memcpy(roi->writeMap<float>(), roiData.data(), roiData.size() * sizeof(float));
            if (!checkVectorByRelativeError<float>(output->readMap<float>(), outputData.data(), outputData.size(),
                                                   0.001)) {
                MNN_ERROR("%s(%s) test failed!\n", testOpName.c_str(), deviceName.c_str());
                return false;
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(ROIPoolingTest, "op/ROIPooling");
