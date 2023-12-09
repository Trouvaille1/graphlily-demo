#ifndef DEMO_GLOBAL_H_
#define DEMO_GLOBAL_H_

#include <string>
#include <cstdlib>

#include "ap_fixed.h"
#include "xcl2.hpp"

namespace demo {
const std::string root_path = get_root_path();
const std::string device_name = "xilinx_u280_gen3x16_xdma_1_202211_1";

const uint32_t num_hbm_channels = 16;

// Data types (please change this according to the kernel!)
// using val_t = unsigned;
using val_t = ap_ufixed<32, 8, AP_RND, AP_SAT>;//HLS任意精度
// using val_t = float;

const int UFIXED_INF = 255;

// Operation type, named as k<opx><op+>
enum OperationType {
    kMulAdd = 0,
    kLogicalAndOr = 1,
    kAddMin = 2,
};

// 半环
struct SemiringType {
    OperationType op;
    val_t one;  // <x>单位元
    val_t zero;  // <+>单位元
};

const SemiringType ArithmeticSemiring = {kMulAdd, 1, 0};
const SemiringType LogicalSemiring = {kLogicalAndOr, 1, 0};
// const SemiringType TropicalSemiring = {kAddMin, 0, UINT_INF};
const SemiringType TropicalSemiring = {kAddMin, 0, 255};
// const SemiringType TropicalSemiring = {kAddMin, 0, FLOAT_INF};

// Mask type
enum MaskType {
    kNoMask = 0,
    kMaskWriteToZero = 1,
    kMaskWriteToOne = 2,
};


//调用xcl中函数get_xil_devices()寻找设备
//当找到与device_name相同的设备时，返回该设备
cl::Device find_device() {
    auto devices = xcl::get_xil_devices();
    for (size_t i = 0; i < devices.size(); i++) {
        cl::Device device = devices[i];
        if (device.getInfo<CL_DEVICE_NAME>() == device_name) {
            return device;
        }
    }
    std::cout << "Failed to find " << device_name << ", exit!\n";
    exit(EXIT_FAILURE);
}







}// namespace demo

#endif  // GRAPHLILY_GLOBAL_H_