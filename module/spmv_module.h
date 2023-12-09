#ifndef DEMO_SPMV_MODULE_H_
#define DEMO_SPMV_MODULE_H_

#include <cstdint>
#include <vector>
#include <fstream>
#include <chrono>

#include "xcl2.hpp"

#include "demo/global.h"
#include "demo/io/data_loader.h"
// #include "graphlily/io/data_formatter.h"
#include "demo/module/base_module.h"

using demo::io::CSRMatrix;

namespace demo {
namespace module {

//泛型类，模板参数为矩阵和向量的数据类型
template<typename matrix_data_t, typename vector_data_t>
class SpMVModule : public BaseModule {
private:
    using val_t = vector_data_t;//matrix_data_t和vector_data_t应该是一样的，所以用val_t来统一表示这两种类型

    graphlily::MaskType mask_type_;
    graphlily::SemiringType semiring_;
    // uint32_t num_channels_;//暂时不考虑多通道
    // uint32_t out_buf_len_;//输出缓冲区的长度
    // uint32_t num_row_partitions_;//行分区的数量
    // uint32_t vec_buf_len_;//向量缓冲区的长度
    // uint32_t num_col_partitions_;//列分区的数量

    /*! \brief Matrix packets (indices and vals) + partition indptr for each channel */
    // std::vector<aligned_mat_pkt_t> channel_packets_;//暂时不用CPSR格式
    //改为CSR格式.在module中，分别有私有成员变量values_，opencl buffer values_buf_,和send_matrix_host_to_device()函数中的cl_mem_ext_ptr_t values_ext。
    std::vector<val_t> values_;
    std::vector<val_t> columnIndex_;
    std::vector<val_t> rowPtr_;

    std::vector<val_t> vector_;//稠密向量
    std::vector<val_t> mask_;//mask向量
    std::vector<val_t> results_;//结果
    CSRMatrix<float> csr_matrix_float_;//从npz文件中读取的float类型的矩阵，用于计算reference results
    CSRMatrix<val_t> csr_matrix_;//从csr_matrix_float_转换到val_t类型的矩阵，是kernel中真正使用的矩阵


void _check_data_type();//检查矩阵和向量的数据类型是否一致




public:
    // Device buffers
    // std::vector<cl::Buffer> channel_packets_buf;//每个HBM通道的数据包.大小为H=channel_number.matrix占用0~H-1个通道
    //CSR格式矩阵的三个数组的opencl buffer
    cl:Buffer values_buf;
    cl::Buffer col_index_buf;
    cl::Buffer row_ptr_buf;

    cl::Buffer vector_buf;
    cl::Buffer mask_buf;
    cl::Buffer results_buf;

//构造函数
SpMVModule(uint32_t num_channels, uint32_t out_buf_len, uint32_t vec_buf_len) : BaseModule("overlay") {
        this->_check_data_type();
        // this->num_channels_ = num_channels;
        this->out_buf_len_ = out_buf_len;
        this->vec_buf_len_ = vec_buf_len;
}

template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::_check_data_type() {
    assert((std::is_same<matrix_data_t, vector_data_t>::value));
}

// /*----------------- arguments for SpMV --------------------*/
// float *spmv_values,                        //0
// float *spmv_columnIndex,                   //1
// float *spmv_rowPtr,                        //2
// float *spmv_vector,                        //3
// float *spmv_mask,                          //4
// float *spmv_mask_w,                        //5
// float *spmv_out,                           //6
// /*----------------- arguments for SpMSpV -------------------*/
// float *spmspv_values,                      //7
// float *spmspv_rowIndex,                    //8
// float *spmspv_columnPtr,                   //9
// float *spmspv_vector,                      //10
// float *spmspv_mask,                        //11
// float *spmspv_out,                         //12
// /*----------- arguments shared by all kernels --------------*/
// unsigned num_rows,                         //13
// unsigned num_cols,                         //14
// char Op,                                   //15
// char mask_type,                            //16
// unsigned mode,                             //17
// /*---------- arguments for apply kernels -------------------*/
// // val must be the last argument, otherwise fixed point causes an XRT run-time error
// unsigned length,                           //18
// unsigned val                               //19
    void set_unused_args() override {
        // 设置SpMSpV的参数（7～12）
        for (uint32_t i = 7; i < 13; i++) {
            this->kernel_.setArg(i, cl::Buffer(this->context_, 0, 4));
        }
        // Set unused scalar arguments
        this->kernel_.setArg(this->18, (unsigned)NULL);
        // To avoid runtime error of invalid scalar argument size
        if (!(std::is_same<vector_data_t, unsigned>::value || std::is_same<vector_data_t, float>::value)) {
            this->kernel_.setArg(this->19, (long long)NULL);
        } else {
            this->kernel_.setArg(this->19, (unsigned)NULL);
        }
        //如果是无mask
        if (this->mask_type_ == demo::kNoMask) {
            this->kernel_.setArg(this->4, cl::Buffer(this->context_, 0, 4));//mask的读接口。flag=0,size=4
            this->kernel_.setArg(this->5, cl::Buffer(this->context_, 0, 4));//mask的写接口
        }
    }

    void set_mode() override {
        this->kernel_.setArg(17, 1);  // SpMV的mode是0
    }

    void set_semiring(graphlily::SemiringType semiring) {
        this->semiring_ = semiring;
    }

    void set_mask_type(graphlily::MaskType mask_type) {
        this->mask_type_ = mask_type;
    }
    uint32_t get_num_rows() {
        return this->csr_matrix_.num_rows;
    }

    uint32_t get_num_cols() {
        return this->csr_matrix_.num_cols;
    }

    uint32_t get_nnz() {
        return this->csr_matrix_.rowPtr[this->csr_matrix_.num_rows];
    }

template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::load_matrix(CSRMatrix<float> const &csr_matrix_float){//csr_matrix_float是从npz文件中读取的，一定是float类型
    this->csr_matrix_float_ = csr_matrix_float;
    this->csr_matrix_ = demo::io::csr_matrix_convert_from_float<val_t>(csr_matrix_float);
}


//从host到device
// send_matrix_host_to_device()
// send_vector_host_to_device()
// send_mask_host_to_device()
//12.5日遗留问题。channel_packets_怎么构建？（暂时不用作者的实现）
template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::send_matrix_host_to_device() {//先要在host端为matrix分配内存，然后再把内存传到device端
    cl_int err;

    //设置matrix的opencl buffer
    cl_mem_ext_ptr_t values_ext;
    cl_mem_ext_ptr_t columnIndex_ext;
    cl_mem_ext_ptr_t rowPtr_ext;

    values_ext.obj = this->values_.data();
    //以下两行可能没用
    // channel_packets_ext.param = 0;
    // channel_packets_ext.flags = graphlily::DDR[0];
    columnIndex_ext.obj = this->columnIndex_.data();
    rowPtr_ext.obj = this->rowPtr_.data();

    OCL_CHECK(err, this->values_buf = cl::Buffer(this->context_,
        CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,//指定了CL_MEM_USE_HOST_PTR，就必须指定obj
        sizeof(float) * this->values_.size(),
        &values_ext,
        &err));

    //设置results opencl buffer
    cl_mem_ext_ptr_t results_ext;
    results_ext.obj = this->results_.data();
    results_ext.param = 0;
    // results_ext.flags = graphlily::HBM[22];
    OCL_CHECK(err, this->results_buf = cl::Buffer(this->context_,
        CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(val_t) * this->csr_matrix_.num_rows,
        &results_ext,
        &err));

    //设置与matrix有关的参数
    OCL_CHECK(err, err = this->kernel_.setArg(0, this->values_buf));
    OCL_CHECK(err, err = this->kernel_.setArg(1, this->columnIndex_buf));
    OCL_CHECK(err, err = this->kernel_.setArg(2, this->rowPtr_buf));
    OCL_CHECK(err, err = this->kernel_.setArg(6, this->results_buf));
    OCL_CHECK(err, err = this->kernel_.setArg(13, this->csr_matrix_.num_rows));
    OCL_CHECK(err, err = this->kernel_.setArg(14, this->csr_matrix_.num_cols));
    OCL_CHECK(err, err = this->kernel_.setArg(15, (char)this->semiring_.op));
    OCL_CHECK(err, err = this->kernel_.setArg(16, (char)this->mask_type_));

    //发送matrix数据到device端
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({
        this->values_buf,
        this->columnIndex_buf,
        this->rowPtr_buf,
        }, 0));/* 0 means from host*/

    this->command_queue_.finish();

}

template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::send_vector_host_to_device(std::vector<val_t> &vector) {
    this->vector_.assign(vector.begin(), vector.end());
    cl_mem_ext_ptr_t vector_ext;
    vector_ext.obj = this->vector_.data();
    vector_ext.param = 0;
    // vector_ext.flags = graphlily::HBM[20];
    cl_int err;
    OCL_CHECK(err, this->vector_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(val_t) * this->csr_matrix_.num_cols,
                &vector_ext,
                &err));
    OCL_CHECK(err, err = this->kernel_.setArg(3, this->vector_buf));
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->vector_buf}, 0));
    this->command_queue_.finish();
}

template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::send_mask_host_to_device(std::vector<val_t> &mask) {
    this->mask_.assign(mask.begin(), mask.end());
    cl_mem_ext_ptr_t mask_ext;
    mask_ext.obj = this->mask_.data();
    mask_ext.param = 0;
    // mask_ext.flags = graphlily::HBM[21];
    cl_int err;
    OCL_CHECK(err, this->mask_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(val_t) * this->csr_matrix_.num_rows,
                &mask_ext,
                &err));
    OCL_CHECK(err, err = this->kernel_.setArg(4, this->mask_buf));//mask读接口
    OCL_CHECK(err, err = this->kernel_.setArg(5, this->mask_buf));//mask写接口
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->mask_buf}, 0));
    this->command_queue_.finish();
}


//从device到主机：
// send_vector_device_to_host()
// send_mask_device_to_host()
// send_results_device_to_host()
std::vector<val_t> send_vector_device_to_host() {
    this->command_queue_.enqueueMigrateMemObjects({this->vector_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    this->command_queue_.finish();
    return this->vector_;
}

std::vector<val_t> send_mask_device_to_host() {
    this->command_queue_.enqueueMigrateMemObjects({this->mask_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    this->command_queue_.finish();
    return this->mask_;
}

std::vector<val_t> send_results_device_to_host() {
    this->command_queue_.enqueueMigrateMemObjects({this->results_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    this->command_queue_.finish();
    return this->results_;
}

template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::bind_mask_buf(cl::Buffer src_buf) {
    this->mask_buf = src_buf;
    this->kernel_.setArg(4, this->mask_buf);
    this->kernel_.setArg(5, this->mask_buf);
}

template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::run() {
    cl_int err;
    OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->kernel_));
    this->command_queue_.finish();
}

};

}  // namespace module
}  // namespace demo




#endif  // DEMO_SPMV_MODULE_H_