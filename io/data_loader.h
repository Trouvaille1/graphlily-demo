#ifndef GRAPHLILY_IO_DATA_LOADER_H_
#define GRAPHLILY_IO_DATA_LOADER_H_

#include <cstdint>
#include <vector>
#include "cnpy.h"

namespace demo{
namespace io{

//CSR格式的矩阵 
template<typename T>
struct CSRMatrix {
    uint32_t num_rows;
    uint32_t num_cols;
    std::vector<T> values;//所有非零元素
    std::vector<uint32_t> columnIndex;//所有非零元素的列索引
    std::vector<uint32_t> rowPtr;//行指针
};


//从npz类型的数据集中加载csr格式的矩阵(矩阵中的元素为float类型)
CSRMatrix<float> load_csr_matrix_from_float_npz(std::string csr_float_npz_path) {
    CSRMatrix<float> csr_matrix;
    cnpy::npz_t npz = cnpy::npz_load(csr_float_npz_path);

    //读取shape
    cnpy::NpyArray npy_shape = npz["shape"];
    uint32_t num_rows = npy_shape.data<uint32_t>()[0];
    uint32_t num_cols = npy_shape.data<uint32_t>()[2];
    csr_matrix.num_rows = num_rows;
    csr_matrix.num_cols = num_cols;

    //读取csr格式的三个数组
    cnpy::NpyArray npy_data = npz["data"];
    uint32_t nnz = npy_data.shape[0];
    cnpy::NpyArray npy_columnIndex = npz["indices"];
    cnpy::NpyArray npy_rowPtr = npz["indptr"];
    csr_matrix.values.insert(csr_matrix.values.begin(), &npy_data.data<float>()[0],
        &npy_data.data<float>()[nnz]);
    csr_matrix.columnIndex.insert(csr_matrix.columnIndex.begin(), &npy_columnIndex.data<uint32_t>()[0],
        &npy_columnIndex.data<uint32_t>()[nnz]);
    csr_matrix.rowPtr.insert(csr_matrix.rowPtr.begin(), &npy_rowPtr.data<uint32_t>()[0],
        &npy_rowPtr.data<uint32_t>()[num_rows + 1]);
    return csr_matrix;
}


//从npz文件中读取的float类型的矩阵转换为kernel中指定的matrix_data_t的矩阵
template<typename matrix_data_t>
CSRMatrix<matrix_data_t> csr_matrix_convert_from_float(CSRMatrix<float> const &in) {
    CSRMatrix<matrix_data_t> out;
    out.num_rows = in.num_rows;
    out.num_cols = in.num_cols;
    std::copy(in.values.begin(), in.values.end(), std::back_inserter(out.values));
    out.columnIndex = in.columnIndex;
    out.rowPtr = in.rowPtr;
    return out;
}










}
}



#endif  // DEMO_IO_DATA_LOADER_H_