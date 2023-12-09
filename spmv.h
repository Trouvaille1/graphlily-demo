#ifndef __SPMV_H__
#define __SPMV_H__

const static int SIZE = 256; // size=矩阵列数=向量长度
const static int NNZ = 3277; //Number of non-zero elements
const static int NUM_ROWS = 256;// 矩阵行数
typedef float DTYPE;
void spmv(int rowPtr[NUM_ROWS+1], int columnIndex[NNZ],
		  DTYPE values[NNZ], DTYPE y[SIZE], DTYPE x[SIZE]);

void spmv_tiling_stream(int rowPtr[NUM_ROWS + 1], int cols[NNZ], DTYPE values[NNZ], DTYPE y[NUM_ROWS], DTYPE x[SIZE]);

#endif