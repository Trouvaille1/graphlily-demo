#include "spmv.h"

//spmv原始版本
void spmv(int rowPtr[NUM_ROWS+1], int columnIndex[NNZ],
		DTYPE values[NNZ], DTYPE y[NUM_ROWS], DTYPE x[SIZE])
{
L1: for (int i = 0; i < NUM_ROWS; i++) {//外层循环不能pipeline，因为内层循环的tripcount不确定
		DTYPE y0 = 0;
	L2: for (int k = rowPtr[i]; k < rowPtr[i+1]; k++) {//遍历原矩阵第i行的所有非零元素
//    #pragma HLS pipeline off
			y0 += values[k] * x[columnIndex[k]];//x[columnIndex[k]]为向量对应元素
		}
		y[i] = y0;
	}
}

