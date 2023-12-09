#include "./overlay.h"

#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <iomanip>

#include <hls_stream.h>
#include <ap_fixed.h>

#define II 9//tiling factor

void spmv_stream(
		int rows_length[NUM_ROWS],
		int rows_length_pad[NUM_ROWS],
		int spmv_columnIndex[NNZ],
		float spmv_values[NNZ],
		float spmv_out[NUM_ROWS],
		float spmv_vector[SIZE],
		int new_nnz)
{
#pragma HLS DATAFLOW

	int row_length_pad = 0, row_length = 0, k = 0, row_counter = 0;
	hls::stream<float> spmv_values_fifo;
	hls::stream<int>   spmv_columnIndex_fifo;
	hls::stream<float> results_fifo;

	float sum = 0;
	float value;
	int col;
	float term[II];

	for (int i = 0; i < NNZ; i++) {
#pragma HLS PIPELINE
		spmv_values_fifo << spmv_values[i];
		spmv_columnIndex_fifo   << spmv_columnIndex[i];
	}

	for (int i = 0; i < new_nnz; i+=II) {
#pragma HLS PIPELINE
        //处理完当前行切换到下一行时，读入相关数据
		if (row_length_pad == 0) {
			row_length_pad = rows_length_pad[k];
			row_length = rows_length[k++];
			row_counter = 0;
			sum = 0;
		}

		for (int j = 0; j < II; j++) {
			row_counter++;
            //对于rows_length_pad，当row_counter>row_length时，说明原矩阵中该行真正的非零元素已经处理完毕，term[j]置0
			if (row_counter > row_length) {
				term[j] = 0;
			} else {
				value = spmv_values_fifo.read();
				col   = spmv_columnIndex_fifo.read();
				term[j] = value * spmv_vector[col];
			}
		}

		float sum_tmp = 0;
		for (int j = 0; j < II; j++) {
			sum_tmp += term[j];
		}
		sum += sum_tmp;

		row_length_pad -= II;
		if (row_length_pad == 0) {//每处理完一行，输出一个结果
			results_fifo << sum;
		}
	}

	for (int i = 0; i < NUM_ROWS; i++) {
#pragma HLS PIPELINE
		spmv_out[i] = results_fifo.read();
	}
}


void kernel_spmv(
/*----------------- arguments for SpMV --------------------*/
float *spmv_values,                        //0      values[NNZ]
float *spmv_columnIndex,                   //1      cols[NNZ]
float *spmv_rowPtr,                        //2      rowPtr[NUM_ROWS + 1]
float *spmv_vector,                        //3      x[SIZE]
float *spmv_mask,                          //4
float *spmv_out,                           //5      y[NUM_ROWS]
int num_rows,                              //6
int num_cols,                              //7
int Op,                                    //8
int mask_type                              //9
){
#pragma HLS INTERFACE m_axi port=spmv_values offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=spmv_columnIndex offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=spmv_rowPtr offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=spmv_vector offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=spmv_mask offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=spmv_out offset=slave bundle=gmem5

    unsigned Zero;
    switch (Op) {
    case MULADD:
        Zero = MulAddZero;
        break;
    case ANDOR:
        Zero = AndOrZero;
        break;
    case ADDMIN:
        Zero = AddMinZero;
        break;
    default:
        Zero = 0;
        break;
    }


	// rowPtr to rows_length.从CSR格式转变到MCSR格式。现在rows_length存储的就是每行非零元素的个数
	int rows_length[num_rows] = {0};
	for (int i = 1; i < num_rows + 1; i++) {
#pragma HLS PIPELINE
		rows_length[i - 1] = spmv_rowPtr[i] - spmv_rowPtr[i - 1];
	}

//计算在tiling factor(II)下，新的nnz(new_nnz>=nnz)，以及填充新的rows_length_pad
	int rows_length_pad[num_rows];
	int new_nnz = 0;
	for (int i = 0; i < num_rows; i++) {
#pragma HLS PIPELINE
		int r = rows_length[i];
		int r_diff = r % II;
		if (r == 0) {//原矩阵第i行非零元素个数为0，则在第i行后面补II个非零元素个数
			rows_length_pad[i] = II;
			new_nnz += II;
		} else if (r_diff != 0) {//原矩阵第i行非零元素个数不能被II整除，则在第i行后面补非零元素个数，使得第i行非零元素个数能被II整除
			rows_length_pad[i] = r + (II - r_diff);
			new_nnz += r + (II - r_diff);
		} else {//原矩阵第i行非零元素个数能被II整除，则加上r即可
			rows_length_pad[i] = r;
			new_nnz += r;
		}
	}

    //执行stream版本的spmv
	spmv_stream(rows_length, rows_length_pad, spmv_columnIndex, spmv_values, spmv_out, spmv_vector, new_nnz);
}