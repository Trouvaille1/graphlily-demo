#include "spmv.h"
#include <hls_stream.h>

#define II 9//tiling factor

void spmv_stream(
		int rows_length[NUM_ROWS],
		int rows_length_pad[NUM_ROWS],
		int cols[NNZ],
		DTYPE values[NNZ],
		DTYPE y[NUM_ROWS],
		DTYPE x[SIZE],
		int new_nnz)
{
#pragma HLS DATAFLOW

	int row_length_pad = 0, row_length = 0, k = 0, row_counter = 0;
	hls::stream<DTYPE> values_fifo;
	hls::stream<int>   cols_fifo;
	hls::stream<DTYPE> results_fifo;

	DTYPE sum = 0;
	DTYPE value;
	int col;
	DTYPE term[II];

	for (int i = 0; i < NNZ; i++) {
#pragma HLS PIPELINE
		values_fifo << values[i];
		cols_fifo   << cols[i];
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
				value = values_fifo.read();
				col   = cols_fifo.read();
				term[j] = value * x[col];
			}
		}

		DTYPE sum_tmp = 0;
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
		y[i] = results_fifo.read();
	}
}

void spmv_tiling_stream(int rowPtr[NUM_ROWS + 1], int cols[NNZ], DTYPE values[NNZ], DTYPE y[NUM_ROWS], DTYPE x[SIZE])
{
	// rowPtr to rows_length.从CSR格式转变到MCSR格式。现在rows_length存储的就是每行非零元素的个数
	int rows_length[NUM_ROWS] = {0};
	for (int i = 1; i < NUM_ROWS + 1; i++) {
#pragma HLS PIPELINE
		rows_length[i - 1] = rowPtr[i] - rowPtr[i - 1];
	}

//计算在tiling factor(II)下，新的nnz(new_nnz>=nnz)，以及填充新的rows_length_pad
	int rows_length_pad[NUM_ROWS];
	int new_nnz = 0;
	for (int i = 0; i < NUM_ROWS; i++) {
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
	spmv_stream(rows_length, rows_length_pad, cols, values, y, x, new_nnz);
}
