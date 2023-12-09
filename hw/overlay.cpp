#include "./overlay.h"
#include "./kernel_spmv_impl.h"

extern "C" {
void overlay(
/*----------------- arguments for SpMV --------------------*/
float *spmv_values,                        //0
float *spmv_columnIndex,                   //1
float *spmv_rowPtr,                        //2
float *spmv_vector,                        //3
float *spmv_mask,                          //4
float *spmv_mask_w,                        //5
float *spmv_out,                           //6
/*----------------- arguments for SpMSpV -------------------*/
float *spmspv_values,                      //7
float *spmspv_rowIndex,                    //8
float *spmspv_columnPtr,                   //9
float *spmspv_vector,                      //10
float *spmspv_mask,                        //11
float *spmspv_out,                         //12
/*----------- arguments shared by all kernels --------------*/
unsigned num_rows,                         //13
unsigned num_cols,                         //14
char Op,                                   //15
char mask_type,                            //16
unsigned mode,                             //17
/*---------- arguments for apply kernels -------------------*/
// val must be the last argument, otherwise fixed point causes an XRT run-time error
unsigned length,                           //18
unsigned val                               //19
){
/*----------------- arguments for SpMV -------------------*/
#pragma HLS INTERFACE axis port=spmv_values offset=slave bundle=gmem0
#pragma HLS INTERFACE axis port=spmv_columnIndex offset=slave bundle=gmem1
#pragma HLS INTERFACE axis port=spmv_rowPtr offset=slave bundle=gmem2
#pragma HLS INTERFACE axis port=spmv_vector offset=slave bundle=gmem3
#pragma HLS INTERFACE axis port=spmv_mask offset=slave bundle=gmem4
#pragma HLS INTERFACE axis port=spmv_mask_w offset=slave bundle=gmem5
#pragma HLS INTERFACE axis port=spmv_out offset=slave bundle=gmem6

/*----------------- arguments for SpMSpV -------------------*/
#pragma HLS INTERFACE axis port=spmspv_values offset=slave bundle=gmem7
#pragma HLS INTERFACE axis port=spmspv_rowIndex offset=slave bundle=gmem8
#pragma HLS INTERFACE axis port=spmspv_columnPtr offset=slave bundle=gmem9
#pragma HLS INTERFACE axis port=spmspv_vector offset=slave bundle=gmem10
#pragma HLS INTERFACE axis port=spmspv_mask offset=slave bundle=gmem11
#pragma HLS INTERFACE axis port=spmspv_out offset=slave bundle=gmem12

/*----------- arguments shared by all kernels --------------*/
#pragma HLS INTERFACE s_axilite port=num_rows bundle=control
#pragma HLS INTERFACE s_axilite port=num_cols bundle=control
#pragma HLS INTERFACE s_axilite port=Op bundle=control
#pragma HLS INTERFACE s_axilite port=mask_type bundle=control
#pragma HLS INTERFACE s_axilite port=mode bundle=control


/*---------- arguments for apply kernels -------------------*/
#pragma HLS INTERFACE s_axilite port=length bundle=control
#pragma HLS INTERFACE s_axilite port=val bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    switch (mode) {
        case 1:
            std::cout << "Running SpMV" << std::endl;
            break;
        case 2:
            std::cout << "Running SpMSpV" << std::endl;
            break;
        case 3:
            std::cout << "Running kernel_add_scalar_vector_dense" << std::endl;
            break;
        case 4:
            std::cout << "Running kernel_assign_vector_dense" << std::endl;
            break;
        case 5:
            std::cout << "Running kernel_assign_vector_sparse_no_new_frontier" << std::endl;
            break;
        case 6:
            std::cout << "Running kernel_assign_vector_sparse_new_frontier" << std::endl;
            break;
        default:
            std::cout << "ERROR! Unsupported mode: " << mode << std::endl;
            break;
    }

    switch (mode) {
        case 1:
        kernel_spmv(
            spmv_values,
            spmv_columnIndex,
            spmv_rowPtr,
            spmv_vector,
            spmv_mask,//mask读接口
            spmv_out,
            num_rows,
            num_cols,
            Op,
            mask_type
            // out_uram_spmv 暂时不用这个参数。因为不使用HBM.在graphlily中作为compute_spmv_one_channel()函数的参数，每个通道输出的结果都存储在这里。最后调用write_to_out_ddr()函数将结果写入DDR
        );
        break;



        default:
        break;
    }

}

}