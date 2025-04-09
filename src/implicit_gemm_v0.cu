#include <cuda_runtime.h>
#include <cuda_fp16.h> 
#include "conv2d.h"

extern "C" __global__ void implicit_gemm_v0(mykernelParamType param)
{ 
    int OhOw = PLACEHOLDER;
    int k = PLACEHOLDER;
    int n = PLACEHOLDER;
    if(OhOw >= param.Oh*param.Ow || k >= param.k || n >= param.n)
        return;

    int oh = PLACEHOLDER;
    int ow = PLACEHOLDER;
    int input_addr, weight_addr, output_addr;
    float sum = 0.0;
    output_addr = PLACEHOLDER;

    for(int r = 0; r < param.r; r++) {
        for(int s = 0; s < param.s; s++) {
            int ih = PLACEHOLDER;
            int iw = PLACEHOLDER;

            if (ih >= 0 && ih < param.h && iw >= 0 && iw < param.w) {
                for(int c = 0; c < param.c; c++) {
                    input_addr = PLACEHOLDER;
                    weight_addr = PLACEHOLDER;

                    sum += param.pin[input_addr] * param.pweight[weight_addr];
                }
            }
        }
    }
    param.pout[output_addr] = sum;
}

void launch_implicit_gemm_v0(unsigned int outh, unsigned int outw, unsigned int k, unsigned int n, mykernelParamType* param) {
    int blockx = PLACEHOLDER;
    int blocky = PLACEHOLDER;
    int blockz = PLACEHOLDER;
    int threadx = 16;        
    int thready = 16;
    int threadz = 1;
    dim3 block(threadx, thready, threadz);
    dim3 grid(blockx, blocky, blockz);
    implicit_gemm_v0<<<grid, block>>>(*param);
}