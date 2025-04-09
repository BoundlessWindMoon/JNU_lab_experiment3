#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "conv2d.h"
#include <assert.h>
#include <stdio.h>

convParamType testcase_1 = {64, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1};
convParamType testcase_2 = {256, 192, 14, 14, 192, 3, 3, 1, 1, 1, 1};
convParamType testcase_3 = {16, 256, 26, 26, 512, 3, 3, 1, 1, 1, 1};
convParamType testcase_4 = {32, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1};
convParamType testcase_5 = {2, 1280, 16, 16, 1280, 3, 3, 1, 1, 1, 1};
convParamType testcase_6 = {2, 960, 64, 64, 32, 3, 3, 1, 1, 1, 1};

#define UNROLL_PARAM(param) unsigned int n = param->n; \
                            unsigned int c = param->c; \
                            unsigned int h = param->h; \
                            unsigned int w = param->w; \
                            unsigned int k = param->k; \
                            unsigned int r = param->r; \
                            unsigned int s = param->s; \
                            unsigned int p = param->p; \
                            unsigned int q = param->q; \
                            unsigned int u = param->u; \
                            unsigned int v = param->v; \
                            unsigned int outh = param->Oh; \
                            unsigned int outw = param->Ow;

#define UNROLL_PROBLEM(problem) unsigned int n = problem->n; \
                            unsigned int c = problem->c; \
                            unsigned int h = problem->h; \
                            unsigned int w = problem->w; \
                            unsigned int k = problem->k; \
                            unsigned int r = problem->r; \
                            unsigned int s = problem->s; \
                            unsigned int u = problem->u; \
                            unsigned int v = problem->v; \
                            unsigned int p = problem->p; \
                            unsigned int q = problem->q; 

void print_log(problem_t *problem) {
    UNROLL_PROBLEM(problem);
    unsigned int outh = (h - r + 2 * p) / u + 1;
    unsigned int outw = (w - s + 2 * q) / v + 1;
    printf("n=%d\tc=%d\th=%d\tw=%d\nk=%d\tr=%d\ts=%d\nu=%d\tv=%d\tp=%d\tq=%d\n", n, c, h, w, k, r, s, u, v, p, q);
}

void impl_gemm_init(mykernelParamType * param) {
}

void impl_gemm_exit(mykernelParamType * param) {
}

void impl_gemm_v0_run(mykernelParamType * param) {
    UNROLL_PARAM(param);
    launch_implicit_gemm_v0(outh, outw, k, n, param);
}
void impl_gemm_v1_run(mykernelParamType * param) {
    UNROLL_PARAM(param);
    launch_implicit_gemm_v1(outh, outw, k, n, param);
}
void impl_gemm_v2_run(mykernelParamType * param) {
    UNROLL_PARAM(param);
    launch_implicit_gemm_v2(outh, outw, k, n, param);
}

void umimplement_init(mykernelParamType* param) {
    printf("unimplement init\n");
    assert(0);
}

void umimplement_run(mykernelParamType* param) {
    printf("unimplement run\n");
    assert(0);
}

void umimplement_exit(mykernelParamType* param) {
    printf("unimplement exit\n");
    assert(0);
}

convPlanType conv_plans[7] = {
    {"testcase_1", impl_gemm_init, impl_gemm_v0_run, impl_gemm_exit},
    {"testcase_2", impl_gemm_init, impl_gemm_v0_run, impl_gemm_exit},
    {"testcase_3", impl_gemm_init, impl_gemm_v0_run, impl_gemm_exit},
    {"testcase_4", impl_gemm_init, impl_gemm_v0_run, impl_gemm_exit},
    {"testcase_5", impl_gemm_init, impl_gemm_v0_run, impl_gemm_exit},
    {"testcase_6", impl_gemm_init, impl_gemm_v0_run, impl_gemm_exit},
    {"umimplement", impl_gemm_init, impl_gemm_v0_run, impl_gemm_exit},
};

convPlanType scheduler(problem_t *problem, mykernelParamType * param) {
    print_log(problem);
    convParamType in_param = {problem->n, problem->c, problem->h, problem->w, problem->k, problem->r, problem->s, problem->u, problem->v, problem->p, problem->q};
    if (PARAM_EQUAL(testcase_1, in_param)) {
        return conv_plans[0];
    } else if (PARAM_EQUAL(testcase_2, in_param)) {
        return conv_plans[1];
    } else if (PARAM_EQUAL(testcase_3, in_param)) {
        return conv_plans[2];
    } else if (PARAM_EQUAL(testcase_4, in_param)) {
        return conv_plans[3];
    } else if (PARAM_EQUAL(testcase_5, in_param)) {
        return conv_plans[4];
    } else if (PARAM_EQUAL(testcase_6, in_param)) {
        return conv_plans[5];
    } else 
        return conv_plans[6];
}
