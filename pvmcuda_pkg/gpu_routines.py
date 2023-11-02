# GPU PVM Implementation
# (C) 2023 Filip Piekniewski 
# filip@piekniewski.info
import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import time
from pycuda.compiler import SourceModule
import pycuda.driver as drv


def py_dot(mat, vec, res, total_obj):
    """
    Simple dot product matrix times vector
    :param mat:
    :param vec:
    :param res:
    :param total_obj:
    :return:
    """
    for i in range(total_obj):
        res[i][:] = np.dot(mat[i], vec[i])


mod2 = SourceModule("""
__global__ void gpu_dot(float *mem1,
                        float *mem2,
                        float *mem3,
                        int *ptr1,
                        int *ptr2,
                        int *ptr3,
                        int *shape0,
                        int *shape1,
                        int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  float * mat = &mem1[ptr1[i]];
  float * vec = &mem2[ptr2[i]];
  float * res = &mem3[ptr3[i]];
  for(int k = 0; k<shape0[i]; k++)
  {
     float dot = 0;
     for(int l = 0; l<shape1[i]; l++)
     {
        dot += mat[k*shape1[i]+l] * vec[l];
     }
     res[k] = dot;
  }
}

__global__ void gpu_sgn(float *mem1,
                        int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  mem1[i]=copysignf(1.0, mem1[i]);
}
""")


def py_dot_transpose(mat, vec, res, total_obj, bias=False):
    """
    Dot product by a transposed matrix. If bias set to true,
    the input vector will be trimmed by one element to acomodate
    for bias value which should not be used by this calculation
    in MLP code.
    :param mat:
    :param vec:
    :param res:
    :param total_obj:
    :param bias:
    :return:
    """
    if bias:
        for i in range(total_obj):
            res[i][:] = np.dot(mat[i].T, vec[i][:-1])
    else:
        for i in range(total_obj):
            res[i][:] = np.dot(mat[i].T, vec[i])



mod4 = SourceModule("""
__global__ void gpu_dot_transpose(float *mem1,
                                 float *mem2,
                                 float *mem3,
                                 int *ptr1,
                                 int *ptr2,
                                 int *ptr3,
                                 int *shape0,
                                 int *shape1,
                                 int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  float * mat = &mem1[ptr1[i]];
  float * vec = &mem2[ptr2[i]];
  float * res = &mem3[ptr3[i]];
  int shape0_ = shape0[i];
  int shape1_ = shape1[i];
  for(int k = 0; k<shape1_; k++)
  {
     res[k]=0;
  }
  for(int l = 0; l<shape0_; l++)
  {
     for(int k = 0; k<shape1_; k++)
     {
     //float dot = 0;
        res[k] += mat[l * shape1_ + k] * vec[l];
     }
     //res[k] = dot;
  }
}
""")

mod41 = SourceModule("""
__global__ void gpu_dot_transpose_fast(float *mem0,
                                       float *mem1,
                                       float *mem2,
                                       int *ptr1,
                                       int *ptr2,
                                       int *shape0,
                                       int *shape1,
                                       int *obj_id,
                                       int *row_id,
                                       int total_threads)
{

  const int thread_id = (blockIdx.x * blockDim.x + threadIdx.x);
  if (thread_id >= total_threads)
      return;
  const int i = obj_id[thread_id];
  const int k = row_id[thread_id];
  const int shape0_ = shape0[i];
  const int shape1_ = shape1[i];
  float * mat = &mem0[ptr1[i]];
  float * mat_buf = &mem1[ptr1[i]];
  float * vec = &mem2[ptr2[i]];
  for(int l = 0; l<shape0_; l++)
  {
        mat_buf[l * shape1_ + k] =  mat[l * shape1_ + k] * vec[l];
  }
}

__global__ void gpu_sum_dot_transpose(float * mem0,
                                      float * mem1,
                                      int * ptr1,
                                      int * ptr3,
                                      int *shape0,
                                      int *shape1,
                                      int *obj_id,
                                      int *row_id,
                                      int total_threads)
{
  const int thread_id = (blockIdx.x * blockDim.x + threadIdx.x);
  if (thread_id >= total_threads)
      return;
  const int i = obj_id[thread_id];
  const int k = row_id[thread_id];
  float * mat = &mem0[ptr1[i]];
  float * res = &mem1[ptr3[i]];
  const int shape0_ = shape0[i];
  const int shape1_ = shape1[i];
  for(int l = 0; l<shape0_; l++)
  {
        res[k] += mat[l * shape1_ + k];
  }

}
""")


def py_dot_sigmoid(mat, vec, res, beta, total_obj):
    """
    Dot product matrix times a vector passed through a
    sigmoid activation function.
    :param mat:
    :param vec:
    :param res:
    :param beta:
    :param total_obj:
    :return:
    """
    for i in range(total_obj):
        res[i][:mat[i].shape[0]] = 1.0/(1.0+np.exp(-beta[i]*np.dot(mat[i], vec[i])))


mod5 = SourceModule("""
__global__ void gpu_dot_sigmoid(float *mem1,
                                float *mem2,
                                float *mem3,
                                int *ptr1,
                                int *ptr2,
                                int *ptr3,
                                float *beta,
                                int *shape0,
                                int *shape1,
                                int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  float * mat = &mem1[ptr1[i]];
  float * vec = &mem2[ptr2[i]];
  float * res = &mem3[ptr3[i]];
  for(int k = 0; k<shape0[i]; k++)
  {
     float dot = 0;
     for(int l = 0; l<shape1[i]; l++)
     {
        dot += mat[k*shape1[i]+l] * vec[l];
     }
     res[k] = 1.0/(1.0+expf(-beta[i]*dot));
  }
}
""")

mod51 = SourceModule("""
__global__ void gpu_dot_fast(float *mem1,
                             float *mem2,
                             float *mem3,
                             int *ptr1,
                             int *ptr2,
                             int *ptr3,
                             int *shape0,
                             int *shape1,
                             int *obj_id,
                             int *row_id,
                             int total_threads)
{
  const int thread_id = (blockIdx.x * blockDim.x + threadIdx.x);
  if (thread_id >= total_threads)
      return;
  const int i = obj_id[thread_id];
  const int l = row_id[thread_id];
  float * mat = &mem1[ptr1[i]];
  float * vec = &mem2[ptr2[i]];
  float * res = &mem3[ptr3[i]];
  // Assumed that res[k] has been zeroed before this 
  // kernel is called
  for(int k = 0; k<shape0[i]; k++)
  {
     atomicAdd(&res[k], mat[k*shape1[i]+l] * vec[l]);
  }
}

__global__ void gpu_dot_fast_set_bias(float *mem1,
                             float *mem2,
                             float *mem3,
                             int *ptr1,
                             int *ptr2,
                             int *ptr3,
                             int *shape0,
                             int *shape1,
                             int *obj_id,
                             int *row_id,
                             int total_threads)
{
  const int thread_id = (blockIdx.x * blockDim.x + threadIdx.x);
  if (thread_id >= total_threads)
      return;
  const int i = obj_id[thread_id];
  const int l = row_id[thread_id];
  float * mat = &mem1[ptr1[i]];
  float * vec = &mem2[ptr2[i]];
  float * res = &mem3[ptr3[i]];
  // Assumed that res[k] has been zeroed before this 
  // kernel is called
  for(int k = 0; k<shape0[i]; k++)
  {
     atomicAdd(&res[k], mat[k*shape1[i]+l] * vec[l]);
  }
  res[shape0[i]]=1.0;
}


__global__ void gpu_dot_slow(float *mem1,
                                float *mem2,
                                float *mem3,
                                int *ptr1,
                                int *ptr2,
                                int *ptr3,
                                float *beta,
                                int *shape0,
                                int *shape1,
                                int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  float * mat = &mem1[ptr1[i]];
  float * vec = &mem2[ptr2[i]];
  float * res = &mem3[ptr3[i]];
  for(int k = 0; k<shape0[i]; k++)
  {
     float dot = 0;
     for(int l = 0; l<shape1[i]; l++)
     {
        dot += mat[k*shape1[i]+l] * vec[l];
     }
     res[k] = dot;
  }
}


__global__ void gpu_sigmoid_fast(float *mem1,
                                 int *ptr1,
                                 float *beta,
                                 int *shape0,
                                 int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  float * res = &mem1[ptr1[i]];
  for(int k = 0; k<shape0[i]; k++)
  {
     res[k] = 1.0/(1.0+expf(-beta[i]*res[k]));
  }
}

__global__ void gpu_sigmoid_poly_fast(float *mem1,
                                      int *ptr1,
                                      float *beta,
                                      int *shape0,
                                      int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  float * res = &mem1[ptr1[i]];
  for(int k = 0; k<shape0[i]; k++)
  {
     res[k] = (res[k] / (2 * (fabs(res[k]) + 1))) + 0.5;
  }
}

""")


def py_dot_sigmoid_poly(mat, vec, res, total_obj):
    """
    Dot product matrix times a vector passed through a
    rational (composed out of rational functions) sigmoid activation function.
    :param mat:
    :param vec:
    :param res:
    :param beta:
    :param total_obj:
    :return:
    """
    for i in range(total_obj):
        s0 = np.dot(mat[i], vec[i])
        res[i][:mat[i].shape[0]] = (s0/(2*(np.abs(s0)+1)))+0.5


mod7 = SourceModule("""
__global__ void gpu_dot_sigmoid_poly(float *mem1,
                                     float *mem2,
                                     float *mem3,
                                     int *ptr1,
                                     int *ptr2,
                                     int *ptr3,
                                     int *shape0,
                                     int *shape1,
                                     int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  float * mat = &mem1[ptr1[i]];
  float * vec = &mem2[ptr2[i]];
  float * res = &mem3[ptr3[i]];
  for(int k = 0; k<shape0[i]; k++)
  {
     float s0 = 0;
     for(int l = 0; l<shape1[i]; l++)
     {
        s0 += mat[k*shape1[i]+l] * vec[l];
     }
     res[k] = (s0 / (2 * (fabs(s0) + 1))) + 0.5;
  }
}
""")


def py_sigmoid_der_mul(vec0, vec1, res, total_obj):
    """
    Compute sigmoid derivative on vec0 and multiply
    pointwise by vec1
    :param vec0:
    :param vec1:
    :param res:
    :param total_obj:
    :return:
    """
    for i in range(total_obj):
        res[i][:] = vec0[i] * (1 - vec0[i]) * vec1[i]


mod8 = SourceModule("""
__global__ void gpu_sigmoid_der_mul(float *mem1,
                                    float *mem2,
                                    float *mem3,
                                    int *ptr1,
                                    int *ptr2,
                                    int *ptr3,
                                    int *shape0,
                                    int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  float * vec0 = &mem1[ptr1[i]];
  float * vec1 = &mem2[ptr2[i]];
  float * res = &mem3[ptr3[i]];
  for(int k = 0; k<shape0[i]; k++)
  {
     res[k] = vec0[k] * (1 - vec0[k]) * vec1[k];
  }
}
""")


def py_sigmoid_poly_der_mul(vec0, vec1, res, total_obj):
    """
    Compute rational sigmoid derivative on vec0 and multiply
    pointwise by vec1
    :param vec0:
    :param vec1:
    :param res:
    :param total_obj:
    :return:
    """
    for i in range(total_obj):
        x = np.zeros_like(vec0[i])
        for j in range(vec0[i].shape[0]):
            if vec0[i][j] >= 0.5:
                x[j] = (2 * vec0[i][j] - 1)/(1 - (2 * vec0[i][j] - 1))
            else:
                x[j] = (2 * vec0[i][j] - 1)/(1 + (2 * vec0[i][j] - 1))
        x = np.abs(x) + 1
        res[i][:] = (1.0 / (2 * (x * x))) * vec1[i]


mod9 = SourceModule("""
__global__ void gpu_sigmoid_poly_der_mul(float *mem1,
                                         float *mem2,
                                         float *mem3,
                                         int *ptr1,
                                         int *ptr2,
                                         int *ptr3,
                                         int *shape0,
                                         int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  float * vec0 = &mem1[ptr1[i]];
  float * vec1 = &mem2[ptr2[i]];
  float * res = &mem3[ptr3[i]];
  for(int j = 0; j<shape0[i]; j++)
  {
      float x;
      float m = copysignf(1.0, vec0[j]-0.5);
      //if (vec0[j]>=0.5) {
      //   x=(2*vec0[j]-1)/(1-(2*vec0[j]-1));
      //} else {
      //   x=(2*vec0[j]-1)/(1+(2*vec0[j]-1));
      //}
      x = fmaf(2.0, vec0[j], -1.0)/(1+m*fmaf(2.0, vec0[j], -1.0));
      x = fabs(x) + 1;
      res[j] =(1.0/(2*(x*x))) * vec1[j];
  }
}
""")


def py_outer_simple(vec0, vec1, mat, total_obj):
    """
    Compute simple outer product of vectors and store in matrix
    :param vec0:
    :param vec1:
    :param mat:
    :param total_obj:
    :return:
    """
    for i in range(total_obj):
        mat[i][:, :] = np.outer(vec0[i], vec1[i])


mod3 = SourceModule("""
__global__ void gpu_outer_simple(float *mem1,
                                  float *mem2,
                                  float *mem3,
                                  int *ptr1,
                                  int *ptr2,
                                  int *ptr3,
                                  int *shape0,
                                  int *shape1,
                                  int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  float * mat = &mem3[ptr3[i]];
  float * vec0 = &mem1[ptr1[i]];
  float * vec1 = &mem2[ptr2[i]];
  for(int k = 0; k<shape0[i]; k++)
  {
     for(int l = 0; l<shape1[i]; l++)
     {
        mat[k*shape1[i]+l] = vec0[k] * vec1[l];
     }
  }
}
""")


def py_generalized_outer(vec0, vec1, mat, res, alpha, beta, total_obj, bias=False):
    """
    Generalized outer. Calculate

    res = \alpha * vec0 x vec1 + \beta * mat

    When bias is true, the vec0 will be trimmed by one element as bias should not be used for this calculation
    in MLP training execution

    :param vec0:
    :param vec1:
    :param mat:
    :param res:
    :param alpha:
    :param beta:
    :param total_obj:
    :param bias:
    :return:
    """
    if bias:
        for i in range(total_obj):
            res[i][:] = alpha[i]*np.outer(vec0[i][:-1], vec1[i]) + beta[i]*mat[i]
    else:
        for i in range(total_obj):
            res[i][:] = alpha[i]*np.outer(vec0[i][:], vec1[i]) + beta[i]*mat[i]


mod6 = SourceModule("""
__global__ void gpu_generalized_outer(float *mem1,
                                      float *mem2,
                                      float *mem3,
                                      float *mem4,
                                      int *ptr1,
                                      int *ptr2,
                                      int *ptr3,
                                      int *ptr4,
                                      int *shape0,
                                      int *shape1,
                                      float *alpha,
                                      float *beta,
                                      int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  float * mat0 = &mem3[ptr3[i]];
  float * res = &mem4[ptr4[i]];
  float * vec0 = &mem1[ptr1[i]];
  float * vec1 = &mem2[ptr2[i]];
  float beta_ = beta[i];
  float alpha_ = alpha[i];
  int shape1_ = shape1[i];
  int shape0_ = shape0[i];
  int idx = 0;
  for(int k = 0; k<shape0_; k++)
  {
     float vec0k = vec0[k];
     idx = k * shape1_;
     for(int l = 0; l<shape1_; l++)
     {
        res[idx + l] = beta_*mat0[idx + l] + alpha_ * vec0k * vec1[l];
     }
  }
}
""")

mod61 = SourceModule("""
__global__ void gpu_generalized_outer_fast(float *mem1,
                                           float *mem2,
                                           float *mem3,
                                           float *mem4,
                                           int *ptr1,
                                           int *ptr2,
                                           int *ptr3,
                                           int *ptr4,
                                           int *shape0,
                                           int *shape1,
                                           float *alpha,
                                           float *beta,
                                           int *obj_id,
                                           int *row_id,
                                           int total_threads)
{

  const int thread_id = (blockIdx.x * blockDim.x + threadIdx.x);
  if (thread_id >= total_threads)
      return;
  const int i = obj_id[thread_id];
  const int k = row_id[thread_id];
  const int shape0_ = shape0[i];
  const int shape1_ = shape1[i];
  float * mat0 = &mem3[ptr3[i]];
  float * res = &mem4[ptr4[i]];
  float * vec0 = &mem1[ptr1[i]];
  float * vec1 = &mem2[ptr2[i]];
  const float beta_ = beta[i];
  const float alpha_ = alpha[i];
  for(int l = 0; l<shape0_; l++)
  {
     res[l * shape1_ + k] = beta_ * mat0[l * shape1_ + k] + alpha_ * vec0[l] * vec1[k];
  }
}
""")


mod62 = SourceModule("""
__global__ void gpu_generalized_outer_fast2(float *mem1,
                                           float *mem2,
                                           float *mem3,
                                           float *mem4,
                                           int *ptr1,
                                           int *ptr2,
                                           int *ptr3,
                                           int *ptr4,
                                           int *shape0,
                                           int *shape1,
                                           float alpha,
                                           float beta,
                                           int *obj_id,
                                           int *row_id,
                                           int total_threads)
{

  const int thread_id = (blockIdx.x * blockDim.x + threadIdx.x);
  if (thread_id >= total_threads)
      return;
  const int i = obj_id[thread_id];
  const int k = row_id[thread_id];
  const int shape0_ = shape0[i];
  const int shape1_ = shape1[i];
  float * mat0 = &mem3[ptr3[i]];
  float * res = &mem4[ptr4[i]];
  float * vec0 = &mem1[ptr1[i]];
  float * vec1 = &mem2[ptr2[i]];
  for(int l = 0; l<shape0_; l++)
  {
     res[l * shape1_ + k] = beta * mat0[l * shape1_ + k] + alpha * vec0[l] * vec1[k];
  }
}

__global__ void gpu_generalized_outer_fast3(float *mem1,
                                           float *mem2,
                                           float *mem3,
                                           float *mem4,
                                           int *ptr1,
                                           int *ptr2,
                                           int *ptr3,
                                           int *ptr4,
                                           int *shape0,
                                           int *shape1,
                                           float *alpha,
                                           float *beta,
                                           int *obj_id,
                                           int *row_id,
                                           int total_threads)
{

  const int thread_id = (blockIdx.x * blockDim.x + threadIdx.x);
  if (thread_id >= total_threads)
      return;
  const int i = obj_id[thread_id];
  const int k = row_id[thread_id];
  const int shape0_ = shape0[i];
  const int shape1_ = shape1[i];
  float * mat0 = &mem3[ptr3[i]];
  float * res = &mem4[ptr4[i]];
  float * vec0 = &mem1[ptr1[i]];
  float * vec1 = &mem2[ptr2[i]];
  for(int l = 0; l<shape0_; l++)
  {
     res[l * shape1_ + k] = fmaf(beta[i], mat0[l * shape1_ + k], alpha[i] * vec0[l] * vec1[k]);
  }
}
""")


mod10 = SourceModule("""
__global__ void gpu_add(float *mem1,
                                      float *mem2,
                                      int *ptr1,
                                      int *ptr2,
                                      int *shape0,
                                      int *shape1,
                                      int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  float * mat0 = &mem1[ptr1[i]];
  float * mat1 = &mem2[ptr2[i]];
  int shape1_ = shape1[i];
  int shape0_ = shape0[i];
  for(int k = 0; k<shape0_; k++)
  {
     for(int l = 0; l<shape1_; l++)
     {
        mat1[k*shape1_+l] += mat0[k*shape1_+l];
     }
  }
}
""")

mod11 = SourceModule("""
__global__ void gpu_mov(float *mem1,
                                      float *mem2,
                                      int *ptr1,
                                      int *ptr2,
                                      int *shape0,
                                      int *shape1,
                                      int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  float * mat0 = &mem1[ptr1[i]];
  float * mat1 = &mem2[ptr2[i]];
  for(int k = 0; k<shape0[i]; k++)
  {
     for(int l = 0; l<shape1[i]; l++)
     {
        mat1[k*shape1[i]+l] = mat0[k*shape1[i]+l];
     }
  }
}
""")

mod12 = SourceModule("""
//#include <stdio.h>
__global__ void gpu_dist_frame(float *frame_arr,
                               float *input_obj_mem,
                               int *ptr2,
                               int shape0,
                               int shape1,
                               int dx,
                               int dy,
                               int sx,
                               int sy,
                               int input_offset,
                               int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;

  int x_block_ind = i / dy;
  int y_block_ind = i % dy;
  int x0 = x_block_ind * sx;
  int y0 = y_block_ind * sy;

  if (0)
     return;
  for (int j=0; j<sy; j++)
  {
      int y = y0 + j;
      for (int k=0; k<sx; k++)
      {
          int x = x0 + k;
          int mem_ind = 3*(shape0 * x + y);
          input_obj_mem[ptr2[i] + input_offset + 3*(k*sy + j)] = frame_arr[mem_ind];
          input_obj_mem[ptr2[i] + input_offset + 3*(k*sy + j) + 1] = frame_arr[mem_ind + 1];
          input_obj_mem[ptr2[i] + input_offset + 3*(k*sy + j) + 2] = frame_arr[mem_ind + 2];
      }
  }
}

__global__ void gpu_dist_frame4(float *frame_arr,
                                float *input_obj_mem,
                                int *ptr2,
                                int shape0,
                                int shape1,
                                int dx,
                                int dy,
                                int sx,
                                int sy,
                                int input_offset,
                                int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;

  int x_block_ind = i / dy;
  int y_block_ind = i % dy;
  int x0 = x_block_ind * sx;
  int y0 = y_block_ind * sy;

  if (0)
     return;
  for (int j=0; j<sy; j++)
  {
      int y = y0 + j;
      for (int k=0; k<sx; k++)
      {
          int x = x0 + k;
          int mem_ind = 4*(shape0 * x + y);
          input_obj_mem[ptr2[i] + input_offset + 4*(k*sy + j)] = frame_arr[mem_ind];
          input_obj_mem[ptr2[i] + input_offset + 4*(k*sy + j) + 1] = frame_arr[mem_ind + 1];
          input_obj_mem[ptr2[i] + input_offset + 4*(k*sy + j) + 2] = frame_arr[mem_ind + 2];
          input_obj_mem[ptr2[i] + input_offset + 4*(k*sy + j) + 3] = frame_arr[mem_ind + 3];
      }
  }
}

""")

mod13 = SourceModule("""
#include <stdio.h>
__global__ void gpu_calc_error_frame(float *frame_arr,
                               float *out_obj_mem,
                               int *ptr2,
                               float *error_obj_mem,
                               int *ptr3,
                               int shape0,
                               int shape1,
                               int dx,
                               int dy,
                               int sx,
                               int sy,
                               int input_offset,
                               int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;

  int x_block_ind = i / dy;
  int y_block_ind = i % dy;
  int x0 = x_block_ind * sx;
  int y0 = y_block_ind * sy;

  for (int j=0; j<sy; j++)
  {
      int y = y0 + j;
      for (int k=0; k<sx; k++)
      {
          int x = x0 + k;
          int mem_ind = 3*(shape0 * x + y);
          error_obj_mem[ptr3[i] + input_offset + 3*(k*sy + j)] = - out_obj_mem[ptr2[i] + input_offset + 3*(k*sy + j)] + frame_arr[mem_ind];
          error_obj_mem[ptr3[i] + input_offset + 3*(k*sy + j) + 1] = - out_obj_mem[ptr2[i] + input_offset + 3*(k*sy + j) + 1] + frame_arr[mem_ind + 1];
          error_obj_mem[ptr3[i] + input_offset + 3*(k*sy + j) + 2] = - out_obj_mem[ptr2[i] + input_offset + 3*(k*sy + j) + 2] + frame_arr[mem_ind + 2];
      }
  }
}

__global__ void gpu_calc_error_frame_1ch(float *frame_arr,
                               float *out_obj_mem,
                               int *ptr2,
                               float *error_obj_mem,
                               int *ptr3,
                               int shape0,
                               int shape1,
                               int dx,
                               int dy,
                               int sx,
                               int sy,
                               int input_offset,
                               int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;

  int x_block_ind = i / dy;
  int y_block_ind = i % dy;
  int x0 = x_block_ind * sx;
  int y0 = y_block_ind * sy;

  for (int j=0; j<sy; j++)
  {
      int y = y0 + j;
      for (int k=0; k<sx; k++)
      {
          int x = x0 + k;
          int mem_ind =(shape0 * x + y);
          error_obj_mem[ptr3[i] + input_offset + (k*sy + j)] = - out_obj_mem[ptr2[i] + input_offset + (k*sy + j)] + frame_arr[mem_ind];
      }
  }
}

__global__ void gpu_calc_abs_diff_error_frame_1ch(float *frame_arr,
                                                  float *out_obj_mem,
                                                  int *ptr2,
                                                  float *error_obj_mem,
                                                  int *ptr3,
                                                  int shape0,
                                                  int shape1,
                                                  int dx,
                                                  int dy,
                                                  int sx,
                                                  int sy,
                                                  int input_offset,
                                                  int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;

  int x_block_ind = i / dy;
  int y_block_ind = i % dy;
  int x0 = x_block_ind * sx;
  int y0 = y_block_ind * sy;

  for (int j=0; j<sy; j++)
  {
      int y = y0 + j;
      for (int k=0; k<sx; k++)
      {
          int x = x0 + k;
          int mem_ind =(shape0 * x + y);
          error_obj_mem[ptr3[i] + input_offset + (k*sy + j)] = copysignf(1.0, - out_obj_mem[ptr2[i] + input_offset + (k*sy + j)] + frame_arr[mem_ind]);
      }
  }
}
""")

mod14 = SourceModule("""
//#include <stdio.h>
__global__ void gpu_collect_frame4(float *frame_arr,
                                  float *input_obj_mem,
                                  int *ptr2,
                                  int shape0,
                                  int shape1,
                                  int dx,
                                  int dy,
                                  int sx,
                                  int sy,
                                  int input_offset,
                                  int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;

  int x_block_ind = i / dy;
  int y_block_ind = i % dy;
  int x0 = x_block_ind * sx;
  int y0 = y_block_ind * sy;

  for (int j=0; j<sy; j++)
  {
      int y = y0 + j;
      for (int k=0; k<sx; k++)
      {
          int x = x0 + k;
          int mem_ind = 4*(shape0 * x + y);
          frame_arr[mem_ind] = input_obj_mem[ptr2[i] + input_offset + 4*(k*sy + j)];
          frame_arr[mem_ind + 1] = input_obj_mem[ptr2[i] + input_offset + 4*(k*sy + j) + 1];
          frame_arr[mem_ind + 2] = input_obj_mem[ptr2[i] + input_offset + 4*(k*sy + j) + 2];
          frame_arr[mem_ind + 3] = input_obj_mem[ptr2[i] + input_offset + 4*(k*sy + j) + 3];
      }
  }
}

__global__ void gpu_collect_frame(float *frame_arr,
                                  float *input_obj_mem,
                                  int *ptr2,
                                  int shape0,
                                  int shape1,
                                  int dx,
                                  int dy,
                                  int sx,
                                  int sy,
                                  int input_offset,
                                  int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;

  int x_block_ind = i / dy;
  int y_block_ind = i % dy;
  int x0 = x_block_ind * sx;
  int y0 = y_block_ind * sy;

  for (int j=0; j<sy; j++)
  {
      int y = y0 + j;
      for (int k=0; k<sx; k++)
      {
          int x = x0 + k;
          int mem_ind = 3*(shape0 * x + y);
          frame_arr[mem_ind] = input_obj_mem[ptr2[i] + input_offset + 3*(k*sy + j)];
          frame_arr[mem_ind + 1] = input_obj_mem[ptr2[i] + input_offset + 3*(k*sy + j) + 1];
          frame_arr[mem_ind + 2] = input_obj_mem[ptr2[i] + input_offset + 3*(k*sy + j) + 2];
      }
  }
}

__global__ void gpu_collect_frame_1ch(float *frame_arr,
                                      float *input_obj_mem,
                                      int *ptr2,
                                      int shape0,
                                      int shape1,
                                      int dx,
                                      int dy,
                                      int sx,
                                      int sy,
                                      int input_offset,
                                      int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;

  int x_block_ind = i / dy;
  int y_block_ind = i % dy;
  int x0 = x_block_ind * sx;
  int y0 = y_block_ind * sy;

  for (int j=0; j<sy; j++)
  {
      int y = y0 + j;
      for (int k=0; k<sx; k++)
      {
          int x = x0 + k;
          int mem_ind = (shape0 * x + y);
          frame_arr[mem_ind] = input_obj_mem[ptr2[i] + input_offset + (k*sy + j)];
      }
  }
}
""")

mod141 = SourceModule("""
__global__ void gpu_collect_activ(unsigned int *frame_arr,
                                  float *input_obj_mem,
                                  int *ptr2,
                                  int shape0,
                                  int shape1,
                                  int dx,
                                  int dy,
                                  int sx,
                                  int sy,
                                  int ptr_offset,
                                  int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;

  int x_block_ind = i / dy;
  int y_block_ind = i % dy;
  int x0 = x_block_ind * sx;
  int y0 = y_block_ind * sy;

  for (int j=0; j<sy; j++)
  {
      int y = y0 + j;
      for (int k=0; k<sx; k++)
      {
          int x = x0 + k;
          int mem_ind = (shape0 * x + y);
          frame_arr[mem_ind] = min(255, __float2uint_rn(255*input_obj_mem[ptr2[i + ptr_offset] + (k*sy + j)]));
      }
  }
}
""")

def copy_blocks_py(from_arr, to_arr, from_ptr, from_qnt, to_ptr, nblocks):
    for k in range(nblocks):
        to_arr[to_ptr[k]:to_ptr[k]+from_qnt[k]] = from_arr[from_ptr[k]:from_ptr[k]+from_qnt[k]]


mod15 = SourceModule("""
__global__ void gpu_copy_blocks(float *from_arr,
                                float *to_arr,
                                int *from_ptr,
                                int *from_qnt,
                                int *to_ptr,
                                int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  int n = from_qnt[i];
  for (int j=0; j<n; j++)
  {
      to_arr[to_ptr[i]+j] = from_arr[from_ptr[i]+j];
  }
}

__global__ void gpu_copy_blocks_fixed(float *from_arr,
                                      float *to_arr,
                                      int *from_ptr,
                                      int *to_ptr,
                                      int from_qnt,
                                      float mul,
                                      int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  for (int j=0; j<from_qnt; j++)
  {
      to_arr[to_ptr[i]+j] = mul*from_arr[from_ptr[i]+j];
  }
}


__global__ void gpu_set_one_hot_error(float *from_arr,
                                      int *from_ptr,
                                      float *to_arr,
                                      int *to_ptr,
                                      int hot,
                                      int length,
                                      int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  for (int j=0; j<length; j++)
  {
      to_arr[to_ptr[i]+j] = 0 - from_arr[from_ptr[i]+j];
      if (j == hot)
         to_arr[to_ptr[i]+j] = 1 - from_arr[from_ptr[i]+j];
  }
}

__global__ void gpu_copy_blocks_comp(float *from_arr,
                                     float *to_arr,
                                     int *from_ptr,
                                     int *from_qnt,
                                     int *to_ptr,
                                     int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  int n = from_qnt[i];
  for (int j=0; j<n; j++)
  {
      to_arr[to_ptr[i]+j] = 0.8 * from_arr[from_ptr[i]+j] + 0.1;
  }
}

__global__ void gpu_copy_blocks_sigmoid(float *from_arr,
                                        float *to_arr,
                                        int *from_ptr,
                                        int *from_qnt,
                                        int *to_ptr,
                                        float beta,
                                        int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  int n = from_qnt[i];
  for (int j=0; j<n; j++)
  {
      to_arr[to_ptr[i]+j] = 1.0/(1.0+expf(-beta*from_arr[from_ptr[i]+j]));
  }
}

""")

mod16 = SourceModule("""
__global__ void gpu_clip(float *from_arr,
                         float val0,
                         float val1,
                         int total_obj)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i>=total_obj)
      return;
  if (from_arr[i]>val1)
  {
     from_arr[i]=val1;
  } else 
  {
    if (from_arr[i]<val0)
    {
       from_arr[i]=val0;
    }
  }
}
""")



def gpu_array_copy(arr_from, arr_to, nbytes=0):
    if nbytes == 0:
        nbytes = arr_from.dtype.itemsize * arr_from.size
    drv.memcpy_dtod(arr_to.ptr, arr_from.ptr, nbytes)


gpu_dot = mod2.get_function("gpu_dot")
gpu_sgn = mod2.get_function("gpu_sgn")
gpu_dot_transpose = mod4.get_function("gpu_dot_transpose")
gpu_dot_transpose_fast = mod41.get_function("gpu_dot_transpose_fast")
gpu_sum_dot_transpose = mod41.get_function("gpu_sum_dot_transpose")
gpu_dot_sigmoid = mod5.get_function("gpu_dot_sigmoid")
gpu_dot_fast = mod51.get_function("gpu_dot_fast")
gpu_dot_fast_set_bias = mod51.get_function("gpu_dot_fast_set_bias")
gpu_sigmoid_fast = mod51.get_function("gpu_sigmoid_fast")
gpu_sigmoid_poly_fast = mod51.get_function("gpu_sigmoid_poly_fast")
gpu_outer_simple = mod3.get_function("gpu_outer_simple")
gpu_generalized_outer = mod6.get_function("gpu_generalized_outer")
gpu_generalized_outer_fast = mod61.get_function("gpu_generalized_outer_fast")
gpu_generalized_outer_fast2 = mod62.get_function("gpu_generalized_outer_fast2")
gpu_generalized_outer_fast3 = mod62.get_function("gpu_generalized_outer_fast3")
gpu_dot_sigmoid_poly = mod7.get_function("gpu_dot_sigmoid_poly")
gpu_sigmoid_der_mul = mod8.get_function("gpu_sigmoid_der_mul")
gpu_sigmoid_poly_der_mul = mod9.get_function("gpu_sigmoid_poly_der_mul")
gpu_add = mod10.get_function("gpu_add")
gpu_mov = mod11.get_function("gpu_mov")
gpu_dist_frame = mod12.get_function("gpu_dist_frame")
gpu_dist_frame4 = mod12.get_function("gpu_dist_frame4")
gpu_calc_error_frame = mod13.get_function("gpu_calc_error_frame")
gpu_calc_error_frame_1ch = mod13.get_function("gpu_calc_error_frame_1ch")
gpu_calc_abs_diff_error_frame_1ch = mod13.get_function("gpu_calc_abs_diff_error_frame_1ch")
gpu_collect_frame = mod14.get_function("gpu_collect_frame")
gpu_collect_frame4 = mod14.get_function("gpu_collect_frame4")
gpu_collect_frame_1ch = mod14.get_function("gpu_collect_frame_1ch")
gpu_collect_activ = mod141.get_function("gpu_collect_activ")
gpu_copy_blocks = mod15.get_function("gpu_copy_blocks")
gpu_copy_blocks_fixed = mod15.get_function("gpu_copy_blocks_fixed")
gpu_copy_blocks_comp = mod15.get_function("gpu_copy_blocks_comp")
gpu_copy_blocks_sigm = mod15.get_function("gpu_copy_blocks_sigmoid")
gpu_set_one_hot_error = mod15.get_function("gpu_set_one_hot_error")
gpu_clip = mod16.get_function("gpu_clip")
