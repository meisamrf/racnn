#pragma once

void im2col(const float *in, float *out_mat, int height, int width, int dim);
void im2col7x7rgb(const float *in, float *out_mat, int height, int width);
int im2col8_mask(const float *in, float *out_mat,
	int height, int width, int dim, float *mask, int stride_jump, float mask_bias);
void col2im8_mask(float *out, const float *in_mat3, float *in_mat1,
	const float *bias3, const float *bias1, int rows, int dim);
void bias_relu_pool2_s2(float *out, const float *in_mat, const float *bias,
	int height, int width, int dim);
void bias_relu_pool3_s2(float *out, const float *in_mat, const float *bias,
	int height, int width, int dim);
void avg_pool2(float *out, const float *in_mat,
	int height, int width, int dim);
void max_pool2(float *out, const float *in_mat,
	int height, int width, int dim);
void bias_relu(float *in_out, const float *bias,
	int rows, int dim);
void col2im8_mask_pool(float *out, const float *in_mat3, float *in_mat1,
	const float *bias3, const float *bias1, int width, int height, int dim);
void add_bias_relu(float *in_out, const float *to_add, const float *bias,
	int rows, int dim);
