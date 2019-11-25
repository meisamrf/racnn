// RACNN GPU python wrapper 
// by Meisam Rakhshanfar

#include "libcuracnn.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <string>
#include <algorithm> 
#include <fstream> 

#define CUDA_DEVICE_NAME_SIZE 256
#define returnOnCudaError(err) if (CUDA_SUCCESS != err) {printCudaErrors(err); return err;}
#define PTX_FILE "./racnn64.ptx"

struct CUMat {
	CUdeviceptr mat;
	int size;
	int w, h;
	float value;
};

struct CUkernel
{
	CUfunction kernel;
	const char *name;
};

struct DeviceHandle
{
	CUdevice cuDevice;
	CUcontext cuContext;
	CUmodule cuModule;
	size_t totalGlobalMem;
	cublasHandle_t cublas_handle;
	CUkernel *cukern;

};


class curacnn
{
public:
	curacnn();
	~curacnn();
	int open();
	void release();
	bool predict(float *input_data, float *h_D);
	int build(int model_type);
	bool load_weight(int layer_num, int rows, int cols, float *weight_data);
	char *get_device_name();
	bool is_open();
	int get_weight_num();
	void get_inout_size(int *inout_size);
	void set_speed_test(bool state);
	unsigned long long mem_size();
private:
	char deviceName[CUDA_DEVICE_NAME_SIZE];
	DeviceHandle dh;
	bool device_open;
	CUMat cumat[6];
	CUMat *cuweight;
	const int *layer_size_w;
	const int *layer_size_h;
	int cuweight_num;
	int model_type;
	bool speed_test;
};


void printCudaErrors(CUresult err)
{
	printf("error = %04d\n", err);
}


const char * const cudaFuncs[] = { "im2col", "im2col7x7rgb", "im2col_bias_relu",
	"im2col_bias_maxpool_relu", "maxpool2_bias_relu", "add_bias_relu", "add_bias",
	"fill_index_ref", "im2col8_mask", "col2im8_mask_bias_relu", "maxpool2",
	"maxpool3_bias_relu", "bias_relu", "sum_bias_relu", "resample2", "avgpool2" };

enum CUfuncEnum
{
	cu_im2col, cu_im2col7x7rgb, cu_im2col_bias_relu, cu_im2col_bias_maxpool_relu,
	cu_maxpool2_bias_relu, cu_add_bias_relu, cu_add_bias,
	cu_fill_index_ref, cu_im2col8_mask, col2im8_mask_bias_relu, cu_maxpool2,
	cu_maxpool3_bias_relu, cu_bias_relu, cu_sum_bias_relu, cu_resample2, cu_avgpool2
};

int divUp(int num, int den) {
	return (num + den - 1) / den;
}

void softMax(float *datap, int data_size) {

	float maxval = *datap;
	float *data = datap;
	for (int k = 0; k < data_size; ++k) {
		float d = *data++;
		if (d > maxval) {
			maxval = d;
		}
	}

	data = datap;
	float sum = 0;
	for (int k = 0; k < data_size; ++k) {
		float d = *data;
		d = expf(d - maxval);
		sum += d;
		*data++ = d;
	}
	data = datap;
	for (int k = 0; k < data_size; ++k) {
		*data++ /= sum;
	}
}

CUresult load_weight_bias(float *h_weight, int h, int w,
	CUMat &d_weight) {

	CUresult cures;

	cures = cuMemAlloc(&d_weight.mat, w * h * sizeof(float));
	returnOnCudaError(cures);
	d_weight.size = w * h * sizeof(float);
	d_weight.w = w;
	d_weight.h = h;

	cures = cuMemcpyHtoD(d_weight.mat, h_weight, d_weight.size);
	returnOnCudaError(cures);

	return cures;
}

CUresult initCUDA(DeviceHandle &dh, char *deviceName)
{
	//CUfunction cuFunction = 0;
	CUresult status;
	int deviceCount = 0;
	// pick the device with highest Gflops/s
	int devID = 0;
	status = cuInit(0);
	returnOnCudaError(status);

	status = cuDeviceGetCount(&deviceCount);
	if (deviceCount == 0 || status != 0)
	{
		printf("cudaDeviceInit error: no devices supporting CUDA\n");
	}

	status = cuDeviceGet(&dh.cuDevice, devID);
	returnOnCudaError(status);

	// get compute capabilities and the devicename
	status = cuDeviceGetName(deviceName, CUDA_DEVICE_NAME_SIZE, dh.cuDevice);
	returnOnCudaError(status);

	status = cuDeviceTotalMem(&dh.totalGlobalMem, dh.cuDevice);
	returnOnCudaError(status);

	status = cuCtxCreate(&dh.cuContext, 0, dh.cuDevice);
	if (CUDA_SUCCESS != status) {
		cuCtxDestroy(dh.cuContext);
		return status;
	}

	status = cuModuleLoad(&dh.cuModule, PTX_FILE);
	if (CUDA_SUCCESS != status) {
		cuCtxDestroy(dh.cuContext);
		return status;
	}

	int kern_num = sizeof(cudaFuncs) / sizeof(cudaFuncs[0]);
	dh.cukern = new CUkernel[kern_num];

	for (int k = 0; k < kern_num; k++) {
		dh.cukern[k].name = cudaFuncs[k];
		status = cuModuleGetFunction(&dh.cukern[k].kernel, dh.cuModule, dh.cukern[k].name);
		if (CUDA_SUCCESS != status) {
			cuCtxDestroy(dh.cuContext);
			return status;
		}
	}

	int res = cublasCreate(&dh.cublas_handle);
	returnOnCudaError((CUresult)res);

	return CUDA_SUCCESS;
}

CUresult device_conv2_racnn(CUfunction &cuk_index_ref, CUfunction &cuk_im2col8_mask,
	CUfunction &cuk_col2im8_mask_bias_relu, CUdeviceptr *img_in,
	CUdeviceptr *buffer_out1x1, CUdeviceptr *buffer_out3x3,
	CUdeviceptr *weight1x1_det, CUdeviceptr *weight3x3_noc,
	CUdeviceptr *bias_1x1, CUdeviceptr *bias_3x3, CUdeviceptr *bymask_index,
	CUdeviceptr *bymask_page_ptr,
	float bias_bymask,
	cublasHandle_t &handle,
	int img_width, int img_height, int img_dim, int filters) {


	const float alpha = 1.0f;
	const float beta = 0.0f;

	float *d_A, *d_B, *d_C;
	d_A = (float *)(*img_in);
	d_B = (float *)(*weight1x1_det);
	d_C = (float *)(*buffer_out1x1);

	//       K				  N					 N
	//   |------          |-------			|----------
	// M |			X   K |             = M |
	//   |                |					|
	// Because it's column major we have to put weights first

	int mm_N = img_width * img_height;
	int mm_K = img_dim;
	int mm_M = (filters + 1);

	CUresult res = (CUresult)cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, mm_M, mm_N, mm_K, &alpha, d_B, mm_M, d_A, mm_K, &beta, d_C, mm_M);

	int divisions = 512;
	dim3 block(divisions, 1, 1);
	dim3 grid(1, 1, 1);
	void *args[10];
	int mat_rows = img_width * img_height;
	int stride_jump = (filters + 1);

	args[0] = bymask_index;
	args[1] = buffer_out1x1;
	args[2] = bymask_page_ptr;
	args[3] = &mat_rows;
	args[4] = &stride_jump;
	args[5] = &bias_bymask;

	res = cuLaunchKernel(cuk_index_ref, grid.x, grid.y, grid.z,
		block.x, block.y, block.z,
		512 * sizeof(float), NULL, args, NULL);

	returnOnCudaError(res);


	args[0] = img_in;
	args[1] = buffer_out3x3;
	args[2] = bymask_index;
	args[3] = bymask_page_ptr;
	args[4] = &img_width;
	args[5] = &img_height;
	args[6] = &img_dim;
	args[7] = &divisions;


	block.x = 512;
	grid.x = (img_width * img_height*img_dim) / block.x;
	if ((img_width * img_height*img_dim) % block.x) {
		return (CUresult)1;
	}

	res = cuLaunchKernel(cuk_im2col8_mask, grid.x, grid.y, grid.z,
		block.x, block.y, block.z,
		512 * sizeof(float), NULL, args, NULL);

	returnOnCudaError(res);

	int acc[1];
	res = cuMemcpyDtoH((void *)(acc), (*bymask_page_ptr) + divisions * sizeof(int), 1 * sizeof(int));
	returnOnCudaError(res);

	if (acc[0] > 0) {

		mm_N = acc[0];
		mm_K = img_dim * 8;
		mm_M = filters;

		d_A = (float *)(*buffer_out3x3);
		d_B = (float *)(*weight3x3_noc);
		d_C = (float *)(*img_in);

		res = (CUresult)cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, mm_M, mm_N, mm_K, &alpha, d_B, mm_M, d_A, mm_K, &beta, d_C, mm_M);

	}

	block.x = 512;
	grid.x = (img_width * img_height*filters) / block.x;
	if ((img_width * img_height*filters) % block.x) {
		return (CUresult)1;
	}

	args[0] = buffer_out3x3;
	args[1] = img_in;
	args[2] = buffer_out1x1;
	args[3] = bias_1x1;
	args[4] = bias_3x3;
	args[5] = bymask_index;
	args[6] = bymask_page_ptr;
	args[7] = &mat_rows;
	args[8] = &filters;
	args[9] = &divisions;

	res = cuLaunchKernel(cuk_col2im8_mask_bias_relu, grid.x, grid.y, grid.z,
		block.x, block.y, block.z,
		512 * sizeof(float), NULL, args, NULL);

	return res;
}

CUresult device_conv7x7_pool3(CUfunction &cu_im2col, CUfunction &cu_maxpool3, CUdeviceptr *img_in,
	CUdeviceptr *img_out, CUdeviceptr *img_tmp, CUdeviceptr *weights, CUdeviceptr *bias,
	cublasHandle_t &handle, int &img_width, int &img_height, int &img_dim, int filters) {

	dim3 block(512, 1, 1);
	dim3 grid(1, 1, 1);
	grid.x = divUp((img_width / 2)* (img_height / 2) * 7 * img_dim, block.x);
	void *args[6];

	args[0] = img_in;
	args[1] = img_tmp;
	args[2] = &img_width;
	args[3] = &img_height;

	CUresult res = cuLaunchKernel(cu_im2col, grid.x, grid.y, grid.z,
		block.x, block.y, block.z,
		512 * sizeof(float),
		NULL, args, NULL);

	returnOnCudaError(res);

	const float alpha = 1.0f;
	const float beta = 0.0f;

	float *d_A, *d_B, *d_C;
	d_A = (float *)(*img_tmp);
	d_B = (float *)(*weights);
	d_C = (float *)(*img_in);

	img_width /= 2;
	img_height /= 2;

	int mm_N = img_width * img_height;
	int mm_K = 7 * 7 * img_dim;
	int mm_M = filters;

	res = (CUresult)cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, mm_M, mm_N, mm_K, &alpha, d_B, mm_M, d_A, mm_K, &beta, d_C, mm_M);
	returnOnCudaError(res);


	img_dim = filters;

	args[0] = img_in;
	args[1] = bias;
	args[2] = &img_width;
	args[3] = &img_height;
	args[4] = &filters;
	args[5] = img_out;

	grid.x = divUp((img_width / 2)* (img_height / 2) * filters, block.x);

	res = cuLaunchKernel(cu_maxpool3, grid.x, grid.y, grid.z,
		block.x, block.y, block.z,
		512 * sizeof(float),
		NULL, args, NULL);

	returnOnCudaError(res);

	img_width /= 2;
	img_height /= 2;

	return res;
}


CUresult device_conv2_1x1(CUfunction &bias_relu, CUdeviceptr *img_in, CUdeviceptr *img_out,
	CUdeviceptr *weights, CUdeviceptr *bias, cublasHandle_t &handle,
	int img_width, int img_height, int &img_dim, int filters) {


	const float alpha = 1.0f;
	const float beta = 0.0f;

	float *d_A, *d_B, *d_C;
	d_A = (float *)(*img_in);
	d_B = (float *)(*weights);
	d_C = (float *)(*img_out);

	int mm_N = img_width * img_height;
	int mm_K = img_dim;
	int mm_M = filters;

	CUresult res = (CUresult)cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, mm_M, mm_N, mm_K, &alpha, d_B, mm_M, d_A, mm_K, &beta, d_C, mm_M);
	returnOnCudaError(res);


	img_dim = filters;

	if (bias == nullptr)
		return res;


	int total_length = img_height * img_width*img_dim;
	dim3 block(512, 1, 1);
	dim3 grid(total_length / block.x, 1, 1);
	void *args[4];

	args[0] = img_out;
	args[1] = bias;
	args[2] = &total_length;
	args[3] = &img_dim;

	res = cuLaunchKernel(bias_relu, grid.x, grid.y, grid.z,
		block.x, block.y, block.z,
		512 * sizeof(float),
		NULL, args, NULL);

	returnOnCudaError(res);

	return res;
}

CUresult device_sum_bias_relu(CUfunction &bias_relu, CUdeviceptr *img1, CUdeviceptr *img2,
	CUdeviceptr *bias, int img_width, int img_height, int img_dim) {

	int total_length = img_height * img_width*img_dim;
	dim3 block(512, 1, 1);
	dim3 grid(total_length / block.x, 1, 1);
	void *args[5];

	args[0] = img1;
	args[1] = img2;
	args[2] = bias;
	args[3] = &total_length;
	args[4] = &img_dim;

	CUresult res = cuLaunchKernel(bias_relu, grid.x, grid.y, grid.z,
		block.x, block.y, block.z,
		512 * sizeof(float),
		NULL, args, NULL);

	return res;
}

CUresult device_resample2(CUfunction &cu_resample, CUdeviceptr *img_in, CUdeviceptr *img_out,
	int &img_width, int &img_height, int img_dim) {

	dim3 block(512, 1, 1);
	dim3 grid((img_height / 2) * (img_width / 2)*img_dim / block.x, 1, 1);
	void *args[5];

	args[0] = img_in;
	args[1] = img_out;
	args[2] = &img_width;
	args[3] = &img_height;
	args[4] = &img_dim;

	CUresult res = cuLaunchKernel(cu_resample, grid.x, grid.y, grid.z,
		block.x, block.y, block.z,
		512 * sizeof(float),
		NULL, args, NULL);

	img_width /= 2;
	img_height /= 2;
	return res;
}


CUresult device_conv2_3x3(CUfunction &cu_im2col, CUfunction &cu_bias_relu, CUdeviceptr *img_in, CUdeviceptr *img_out,
	CUdeviceptr *mat_img, CUdeviceptr *weights, CUdeviceptr *bias, cublasHandle_t &handle,
	int img_width, int img_height, int img_dim, int filters) {


	dim3 block(512, 1, 1);
	dim3 grid(img_height* img_width*img_dim / block.x, 1, 1);
	void *args[5];

	args[0] = img_in;
	args[1] = &img_width;
	args[2] = &img_height;
	args[3] = &img_dim;
	args[4] = mat_img;

	CUresult res = cuLaunchKernel(cu_im2col, grid.x, grid.y, grid.z,
		block.x, block.y, block.z,
		512 * sizeof(float),
		NULL, args, NULL);

	returnOnCudaError(res);

	const float alpha = 1.0f;
	const float beta = 0.0f;

	float *d_A, *d_B, *d_C;
	d_A = (float *)(*mat_img);
	d_B = (float *)(*weights);
	d_C = (float *)(*img_out);

	int mm_N = img_width * img_height;
	int mm_K = 9 * img_dim;
	int mm_M = filters;

	res = (CUresult)cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, mm_M, mm_N, mm_K, &alpha, d_B, mm_M, d_A, mm_K, &beta, d_C, mm_M);
	returnOnCudaError(res);

	img_dim = filters;
	int total_length = img_height * img_width*img_dim;
	grid.x = total_length / block.x;

	args[0] = img_out;
	args[1] = bias;
	args[2] = &total_length;
	args[3] = &img_dim;

	res = cuLaunchKernel(cu_bias_relu, grid.x, grid.y, grid.z,
		block.x, block.y, block.z,
		512 * sizeof(float),
		NULL, args, NULL);

	return res;
}




CUresult device_conv2(CUfunction &cu_im2col, CUdeviceptr *img_in, CUdeviceptr *img_out,
	CUdeviceptr *mat_img, CUdeviceptr *weights, CUdeviceptr *bias, cublasHandle_t &handle,
	int img_width, int img_height, int img_dim, int filters) {


	dim3 block(512, 1, 1);
	dim3 grid(img_height* img_width*img_dim / block.x, 1, 1);
	void *args[6];

	if (bias == nullptr) {
		args[0] = img_in;
		args[1] = &img_width;
		args[2] = &img_height;
		args[3] = &img_dim;
		args[4] = mat_img;
	}
	else {
		args[0] = img_in;
		args[1] = &img_width;
		args[2] = &img_height;
		args[3] = &img_dim;
		args[4] = mat_img;
		args[5] = bias;
	}

	CUresult res = cuLaunchKernel(cu_im2col, grid.x, grid.y, grid.z,
		block.x, block.y, block.z,
		512 * sizeof(float),
		NULL, args, NULL);

	returnOnCudaError(res);

	const float alpha = 1.0f;
	const float beta = 0.0f;

	float *d_A, *d_B, *d_C;
	d_A = (float *)(*mat_img);
	d_B = (float *)(*weights);
	d_C = (float *)(*img_out);

	int mm_N = img_width * img_height;
	int mm_K = 9 * img_dim;
	int mm_M = filters;

	res = (CUresult)cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, mm_M, mm_N, mm_K, &alpha, d_B, mm_M, d_A, mm_K, &beta, d_C, mm_M);

	return res;
}


CUresult device_maxpool2_bias(CUfunction &cu_maxpool2, CUdeviceptr *img_in,
	CUdeviceptr *img_out, CUdeviceptr *bias, int img_width, int img_height, int img_dim) {

	dim3 block(512, 1, 1);

	dim3 grid(divUp((img_width / 2)* (img_height / 2)*img_dim, block.x), 1, 1);
	void *args[6] = { img_in, bias, &img_width, &img_height, &img_dim, img_out };

	CUresult res = cuLaunchKernel(cu_maxpool2, grid.x, grid.y, grid.z,
		block.x, block.y, block.z,
		512 * sizeof(float),
		NULL, args, NULL);

	return res;
}

CUresult device_maxpool2(CUfunction &cu_maxpool2, CUdeviceptr *img_in,
	CUdeviceptr *img_out, int img_width, int img_height, int img_dim) {

	dim3 block(512, 1, 1);

	dim3 grid(divUp((img_width / 2)* (img_height / 2)*img_dim, block.x), 1, 1);
	void *args[5] = { img_in, &img_width, &img_height, &img_dim, img_out };

	CUresult res = cuLaunchKernel(cu_maxpool2, grid.x, grid.y, grid.z,
		block.x, block.y, block.z,
		512 * sizeof(float),
		NULL, args, NULL);

	return res;
}


CUresult device_avgpool2(CUfunction &cu_avgpool2, CUdeviceptr *img_in,
	CUdeviceptr *img_out, int &img_width, int &img_height, int img_dim) {

	dim3 block(512, 1, 1);

	dim3 grid(divUp((img_width / 2)* (img_height / 2)*img_dim, block.x), 1, 1);
	void *args[5] = { img_in, &img_width, &img_height, &img_dim, img_out };

	CUresult res = cuLaunchKernel(cu_avgpool2, grid.x, grid.y, grid.z,
		block.x, block.y, block.z,
		512 * sizeof(float),
		NULL, args, NULL);

	img_width /= 2;
	img_height /= 2;

	return res;
}

CUresult device_dense_mul(CUfunction &cu_add_bias_relu, CUdeviceptr *img_in, CUdeviceptr *img_out, CUdeviceptr *weights,
	CUdeviceptr *bias, int img_width, int filters, cublasHandle_t &handle) {

	const float alpha = 1.0f;
	const float beta = 0.0f;

	float *d_A, *d_B, *d_C;
	d_A = (float *)(*img_in);
	d_B = (float *)(*weights);
	d_C = (float *)(*img_out);

	int mm_N = 1;
	int mm_K = img_width;
	int mm_M = filters;

	CUresult res = (CUresult)cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, mm_M, mm_N, mm_K, &alpha, d_B, mm_M, d_A, mm_K, &beta, d_C, mm_M);
	returnOnCudaError(res);

	dim3 block(std::min(512, filters), 1, 1);
	dim3 grid(divUp(filters, block.x), 1, 1);
	void *args[3] = { img_out, bias, &filters };

	res = cuLaunchKernel(cu_add_bias_relu, grid.x, grid.y, grid.z,
		block.x, block.y, block.z,
		512 * sizeof(float),
		NULL, args, NULL);

	return res;
}



CUresult vgg16_classifier(DeviceHandle &dh, CUdeviceptr &input, CUdeviceptr &output, CUdeviceptr &mulbuf,
	CUMat *weights, float *input_data, float *h_D, bool speed_test=false) {

	CUresult err;
	int layer_width = 224;
	int layer_height = 224;
	int layer_depth = 3;

	int mem_size_input = layer_width * layer_height * layer_depth * sizeof(float);
	err = cuMemcpyHtoD(input, input_data, mem_size_input);
	returnOnCudaError(err);


	err = device_conv2(dh.cukern[cu_im2col].kernel, &input, &output, &mulbuf,
		&weights[0].mat, nullptr, dh.cublas_handle, layer_width, layer_height, layer_depth,
		weights[0].w);
	returnOnCudaError(err);

	if (speed_test) {
		int sync[1];
		err = cuMemcpyDtoH((void *)(sync), output, sizeof(int));
		returnOnCudaError(err);
	}

	layer_depth = weights[0].w;
	err = device_conv2(dh.cukern[cu_im2col_bias_relu].kernel, &output, &input, &mulbuf,
		&weights[2].mat, &weights[1].mat, dh.cublas_handle, layer_width, layer_height, layer_depth,
		weights[2].w);
	returnOnCudaError(err);

	if (speed_test) {
		int sync[1];
		err = cuMemcpyDtoH((void *)(sync), output, sizeof(int));
		returnOnCudaError(err);
	}

	layer_depth = weights[2].w;
	layer_width /= 2;
	layer_height /= 2;

	err = device_conv2(dh.cukern[cu_im2col_bias_maxpool_relu].kernel, &input, &output, &mulbuf,
		&weights[4].mat, &weights[3].mat, dh.cublas_handle, layer_width, layer_height, layer_depth,
		weights[4].w);
	returnOnCudaError(err);

	if (speed_test) {
		int sync[1];
		err = cuMemcpyDtoH((void *)(sync), output, sizeof(int));
		returnOnCudaError(err);
	}

	layer_depth = weights[4].w;
	err = device_conv2(dh.cukern[cu_im2col_bias_relu].kernel, &output, &input, &mulbuf,
		&weights[6].mat, &weights[5].mat, dh.cublas_handle, layer_width, layer_height, layer_depth,
		weights[6].w);
	returnOnCudaError(err);

	if (speed_test) {
		int sync[1];
		err = cuMemcpyDtoH((void *)(sync), output, sizeof(int));
		returnOnCudaError(err);
	}

	layer_depth = weights[6].w;
	layer_width /= 2;
	layer_height /= 2;
	err = device_conv2(dh.cukern[cu_im2col_bias_maxpool_relu].kernel, &input, &output, &mulbuf,
		&weights[8].mat, &weights[7].mat, dh.cublas_handle, layer_width, layer_height, layer_depth,
		weights[8].w);
	returnOnCudaError(err);

	if (speed_test) {
		int sync[1];
		err = cuMemcpyDtoH((void *)(sync), output, sizeof(int));
		returnOnCudaError(err);
	}

	layer_depth = weights[8].w;
	err = device_conv2(dh.cukern[cu_im2col_bias_relu].kernel, &output, &input, &mulbuf,
		&weights[10].mat, &weights[9].mat, dh.cublas_handle, layer_width, layer_height, layer_depth,
		weights[10].w);
	returnOnCudaError(err);

	if (speed_test) {
		int sync[1];
		err = cuMemcpyDtoH((void *)(sync), output, sizeof(int));
		returnOnCudaError(err);
	}

	layer_depth = weights[10].w;
	err = device_conv2(dh.cukern[cu_im2col_bias_relu].kernel, &input, &output, &mulbuf,
		&weights[12].mat, &weights[11].mat, dh.cublas_handle, layer_width, layer_height, layer_depth,
		weights[12].w);
	returnOnCudaError(err);

	if (speed_test) {
		int sync[1];
		err = cuMemcpyDtoH((void *)(sync), output, sizeof(int));
		returnOnCudaError(err);
	}

	layer_depth = weights[12].w;
	layer_width /= 2;
	layer_height /= 2;
	err = device_conv2(dh.cukern[cu_im2col_bias_maxpool_relu].kernel, &output, &input, &mulbuf,
		&weights[14].mat, &weights[13].mat, dh.cublas_handle, layer_width, layer_height, layer_depth,
		weights[14].w);
	returnOnCudaError(err);

	if (speed_test) {
		int sync[1];
		err = cuMemcpyDtoH((void *)(sync), output, sizeof(int));
		returnOnCudaError(err);
	}

	layer_depth = weights[14].w;
	err = device_conv2(dh.cukern[cu_im2col_bias_relu].kernel, &input, &output, &mulbuf,
		&weights[16].mat, &weights[15].mat, dh.cublas_handle, layer_width, layer_height, layer_depth,
		weights[16].w);
	returnOnCudaError(err);

	if (speed_test) {
		int sync[1];
		err = cuMemcpyDtoH((void *)(sync), output, sizeof(int));
		returnOnCudaError(err);
	}

	layer_depth = weights[16].w;
	err = device_conv2(dh.cukern[cu_im2col_bias_relu].kernel, &output, &input, &mulbuf,
		&weights[18].mat, &weights[17].mat, dh.cublas_handle, layer_width, layer_height, layer_depth,
		weights[18].w);
	returnOnCudaError(err);

	if (speed_test) {
		int sync[1];
		err = cuMemcpyDtoH((void *)(sync), output, sizeof(int));
		returnOnCudaError(err);
	}

	layer_depth = weights[18].w;
	layer_width /= 2;
	layer_height /= 2;
	err = device_conv2(dh.cukern[cu_im2col_bias_maxpool_relu].kernel, &input, &output, &mulbuf,
		&weights[20].mat, &weights[19].mat, dh.cublas_handle, layer_width, layer_height, layer_depth,
		weights[20].w);
	returnOnCudaError(err);

	if (speed_test) {
		int sync[1];
		err = cuMemcpyDtoH((void *)(sync), output, sizeof(int));
		returnOnCudaError(err);
	}

	layer_depth = weights[20].w;
	err = device_conv2(dh.cukern[cu_im2col_bias_relu].kernel, &output, &input, &mulbuf,
		&weights[22].mat, &weights[21].mat, dh.cublas_handle, layer_width, layer_height, layer_depth,
		weights[22].w);
	returnOnCudaError(err);

	if (speed_test) {
		int sync[1];
		err = cuMemcpyDtoH((void *)(sync), output, sizeof(int));
		returnOnCudaError(err);
	}

	layer_depth = weights[22].w;
	err = device_conv2(dh.cukern[cu_im2col_bias_relu].kernel, &input, &output, &mulbuf,
		&weights[24].mat, &weights[23].mat, dh.cublas_handle, layer_width, layer_height, layer_depth,
		weights[24].w);
	returnOnCudaError(err);

	if (speed_test) {
		int sync[1];
		err = cuMemcpyDtoH((void *)(sync), output, sizeof(int));
		returnOnCudaError(err);
	}

	layer_depth = weights[24].w;
	err = device_maxpool2_bias(dh.cukern[cu_maxpool2_bias_relu].kernel, &output,
		&input, &weights[25].mat, layer_width, layer_height, layer_depth);
	returnOnCudaError(err);


	layer_width /= 2;
	layer_height /= 2;

	err = device_dense_mul(dh.cukern[cu_add_bias_relu].kernel, &input, &output, &weights[26].mat,
		&weights[27].mat, layer_width * layer_height * layer_depth, weights[26].w, dh.cublas_handle);
	returnOnCudaError(err);

	err = device_dense_mul(dh.cukern[cu_add_bias_relu].kernel, &output, &input, &weights[28].mat,
		&weights[29].mat, weights[26].w, weights[28].w, dh.cublas_handle);
	returnOnCudaError(err);

	err = device_dense_mul(dh.cukern[cu_add_bias].kernel, &input, &output, &weights[30].mat,
		&weights[31].mat, weights[28].w, weights[30].w, dh.cublas_handle);
	returnOnCudaError(err);

	err = cuMemcpyDtoH((void *)h_D, output, weights[30].w * sizeof(float));
	softMax(h_D, weights[30].w);
	return err;
}

CUresult vgg_racnn_block(DeviceHandle &dh, CUdeviceptr *input, CUdeviceptr *output,
	CUdeviceptr *tmp, CUMat *weights, CUdeviceptr *index, CUdeviceptr *page, int &layer_width,
	int &layer_height, int &layer_depth, int &weight_index, int block_num) {

	CUresult err;
	CUdeviceptr *swap;

	for (int blk = 0; blk < block_num; ++blk) {
		CUMat *cuweight = weights + weight_index;

		err = device_conv2_racnn(dh.cukern[cu_fill_index_ref].kernel, dh.cukern[cu_im2col8_mask].kernel,
			dh.cukern[col2im8_mask_bias_relu].kernel, input, tmp, output,
			&cuweight[0].mat, &cuweight[1].mat, &cuweight[2].mat, &cuweight[3].mat,
			index, page,
			cuweight[4].value, dh.cublas_handle,
			layer_width, layer_height, layer_depth,
			cuweight[0].w - 1);
		returnOnCudaError(err);
		weight_index += 5;
		layer_depth = cuweight[0].w - 1;

		swap = input;
		input = output;
		output = swap;

	}

	err = device_maxpool2(dh.cukern[cu_maxpool2].kernel, input,
		output, layer_width, layer_height, layer_depth);
	returnOnCudaError(err);
	layer_width /= 2;
	layer_height /= 2;

	return err;

}



CUresult vgg_conv_block(DeviceHandle &dh, CUdeviceptr *input, CUdeviceptr *output,
	CUdeviceptr *tmp, CUMat *weights, int &layer_width,
	int &layer_height, int &layer_depth, int &weight_index, int block_num, bool first_pool) {

	CUresult err;
	CUdeviceptr *swap;

	for (int blk = 0; blk < block_num; ++blk) {
		CUMat *cuweight = weights + weight_index;
		CUfunction i2c;
		CUdeviceptr *bias;
		int wi;
		if (blk == 0) {
			if (!first_pool) {
				i2c = dh.cukern[cu_im2col].kernel;
				bias = nullptr;
				wi = 0;
			}
			else {
				layer_width /= 2;
				layer_height /= 2;
				i2c = dh.cukern[cu_im2col_bias_maxpool_relu].kernel;
				bias = &cuweight[0].mat;
				wi = 1;
			}
		}
		else {
			i2c = dh.cukern[cu_im2col_bias_relu].kernel;
			bias = &cuweight[0].mat;
			wi = 1;
		}
		err = device_conv2(i2c, input, output, tmp,
			&cuweight[wi].mat, bias, dh.cublas_handle, layer_width, layer_height, layer_depth,
			cuweight[wi].w);
		returnOnCudaError(err);

		layer_depth = cuweight[wi].w;
		weight_index += (wi + 1);

		swap = input;
		input = output;
		output = swap;

	}

	return err;

}


CUresult vgg_racnn_classifier(DeviceHandle &dh, CUMat *buffs, CUMat *weights,
	float *input_data, float *h_D) {

	CUresult err;
	int layer_width = 224;
	int layer_height = 224;
	int layer_depth = 3;

	int mem_size_input = layer_width * layer_height * layer_depth * sizeof(float);
	err = cuMemcpyHtoD(buffs[0].mat, input_data, mem_size_input);
	returnOnCudaError(err);
	int weight_index = 0;

	err = vgg_racnn_block(dh, &buffs[0].mat, &buffs[1].mat, &buffs[2].mat, weights,
		&buffs[3].mat, &buffs[4].mat,
		layer_width, layer_height, layer_depth, weight_index, 2);
	returnOnCudaError(err);

	err = vgg_racnn_block(dh, &buffs[1].mat, &buffs[0].mat, &buffs[2].mat, weights,
		&buffs[3].mat, &buffs[4].mat,
		layer_width, layer_height, layer_depth, weight_index, 2);
	returnOnCudaError(err);

	err = vgg_racnn_block(dh, &buffs[0].mat, &buffs[1].mat, &buffs[2].mat, weights,
		&buffs[3].mat, &buffs[4].mat,
		layer_width, layer_height, layer_depth, weight_index, 3);
	returnOnCudaError(err);

	err = vgg_conv_block(dh, &buffs[0].mat, &buffs[1].mat, &buffs[2].mat, weights, layer_width,
		layer_height, layer_depth, weight_index, 3, false);
	returnOnCudaError(err);


	err = vgg_conv_block(dh, &buffs[1].mat, &buffs[0].mat, &buffs[2].mat, weights, layer_width,
		layer_height, layer_depth, weight_index, 3, true);

	CUMat *cuweight = weights + weight_index;
	err = device_maxpool2_bias(dh.cukern[cu_maxpool2_bias_relu].kernel, &buffs[0].mat,
		&buffs[1].mat, &cuweight[0].mat, layer_width, layer_height, layer_depth);
	returnOnCudaError(err);

	layer_width /= 2;
	layer_height /= 2;
	weight_index++;

	cuweight = weights + weight_index;
	err = device_dense_mul(dh.cukern[cu_add_bias_relu].kernel, &buffs[1].mat, &buffs[0].mat, &cuweight[0].mat,
		&cuweight[1].mat, layer_width * layer_height * layer_depth, cuweight[0].w, dh.cublas_handle);
	returnOnCudaError(err);
	layer_depth = cuweight[0].w;
	weight_index += 2;

	cuweight = weights + weight_index;
	err = device_dense_mul(dh.cukern[cu_add_bias_relu].kernel, &buffs[0].mat, &buffs[1].mat, &cuweight[0].mat,
		&cuweight[1].mat, layer_depth, cuweight[0].w, dh.cublas_handle);
	returnOnCudaError(err);
	layer_depth = cuweight[0].w;
	weight_index += 2;

	cuweight = weights + weight_index;
	err = device_dense_mul(dh.cukern[cu_add_bias].kernel, &buffs[1].mat, &buffs[0].mat, &cuweight[0].mat,
		&cuweight[1].mat, layer_depth, cuweight[0].w, dh.cublas_handle);
	returnOnCudaError(err);
	layer_depth = cuweight[0].w;

	err = cuMemcpyDtoH((void *)h_D, buffs[0].mat, cuweight[0].w * sizeof(float));
	softMax(h_D, cuweight[0].w);

	return err;

}

CUresult resnet_conv_block(DeviceHandle &dh, CUdeviceptr *shortcut, CUdeviceptr *output,
	CUdeviceptr *tmp1, CUdeviceptr *tmp2, CUMat *weights, int &layer_width,
	int &layer_height, int &layer_depth, bool speed_test=false) {

	int layer_depth_shortcut = layer_depth;
	CUresult err = device_conv2_1x1(dh.cukern[cu_bias_relu].kernel, shortcut, tmp1,
		&weights[0].mat, &weights[1].mat, dh.cublas_handle,
		layer_width, layer_height, layer_depth, weights[0].w);
	returnOnCudaError(err);

	device_conv2_3x3(dh.cukern[cu_im2col].kernel, dh.cukern[cu_bias_relu].kernel,
		tmp1, tmp2, output,
		&weights[2].mat, &weights[3].mat, dh.cublas_handle,
		layer_width, layer_height, layer_depth, weights[2].w);
	returnOnCudaError(err);

	if (speed_test) {
		int sync[1];
		err = cuMemcpyDtoH((void *)(sync), (*output), sizeof(int));
		returnOnCudaError(err);
	}

	err = device_conv2_1x1(dh.cukern[cu_bias_relu].kernel, tmp2, tmp1,
		&weights[4].mat, nullptr, dh.cublas_handle,
		layer_width, layer_height, layer_depth, weights[4].w);

	returnOnCudaError(err);

	err = device_conv2_1x1(dh.cukern[cu_bias_relu].kernel, shortcut, output,
		&weights[5].mat, nullptr, dh.cublas_handle,
		layer_width, layer_height, layer_depth_shortcut, weights[5].w);

	returnOnCudaError(err);

	err = device_sum_bias_relu(dh.cukern[cu_sum_bias_relu].kernel, tmp1, output,
		&weights[6].mat, layer_width, layer_height, layer_depth);

	return err;
}



CUresult resnet_identity_block(DeviceHandle &dh, CUdeviceptr *shortcut, CUdeviceptr *output,
	CUdeviceptr *tmp1, CUdeviceptr *tmp2, CUMat *weights, int &layer_width,
	int &layer_height, int &layer_depth, bool speed_test = false) {

	CUresult err = device_conv2_1x1(dh.cukern[cu_bias_relu].kernel, shortcut, tmp1,
		&weights[0].mat, &weights[1].mat, dh.cublas_handle,
		layer_width, layer_height, layer_depth, weights[0].w);
	returnOnCudaError(err);

	device_conv2_3x3(dh.cukern[cu_im2col].kernel, dh.cukern[cu_bias_relu].kernel,
		tmp1, tmp2, output,
		&weights[2].mat, &weights[3].mat, dh.cublas_handle,
		layer_width, layer_height, layer_depth, weights[2].w);
	returnOnCudaError(err);

	if (speed_test) {
		int sync[1];
		err = cuMemcpyDtoH((void *)(sync), (*output), sizeof(int));
		returnOnCudaError(err);
	}

	err = device_conv2_1x1(dh.cukern[cu_bias_relu].kernel, tmp2, tmp1,
		&weights[4].mat, nullptr, dh.cublas_handle,
		layer_width, layer_height, layer_depth, weights[4].w);

	returnOnCudaError(err);

	err = device_sum_bias_relu(dh.cukern[cu_sum_bias_relu].kernel, tmp1, shortcut,
		&weights[5].mat, layer_width, layer_height, layer_depth);

	return err;
}


CUresult resnet_macro_block(DeviceHandle &dh, CUdeviceptr *shortcut, CUdeviceptr *output,
	CUdeviceptr *tmp1, CUdeviceptr *tmp2, CUMat *weights, int &layer_width,
	int &layer_height, int &layer_depth, int block_num, bool speed_test) {



	CUresult err = resnet_conv_block(dh, shortcut, output,
		tmp1, tmp2, weights, layer_width,
		layer_height, layer_depth, speed_test);
	returnOnCudaError(err);

	weights += 7;

	CUdeviceptr *swap;
	CUdeviceptr *blk_in = tmp1;
	CUdeviceptr *blk_out = shortcut;

	for (int blk = 0; blk < block_num; ++blk) {
		err = resnet_identity_block(dh, blk_in, output,
			blk_out, tmp2, weights, layer_width,
			layer_height, layer_depth, speed_test);

		returnOnCudaError(err);

		weights += 6;

		swap = blk_in;
		blk_in = blk_out;
		blk_out = swap;

	}

	return err;
}





CUresult resnet_racnn_conv_block(DeviceHandle &dh, CUdeviceptr *shortcut, CUdeviceptr *output,
	CUdeviceptr *tmp1, CUdeviceptr *tmp2, CUdeviceptr *index, CUdeviceptr *page,
	CUMat *weights, int &layer_width,
	int &layer_height, int &layer_depth) {

	int layer_depth_shortcut = layer_depth;
	CUresult err = device_conv2_1x1(dh.cukern[cu_bias_relu].kernel, shortcut, tmp1,
		&weights[0].mat, &weights[1].mat, dh.cublas_handle,
		layer_width, layer_height, layer_depth, weights[0].w);
	returnOnCudaError(err);

	err = device_conv2_racnn(dh.cukern[cu_fill_index_ref].kernel, dh.cukern[cu_im2col8_mask].kernel,
		dh.cukern[col2im8_mask_bias_relu].kernel, tmp1, tmp2, output,
		&weights[2].mat, &weights[3].mat, &weights[4].mat, &weights[5].mat,
		index, page,
		weights[6].value, dh.cublas_handle,
		layer_width, layer_height, layer_depth,
		weights[2].w - 1);

	returnOnCudaError(err);

	err = device_conv2_1x1(dh.cukern[cu_bias_relu].kernel, output, tmp1,
		&weights[7].mat, nullptr, dh.cublas_handle,
		layer_width, layer_height, layer_depth, weights[7].w);

	returnOnCudaError(err);

	err = device_conv2_1x1(dh.cukern[cu_bias_relu].kernel, shortcut, output,
		&weights[8].mat, nullptr, dh.cublas_handle,
		layer_width, layer_height, layer_depth_shortcut, weights[8].w);

	returnOnCudaError(err);

	err = device_sum_bias_relu(dh.cukern[cu_sum_bias_relu].kernel, tmp1, output,
		&weights[9].mat, layer_width, layer_height, layer_depth);

	return err;
}



CUresult resnet_racnn_identity_block(DeviceHandle &dh, CUdeviceptr *shortcut, CUdeviceptr *output,
	CUdeviceptr *tmp1, CUdeviceptr *tmp2, CUdeviceptr *index, CUdeviceptr *page,
	CUMat *weights, int &layer_width, int &layer_height, int &layer_depth) {

	CUresult err = device_conv2_1x1(dh.cukern[cu_bias_relu].kernel, shortcut, tmp1,
		&weights[0].mat, &weights[1].mat, dh.cublas_handle,
		layer_width, layer_height, layer_depth, weights[0].w);
	returnOnCudaError(err);

	err = device_conv2_racnn(dh.cukern[cu_fill_index_ref].kernel, dh.cukern[cu_im2col8_mask].kernel,
		dh.cukern[col2im8_mask_bias_relu].kernel, tmp1, tmp2, output,
		&weights[2].mat, &weights[3].mat, &weights[4].mat, &weights[5].mat,
		index, page,
		weights[6].value, dh.cublas_handle,
		layer_width, layer_height, layer_depth,
		weights[2].w - 1);

	returnOnCudaError(err);


	err = device_conv2_1x1(dh.cukern[cu_bias_relu].kernel, output, tmp1,
		&weights[7].mat, nullptr, dh.cublas_handle,
		layer_width, layer_height, layer_depth, weights[7].w);

	returnOnCudaError(err);


	err = device_sum_bias_relu(dh.cukern[cu_sum_bias_relu].kernel, tmp1, shortcut,
		&weights[8].mat, layer_width, layer_height, layer_depth);

	return err;
}



CUresult resnet_racnn_macro_block(DeviceHandle &dh, CUdeviceptr *shortcut, CUdeviceptr *output,
	CUdeviceptr *tmp1, CUdeviceptr *tmp2, CUdeviceptr *index, CUdeviceptr *page, CUMat *weights, int &layer_width,
	int &layer_height, int &layer_depth, int block_num) {

	CUresult err = resnet_racnn_conv_block(dh, shortcut, output,
		tmp1, tmp2, index, page, weights, layer_width,
		layer_height, layer_depth);
	returnOnCudaError(err);


	weights += 10;

	CUdeviceptr *swap;
	CUdeviceptr *blk_in = tmp1;
	CUdeviceptr *blk_out = shortcut;

	for (int blk = 0; blk < block_num; ++blk) {
		err = resnet_racnn_identity_block(dh, blk_in, output,
			blk_out, tmp2, index, page, weights, layer_width,
			layer_height, layer_depth);

		returnOnCudaError(err);

		weights += 9;

		swap = blk_in;
		blk_in = blk_out;
		blk_out = swap;

	}

	return err;
}


CUresult resnet50_classifier(DeviceHandle &dh, CUMat *buffs, CUMat *weights,
	float *input_data, float *h_D, bool speed_test=false) {

	CUresult err;
	int layer_width = 256;
	int layer_height = 256;
	int layer_depth = 3;

	int mem_size_input = layer_width * layer_height * layer_depth * sizeof(float);
	err = cuMemcpyHtoD(buffs[1].mat, input_data, mem_size_input);
	returnOnCudaError(err);

	err = device_conv7x7_pool3(dh.cukern[cu_im2col7x7rgb].kernel,
		dh.cukern[cu_maxpool3_bias_relu].kernel, &buffs[1].mat,
		&buffs[3].mat, &buffs[0].mat, &weights[0].mat, &weights[1].mat,
		dh.cublas_handle, layer_width, layer_height, layer_depth, weights[0].w);
	returnOnCudaError(err);

	weights += 2;

	err = resnet_macro_block(dh, &buffs[3].mat, &buffs[0].mat,
		&buffs[1].mat, &buffs[2].mat, weights, layer_width,
		layer_height, layer_depth, 2, speed_test);

	weights += (7 + 6 * 2);

	returnOnCudaError(err);

	err = device_resample2(dh.cukern[cu_resample2].kernel, &buffs[1].mat, &buffs[3].mat,
		layer_width, layer_height, layer_depth);

	returnOnCudaError(err);

	err = resnet_macro_block(dh, &buffs[3].mat, &buffs[0].mat,
		&buffs[1].mat, &buffs[2].mat, weights, layer_width,
		layer_height, layer_depth, 3, speed_test);

	returnOnCudaError(err);

	weights += (7 + 6 * 3);

	err = device_resample2(dh.cukern[cu_resample2].kernel, &buffs[3].mat, &buffs[1].mat,
		layer_width, layer_height, layer_depth);

	returnOnCudaError(err);

	err = resnet_macro_block(dh, &buffs[1].mat, &buffs[0].mat,
		&buffs[3].mat, &buffs[2].mat, weights, layer_width,
		layer_height, layer_depth, 5, speed_test);

	returnOnCudaError(err);
	weights += (7 + 6 * 5);

	err = device_resample2(dh.cukern[cu_resample2].kernel, &buffs[1].mat, &buffs[3].mat,
		layer_width, layer_height, layer_depth);

	returnOnCudaError(err);

	err = resnet_macro_block(dh, &buffs[3].mat, &buffs[0].mat,
		&buffs[1].mat, &buffs[2].mat, weights, layer_width,
		layer_height, layer_depth, 2, speed_test);

	returnOnCudaError(err);
	weights += (7 + 6 * 2);

	err = device_avgpool2(dh.cukern[cu_avgpool2].kernel, &buffs[1].mat, &buffs[3].mat,
		layer_width, layer_height, layer_depth);


	err = device_dense_mul(dh.cukern[cu_add_bias].kernel, &buffs[3].mat, &buffs[1].mat,
		&weights[0].mat, &weights[1].mat, weights[0].h, weights[1].w, dh.cublas_handle);
	returnOnCudaError(err);

	err = cuMemcpyDtoH((void *)h_D, buffs[1].mat, weights[1].w * sizeof(float));
	softMax(h_D, weights[1].w);

	return err;
}



CUresult resnet_racnn_classifier(DeviceHandle &dh, CUMat *buffs, CUMat *weights,
	float *input_data, float *h_D) {

	CUresult err;
	int layer_width = 256;
	int layer_height = 256;
	int layer_depth = 3;

	int mem_size_input = layer_width * layer_height * layer_depth * sizeof(float);
	err = cuMemcpyHtoD(buffs[1].mat, input_data, mem_size_input);
	returnOnCudaError(err);

	err = device_conv7x7_pool3(dh.cukern[cu_im2col7x7rgb].kernel,
		dh.cukern[cu_maxpool3_bias_relu].kernel, &buffs[1].mat,
		&buffs[3].mat, &buffs[0].mat, &weights[0].mat, &weights[1].mat,
		dh.cublas_handle, layer_width, layer_height, layer_depth, weights[0].w);
	returnOnCudaError(err);

	weights += 2;

	err = resnet_racnn_macro_block(dh, &buffs[3].mat, &buffs[0].mat,
		&buffs[1].mat, &buffs[2].mat, &buffs[4].mat, &buffs[5].mat, weights, layer_width,
		layer_height, layer_depth, 2);

	weights += (10 + 9 * 2);

	returnOnCudaError(err);

	err = device_resample2(dh.cukern[cu_resample2].kernel, &buffs[1].mat, &buffs[3].mat,
		layer_width, layer_height, layer_depth);

	returnOnCudaError(err);

	err = resnet_racnn_macro_block(dh, &buffs[3].mat, &buffs[0].mat,
		&buffs[1].mat, &buffs[2].mat, &buffs[4].mat, &buffs[5].mat, weights, layer_width,
		layer_height, layer_depth, 3);

	returnOnCudaError(err);

	weights += (10 + 9 * 3);

	err = device_resample2(dh.cukern[cu_resample2].kernel, &buffs[3].mat, &buffs[1].mat,
		layer_width, layer_height, layer_depth);

	returnOnCudaError(err);

	err = resnet_racnn_macro_block(dh, &buffs[1].mat, &buffs[0].mat,
		&buffs[3].mat, &buffs[2].mat, &buffs[4].mat, &buffs[5].mat, weights, layer_width,
		layer_height, layer_depth, 5);

	returnOnCudaError(err);

	weights += (10 + 9 * 5);

	err = device_resample2(dh.cukern[cu_resample2].kernel, &buffs[1].mat, &buffs[3].mat,
		layer_width, layer_height, layer_depth);

	returnOnCudaError(err);

	err = resnet_racnn_macro_block(dh, &buffs[3].mat, &buffs[0].mat,
		&buffs[1].mat, &buffs[2].mat, &buffs[4].mat, &buffs[5].mat, weights, layer_width,
		layer_height, layer_depth, 2);

	returnOnCudaError(err);

	weights += (10 + 9 * 2);

	err = device_avgpool2(dh.cukern[cu_avgpool2].kernel, &buffs[1].mat, &buffs[3].mat,
		layer_width, layer_height, layer_depth);

	err = device_dense_mul(dh.cukern[cu_add_bias].kernel, &buffs[3].mat, &buffs[1].mat,
		&weights[0].mat, &weights[1].mat, weights[0].h, weights[1].w, dh.cublas_handle);
	returnOnCudaError(err);

	err = cuMemcpyDtoH((void *)h_D, buffs[1].mat, weights[1].w * sizeof(float));
	softMax(h_D, weights[1].w);

	return err;
}



char * curacnn::get_device_name() {
	return deviceName;
}

curacnn::curacnn()
{
	device_open = false;
	cuweight_num = 0;
	for (int k = 0; k < 6; ++k) {
		cumat[k].size = -1;
	}
	dh.cublas_handle = nullptr;
	dh.cuContext = nullptr;
	model_type = -1;
	speed_test = false;
}


curacnn::~curacnn()
{
	release();
}

bool curacnn::is_open()
{
	return device_open;
}

unsigned long long curacnn::mem_size() {

	return dh.totalGlobalMem;
}

int curacnn::get_weight_num() {

	return cuweight_num;
}

void curacnn::set_speed_test(bool state) {

	speed_test = state;
}


void curacnn::get_inout_size(int *inout_size) {

	if (!is_open()) {
		return;
	}
	inout_size[0] = 0;
	inout_size[1] = 0;
	inout_size[2] = 0;
	inout_size[3] = 0;

	if (model_type==0 || model_type == 2) {
		inout_size[0] = 224;
		inout_size[1] = 224;
		inout_size[2] = 3;
		inout_size[3] = 80;
	}else if (model_type == 1 || model_type == 3) {
		inout_size[0] = 256;
		inout_size[1] = 256;
		inout_size[2] = 3;
		inout_size[3] = 80;
	}
}

int curacnn::open()
{
	if (device_open) {
		printf("device is open\n");
		return 1;
	}
	CUresult error_id = initCUDA(dh, deviceName);
	returnOnCudaError(error_id);

	int res = cublasCreate(&dh.cublas_handle);
	returnOnCudaError((CUresult)res);

	device_open = true;
	cuweight_num = 0;
	layer_size_w = nullptr;
	layer_size_h = nullptr;
	cuweight = nullptr;
	model_type = -1;
	speed_test = false;
	return 0;
}

void curacnn::release()
{
	if (!is_open()) {
		return;
	}

	for (int k = 0; k < 6; ++k) {
		if (cumat[k].size > 0) {
			cuMemFree(cumat[k].mat);
		}
		cumat[k].size = -1;
	}
	if (cuweight != nullptr) {
		for (int k = 0; k < cuweight_num; ++k) {
			if (cuweight[k].size > 0) {
				cuMemFree(cuweight[k].mat);
			}
			cuweight[k].size = -1;
		}
		delete[] cuweight;
		cuweight = nullptr;
	}

	cuweight_num = 0;
	if (layer_size_w != nullptr) {
		layer_size_w = nullptr;
	}
	if (layer_size_h != nullptr) {
		layer_size_h = nullptr;
	}

	if (dh.cublas_handle != nullptr) {
		cublasDestroy(dh.cublas_handle);
	}
	if (dh.cuContext != nullptr) {
		cuCtxDestroy(dh.cuContext);
	}

	dh.cublas_handle = nullptr;
	dh.cuContext = nullptr;

	device_open = false;
	model_type = -1;

}

bool curacnn::load_weight(int layer_num, int rows, int cols, float *weight_data) {

	if (!is_open() || model_type < 0 || model_type>3) {
		return false;
	}
	if (layer_num >= cuweight_num || layer_num < 0) {
		return false;
	}
	if (rows != cuweight[layer_num].h || cols != cuweight[layer_num].w) {
		return false;
	}
	if (rows == 1 && cols == 1) {
		cuweight[layer_num].value = weight_data[0];
		return true;
	}
	int error_id = cuMemcpyHtoD(cuweight[layer_num].mat, weight_data,
		cuweight[layer_num].h*cuweight[layer_num].w * sizeof(float));
	if (error_id != CUDA_SUCCESS) return false;
	return true;
}

int curacnn::build(int model_type) {

	const int weight_size_vgg16_w[32] = {64,64,64,64,128,128,128,128,
	256,256,256,256,256,256,512,512,512,512,512,512,
	512,512,512,512,512,512,4096,4096,4096,4096,80,80 };

	const int weight_size_vgg16_h[32] = {9*3, 1, 9*64, 1,9 * 64,1,9 * 128,1,
	9 * 128,1,9 * 256,1,9 * 256,1,9 * 256,1,9 * 512,1,9 * 512,1,
	9 * 512,1,9 * 512,1,9 * 512,1,7 * 7 * 512,1,4096,1,4096,1 };

	const int weight_size_vgg16_racnn_h[53] = { 3, 24, 1, 1, 1, 64, 512, 1, 1,
		1, 64, 512, 1, 1, 1, 128, 1024, 1, 1, 1, 128, 1024, 1, 1, 1, 256, 2048,
		1, 1, 1, 256, 2048, 1, 1, 1, 2304, 1, 4608, 1, 4608, 1, 4608, 1, 4608, 1,
		4608, 1, 25088, 1, 4096, 1, 4096, 1 };

	const int weight_size_vgg16_racnn_w[53] = { 65, 64, 64, 64, 1, 65, 64, 64, 64, 1, 129,
		128, 128, 128, 1, 129, 128, 128, 128, 1, 257, 256,
		256, 256, 1, 257, 256, 256, 256, 1, 257, 256, 256,
		256, 1, 512, 512, 512, 512, 512, 512, 512, 512, 512,
		512, 512, 512, 4096, 4096, 4096, 4096, 80, 80 };

	const int weight_size_resnet50_h[104] = { 147, 1, 64, 1, 576, 1, 64, 64, 1, 256, 1,
		576, 1, 64, 1, 256, 1, 576, 1, 64, 1, 256, 1, 1152, 1, 128, 256, 1, 512, 1, 1152, 1, 128,
		1, 512, 1, 1152, 1, 128, 1, 512, 1, 1152, 1, 128, 1, 512, 1, 2304, 1, 256, 512, 1, 1024, 1,
		2304, 1, 256, 1, 1024, 1, 2304, 1, 256, 1, 1024, 1, 2304, 1, 256, 1, 1024, 1, 2304, 1, 256, 1,
		1024, 1, 2304, 1, 256, 1, 1024, 1, 4608, 1, 512, 1024, 1, 2048, 1, 4608, 1, 512, 1, 2048, 1, 4608,
		1, 512, 1, 32768, 1 };

	const int weight_size_resnet50_w[104] = { 64, 64, 64, 64, 64, 64, 256, 256, 256, 64, 64,
		64, 64, 256, 256, 64, 64, 64, 64, 256, 256, 128, 128, 128, 128, 512, 512, 512, 128, 128, 128, 128, 512,
		512, 128, 128, 128, 128, 512, 512, 128, 128, 128, 128, 512, 512, 256, 256, 256, 256, 1024, 1024, 1024, 256, 256,
		256, 256, 1024, 1024, 256, 256, 256, 256, 1024, 1024, 256, 256, 256, 256, 1024, 1024, 256, 256, 256, 256, 1024, 1024,
		256, 256, 256, 256, 1024, 1024, 512, 512, 512, 512, 2048, 2048, 2048, 512, 512, 512, 512, 2048, 2048, 512, 512, 512,
		512, 2048, 2048, 80, 80 };

	const int weight_size_resnet_racnn_h[152] = { 147, 1, 64, 1, 64, 512, 1, 1, 1,
		  64, 64, 1, 256, 1, 64, 512, 1, 1, 1, 64, 1, 256, 1, 64, 512, 1, 1,
		   1, 64, 1, 256, 1, 128, 1024, 1, 1, 1, 128, 256, 1, 512, 1, 128, 1024, 1,
		   1, 1, 128, 1, 512, 1, 128, 1024, 1, 1, 1, 128, 1, 512, 1, 128, 1024, 1,
		   1, 1, 128, 1, 512, 1, 256, 2048, 1, 1, 1, 256, 512, 1, 1024, 1, 256, 2048,
		   1, 1, 1, 256, 1, 1024, 1, 256, 2048, 1, 1, 1, 256, 1, 1024, 1, 256, 2048,
		   1, 1, 1, 256, 1, 1024, 1, 256, 2048, 1, 1, 1, 256, 1, 1024, 1, 256, 2048,
		   1, 1, 1, 256, 1, 1024, 1, 512, 4096, 1, 1, 1, 512, 1024, 1, 2048, 1, 512,
		4096, 1, 1, 1, 512, 1, 2048, 1, 512, 4096, 1, 1, 1, 512, 1, 32768, 1 };

	const int weight_size_resnet_racnn_w[152] = { 64, 64, 64, 64, 65, 64, 64, 64, 1, 256, 256,
		256, 64, 64, 65, 64, 64, 64, 1, 256, 256, 64, 64, 65, 64, 64, 64, 1, 256, 256, 128, 128, 129,
		128, 128, 128, 1, 512, 512, 512, 128, 128, 129, 128, 128, 128, 1, 512, 512, 128, 128, 129, 128, 128, 128,
		  1, 512, 512, 128, 128, 129, 128, 128, 128, 1, 512, 512, 256, 256, 257, 256, 256, 256, 1, 1024, 1024, 1024,
		256, 256, 257, 256, 256, 256, 1, 1024, 1024, 256, 256, 257, 256, 256, 256, 1, 1024, 1024, 256, 256, 257, 256,
		256, 256, 1, 1024, 1024, 256, 256, 257, 256, 256, 256, 1, 1024, 1024, 256, 256, 257, 256, 256, 256, 1, 1024,
	   1024, 512, 512, 513, 512, 512, 512, 1, 2048, 2048, 2048,	512, 512, 513, 512, 512, 512, 1, 2048, 2048, 512, 512,
		513, 512, 512, 512, 1, 2048, 2048, 80, 80 };

	const int buff_num_vgg16 = 3;
	const int buff_num_vgg_racnn = 5;
	const int buff_num_resnet50 = 4;
	const int buff_num_resnet_racnn = 6;

	const int buff_size_vgg16[3] = { 224 * 224 * 64,224 * 224 * 64, 224 * 224 * 64 * 9 };
	const int buff_size_vgg_racnn[5] = { 224 * 224 * 64 * 8, 224 * 224 * 64 * 4, 224 * 224 * 9 * 8,
									224 * 224, 513 };
	const int buff_size_resnet50[4] = { 128 * 128 * 3 * 49, 128 * 128 * 64, 64 * 64 * 64,
									128 * 128 * 64 };
	const int buff_size_resnet_racnn[6] = { 128 * 128 * 3 * 49, 128 * 128 * 64, 64 * 64 * 65,
									128 * 128 * 64, 256 * 256, 513 };

	int model_buff_num = 0;
	const int *model_buff_size = nullptr;
	if (model_type == 0) {
		model_buff_num = buff_num_vgg16;
		model_buff_size = buff_size_vgg16;
		cuweight_num = sizeof(weight_size_vgg16_h) / sizeof(int);
		layer_size_h = weight_size_vgg16_h;
		layer_size_w = weight_size_vgg16_w;
	}
	else if (model_type == 1) {
		model_buff_num = buff_num_resnet50;
		model_buff_size = buff_size_resnet50;
		cuweight_num = sizeof(weight_size_resnet50_h) / sizeof(int);
		layer_size_h = weight_size_resnet50_h;
		layer_size_w = weight_size_resnet50_w;
	}
	else if (model_type == 2) {
		model_buff_num = buff_num_vgg_racnn;
		model_buff_size = buff_size_vgg_racnn;
		cuweight_num = sizeof(weight_size_vgg16_racnn_h) / sizeof(int);
		layer_size_h = weight_size_vgg16_racnn_h;
		layer_size_w = weight_size_vgg16_racnn_w;
	}	
	else if (model_type == 3) {
		model_buff_num = buff_num_resnet_racnn;
		model_buff_size = buff_size_resnet_racnn;
		cuweight_num = sizeof(weight_size_resnet_racnn_h) / sizeof(int);
		layer_size_h = weight_size_resnet_racnn_h;
		layer_size_w = weight_size_resnet_racnn_w;
	}

	for (int k = 0; k < model_buff_num; ++k) {
		int error_id = cuMemAlloc(&cumat[k].mat, model_buff_size[k] * sizeof(float));
		if (error_id != CUDA_SUCCESS) return 0;
		cumat[k].size = model_buff_size[k] * sizeof(float);
	}

	cuweight = new CUMat[cuweight_num];

	for (int k = 0; k < cuweight_num; ++k) {
		int lsize = layer_size_h[k] * layer_size_w[k] * sizeof(float);
		int error_id = cuMemAlloc(&cuweight[k].mat, lsize);
		if (error_id != CUDA_SUCCESS) return 0;
		cuweight[k].size = lsize;
		cuweight[k].h = layer_size_h[k];
		cuweight[k].w = layer_size_w[k];
	}

	this->model_type = model_type;
	return cuweight_num;
}


bool curacnn::predict(float *input_data, float *h_D)
{
	int res = 1;
	if (model_type == 0) {
		res = vgg16_classifier(dh, cumat[0].mat, cumat[1].mat, cumat[2].mat,
			cuweight, input_data, h_D, speed_test);
	}
	else if (model_type == 1) {
		res = resnet50_classifier(dh, cumat,
			cuweight, input_data, h_D, speed_test);
	}
	else if (model_type == 2) {
		res = vgg_racnn_classifier(dh, cumat,
			cuweight, input_data, h_D);
	}	
	else if (model_type == 3) {
		res = resnet_racnn_classifier(dh, cumat,
			cuweight, input_data, h_D);
	}

	if (res != CUDA_SUCCESS) {
		return false;
	}
	return true;
}


libcuracnn::libcuracnn()
{
	cucnn = nullptr;
}

bool libcuracnn::open(int model_type)
{
	std::ifstream f(PTX_FILE);
	if (!f.good()) {
		printf("cannot find ptx file\n");
		return false;
	}

	curacnn *d_cucnn = new curacnn();
	cucnn = d_cucnn;	
	int res = d_cucnn->open();
	if (res!=0) {
		return false;
	}
	int lnum = d_cucnn->build(model_type);
	if (lnum == 0) {
		return false;
	}
	return true;
}

bool libcuracnn::predict(float *input_data, float *ouput_data) {
	if (cucnn == nullptr)
		return false;
	return ((curacnn *)cucnn)->predict(input_data, ouput_data);
}

bool libcuracnn::load_weight(int layer_num, int rows, int cols, float *weight_data) {
	if (cucnn == nullptr)
		return false;
	return ((curacnn *)cucnn)->load_weight(layer_num, rows, cols, weight_data);
}

void libcuracnn::get_device_name(char *name, int size) {
	if (cucnn == nullptr)
		return;
	char *ss = ((curacnn *)cucnn)->get_device_name();
	strncpy(name, ss, size);
}

bool libcuracnn::is_open() {
	if (cucnn == nullptr)
		return false;
	return ((curacnn *)cucnn)->is_open();
}

int libcuracnn::get_weight_num() {
	if (cucnn == nullptr)
		return 0;
	return ((curacnn *)cucnn)->get_weight_num();
}

void libcuracnn::set_speed_test(bool state) {
	if (cucnn == nullptr)
		return;
	((curacnn *)cucnn)->set_speed_test(state);
}

void libcuracnn::get_inout_size(int *inout_size) {
	if (cucnn == nullptr)
		return;
	return ((curacnn *)cucnn)->get_inout_size(inout_size);
}

unsigned long long  libcuracnn::get_mem_size() {
	if (cucnn == nullptr)
		return 0;
	return ((curacnn *)cucnn)->mem_size();
}

void libcuracnn::release() {
	if (cucnn == nullptr)
		return;
	((curacnn *)cucnn)->release();
}


libcuracnn::~libcuracnn()
{
	if (cucnn == nullptr)
		return;
	((curacnn *)cucnn)->release();
	cucnn = nullptr;
}
