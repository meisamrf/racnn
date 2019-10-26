#pragma once
class libcuracnn
{
public:
	libcuracnn();
	~libcuracnn();
	// model_type 
	// 0 VGG16
	// 1 VGG16+racnn
	// 2 ResNet50
	// 3 ResNet50+racnn
	bool open(int model_type);
	void release();
	bool predict(float *input_data, float *ouput_data);	
	bool load_weight(int layer_num, int rows, int cols, float *weight_data);
	void get_device_name(char *name, int size);
	bool is_open();
	int get_weight_num();
	void set_speed_test(bool state);
	void get_inout_size(int *inout_size);
	unsigned long long get_mem_size();
private:
	void *cucnn;
};

