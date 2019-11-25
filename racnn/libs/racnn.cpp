#include "racnn.h"

#ifdef AVX_AVX2
#include <immintrin.h>
#elif defined(ARM_NEON)
#include <arm_neon.h>
#endif


void copy3x3(const float *in, float *out, int x, int y, int width, int dim) {

	const float *pin;

	pin = &in[(x - 1 + (y - 1)*width)*dim];
	for (int d = dim * 3; d > 0; d--) {
		*out++ = *pin++;
	}

	pin = &in[(x - 1 + (y)*width)*dim];
	for (int d = dim * 3; d > 0; d--) {
		*out++ = *pin++;
	}

	pin = &in[(x - 1 + (y + 1)*width)*dim];
	for (int d = dim * 3; d > 0; d--) {
		*out++ = *pin++;
	}

}

#ifdef AVX_AVX2
void copy3x3_opt(const float *in, float *out, int x, int y, int width, int dim) {

	
	const int VEC = 8;
	const float *pin;

	pin = &in[(x - 1 + (y - 1)*width)*dim];
	int dim_n_3 = dim * 3 / VEC;
	for (int d = dim_n_3; d > 0; d--) {
		__m256 a = _mm256_load_ps(pin);
		_mm256_store_ps(out, a);
		out += VEC;
		pin += VEC;
	}

	pin = &in[(x - 1 + (y)*width)*dim];
	for (int d = dim_n_3; d > 0; d--) {
		__m256 a = _mm256_load_ps(pin);
		_mm256_store_ps(out, a);
		out += VEC;
		pin += VEC;
	}

	pin = &in[(x - 1 + (y + 1)*width)*dim];
	for (int d = dim_n_3; d > 0; d--) {
		__m256 a = _mm256_load_ps(pin);
		_mm256_store_ps(out, a);
		out += VEC;
		pin += VEC;
	}

}

#elif defined(ARM_NEON)
void copy3x3_opt(const float *in, float *out, int x, int y, int width, int dim) {

	const int VEC = 4;
	const float *pin;
	pin = &in[(x - 1 + (y - 1)*width)*dim];
	int dim_n_3 = dim * 3 / VEC;
	for (int d = dim_n_3; d > 0; d--) {
		float32x4_t a = vld1q_f32(pin);
		vst1q_f32(out, a);
		out += VEC;
		pin += VEC;
	}

	pin = &in[(x - 1 + (y)*width)*dim];
	for (int d = dim_n_3; d > 0; d--) {
		float32x4_t a = vld1q_f32(pin);
		vst1q_f32(out, a);
		out += VEC;
		pin += VEC;
	}

	pin = &in[(x - 1 + (y + 1)*width)*dim];
	for (int d = dim_n_3; d > 0; d--) {
		float32x4_t a = vld1q_f32(pin);
		vst1q_f32(out, a);
		out += VEC;
		pin += VEC;
	}

}
#endif


#ifdef AVX_AVX2
void max3x3_opt(const float *in, float *out, const float *bias, int xo, int yo, int width, int dim) {

	const int VEC = 8;

	const float *pin;
	__m256 zero = _mm256_set1_ps(0);
	
	pin = &in[(xo + yo * width)*dim];
	float *outp = out;
	int dim_n = dim / VEC;
	for (int d = dim_n; d > 0; d--) {
		__m256 a = _mm256_load_ps(pin);
		_mm256_store_ps(outp, a);
		outp += VEC;
		pin += VEC;
	}

	for (int i = 0; i < 9; ++i) {
		if (i == 4) {
			continue;
		}
		int x = xo - 1 + (i % 3);
		int y = yo - 1 + (i / 3);
		pin = &in[(x + y * width)*dim];
		outp = out;
		for (int d = dim_n; d > 0; d--) {
			__m256 z = _mm256_load_ps(pin);
			__m256 a = _mm256_load_ps(outp);
			z = _mm256_max_ps(z, a);
			_mm256_store_ps(outp, z);
			outp += VEC;
			pin += VEC;
		}
	}
	outp = out;
	for (int d = dim_n; d > 0; d--) {
		__m256 a = _mm256_load_ps(outp);
		__m256 b = _mm256_load_ps(bias);
		a = _mm256_add_ps(a, b);
		a = _mm256_max_ps(a, zero);
		_mm256_store_ps(outp, a);
		outp += VEC;
		bias += VEC;
	}
}

void max3x3cond_opt(const float *in, float *out, const float *bias, int xo,
	int yo, int height, int width, int dim) {

	const int VEC = 8;

	const float *pin;
	__m256 zero = _mm256_set1_ps(0);

	pin = &in[(xo + yo * width)*dim];
	float *outp = out;
	int dim_n = dim / VEC;
	for (int d = dim_n; d > 0; d--) {
		__m256 a = _mm256_load_ps(pin);
		_mm256_store_ps(outp, a);
		outp += VEC;
		pin += VEC;
	}

	for (int i = 0; i < 9; ++i) {
		if (i == 4) {
			continue;
		}
		int x = xo - 1 + (i % 3);
		int y = yo - 1 + (i / 3);
		if (y < 0 || x < 0 || x >= width || y >= height) {
			continue;
		}
		else {
			pin = &in[(x + y * width)*dim];
			outp = out;
			for (int d = dim_n; d > 0; d--) {
				__m256 z = _mm256_load_ps(pin);
				__m256 a = _mm256_load_ps(outp);
				z = _mm256_max_ps(z, a);
				_mm256_store_ps(outp, z);
				outp += VEC;
				pin += VEC;
			}
		}
	}
	outp = out;
	for (int d = dim_n; d > 0; d--) {
		__m256 a = _mm256_load_ps(outp);
		__m256 b = _mm256_load_ps(bias);
		a = _mm256_add_ps(a, b);
		a = _mm256_max_ps(a, zero);
		_mm256_store_ps(outp, a);
		outp += VEC;
		bias += VEC;
	}
}

#elif defined(ARM_NEON)
void max3x3_opt(const float *in, float *out, const float *bias, int xo, int yo, int width, int dim) {

	const int VEC = 4;
	
	const float *pin;
	float32x4_t zero = vdupq_n_f32(0);

	pin = &in[(xo + yo * width)*dim];
	float *outp = out;
	

	int dim_n = dim / VEC;
	for (int d = dim_n; d > 0; d--) {
		float32x4_t a = vld1q_f32(pin);
		vst1q_f32(outp, a);
		outp += VEC;
		pin += VEC;
	}

	for (int i = 0; i < 9; ++i) {
		if (i == 4) {
			continue;
		}
		int x = xo - 1 + (i % 3);
		int y = yo - 1 + (i / 3);
		pin = &in[(x + y * width)*dim];
		outp = out;
		for (int d = dim_n; d > 0; d--) {
			float32x4_t z = vld1q_f32(pin);
			float32x4_t a = vld1q_f32(outp);
			z = vmaxq_f32(z, a);
			vst1q_f32(outp, z);
			outp += VEC;
			pin += VEC;
		}
	}
	outp = out;
	for (int d = dim_n; d > 0; d--) {
		float32x4_t a = vld1q_f32(outp);
		float32x4_t b = vld1q_f32(bias);
		a = vaddq_f32(a, b);
		a = vmaxq_f32(a, zero);
		vst1q_f32(outp, a);
		outp += VEC;
		bias += VEC;
	}
}


void max3x3cond_opt(const float *in, float *out, const float *bias, int xo,
	int yo, int height, int width, int dim) {

	const int VEC = 4;

	const float *pin;
	float32x4_t zero = vdupq_n_f32(0);

	pin = &in[(xo + yo * width)*dim];
	float *outp = out;
	int dim_n = dim / VEC;

	for (int d = dim_n; d > 0; d--) {
		float32x4_t a = vld1q_f32(pin);
		vst1q_f32(outp, a);
		outp += VEC;
		pin += VEC;
	}

	for (int i = 0; i < 9; ++i) {
		if (i == 4) {
			continue;
		}
		int x = xo - 1 + (i % 3);
		int y = yo - 1 + (i / 3);
		if (y < 0 || x < 0 || x >= width || y >= height) {
			continue;
		}
		else {
			pin = &in[(x + y * width)*dim];
			outp = out;
			for (int d = dim_n; d > 0; d--) {
				float32x4_t z = vld1q_f32(pin);
				float32x4_t a = vld1q_f32(outp);
				z = vmaxq_f32(z, a);
				vst1q_f32(outp, z);
				outp += VEC;
				pin += VEC;
			}
		}
	}
	outp = out;
	for (int d = dim_n; d > 0; d--) {
		float32x4_t a = vld1q_f32(outp);
		float32x4_t b = vld1q_f32(bias);
		a = vaddq_f32(a, b);
		a = vmaxq_f32(a, zero);
		vst1q_f32(outp, a);
		outp += VEC;
		bias += VEC;
	}
}


#endif //  NEON

void max3x3(const float *in, float *out, const float *bias, int xo, int yo, int width, int dim) {

	const float *pin;
	pin = &in[(xo + yo * width)*dim];
	float *outp = out;
	for (int d = dim; d > 0; d--) {
		*outp++ = *pin++;
	}

	for (int i = 0; i < 9; ++i) {
		if (i == 4) {
			continue;
		}
		int x = xo - 1 + (i % 3);
		int y = yo - 1 + (i / 3);
		pin = &in[(x + y * width)*dim];
		outp = out;
		for (int d = dim; d > 0; d--) {
			float z = *pin++;
			if (*outp < z)
				*outp = z;
			outp++;
		}
	}
	outp = out;
	for (int d = dim; d > 0; d--) {
		float m = *outp;
		m += (*bias++);
		if (m < 0) m = 0;
		*outp++ = m;
	}
}

void max3x3cond(const float *in, float *out, const float *bias, int xo,
	int yo, int height, int width, int dim) {

	const float *pin;
	pin = &in[(xo + yo * width)*dim];
	float *outp = out;
	for (int d = dim; d > 0; d--) {
		*outp++ = *pin++;
	}

	for (int i = 0; i < 9; ++i) {
		if (i == 4) {
			continue;
		}
		int x = xo - 1 + (i % 3);
		int y = yo - 1 + (i / 3);
		if (y < 0 || x < 0 || x >= width || y >= height) {
			continue;
		}
		else {
			pin = &in[(x + y * width)*dim];
			outp = out;
			for (int d = dim; d > 0; d--) {
				float z = *pin++;
				if (*outp < z)
					*outp = z;
				outp++;
			}
		}
	}
	outp = out;
	for (int d = dim; d > 0; d--) {
		float m = *outp;
		m += (*bias++);
		if (m < 0) m = 0;
		*outp++ = m;
	}
}

inline void copy7x7rgb(const float *in, float *out, int x, int y, int width) {

	const float *pin;
	const int dim = 3;
	const int vec_size = dim * 7;

	for (int m = -3; m <= 3; m++) {
		pin = &in[(x - 3 + (y + m)*width)*dim];
		for (int d = vec_size; d > 0; d--) {
			*out++ = *pin++;
		}
	}
}


void copy3x3_nc(const float *in, float *out, int x, int y, int width, int dim) {

	const float *pin;

	pin = &in[(x - 1 + (y - 1)*width)*dim];
	for (int d = dim * 3; d > 0; d--) {
		*out++ = *pin++;
	}

	pin = &in[(x - 1 + (y)*width)*dim];
	for (int d = dim; d > 0; d--) {
		*out++ = *pin++;
	}

	pin = &in[(x + 1 + (y)*width)*dim];
	for (int d = dim; d > 0; d--) {
		*out++ = *pin++;
	}

	pin = &in[(x - 1 + (y + 1)*width)*dim];
	for (int d = dim * 3; d > 0; d--) {
		*out++ = *pin++;
	}

}

#ifdef AVX_AVX2
inline void copy3x3_nc_opt(const float *in, float *out, int x, int y, int width, int dim) {

	const int VEC = 8;

	const float *pin;
	int dim_n_3 = (dim / VEC) * 3;
	int d_VEC = dim / VEC;

	pin = &in[(x - 1 + (y - 1)*width)*dim];
	for (int d = dim_n_3; d > 0; d--) {
		__m256 a = _mm256_load_ps(pin);
		_mm256_store_ps(out, a);
		out += VEC;
		pin += VEC;
	}

	pin = &in[(x - 1 + (y)*width)*dim];
	for (int d = d_VEC; d > 0; d--) {
		__m256 a = _mm256_load_ps(pin);
		_mm256_store_ps(out, a);
		out += VEC;
		pin += VEC;
	}

	pin = &in[(x + 1 + (y)*width)*dim];
	for (int d = d_VEC; d > 0; d--) {
		__m256 a = _mm256_load_ps(pin);
		_mm256_store_ps(out, a);
		out += VEC;
		pin += VEC;
	}

	pin = &in[(x - 1 + (y + 1)*width)*dim];
	for (int d = dim_n_3; d > 0; d--) {
		__m256 a = _mm256_load_ps(pin);
		_mm256_store_ps(out, a);
		out += VEC;
		pin += VEC;
	}

}
#elif defined(ARM_NEON)

inline void copy3x3_nc_opt(const float *in, float *out, int x, int y, int width, int dim) {

	const int VEC = 4;
	const float *pin;

	int dim_n_3 = (dim / VEC) * 3;
	int d_n = dim / VEC;

	pin = &in[(x - 1 + (y - 1)*width)*dim];
	for (int d = dim_n_3; d > 0; d--) {
		float32x4_t a = vld1q_f32(pin);
		vst1q_f32(out, a);
		out += VEC;
		pin += VEC;
	}

	pin = &in[(x - 1 + (y)*width)*dim];
	for (int d = d_n; d > 0; d--) {
		float32x4_t a = vld1q_f32(pin);
		vst1q_f32(out, a);
		out += VEC;
		pin += VEC;
	}

	pin = &in[(x + 1 + (y)*width)*dim];
	for (int d = d_n; d > 0; d--) {
		float32x4_t a = vld1q_f32(pin);
		vst1q_f32(out, a);
		out += VEC;
		pin += VEC;
	}

	pin = &in[(x - 1 + (y + 1)*width)*dim];
	for (int d = dim_n_3; d > 0; d--) {
		float32x4_t a = vld1q_f32(pin);
		vst1q_f32(out, a);
		out += VEC;
		pin += VEC;
	}

}
#endif

void copy7x7cond_rgb(const float *in, float *out, int xc, int yc, int height, int width) {

	// RGB
	const int dim = 3;
	const float *pin;

	for (int yo = -3; yo <= 3; yo++) {
		int y = yc + yo;
		if (y < 0 || y >= height) {
			for (int xo = -3; xo <= 3; xo++) {
				*out++ = 0; *out++ = 0; *out++ = 0;
			}
		}
		else {
			for (int xo = -3; xo <= 3; xo++) {
				int x = xc + xo;
				if (x < 0 || x >= width) {
					*out++ = 0; *out++ = 0; *out++ = 0;
				}
				else {
					pin = &in[(x + y * width)*dim];
					*out++ = *pin++;
					*out++ = *pin++;
					*out++ = *pin++;
				}
			}
		}
	}
}

void copy3x3cond(const float *in, float *out, int x, int y, int height, int width, int dim) {

	const float *pin;
	if (y == 0) {
		for (int d = dim * 3; d > 0; d--) {
			*out++ = 0;
		}
	}
	else {

		if (x == 0) {
			for (int d = dim; d > 0; d--) {
				*out++ = 0;
			}
		}
		else {
			pin = &in[(x - 1 + (y - 1)*width)*dim];
			for (int d = dim; d > 0; d--) {
				*out++ = *pin++;
			}
		}

		pin = &in[(x + (y - 1)*width)*dim];
		for (int d = dim; d > 0; d--) {
			*out++ = *pin++;
		}


		if (x == width - 1) {
			for (int d = dim; d > 0; d--) {
				*out++ = 0;
			}
		}
		else {
			pin = &in[(x + 1 + (y - 1)*width)*dim];
			for (int d = dim; d > 0; d--) {
				*out++ = *pin++;
			}
		}
	}

	if (x == 0) {
		for (int d = dim; d > 0; d--) {
			*out++ = 0;
		}
	}
	else {
		pin = &in[(x - 1 + (y)*width)*dim];
		for (int d = dim; d > 0; d--) {
			*out++ = *pin++;
		}
	}

	pin = &in[(x + (y)*width)*dim];
	for (int d = dim; d > 0; d--) {
		*out++ = *pin++;
	}
	if (x == width - 1) {
		for (int d = dim; d > 0; d--) {
			*out++ = 0;
		}
	}
	else {
		pin = &in[(x + 1 + (y)*width)*dim];
		for (int d = dim; d > 0; d--) {
			*out++ = *pin++;
		}
	}


	if (y == height - 1) {
		for (int d = dim * 3; d > 0; d--) {
			*out++ = 0;
		}
	}
	else {

		if (x == 0) {
			for (int d = dim; d > 0; d--) {
				*out++ = 0;
			}
		}
		else {
			pin = &in[(x - 1 + (y + 1)*width)*dim];
			for (int d = dim; d > 0; d--) {
				*out++ = *pin++;
			}
		}

		pin = &in[(x + (y + 1)*width)*dim];
		for (int d = dim; d > 0; d--) {
			*out++ = *pin++;
		}

		if (x == width - 1) {
			for (int d = dim; d > 0; d--) {
				*out++ = 0;
			}
		}
		else {
			pin = &in[(x + 1 + (y + 1)*width)*dim];
			for (int d = dim; d > 0; d--) {
				*out++ = *pin++;
			}
		}
	}

}

// no center
void copy3x3_nc_cond(const float *in, float *out, int x, int y, int height, int width, int dim) {

	const float *pin;
	if (y == 0) {
		for (int d = dim * 3; d > 0; d--) {
			*out++ = 0;
		}
	}
	else {

		if (x == 0) {
			for (int d = dim; d > 0; d--) {
				*out++ = 0;
			}
		}
		else {
			pin = &in[(x - 1 + (y - 1)*width)*dim];
			for (int d = dim; d > 0; d--) {
				*out++ = *pin++;
			}
		}

		pin = &in[(x + (y - 1)*width)*dim];
		for (int d = dim; d > 0; d--) {
			*out++ = *pin++;
		}
		if (x == width - 1) {
			for (int d = dim; d > 0; d--) {
				*out++ = 0;
			}
		}
		else {
			pin = &in[(x + 1 + (y - 1)*width)*dim];
			for (int d = dim; d > 0; d--) {
				*out++ = *pin++;
			}
		}
	}

	if (x == 0) {
		for (int d = dim; d > 0; d--) {
			*out++ = 0;
		}
	}
	else {
		pin = &in[(x - 1 + (y)*width)*dim];
		for (int d = dim; d > 0; d--) {
			*out++ = *pin++;
		}
	}

	if (x == width - 1) {
		for (int d = dim; d > 0; d--) {
			*out++ = 0;
		}
	}
	else {
		pin = &in[(x + 1 + (y)*width)*dim];
		for (int d = dim; d > 0; d--) {
			*out++ = *pin++;
		}
	}
	if (y == height - 1) {
		for (int d = dim * 3; d > 0; d--) {
			*out++ = 0;
		}
	}
	else {

		if (x == 0) {
			for (int d = dim; d > 0; d--) {
				*out++ = 0;
			}
		}
		else {
			pin = &in[(x - 1 + (y + 1)*width)*dim];
			for (int d = dim; d > 0; d--) {
				*out++ = *pin++;
			}
		}

		pin = &in[(x + (y + 1)*width)*dim];
		for (int d = dim; d > 0; d--) {
			*out++ = *pin++;
		}

		if (x == width - 1) {
			for (int d = dim; d > 0; d--) {
				*out++ = 0;
			}
		}
		else {
			pin = &in[(x + 1 + (y + 1)*width)*dim];
			for (int d = dim; d > 0; d--) {
				*out++ = *pin++;
			}
		}
	}
}


#ifdef AVX_AVX2
inline void copy3x3_nc_cond_opt(const float *in, float *out, int x, int y, int height, int width, int dim) {

	const int VEC = 8;

	__m256 zero = _mm256_set1_ps(0);
	const float *pin;
	int dim_n_3 = (dim / VEC) * 3;
	int d_VEC = dim / VEC;

	if (y == 0) {
		for (int d = dim_n_3; d > 0; d--) {
			_mm256_store_ps(out, zero);
			out += VEC;
		}
	}
	else {

		if (x == 0) {
			for (int d = d_VEC; d > 0; d--) {
				_mm256_store_ps(out, zero);
				out += VEC;
			}
		}
		else {
			pin = &in[(x - 1 + (y - 1)*width)*dim];
			for (int d = d_VEC; d > 0; d--) {
				__m256 a = _mm256_load_ps(pin);
				_mm256_store_ps(out, a);
				out += VEC;
				pin += VEC;
			}
		}

		pin = &in[(x + (y - 1)*width)*dim];
		for (int d = d_VEC; d > 0; d--) {
			__m256 a = _mm256_load_ps(pin);
			_mm256_store_ps(out, a);
			out += VEC;
			pin += VEC;
		}
		if (x == width - 1) {
			for (int d = d_VEC; d > 0; d--) {
				_mm256_store_ps(out, zero);
				out += VEC;
			}
		}
		else {
			pin = &in[(x + 1 + (y - 1)*width)*dim];
			for (int d = d_VEC; d > 0; d--) {
				__m256 a = _mm256_load_ps(pin);
				_mm256_store_ps(out, a);
				out += VEC;
				pin += VEC;
			}
		}
	}

	if (x == 0) {
		for (int d = d_VEC; d > 0; d--) {
			_mm256_store_ps(out, zero);
			out += VEC;
		}
	}
	else {
		pin = &in[(x - 1 + (y)*width)*dim];
		for (int d = d_VEC; d > 0; d--) {
			__m256 a = _mm256_load_ps(pin);
			_mm256_store_ps(out, a);
			out += VEC;
			pin += VEC;
		}
	}

	if (x == width - 1) {
		for (int d = d_VEC; d > 0; d--) {
			_mm256_store_ps(out, zero);
			out += VEC;
		}
	}
	else {
		pin = &in[(x + 1 + (y)*width)*dim];
		for (int d = d_VEC; d > 0; d--) {
			__m256 a = _mm256_load_ps(pin);
			_mm256_store_ps(out, a);
			out += VEC;
			pin += VEC;
		}
	}
	if (y == height - 1) {
		for (int d = dim_n_3; d > 0; d--) {
			_mm256_store_ps(out, zero);
			out += VEC;
		}
	}
	else {

		if (x == 0) {
			for (int d = d_VEC; d > 0; d--) {
				_mm256_store_ps(out, zero);
				out += VEC;
			}
		}
		else {
			pin = &in[(x - 1 + (y + 1)*width)*dim];
			for (int d = d_VEC; d > 0; d--) {
				__m256 a = _mm256_load_ps(pin);
				_mm256_store_ps(out, a);
				out += VEC;
				pin += VEC;
			}
		}

		pin = &in[(x + (y + 1)*width)*dim];
		for (int d = d_VEC; d > 0; d--) {
			__m256 a = _mm256_load_ps(pin);
			_mm256_store_ps(out, a);
			out += VEC;
			pin += VEC;
		}

		if (x == width - 1) {
			for (int d = d_VEC; d > 0; d--) {
				_mm256_store_ps(out, zero);
				out += VEC;
			}
		}
		else {
			pin = &in[(x + 1 + (y + 1)*width)*dim];
			for (int d = d_VEC; d > 0; d--) {
				__m256 a = _mm256_load_ps(pin);
				_mm256_store_ps(out, a);
				out += VEC;
				pin += VEC;
			}
		}
	}
}
#elif defined(ARM_NEON)

inline void copy3x3_nc_cond_opt(const float *in, float *out, int x, int y, int height, int width, int dim) {

	const int VEC = 4;

	float32x4_t zero = vdupq_n_f32(0);
	const float *pin;
	int dim_n_3 = (dim / VEC) * 3;
	int d_n = dim / VEC;

	if (y == 0) {
		for (int d = dim_n_3; d > 0; d--) {
			vst1q_f32(out, zero);
			out += VEC;
		}
	}
	else {

		if (x == 0) {
			for (int d = d_n; d > 0; d--) {
				vst1q_f32(out, zero);
				out += VEC;
			}
		}
		else {
			pin = &in[(x - 1 + (y - 1)*width)*dim];
			for (int d = d_n; d > 0; d--) {
				float32x4_t a = vld1q_f32(pin);
				vst1q_f32(out, a);
				out += VEC;
				pin += VEC;
			}
		}

		pin = &in[(x + (y - 1)*width)*dim];
		for (int d = d_n; d > 0; d--) {
			float32x4_t a = vld1q_f32(pin);
			vst1q_f32(out, a);
			out += VEC;
			pin += VEC;
		}
		if (x == width - 1) {
			for (int d = d_n; d > 0; d--) {
				vst1q_f32(out, zero);
				out += VEC;
			}
		}
		else {
			pin = &in[(x + 1 + (y - 1)*width)*dim];
			for (int d = d_n; d > 0; d--) {
				float32x4_t a = vld1q_f32(pin);
				vst1q_f32(out, a);
				out += VEC;
				pin += VEC;
			}
		}
	}

	if (x == 0) {
		for (int d = d_n; d > 0; d--) {
			vst1q_f32(out, zero);
			out += VEC;
		}
	}
	else {
		pin = &in[(x - 1 + (y)*width)*dim];
		for (int d = d_n; d > 0; d--) {
			float32x4_t a = vld1q_f32(pin);
			vst1q_f32(out, a);
			out += VEC;
			pin += VEC;
		}
	}

	if (x == width - 1) {
		for (int d = d_n; d > 0; d--) {
			vst1q_f32(out, zero);
			out += VEC;
		}
	}
	else {
		pin = &in[(x + 1 + (y)*width)*dim];
		for (int d = d_n; d > 0; d--) {
			float32x4_t a = vld1q_f32(pin);
			vst1q_f32(out, a);
			out += VEC;
			pin += VEC;
		}
	}
	if (y == height - 1) {
		for (int d = dim_n_3; d > 0; d--) {
			vst1q_f32(out, zero);
			out += VEC;
		}
	}
	else {

		if (x == 0) {
			for (int d = d_n; d > 0; d--) {
				vst1q_f32(out, zero);
				out += VEC;
			}
		}
		else {
			pin = &in[(x - 1 + (y + 1)*width)*dim];
			for (int d = d_n; d > 0; d--) {
				float32x4_t a = vld1q_f32(pin);
				vst1q_f32(out, a);
				out += VEC;
				pin += VEC;
			}
		}

		pin = &in[(x + (y + 1)*width)*dim];
		for (int d = d_n; d > 0; d--) {
			float32x4_t a = vld1q_f32(pin);
			vst1q_f32(out, a);
			out += VEC;
			pin += VEC;
		}

		if (x == width - 1) {
			for (int d = d_n; d > 0; d--) {
				vst1q_f32(out, zero);
				out += VEC;
			}
		}
		else {
			pin = &in[(x + 1 + (y + 1)*width)*dim];
			for (int d = d_n; d > 0; d--) {
				float32x4_t a = vld1q_f32(pin);
				vst1q_f32(out, a);
				out += VEC;
				pin += VEC;
			}
		}
	}
}
#endif

#if defined(AVX_AVX2) || defined(ARM_NEON)

void im2col_opt(const float *in, float *out_mat, int height, int width, int dim) {

	int height_lim = height - 1;
	int width_lim = width - 1;

	int y = 0;
	for (int x = 0; x < width; x++) {
		copy3x3cond(in, out_mat, x, y, height, width, dim);
		out_mat += (dim * 9);
	}
	for (y = 1; y < height_lim; y++) {
		copy3x3cond(in, out_mat, 0, y, height, width, dim);
		out_mat += (dim * 9);
		for (int x = 1; x < width_lim; x++) {
			copy3x3_opt(in, out_mat, x, y, width, dim);
			out_mat += (dim * 9);
		}
		copy3x3cond(in, out_mat, width_lim, y, height, width, dim);
		out_mat += (dim * 9);
	}
	y = height - 1;
	for (int x = 0; x < width; x++) {
		copy3x3cond(in, out_mat, x, y, height, width, dim);
		out_mat += (dim * 9);
	}

}

#endif

void im2col(const float *in, float *out_mat, int height, int width, int dim) {

	
#ifdef AVX_AVX2
	const int VEC = 8;
	if (dim % VEC == 0) {
		im2col_opt(in, out_mat, height, width, dim);
		return;
	}
#elif defined(ARM_NEON) 
	const int VEC = 4;
	if (dim % VEC == 0) {
		im2col_opt(in, out_mat, height, width, dim);
		return;
	}
#endif

	int height_lim = height - 1;
	int width_lim = width - 1;

	int y = 0;
	for (int x = 0; x < width; x++) {
		copy3x3cond(in, out_mat, x, y, height, width, dim);
		out_mat += (dim * 9);
	}
	for (y = 1; y < height_lim; y++) {
		copy3x3cond(in, out_mat, 0, y, height, width, dim);
		out_mat += (dim * 9);
		for (int x = 1; x < width_lim; x++) {
			copy3x3(in, out_mat, x, y, width, dim);
			out_mat += (dim * 9);
		}
		copy3x3cond(in, out_mat, width_lim, y, height, width, dim);
		out_mat += (dim * 9);
	}
	y = height - 1;
	for (int x = 0; x < width; x++) {
		copy3x3cond(in, out_mat, x, y, height, width, dim);
		out_mat += (dim * 9);
	}
}

void im2col7x7rgb(const float *in, float *out_mat, int height, int width) {

	int height_lim = height - 3;
	int width_lim = width - 3;
	int box_size = 7 * 7 * 3;

	for (int y = 0; y < height; y += 2) {
		bool y_cond = (y < 3) || (y >= height_lim);
		for (int x = 0; x < width; x += 2) {
			bool x_cond = (x < 3) || (x >= width_lim);
			if (y_cond || x_cond) {
				copy7x7cond_rgb(in, out_mat, x, y, height, width);
			}
			else {
				copy7x7rgb(in, out_mat, x, y, width);
			}
			out_mat += box_size;
		}
	}
}


#if defined(AVX_AVX2) || defined(ARM_NEON)

int im2col8_mask_opt(const float *in, float *out_mat,
	int height, int width, int dim, float *mask, int stride_jump, float mask_bias) {

	int height_lim = height - 1;
	int width_lim = width - 1;
	float m;
	int count = 0;
	int jump = (dim * 8);

	int y = 0;
	for (int x = 0; x < width; x++) {
		m = *mask; m += mask_bias; *mask = m; mask += stride_jump;
		if (m > 0) {
			copy3x3_nc_cond_opt(in, out_mat, x, y, height, width, dim);
			out_mat += jump;
			count++;
		}
	}
	for (y = 1; y < height_lim; y++) {
		m = *mask; m += mask_bias; *mask = m; mask += stride_jump;
		if (m > 0) {
			copy3x3_nc_cond_opt(in, out_mat, 0, y, height, width, dim);
			out_mat += jump;
			count++;
		}
		for (int x = 1; x < width_lim; x++) {
			m = *mask; m += mask_bias; *mask = m; mask += stride_jump;
			if (m > 0) {
				copy3x3_nc_opt(in, out_mat, x, y, width, dim);
				out_mat += jump;
				count++;
			}
		}
		m = *mask; m += mask_bias; *mask = m; mask += stride_jump;
		if (m > 0) {
			copy3x3_nc_cond_opt(in, out_mat, width_lim, y, height, width, dim);
			out_mat += jump;
			count++;
		}
	}
	y = height - 1;
	for (int x = 0; x < width; x++) {
		m = *mask; m += mask_bias; *mask = m; mask += stride_jump;
		if (m > 0) {
			copy3x3_nc_cond_opt(in, out_mat, x, y, height, width, dim);
			out_mat += jump;
			count++;
		}
	}

	return count;
}
#endif

int im2col8_mask(const float *in, float *out_mat,
	int height, int width, int dim, float *mask, int stride_jump, float mask_bias) {

	int count = 0;

#ifdef AVX_AVX2
	const int VEC = 8;
	if (dim % VEC == 0) {
		count = im2col8_mask_opt(in, out_mat, height, width, dim,
			 mask, stride_jump, mask_bias);
		return count;
	}
#elif defined(ARM_NEON) 
	const int VEC = 4;
	if (dim % VEC == 0) {
		count = im2col8_mask_opt(in, out_mat, height, width, dim,
			 mask, stride_jump, mask_bias);
		return count;
	}	
#endif
	int height_lim = height - 1;
	int width_lim = width - 1;
	float m;
	int jump = (dim * 8);

	int y = 0;
	for (int x = 0; x < width; x++) {
		m = *mask; m += mask_bias; *mask = m; mask += stride_jump;
		if (m > 0) {
			copy3x3_nc_cond(in, out_mat, x, y, height, width, dim);
			out_mat += jump;
			count++;
		}
	}
	for (y = 1; y < height_lim; y++) {
		m = *mask; m += mask_bias; *mask = m; mask += stride_jump;
		if (m > 0) {
			copy3x3_nc_cond(in, out_mat, 0, y, height, width, dim);
			out_mat += jump;
			count++;
		}
		for (int x = 1; x < width_lim; x++) {
			m = *mask; m += mask_bias; *mask = m; mask += stride_jump;
			if (m > 0) {
				copy3x3_nc(in, out_mat, x, y, width, dim);
				out_mat += jump;
				count++;
			}
		}
		m = *mask; m += mask_bias; *mask = m; mask += stride_jump;
		if (m > 0) {
			copy3x3_nc_cond(in, out_mat, width_lim, y, height, width, dim);
			out_mat += jump;
			count++;
		}
	}
	y = height - 1;
	for (int x = 0; x < width; x++) {
		m = *mask; m += mask_bias; *mask = m; mask += stride_jump;
		if (m > 0) {
			copy3x3_nc_cond(in, out_mat, x, y, height, width, dim);
			out_mat += jump;
			count++;
		}
	}

	return count;
}


#ifdef AVX_AVX2

void col2im8_mask_opt(float *out, const float *in_mat3, float *in_mat1,
	const float *bias3, const float *bias1, int rows, int dim) {

	const int VEC = 8;
	__m256 zero = _mm256_set1_ps(0);
	__m256 two = _mm256_set1_ps(2);

	for (int y = 0; y < rows; y++) {
		float m = *in_mat1++;
		// for vector (x8) alignment
		in_mat1 += 7;

		if (m <= 0) {
			for (int d = 0; d < dim; d += VEC) {
				__m256 data1 = _mm256_load_ps(in_mat1);
				__m256 b1 = _mm256_load_ps(&bias1[d]);
				__m256 z = _mm256_add_ps(data1, b1);
				z = _mm256_max_ps(z, zero);
				_mm256_store_ps(out, z);
				out += VEC;
				in_mat1 += VEC;
			}
		}
		else if (m >= 1) {
			for (int d = 0; d < dim; d += VEC) {
				__m256 data1 = _mm256_load_ps(in_mat1);
				__m256 b1 = _mm256_load_ps(&bias1[d]);
				//__m256 z = _mm256_mul_ps(data1, two);
				//z = _mm256_add_ps(z, b1);
				__m256 z = _mm256_fmadd_ps(data1, two, b1);				
				__m256 data3 = _mm256_load_ps(in_mat3);
				__m256 b3 = _mm256_load_ps(&bias3[d]);
				data3 = _mm256_add_ps(data3, b3);
				z = _mm256_add_ps(z, data3);
				z = _mm256_max_ps(z, zero);
				_mm256_store_ps(out, z);

				out += VEC;
				in_mat1 += VEC;
				in_mat3 += VEC;
			}
		}
		else {
			__m256 mplus = _mm256_set1_ps(1 + m);
			__m256 m_ = _mm256_set1_ps(m);
			for (int d = 0; d < dim; d += VEC) {
				__m256 data1 = _mm256_load_ps(in_mat1);
				__m256 b1 = _mm256_load_ps(&bias1[d]);
				//__m256 z = _mm256_mul_ps(data1, mplus);
				//z = _mm256_add_ps(z, b1);
				__m256 z = _mm256_fmadd_ps(data1, mplus, b1);
				__m256 data3 = _mm256_load_ps(in_mat3);
				__m256 b3 = _mm256_load_ps(&bias3[d]);
				data3 = _mm256_add_ps(data3, b3);
				//data3 = _mm256_mul_ps(data3, m_);
				//z = _mm256_add_ps(z, data3);
				z = _mm256_fmadd_ps(data3, m_, z);
				z = _mm256_max_ps(z, zero);
				_mm256_store_ps(out, z);

				out += VEC;
				in_mat1 += VEC;
				in_mat3 += VEC;
			}
		}
	}
}

#elif defined(ARM_NEON)

void col2im8_mask_opt(float *out, const float *in_mat3, float *in_mat1,
	const float *bias3, const float *bias1, int rows, int dim) {

	const int VEC = 4;

	float32x4_t zero = vdupq_n_f32(0);
	float32x4_t two = vdupq_n_f32(2);

	for (int y = 0; y < rows; y++) {
		float m = *in_mat1++;
		// for vector (x8) alignment
		in_mat1 += 7;

		if (m <= 0) {
			for (int d = 0; d < dim; d += VEC) {
				float32x4_t data1 = vld1q_f32(in_mat1);
				float32x4_t b1 = vld1q_f32(&bias1[d]);
				float32x4_t z = vaddq_f32(data1, b1);
				z = vmaxq_f32(z, zero);
				vst1q_f32(out, z);
				out += VEC;
				in_mat1 += VEC;
			}
		}
		else if (m >= 1) {
			for (int d = 0; d < dim; d += VEC) {
				float32x4_t data1 = vld1q_f32(in_mat1);
				float32x4_t b1 = vld1q_f32(&bias1[d]);
				float32x4_t z = vmlaq_f32(b1, data1, two);
				float32x4_t data3 = vld1q_f32(in_mat3);
				float32x4_t b3 = vld1q_f32(&bias3[d]);
				data3 = vaddq_f32(data3, b3);
				z = vaddq_f32(z, data3);
				z = vmaxq_f32(z, zero);
				vst1q_f32(out, z);

				out += VEC;
				in_mat1 += VEC;
				in_mat3 += VEC;
			}
		}
		else {
			float32x4_t mplus = vdupq_n_f32(1 + m);
			float32x4_t m_ = vdupq_n_f32(m);
			for (int d = 0; d < dim; d += VEC) {
				float32x4_t data1 = vld1q_f32(in_mat1);
				float32x4_t b1 = vld1q_f32(&bias1[d]);
				float32x4_t z = vmlaq_f32(b1, data1, mplus);
				float32x4_t data3 = vld1q_f32(in_mat3);
				float32x4_t b3 = vld1q_f32(&bias3[d]);
				data3 = vaddq_f32(data3, b3);
				z = vmlaq_f32(z, data3, m_);
				z = vmaxq_f32(z, zero);
				vst1q_f32(out, z);

				out += VEC;
				in_mat1 += VEC;
				in_mat3 += VEC;
			}
		}
	}
}
#endif


void col2im8_mask(float *out, const float *in_mat3, float *in_mat1,
	const float *bias3, const float *bias1, int rows, int dim) {

	
#ifdef AVX_AVX2
	const int VEC = 8;
	if (dim % VEC == 0) {
		col2im8_mask_opt(out, in_mat3, in_mat1, bias3, bias1, rows, dim);
		return;
	}
#elif defined(ARM_NEON)
	const int VEC = 4;
	if (dim % VEC == 0) {
		col2im8_mask_opt(out, in_mat3, in_mat1, bias3, bias1, rows, dim);
		return;
	}
#endif

	for (int y = 0; y < rows; y++) {
		float m = *in_mat1++;
		// for vector (x8) alignment
		in_mat1 += 7;
		if (m < 0) m = 0;
		if (m > 1) m = 1;

		if (m > 0) {
			const float *bias3p = bias3;
			const float *bias1p = bias1;
			for (int d = dim; d > 0; d--) {
				float data1 = (*in_mat1++);
				float z = data1 * (1 + m) + (*bias1p++);
				float data3 = (*in_mat3++);
				data3 = data3 + (*bias3p++);
				float z3 = data3 * m;
				z += z3;
				if (z < 0) z = 0;
				*out++ = z;
			}
			// stride jump
		}
		else {
			const float *bias1p = bias1;
			for (int d = dim; d > 0; d--) {
				float data1 = (*in_mat1++);
				float z = data1 + (*bias1p++);
				if (z < 0) z = 0;
				*out++ = z;
			}
		}
	}
}

#ifdef AVX_AVX2 
void bias_relu(float *in_out, const float *bias,
	int rows, int dim) {
	
	const int VEC = 8;
	int all = rows * dim / VEC;
	__m256 zero = _mm256_set1_ps(0);
	for (int y = 0; y < all; y++) {
		__m256 a = _mm256_load_ps(in_out);
		int ib = (y << 3) % dim;
		__m256 b = _mm256_load_ps(&bias[ib]);
		__m256 c = _mm256_add_ps(a, b);
		c = _mm256_max_ps(c, zero);
		_mm256_store_ps(in_out, c);
		in_out += VEC;
	}
}

void add_bias_relu(float *in_out, const float *to_add, const float *bias,
	int rows, int dim) {

	const int VEC = 8;
	int all = rows * dim / VEC;
	__m256 zero = _mm256_set1_ps(0);
	for (int y = 0; y < all; y++) {
		__m256 a = _mm256_load_ps(in_out);
		__m256 d = _mm256_load_ps(to_add);
		int ib = (y << 3) % dim;
		__m256 b = _mm256_load_ps(&bias[ib]);
		__m256 c = _mm256_add_ps(a, b);
		c = _mm256_add_ps(c, d);
		c = _mm256_max_ps(c, zero);
		_mm256_store_ps(in_out, c);
		in_out += VEC;
		to_add += VEC;
	}
}

#elif defined(ARM_NEON)
void bias_relu(float *in_out, const float *bias,
	int rows, int dim) {
	
	const int VEC = 4;
	int all = rows * dim / VEC;
	float32x4_t zero = vdupq_n_f32(0);
	for (int y = 0; y < all; y++) {
		float32x4_t a = vld1q_f32(in_out);
		int ib = (y * VEC) % dim;
		float32x4_t b = vld1q_f32(&bias[ib]);
		float32x4_t c = vaddq_f32(a, b);
		c = vmaxq_f32(c, zero);
		vst1q_f32(in_out, c);
		in_out += VEC;
	}
}

void add_bias_relu(float *in_out, const float *to_add, const float *bias,
	int rows, int dim) {

	const int VEC = 4;
	int all = rows * dim / VEC;
	float32x4_t zero = vdupq_n_f32(0);
	for (int y = 0; y < all; y++) {
		float32x4_t a = vld1q_f32(in_out);
		float32x4_t d = vld1q_f32(to_add);
		int ib = (y * VEC) % dim;
		float32x4_t b = vld1q_f32(&bias[ib]);
		float32x4_t c = vaddq_f32(a, b);
		c = vaddq_f32(c, d);
		c = vmaxq_f32(c, zero);
		vst1q_f32(in_out, c);
		in_out += VEC;
		to_add += VEC;
	}
}

#else

void bias_relu(float *in_out, const float *bias,
	int rows, int dim) {
	
	int all = rows*dim;
	for (int y = 0; y < all; y++) {
			float z = (*in_out) + bias[y%dim];
			if (z < 0)
				z = 0;
			*in_out++ = z;			
	}
}
void add_bias_relu(float *in_out, const float *to_add, const float *bias,
	int rows, int dim) {
	
	int all = rows*dim;
	for (int y = 0; y < all; y++) {
			float z = (*in_out) + bias[y%dim];
			z += (*to_add++);
			if (z < 0)
				z = 0;
			*in_out++ = z;			
	}
}
#endif

#if defined(AVX_AVX2) || defined(ARM_NEON)

void bias_relu_pool3_s2(float *out, const float *in_mat, const float *bias,
	int height, int width, int dim) {

	int height_lim = height - 1;
	int width_lim = width - 1;

	for (int y = 1; y < height; y += 2) {
		bool y_cond = (y < 1) || (y >= height_lim);
		for (int x = 1; x < width; x += 2) {
			bool x_cond = (x < 1) || (x >= width_lim);
			if (y_cond || x_cond) {
				max3x3cond_opt(in_mat, out, bias, x, y, height, width, dim);
			}
			else {
				max3x3_opt(in_mat, out, bias, x, y, width, dim);
			}
			out += dim;
		}
	}

}

#else
void bias_relu_pool3_s2(float *out, const float *in_mat, const float *bias,
	int height, int width, int dim) {

	int height_lim = height - 1;
	int width_lim = width - 1;

	for (int y = 1; y < height; y += 2) {
		bool y_cond = (y < 1) || (y >= height_lim);
		for (int x = 1; x < width; x += 2) {
			bool x_cond = (x < 1) || (x >= width_lim);
			if (y_cond || x_cond) {
				max3x3cond(in_mat, out, bias, x, y, height, width, dim);
			}
			else {
				max3x3(in_mat, out, bias, x, y, width, dim);
			}
			out += dim;
		}
	}

}
#endif


#ifdef AVX_AVX2

void bias_relu_pool2_s2(float *out, const float *in_mat, const float *bias,
	int height, int width, int dim) {

	const int VEC = 8;

	int width_pool = width / 2;
	int height_pool = height / 2;

	float *outp = out;
	int mat_size = height_pool * width_pool*dim;
	__m256 zero = _mm256_set1_ps(0);
	int mat_size_n = mat_size / VEC;
	for (int y = 0; y < mat_size_n; y++) {
		_mm256_store_ps(outp, zero);
		outp += VEC;
	}

	for (int y = 0; y < height; y++) {
		int y_off = (y >> 1)*width_pool;
		for (int x = 0; x < width; x++) {
			float *outp = &out[(y_off + (x >> 1))*dim];
			for (int d = 0; d < dim; d+=VEC) {
				__m256 a = _mm256_load_ps(in_mat);
				__m256 b = _mm256_load_ps(&bias[d]);
				__m256 c = _mm256_add_ps(a, b);
				__m256 e = _mm256_load_ps(outp);
				e = _mm256_max_ps(e, c);
				_mm256_store_ps(outp, e);
				in_mat += VEC;
				outp += VEC;
			}
		}
	}
}

#elif defined(ARM_NEON)

void bias_relu_pool2_s2(float *out, const float *in_mat, const float *bias,
	int height, int width, int dim) {

	const int VEC = 4;
	int width_pool = width / 2;
	int height_pool = height / 2;

	float *outp = out;
	int mat_size = height_pool * width_pool*dim;
	float32x4_t zero = vdupq_n_f32(0);
	int mat_size_n = mat_size / VEC;
	for (int y = 0; y < mat_size_n; y++) {
		vst1q_f32(outp, zero);
		outp += VEC;
	}

	for (int y = 0; y < height; y++) {
		int y_off = (y >> 1)*width_pool;
		for (int x = 0; x < width; x++) {
			float *outp = &out[(y_off + (x >> 1))*dim];
			for (int d = 0; d < dim; d+=VEC) {
				float32x4_t a = vld1q_f32(in_mat);
				float32x4_t b = vld1q_f32(&bias[d]);
				float32x4_t c = vaddq_f32(a, b);
				float32x4_t e = vld1q_f32(outp);
				e = vmaxq_f32(e, c);
				vst1q_f32(outp, e);
				in_mat += VEC;
				outp += VEC;
			}
		}
	}
}

#else
void bias_relu_pool2_s2(float *out, const float *in_mat, const float *bias,
	int height, int width, int dim) {

	int width_pool = width / 2;
	int height_pool = height / 2;

	float *outp = out;
	int mat_size = height_pool * width_pool*dim;
	for (int y = 0; y < mat_size; y++) {
		*outp++ = 0;
	}

	for (int y = 0; y < height; y++) {
		int y_off = (y >> 1)*width_pool;
		for (int x = 0; x < width; x++) {
			float *outp = &out[(y_off + (x >> 1))*dim];
			const float *biasp = bias;
			for (int d = dim; d > 0; d--) {
				float z = (*in_mat++) + (*biasp++);
				if (z>(*outp))
					*outp = z;
				//z = fmaxf(z, 0);
				//*outp = fmaxf(z, *outp);
				outp++;
			}
		}
	}
}
#endif

#ifdef AVX_AVX2

void avg_pool2(float *out, const float *in_mat,
	int height, int width, int dim) {

	const int VEC = 8;
	int width_pool = width / 2;
	int height_pool = height / 2;

	float *outp = out;
	int mat_size = width_pool * height_pool*dim;
	__m256 zero = _mm256_set1_ps(0);
	int mat_size_n = mat_size / VEC;
	for (int y = 0; y < mat_size_n; y++) {
		_mm256_store_ps(outp, zero);
		outp += VEC;
	}

	for (int y = 0; y < height; y++) {
		int y_off = (y >> 1)*width_pool;
		for (int x = 0; x < width; x++) {
			float *outp = &out[(y_off + (x >> 1))*dim];
			for (int d = 0; d < dim; d += VEC) {
				__m256 a = _mm256_load_ps(in_mat);
				__m256 b = _mm256_load_ps(outp);
				b = _mm256_add_ps(a, b);
				_mm256_store_ps(outp, b);
				in_mat += VEC;
				outp += VEC;
			}
		}
	}

	outp = out;
	__m256 four = _mm256_set1_ps(4);

	for (int y = 0; y < mat_size_n; y++) {
		__m256 b = _mm256_load_ps(outp);
		b = _mm256_div_ps(b, four);
		_mm256_store_ps(outp, b);
		outp += VEC;
	}
	
}
#elif defined (ARM_NEON)

void avg_pool2(float *out, const float *in_mat,
	int height, int width, int dim) {

	const int VEC = 4;
	int width_pool = width / 2;
	int height_pool = height / 2;

	float *outp = out;
	int mat_size = width_pool * height_pool*dim;
	float32x4_t zero = vdupq_n_f32(0);
	int mat_size_n = mat_size / VEC;
	for (int y = 0; y < mat_size_n; y++) {
		vst1q_f32(outp, zero);
		outp += VEC;
	}

	for (int y = 0; y < height; y++) {
		int y_off = (y >> 1)*width_pool;
		for (int x = 0; x < width; x++) {
			float *outp = &out[(y_off + (x >> 1))*dim];
			for (int d = 0; d < dim; d += VEC) {
				float32x4_t a = vld1q_f32(in_mat);
				float32x4_t b = vld1q_f32(outp);
				b = vaddq_f32(a, b);
				vst1q_f32(outp, b);
				in_mat += VEC;
				outp += VEC;
			}
		}
	}

	outp = out;
	float32x4_t one_four = vdupq_n_f32(1/4.0f);

	for (int y = 0; y < mat_size_n; y++) {
		float32x4_t b = vld1q_f32(outp);
		b = vmulq_f32(b, one_four);
		vst1q_f32(outp, b);
		outp += VEC;
	}
	
}
#else
void avg_pool2(float *out, const float *in_mat,
	int height, int width, int dim) {

	int width_pool = width / 2;
	int height_pool = height / 2;

	float *outp = out;
	int mat_size = width_pool * height_pool*dim;
	for (int y = 0; y < mat_size; y++) {
		*outp++ = 0;
	}
	for (int y = 0; y < height; y++) {
		int y_off = (y >> 1)*width_pool;
		for (int x = 0; x < width; x++) {
			float *outp = &out[(y_off + (x >> 1))*dim];
			for (int d = dim; d > 0; d--) {
				(*outp++) += (*in_mat++);
			}
		}
	}

	outp = out;
	for (int y = 0; y < mat_size; y++) {
		(*outp++) /= 4;
	}

}
#endif

#ifdef AVX_AVX2
void max_pool2(float *out, const float *in_mat,
	int height, int width, int dim) {

	const int VEC = 8;
	int width_pool = width / 2;
	int height_pool = height / 2;

	float *outp = out;
	int mat_size = height_pool * width_pool*dim;
	__m256 zero = _mm256_set1_ps(0);
	int mat_size_n = mat_size / VEC;
	for (int y = 0; y < mat_size_n; y++) {
		_mm256_store_ps(outp, zero);
		outp += VEC;
	}

	for (int y = 0; y < height; y++) {
		int y_off = (y >> 1)*width_pool;
		for (int x = 0; x < width; x++) {
			float *outp = &out[(y_off + (x >> 1))*dim];
			for (int d = 0; d < dim; d += VEC) {
				__m256 a = _mm256_load_ps(in_mat);				
				__m256 e = _mm256_load_ps(outp);
				e = _mm256_max_ps(e, a);
				_mm256_store_ps(outp, e);
				in_mat += VEC;
				outp += VEC;
			}
		}
	}
}

#elif defined(ARM_NEON)

void max_pool2(float *out, const float *in_mat,
	int height, int width, int dim) {

	const int VEC = 4;
	int width_pool = width / 2;
	int height_pool = height / 2;

	float *outp = out;
	int mat_size = height_pool * width_pool*dim;
	float32x4_t zero = vdupq_n_f32(0);
	int mat_size_n = mat_size / VEC;
	for (int y = 0; y < mat_size_n; y++) {
		vst1q_f32(outp, zero);
		outp += VEC;
	}

	for (int y = 0; y < height; y++) {
		int y_off = (y >> 1)*width_pool;
		for (int x = 0; x < width; x++) {
			float *outp = &out[(y_off + (x >> 1))*dim];
			for (int d = 0; d < dim; d += VEC) {
				float32x4_t a = vld1q_f32(in_mat);				
				float32x4_t e = vld1q_f32(outp);
				e = vmaxq_f32(e, a);
				vst1q_f32(outp, e);
				in_mat += VEC;
				outp += VEC;
			}
		}
	}
}

#else
void max_pool2(float *out, const float *in_mat,
	int height, int width, int dim) {

	int width_pool = width / 2;
	int height_pool = height / 2;

	float *outp = out;
	int mat_size = width_pool * height_pool*dim;
	for (int y = 0; y < mat_size; y++) {
		*outp++ = 0;
	}
	for (int y = 0; y < height; y++) {
		int y_off = (y >> 1)*width_pool;
		for (int x = 0; x < width; x++) {
			float *outp = &out[(y_off + (x >> 1))*dim];
			for (int d = dim; d > 0; d--) {
				float z = (*in_mat++);
				if (z>(*outp))
					*outp = z;
				//*outp = fmaxf(z, *outp);
				outp++;
			}
		}
	}
}
#endif

#ifdef AVX_AVX2

void col2imVEC_mask_pool_opt(float *out, const float *in_mat3, float *in_mat1,
	const float *bias3, const float *bias1, int width, int height, int dim) {

	const int VEC = 8;
	__m256 two = _mm256_set1_ps(2);
	__m256 zero = _mm256_set1_ps(0);

	int width_pool = width / 2;
	int height_pool = height / 2;

	float *outp = out;
	int mat_size = height_pool * width_pool*dim;
	
	int mat_size_n = mat_size / VEC;
	for (int y = 0; y < mat_size_n; y++) {
		_mm256_store_ps(outp, zero);
		outp += VEC;
	}

	for (int y = 0; y < height; y++) {
		int y_off = (y >> 1)*width_pool;
		for (int x = 0; x < width; x++) {
			float *outp = &out[(y_off + (x >> 1))*dim];
			float m = *in_mat1++;
			// for vector (xVEC) alignment
			in_mat1 += 7;

			if (m <= 0) {
				for (int d = 0; d < dim; d += VEC) {
					__m256 data1 = _mm256_load_ps(in_mat1);
					__m256 b1 = _mm256_load_ps(&bias1[d]);
					__m256 z = _mm256_add_ps(data1, b1);
					__m256 e = _mm256_load_ps(outp);
					z = _mm256_max_ps(e, z);
					_mm256_store_ps(outp, z);
					in_mat1 += VEC;
					outp += VEC;
				}
			}
			else if (m >= 1) {
				for (int d = 0; d < dim; d += VEC) {
					__m256 data1 = _mm256_load_ps(in_mat1);
					__m256 b1 = _mm256_load_ps(&bias1[d]);
					//__m256 z = _mm256_mul_ps(data1, two);
					//z = _mm256_add_ps(z, b1);
					__m256 z = _mm256_fmadd_ps(data1, two, b1);
					__m256 data3 = _mm256_load_ps(in_mat3);
					__m256 b3 = _mm256_load_ps(&bias3[d]);
					data3 = _mm256_add_ps(data3, b3);
					z = _mm256_add_ps(z, data3);
					__m256 e = _mm256_load_ps(outp);
					z = _mm256_max_ps(e, z);
					_mm256_store_ps(outp, z);
					in_mat1 += VEC;
					in_mat3 += VEC;
					outp += VEC;
				}
			}
			else {
				__m256 m_ = _mm256_set1_ps(m);
				__m256 mplus = _mm256_set1_ps(1 + m);
				for (int d = 0; d < dim; d += VEC) {
					__m256 data1 = _mm256_load_ps(in_mat1);
					__m256 b1 = _mm256_load_ps(&bias1[d]);
					//__m256 z = _mm256_mul_ps(data1, mplus);
					//z = _mm256_add_ps(z, b1); 
					__m256 z = _mm256_fmadd_ps(data1, mplus, b1);
					__m256 data3 = _mm256_load_ps(in_mat3);
					__m256 b3 = _mm256_load_ps(&bias3[d]);
					data3 = _mm256_add_ps(data3, b3);					
					//data3 = _mm256_mul_ps(data3, m_);
					//z = _mm256_add_ps(z, data3);
					z = _mm256_fmadd_ps(data3, m_, z);
					__m256 e = _mm256_load_ps(outp);
					z = _mm256_max_ps(e, z);
					_mm256_store_ps(outp, z);

					in_mat1 += VEC;
					in_mat3 += VEC;
					outp += VEC;
				}
			}
		}
	}
}

#elif defined(ARM_NEON)

void col2im8_mask_pool_opt(float *out, const float *in_mat3, float *in_mat1,
	const float *bias3, const float *bias1, int width, int height, int dim) {

	const int VEC = 4;
	float32x4_t two = vdupq_n_f32(2);
	float32x4_t zero = vdupq_n_f32(0);

	int width_pool = width / 2;
	int height_pool = height / 2;

	float *outp = out;
	int mat_size = height_pool * width_pool*dim;
	
	int mat_size_n = mat_size / VEC;
	for (int y = 0; y < mat_size_n; y++) {
		vst1q_f32(outp, zero);
		outp += VEC;
	}

	for (int y = 0; y < height; y++) {
		int y_off = (y >> 1)*width_pool;
		for (int x = 0; x < width; x++) {
			float *outp = &out[(y_off + (x >> 1))*dim];
			float m = *in_mat1++;
			// for vector (x8) alignment
			in_mat1 += 7;

			if (m <= 0) {
				for (int d = 0; d < dim; d += VEC) {
					float32x4_t data1 = vld1q_f32(in_mat1);
					float32x4_t b1 = vld1q_f32(&bias1[d]);
					float32x4_t z = vaddq_f32(data1, b1);
					float32x4_t e = vld1q_f32(outp);
					z = vmaxq_f32(e, z);
					vst1q_f32(outp, z);
					in_mat1 += VEC;
					outp += VEC;
				}
			}
			else if (m >= 1) {
				for (int d = 0; d < dim; d += VEC) {
					float32x4_t data1 = vld1q_f32(in_mat1);
					float32x4_t b1 = vld1q_f32(&bias1[d]);
					float32x4_t z = vmlaq_f32(b1, data1, two);
					float32x4_t data3 = vld1q_f32(in_mat3);
					float32x4_t b3 = vld1q_f32(&bias3[d]);
					data3 = vaddq_f32(data3, b3);
					z = vaddq_f32(z, data3);
					float32x4_t e = vld1q_f32(outp);
					z = vmaxq_f32(e, z);
					vst1q_f32(outp, z);
					in_mat1 += VEC;
					in_mat3 += VEC;
					outp += VEC;
				}
			}
			else {
				float32x4_t m_ = vdupq_n_f32(m);
				float32x4_t mplus = vdupq_n_f32(1 + m);
				for (int d = 0; d < dim; d += VEC) {
					float32x4_t data1 = vld1q_f32(in_mat1);
					float32x4_t b1 = vld1q_f32(&bias1[d]);
					float32x4_t z = vmlaq_f32(b1, data1, mplus);
					float32x4_t data3 = vld1q_f32(in_mat3);
					float32x4_t b3 = vld1q_f32(&bias3[d]);
					data3 = vaddq_f32(data3, b3);					
					z = vmlaq_f32(z, data3, m_);
					float32x4_t e = vld1q_f32(outp);
					z = vmaxq_f32(e, z);
					vst1q_f32(outp, z);

					in_mat1 += VEC;
					in_mat3 += VEC;
					outp += VEC;
				}
			}
		}
	}
}
#endif

void col2im8_mask_pool(float *out, const float *in_mat3, float *in_mat1,
	const float *bias3, const float *bias1, int width, int height, int dim) {


#ifdef AVX_AVX2
	const int VEC = 8;
		if (dim % VEC == 0) {
		col2imVEC_mask_pool_opt(out, in_mat3, in_mat1,
			bias3, bias1, width, height, dim);
		return;
	}
#elif defined(ARM_NEON)
	const int VEC = 4;
	if (dim % VEC == 0) {
		col2im8_mask_pool_opt(out, in_mat3, in_mat1,
			bias3, bias1, width, height, dim);
		return;
	}
#endif

	int width_pool = width / 2;
	int height_pool = height / 2;

	float *outp = out;
	int mat_size = width_pool * height_pool*dim;
	for (int y = 0; y < mat_size; y++) {
		*outp++ = 0;
	}

	for (int y = 0; y < height; y++) {
		int y_off = (y >> 1)*width_pool;
		for (int x = 0; x < width; x++) {
			float *outp = &out[(y_off + (x >> 1))*dim];
			float m = *in_mat1++;
			// for vector (x8) alignment
			in_mat1 += 7;
			if (m < 0) m = 0;
			if (m > 1) m = 1;

			if (m > 0) {
				const float *bias3p = bias3;
				const float *bias1p = bias1;
				for (int d = dim; d > 0; d--) {
					float data1 = (*in_mat1++);
					float z = data1 * (1 + m) + (*bias1p++);
					float data3 = (*in_mat3++);
					data3 = data3 + (*bias3p++);
					float z3 = data3 * m;
					z += z3;
					if (z < 0) z = 0;
					if (z > *outp) 
						*outp = z;
					outp++;
				}
				// stride jump
			}
			else {
				const float *bias1p = bias1;
				for (int d = dim; d > 0; d--) {
					float data1 = (*in_mat1++);
					float z = data1 + (*bias1p++);
					if (z < 0) z = 0;
					if (z > *outp)
						*outp = z;
					outp++;
				}
			}
		}
	}
}

