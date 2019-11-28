// RACNN python wrapper 
// by Meisam Rakhshanfar

#include <Python.h>
#include "racnn.h"

static PyObject *GenError;

PyObject* py_im2col(PyObject *self, PyObject *args)
{
	PyObject *arg1, *arg2, *arg3;
	Py_buffer b_imgin, b_imgout, b_mask;
	float mask_bias = 0;
	int kernel_size;
	arg3 = Py_None;

	if (!PyArg_ParseTuple(args, "OOi|Of", &arg1, &arg2, &kernel_size, &arg3, &mask_bias))
		return NULL;

		
	if ((PyObject_GetBuffer(arg1, &b_imgin, PyBUF_FULL) < 0) || 
		(PyObject_GetBuffer(arg2, &b_imgout, PyBUF_FULL) < 0))
		return NULL;

	if (b_imgin.itemsize != 4 || b_imgout.itemsize != 4) {
		PyErr_SetString(GenError, "input/output data type error");
		return NULL;
	}
	if (b_imgin.ndim != 3 || b_imgout.ndim != 2) {
		PyErr_SetString(GenError, "input/output dimension type error");
		return NULL;
	}
	
	int img_row = (int)b_imgin.shape[0];
	int img_col = (int)b_imgin.shape[1];
	int img_dim = (int)b_imgin.shape[2];

	int img_row_o = (int)b_imgout.shape[0];
	int img_col_o = (int)b_imgout.shape[1];

	int count = 0;

	if (arg3 != Py_None) {

		if (kernel_size != 3) {
			PyErr_SetString(GenError, "Only kernel size 3 is supported");
			return NULL;
		}

		if (PyObject_GetBuffer(arg3, &b_mask, PyBUF_FULL) < 0)
			return NULL;
		if (b_mask.itemsize != 4 || b_mask.ndim != 1) {
			PyErr_SetString(GenError, "mask data type/dimension error");
			return NULL;
		}
		int img_row_m = (int)b_mask.shape[0];
		int mask_stride = (int)(b_mask.strides[0] / sizeof(float));


		if (img_row_o != (img_row* img_col) || img_col_o != 8 * img_dim || img_row_m != img_row_o) {
			PyErr_SetString(GenError, "Output/mask dimension error .\n");
			return NULL;
		}
		count = im2col8_mask((const float *)b_imgin.buf, (float *)b_imgout.buf, img_row, img_col, img_dim,
			(float *)b_mask.buf, mask_stride, mask_bias);

		PyBuffer_Release(&b_mask);

	}
	else {
		if (!(kernel_size == 3 || kernel_size == 7)) {
			PyErr_SetString(GenError, "Kernel size 3 and 7 is supported");
			return NULL;
		}

		if (kernel_size == 3) {
			if (img_row_o != (img_row* img_col) || img_col_o != kernel_size * kernel_size* img_dim) {
				PyErr_SetString(GenError, "Output dimension error\n");
				return NULL;
			}
			im2col((float *)b_imgin.buf, (float *)b_imgout.buf, img_row, img_col, img_dim);
		}
		else if (kernel_size == 7) {
			if (img_row_o != ((img_row / 2)* (img_col / 2)) || img_col_o != kernel_size * kernel_size* img_dim) {
				PyErr_SetString(GenError, "Output dimension error\n");
				return NULL;
			}
			if (img_dim != 3) {
				PyErr_SetString(GenError, "RGB error");
				return NULL;
			}
			im2col7x7rgb((float *)b_imgin.buf, (float *)b_imgout.buf, img_row, img_col);
		}
	}

	PyObject* res = PyLong_FromLong(count);
	PyBuffer_Release(&b_imgin);
	PyBuffer_Release(&b_imgout);	

	return res;
}

PyObject* py_col2im_mask(PyObject *self, PyObject *args)
{
	PyObject *arg1, *arg2, *arg3, *arg4, *arg5;
	Py_buffer b_imgin3, b_imgin1, b_bias3, b_bias1, b_out;
	int pool = 0;

	if (!PyArg_ParseTuple(args, "OOOOO|i", &arg1, &arg2, &arg3, &arg4, &arg5, &pool))
		return NULL;

	if ((PyObject_GetBuffer(arg1, &b_imgin3, PyBUF_FULL) < 0) || (PyObject_GetBuffer(arg2, &b_imgin1, PyBUF_FULL) < 0)
		|| (PyObject_GetBuffer(arg3, &b_bias3, PyBUF_FULL) < 0)	
		|| (PyObject_GetBuffer(arg4, &b_bias1, PyBUF_FULL) < 0) || (PyObject_GetBuffer(arg5, &b_out, PyBUF_FULL) < 0))
		return NULL;

	if (b_imgin3.itemsize != 4 || b_imgin1.itemsize != 4 || 
		b_bias3.itemsize != 4 || b_bias1.itemsize != 4 || b_out.itemsize != 4) {
		PyErr_SetString(GenError, "data type error");
		return NULL;
	}
	if (b_imgin3.ndim != 2 || b_imgin1.ndim != 2 ||	b_bias3.ndim != 1 
		|| b_bias1.ndim != 1) {
		PyErr_SetString(GenError, "dimension type error");
		return NULL;
	}

	int img_row_o3 = (int)b_imgin3.shape[0];
	int img_col_o3 = (int)b_imgin3.shape[1];

	int img_row_o1 = (int)b_imgin1.shape[0];
	int img_col_o1 = (int)b_imgin1.shape[1];
	int dim = img_col_o1 - 8;

	if (img_row_o3 > img_row_o1 || img_col_o3 != dim ||
		b_bias3.shape[0]!= dim || b_bias1.shape[0] != dim) {
		PyErr_SetString(GenError, "Input dimension error .\n");
		return NULL;
	}

	if (pool == 0) {
		if (b_out.ndim != 2) {
			PyErr_SetString(GenError, "dimension type error");
			return NULL;
		}
		if ((int)b_out.shape[0] != img_row_o1 || (int)b_out.shape[1] != dim) {
			PyErr_SetString(GenError, "Output dimension error .\n");
			return NULL;
		}
		col2im8_mask((float *)b_out.buf, (const float *)b_imgin3.buf, (float *)b_imgin1.buf,
			(const float *)b_bias3.buf, (const float *)b_bias1.buf, img_row_o1, dim);
	}
	else {
		if (b_out.ndim != 3) {
			PyErr_SetString(GenError, "dimension type error");
			return NULL;
		}
		if (((int)b_out.shape[0] * (int)b_out.shape[1]) != (img_row_o1 / 4) || (int)b_out.shape[2] != dim) {
			PyErr_SetString(GenError, "Output dimension error .\n");
			return NULL;
		}
		col2im8_mask_pool((float *)b_out.buf, (const float *)b_imgin3.buf, (float *)b_imgin1.buf,
			(const float *)b_bias3.buf, (const float *)b_bias1.buf,
			(int)b_out.shape[1] * 2, (int)b_out.shape[0] * 2, (int)b_out.shape[2]);
	}

	PyObject* res = PyLong_FromLong(0);
	PyBuffer_Release(&b_imgin3);
	PyBuffer_Release(&b_imgin1);
	PyBuffer_Release(&b_bias3);
	PyBuffer_Release(&b_bias1);
	PyBuffer_Release(&b_out);

	return res;
}

PyObject* py_bias_relu(PyObject *self, PyObject *args)
{
	PyObject *arg1, *arg2, *arg3;
	Py_buffer b_imgout, b_imgin, b_bias;

	arg3 = Py_None;
	int pool_size=0;
	if (!PyArg_ParseTuple(args, "OO|Oi", &arg1, &arg2, &arg3, &pool_size))
		return NULL;

	if ((PyObject_GetBuffer(arg1, &b_imgin, PyBUF_FULL) < 0) || 
		(PyObject_GetBuffer(arg2, &b_bias, PyBUF_FULL) < 0))
		return NULL;

	
	if (b_imgin.itemsize != 4 || b_bias.itemsize != 4) {
		PyErr_SetString(GenError, "data type error");
		return NULL;
	}
	
	int img_row = (int)b_imgin.shape[0];
	int img_col = (int)b_imgin.shape[1];
	int img_dim = (int)b_imgin.shape[2];	

	if (arg3 != Py_None) {

		if (PyObject_GetBuffer(arg3, &b_imgout, PyBUF_FULL) < 0)
			return NULL;
		
		if (b_imgout.itemsize != 4 || b_imgout.ndim != 3) {
			PyErr_SetString(GenError, "output data type/dimension error");
			return NULL;
		}
		
		if (b_imgin.ndim != 3 || b_bias.ndim != 1) {
			PyErr_SetString(GenError, "dimension type error");
			return NULL;
		}

		int img_row_p = (int)b_imgout.shape[0];
		int img_col_p = (int)b_imgout.shape[1];
		int img_dim_p = (int)b_imgout.shape[2];

		if (img_row_p != (img_row / 2) || img_col_p != (img_col / 2) || img_dim_p != img_dim
			|| (int)b_bias.shape[0] != img_dim) {
			PyErr_SetString(GenError, "Output dimension error .\n");
			return NULL;
		}

		if (pool_size == 2) {
			bias_relu_pool2_s2((float *)b_imgout.buf, (const float *)b_imgin.buf, (const float *)b_bias.buf,
				img_row, img_col, img_dim);
		}
		else if (pool_size == 3) {
			bias_relu_pool3_s2((float *)b_imgout.buf, (const float *)b_imgin.buf, (const float *)b_bias.buf,
				img_row, img_col, img_dim);
		}
		else {
			PyErr_SetString(GenError, "pool size error");
			return NULL;
		}		

		PyBuffer_Release(&b_imgout);

	}
	else {
		if (!(b_imgin.ndim == 2 || b_imgin.ndim == 3) || b_bias.ndim != 1) {
			PyErr_SetString(GenError, "dimension type error");
			return NULL;
		}

		int rows = 0;
		int img_dim = 0;
		if (b_imgin.ndim == 2) {
			rows = (int)b_imgin.shape[0];
			img_dim = (int)b_imgin.shape[1];
		}
		else {
			rows = ((int)b_imgin.shape[0])* ((int)b_imgin.shape[1]);
			img_dim = (int)b_imgin.shape[2];
		}

		if ((int)b_bias.shape[0] != img_dim) {
			PyErr_SetString(GenError, "Output dimension error .\n");
			return NULL;
		}

		bias_relu((float *)b_imgin.buf, (const float *)b_bias.buf,
			rows, img_dim);

	}

	PyObject* res = PyLong_FromLong(0);

	PyBuffer_Release(&b_imgin);
	PyBuffer_Release(&b_bias);

	return res;
}

PyObject* py_add_bias_relu(PyObject *self, PyObject *args)
{
	PyObject *arg1, *arg2, *arg3;
	Py_buffer b_imgin, b_imgadd, b_bias;

	if (!PyArg_ParseTuple(args, "OOO", &arg1, &arg2, &arg3))
		return NULL;

	if ((PyObject_GetBuffer(arg1, &b_imgin, PyBUF_FULL) < 0) || (PyObject_GetBuffer(arg2, &b_imgadd, PyBUF_FULL) < 0)
		|| (PyObject_GetBuffer(arg3, &b_bias, PyBUF_FULL) < 0))
		return NULL;

	if (b_imgin.itemsize != 4 || b_bias.itemsize != 4 || b_imgadd.itemsize != 4) {
		PyErr_SetString(GenError, "data type error");
		return NULL;
	}
	if (b_imgin.ndim != 2 || b_imgadd.ndim != 2 || b_bias.ndim != 1) {
		PyErr_SetString(GenError, "dimension type error");
		return NULL;
	}

	int img_row = (int)b_imgin.shape[0];
	int img_dim = (int)b_imgin.shape[1];

	if ((int)b_bias.shape[0] != img_dim || (int)b_bias.shape[0] != (int)b_imgadd.shape[1]||
		img_row != (int)b_imgadd.shape[0]) {
		PyErr_SetString(GenError, "Output dimension error .\n");
		return NULL;
	}

	add_bias_relu((float *)b_imgin.buf, (float *)b_imgadd.buf, (const float *)b_bias.buf,
		img_row, img_dim);

	PyObject* res = PyLong_FromLong(0);
	PyBuffer_Release(&b_imgin);
	PyBuffer_Release(&b_bias);
	PyBuffer_Release(&b_imgadd);

	return res;
}

PyObject* py_avg_pool2(PyObject *self, PyObject *args)
{
	PyObject *arg1, *arg2;
	Py_buffer b_imgout, b_imgin;

	if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2))
		return NULL;

	if ((PyObject_GetBuffer(arg1, &b_imgin, PyBUF_FULL) < 0) ||
		(PyObject_GetBuffer(arg2, &b_imgout, PyBUF_FULL) < 0)) {
		return NULL;
	}

	if (b_imgout.itemsize != 4 || b_imgin.itemsize != 4) {
		PyErr_SetString(GenError, "data type error");
		return NULL;
	}
	if (b_imgout.ndim != 3 || b_imgin.ndim != 3) {
		PyErr_SetString(GenError, "dimension type error");
		return NULL;
	}

	int img_row_p = (int)b_imgout.shape[0];
	int img_col_p = (int)b_imgout.shape[1];
	int img_dim_p = (int)b_imgout.shape[2];

	int img_row = (int)b_imgin.shape[0];
	int img_col = (int)b_imgin.shape[1];
	int img_dim = (int)b_imgin.shape[2];

	if (img_row_p != (img_row / 2) || img_col_p != (img_col / 2) || img_dim_p != img_dim) {
		PyErr_SetString(GenError, "Output dimension error .\n");
		return NULL;
	}

	avg_pool2((float *)b_imgout.buf, (const float *)b_imgin.buf,
		img_row, img_col, img_dim);

	PyObject* res = PyLong_FromLong(0);
	PyBuffer_Release(&b_imgout);
	PyBuffer_Release(&b_imgin);

	return res;
}

PyObject* py_max_pool2(PyObject *self, PyObject *args)
{
	PyObject *arg1, *arg2;
	Py_buffer b_imgout, b_imgin;

	if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2))
		return NULL;

	if ((PyObject_GetBuffer(arg1, &b_imgin, PyBUF_FULL) < 0) ||
		(PyObject_GetBuffer(arg2, &b_imgout, PyBUF_FULL) < 0)) {
		return NULL;
	}

	if (b_imgout.itemsize != 4 || b_imgin.itemsize != 4) {
		PyErr_SetString(GenError, "data type error");
		return NULL;
	}
	if (b_imgout.ndim != 3 || b_imgin.ndim != 3) {
		PyErr_SetString(GenError, "dimension type error");
		return NULL;
	}

	int img_row_p = (int)b_imgout.shape[0];
	int img_col_p = (int)b_imgout.shape[1];
	int img_dim_p = (int)b_imgout.shape[2];

	int img_row = (int)b_imgin.shape[0];
	int img_col = (int)b_imgin.shape[1];
	int img_dim = (int)b_imgin.shape[2];

	if (img_row_p != (img_row / 2) || img_col_p != (img_col / 2) || img_dim_p != img_dim) {
		PyErr_SetString(GenError, "Output dimension error .\n");
		return NULL;
	}

	max_pool2((float *)b_imgout.buf, (const float *)b_imgin.buf,
		img_row, img_col, img_dim);

	PyObject* res = PyLong_FromLong(0);
	PyBuffer_Release(&b_imgout);
	PyBuffer_Release(&b_imgin);

	return res;
}

PyMethodDef racnnlib_methods[] = {
	{ "im2col", (PyCFunction)py_im2col, METH_VARARGS, "Image to column reshape with or without mask" },
	{ "col2im_mask",(PyCFunction)py_col2im_mask,METH_VARARGS,"Column to image with mask" },
	{ "bias_relu",(PyCFunction)py_bias_relu,METH_VARARGS,"Applies bias and relu" },
	{ "add_bias_relu",(PyCFunction)py_add_bias_relu,METH_VARARGS,"Adds two and applies bias and relu on results" },
	{ "avg_pool2",(PyCFunction)py_avg_pool2,METH_VARARGS,"Average pool 2" },
	{ "max_pool2",(PyCFunction)py_max_pool2,METH_VARARGS,"Max pool 2" },
	{ NULL, NULL, 0, NULL }
};

static struct PyModuleDef racnnlib_module = {
	PyModuleDef_HEAD_INIT,
	"racnnlib",
	"racnnlib Module C++",
	-1,
	racnnlib_methods
};

PyMODINIT_FUNC
PyInit_racnnlib(void)
{
	PyObject *m = PyModule_Create(&racnnlib_module);

	if (m == NULL)
		return NULL;

	PyObject * d = PyModule_GetDict(m);
	PyDict_SetItemString(d, "__version__", PyUnicode_FromString("1.0"));
	PyDict_SetItemString(d, "vec_size", PyLong_FromLong(INTR_VEC_SIZE));

	GenError = PyErr_NewException("racnnlib.error", NULL, NULL);
	Py_INCREF(GenError);
	PyModule_AddObject(m, "error", GenError);

	return m;

}
