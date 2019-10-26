// curacnn python wrapper
// By Meisam Rakhshanfar
//

#include <Python.h>
#include "libcuracnn.h"



static PyObject *GenError;


struct curacnn_wrapper_t
{
	PyObject_HEAD
		libcuracnn *v;
};

static PyTypeObject curacnn_wrapper_Type =
{
	PyVarObject_HEAD_INIT(&PyType_Type, 0)
	"cuda.racnn",
	sizeof(curacnn_wrapper_t),
};

static int to_ok(PyTypeObject *to)
{
	to->tp_alloc = PyType_GenericAlloc;
	to->tp_new = PyType_GenericNew;
	to->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	return (PyType_Ready(to) == 0);
}

static PyObject* curacnn_model(PyObject *, PyObject *args)
{

	curacnn_wrapper_t* self = 0;
	self = PyObject_NEW(curacnn_wrapper_t, &curacnn_wrapper_Type);
	libcuracnn *m_curacnn = new libcuracnn();

	if (PyObject_Size(args) == 0) {
		self->v = m_curacnn; // init Ptr with placement new
		return (PyObject*)self;
	}

	if (!PyArg_ParseTuple(args, "")) {
		return NULL;
	}

	self->v = m_curacnn;
	return (PyObject*)self;
}

static void curacnn_wrapper_dealloc(PyObject* self)
{
	((curacnn_wrapper_t*)self)->v->~libcuracnn();
	PyObject_Del(self);
}
static PyObject* curacnn_wrapper_repr(PyObject* self)
{
	char str[1000];
	sprintf(str, "<curacnn %p>", self);
	return PyUnicode_FromString(str);
}

static PyGetSetDef curacnn_wrapper_getseters[] =
{
	{ NULL }  /* Sentinel */
};

PyObject* py_open(PyObject *self, PyObject *args)
{

	if (!PyObject_TypeCheck(self, &curacnn_wrapper_Type)) {
		PyErr_SetString(GenError, "Incorrect type of self (must be 'libcuracnn' or its derivative)");
		return NULL;
	}

	libcuracnn* _self_ = ((curacnn_wrapper_t*)self)->v;

	int model_type;
	if (!PyArg_ParseTuple(args, "i", &model_type)) {
		PyErr_SetString(GenError, "input error");
		return NULL;
	}

	bool ret = _self_->open(model_type);
	return PyBool_FromLong(ret);
	
}

PyObject* py_isopen(PyObject *self, PyObject *args)
{

	if (!PyObject_TypeCheck(self, &curacnn_wrapper_Type)) {
		PyErr_SetString(GenError, "Incorrect type of self (must be 'libcuracnn' or its derivative)");
		return NULL;
	}
	if (!PyArg_ParseTuple(args, "")) {
		PyErr_SetString(GenError, "input error");
		return NULL;
	}

	libcuracnn* _self_ = ((curacnn_wrapper_t*)self)->v;

	if (!_self_->is_open()) {
		return 	PyBool_FromLong(0);
	}

	return 	PyBool_FromLong(1);
}

PyObject* py_release(PyObject *self, PyObject *args)
{

	if (!PyObject_TypeCheck(self, &curacnn_wrapper_Type)) {
		PyErr_SetString(GenError, "Incorrect type of self (must be 'libcuracnn' or its derivative)");
		return NULL;
	}
	if (!PyArg_ParseTuple(args, "")) {
		PyErr_SetString(GenError, "input error");
		return NULL;
	}
	libcuracnn* _self_ = ((curacnn_wrapper_t*)self)->v;

	_self_->release();

	return 	PyBool_FromLong(1);
}

PyObject* py_predict(PyObject *self, PyObject *args)
{

	PyObject *arg1, *arg2;
	Py_buffer py_in, py_out;

	if (!PyObject_TypeCheck(self, &curacnn_wrapper_Type)) {
		PyErr_SetString(GenError, "Incorrect type of self (must be 'libcuracnn' or its derivative)");
		return NULL;
	}
	if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) {
		PyErr_SetString(GenError, "input error");
		return NULL;
	}
	libcuracnn* _self_ = ((curacnn_wrapper_t*)self)->v;

	if (PyObject_GetBuffer(arg1, &py_in, PyBUF_FULL) < 0 ||
		PyObject_GetBuffer(arg2, &py_out, PyBUF_FULL) < 0)
		return NULL;

	if (py_in.itemsize != 4 || py_in.ndim != 3) {
		PyErr_SetString(GenError, "The input do not have the expected size or shape");
		return NULL;
	}

	if (py_out.itemsize != 4 || py_out.ndim != 1) {
		PyErr_SetString(GenError, "The output do not have the expected size or shape");
		return NULL;
	}
	if (!PyBuffer_IsContiguous(&py_in, 'C') || !PyBuffer_IsContiguous(&py_out, 'C')) {
		PyErr_SetString(GenError, "non contiguous memory");
		return NULL;
	}

	int io_size[4];
	_self_->get_inout_size(io_size);

	if (py_in.shape[0] != io_size[0] || py_in.shape[1] != io_size[1] ||
		py_in.shape[2] != io_size[2] || py_out.shape[0] != io_size[3]) {
		PyErr_SetString(GenError, "The input do not have the expected size or shape");
		return NULL;
	}

	int res = _self_->predict((float *)py_in.buf, (float *)py_out.buf);

	PyBuffer_Release(&py_in);
	PyBuffer_Release(&py_out);
	return 	PyBool_FromLong(res);
}

PyObject* py_load_weight(PyObject *self, PyObject *args)
{

	PyObject *arg1;
	Py_buffer py_in;
	int layer_num;

	if (!PyObject_TypeCheck(self, &curacnn_wrapper_Type)) {
		PyErr_SetString(GenError, "Incorrect type of self (must be 'libcuracnn' or its derivative)");
		return NULL;
	}
	if (!PyArg_ParseTuple(args, "Oi", &arg1, &layer_num)) {
		PyErr_SetString(GenError, "input error");
		return NULL;
	}
	libcuracnn* _self_ = ((curacnn_wrapper_t*)self)->v;

	if (PyObject_GetBuffer(arg1, &py_in, PyBUF_FULL) < 0)
		return NULL;

	if (py_in.itemsize != 4 || py_in.ndim != 2) {
		PyErr_SetString(GenError, "The input do not have the expected size or shape");
		return NULL;
	}

	if (!PyBuffer_IsContiguous(&py_in, 'C')) {
		PyErr_SetString(GenError, "non contiguous memory");
		return NULL;
	}

	int res = _self_->load_weight(layer_num, (int)py_in.shape[0], (int)py_in.shape[1], (float *)py_in.buf);

	PyBuffer_Release(&py_in);
	return 	PyBool_FromLong(res);
}



PyObject* py_get_device_name(PyObject *self, PyObject *args)
{

	if (!PyObject_TypeCheck(self, &curacnn_wrapper_Type)) {
		PyErr_SetString(GenError, "Incorrect type of self (must be 'libcuracnn' or its derivative)");
		return NULL;
	}
	if (!PyArg_ParseTuple(args, "")) {
		PyErr_SetString(GenError, "input error");
		return NULL;
	}
	libcuracnn* _self_ = ((curacnn_wrapper_t*)self)->v;

	char dev_name[257];
	dev_name[0] = 0;

	_self_->get_device_name(dev_name, 256);

	return 	PyUnicode_FromStringAndSize(dev_name, 256);
}



PyObject* py_get_mem_size(PyObject *self, PyObject *args)
{

	if (!PyObject_TypeCheck(self, &curacnn_wrapper_Type)) {
		PyErr_SetString(GenError, "Incorrect type of self (must be 'libcuracnn' or its derivative)");
		return NULL;
	}
	if (!PyArg_ParseTuple(args, "")) {
		PyErr_SetString(GenError, "input error");
		return NULL;
	}
	libcuracnn* _self_ = ((curacnn_wrapper_t*)self)->v;

	unsigned long long mem_size = _self_->get_mem_size();

	return 	PyLong_FromLongLong(mem_size);
}



PyObject* py_set_speed_test(PyObject *self, PyObject *args)
{

	int state;
	if (!PyObject_TypeCheck(self, &curacnn_wrapper_Type)) {
		PyErr_SetString(GenError, "Incorrect type of self (must be 'libcuracnn' or its derivative)");
		return NULL;
	}		
	if (!PyArg_ParseTuple(args, "i", &state)) {
		PyErr_SetString(GenError, "input error");
		return NULL;
	}
	libcuracnn* _self_ = ((curacnn_wrapper_t*)self)->v;

	_self_->set_speed_test(state!=0);

	return 	PyLong_FromLongLong(1);
}


PyObject* py_get_weight_num(PyObject *self, PyObject *args)
{

	if (!PyObject_TypeCheck(self, &curacnn_wrapper_Type)) {
		PyErr_SetString(GenError, "Incorrect type of self (must be 'libcuracnn' or its derivative)");
		return NULL;
	}
	if (!PyArg_ParseTuple(args, "")) {
		PyErr_SetString(GenError, "input error");
		return NULL;
	}
	libcuracnn* _self_ = ((curacnn_wrapper_t*)self)->v;

	int weight_num = _self_->get_weight_num();

	return 	PyLong_FromLong(weight_num);
}


static PyMethodDef curacnn_wrapper_methods[] =
{
	{ "open", (PyCFunction)py_open, METH_VARARGS | METH_KEYWORDS, "open(type) -> retvel" },
	{ "is_open", (PyCFunction)py_isopen, METH_VARARGS | METH_KEYWORDS, "is_open() -> retval" },
	{ "release", (PyCFunction)py_release, METH_VARARGS | METH_KEYWORDS, "release() -> retval" },
	{ "predict", (PyCFunction)py_predict, METH_VARARGS | METH_KEYWORDS, "predict(in, out) -> retval" },
	{ "load_weight", (PyCFunction)py_load_weight, METH_VARARGS | METH_KEYWORDS, "load_weight(in, layer_num) -> retval" },
	{ "get_device_name", (PyCFunction)py_get_device_name, METH_VARARGS | METH_KEYWORDS, "get_device_name() -> retval" },
	{ "get_mem_size", (PyCFunction)py_get_mem_size, METH_VARARGS | METH_KEYWORDS, "get_mem_size() -> retval" },
	{ "get_weight_num", (PyCFunction)py_get_weight_num, METH_VARARGS | METH_KEYWORDS, "get_weight_num() -> retval" },
	{ "set_speed_test", (PyCFunction)py_set_speed_test, METH_VARARGS | METH_KEYWORDS, "set_speed_test(bool) -> retval" },
	{ NULL, NULL, 0, NULL }
};

static void curacnn_wrapper_specials(void)
{
	curacnn_wrapper_Type.tp_base = NULL;
	curacnn_wrapper_Type.tp_dealloc = curacnn_wrapper_dealloc;
	curacnn_wrapper_Type.tp_repr = curacnn_wrapper_repr;
	curacnn_wrapper_Type.tp_getset = curacnn_wrapper_getseters;
	curacnn_wrapper_Type.tp_methods = curacnn_wrapper_methods;
}



PyMethodDef curacnn_Methods[] = {
	{ "model", (PyCFunction)curacnn_model, METH_VARARGS | METH_KEYWORDS, "curacnn_wrapper() -> <curacnn_wrapper object>" },
	{ NULL, NULL, 0, NULL }
};


static struct PyModuleDef curacnnmodule = {
	PyModuleDef_HEAD_INIT,
	"curacnn",   /* name of module */
	"curacnn Module C++", /* module documentation, may be NULL */
	-1,       /* size of per-interpreter state of the module,
			  or -1 if the module keeps state in global variables. */
	curacnn_Methods
};

PyMODINIT_FUNC
PyInit_curacnn(void)
{
	curacnn_wrapper_specials();
	if (!to_ok(&curacnn_wrapper_Type))
		return NULL;

	PyObject *m = PyModule_Create(&curacnnmodule);

	if (m == NULL)
		return NULL;

	PyObject * d = PyModule_GetDict(m);
	PyDict_SetItemString(d, "__version__", PyUnicode_FromString("1.0"));

	GenError = PyErr_NewException("curacnn.error", NULL, NULL);
	Py_INCREF(GenError);
	PyModule_AddObject(m, "error", GenError);

	return m;
}
