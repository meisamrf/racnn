from distutils.core import Extension, setup

module = Extension(
    "curacnn",
    sources=["./libs/py_curacnn.cpp"],
    include_dirs=[],
    libraries=['curacnn'],
    library_dirs=["/usr/local/lib", './libs'],
    extra_compile_args=["-std=c++11"],
)

setup(
    name="curacnn",
    version="1.0",
    description="racnn gpu lib tools",
    ext_modules=[module],
)
