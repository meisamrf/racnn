from distutils.core import setup, Extension


module = Extension('racnnlib',
                    sources = ['./libs/py_racnnlib.cpp'],
                    include_dirs = [],
                    libraries = ['racnn'],
                    library_dirs = ['/usr/local/lib', './libs'],                    
                    extra_compile_args=['-std=c++11'])
 
setup(name = 'racnnlib',
      version = '1.0',
      description = 'racnn lib tools',
      ext_modules = [module])
