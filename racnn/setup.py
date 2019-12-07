from distutils.core import setup, Extension
import platform

compiler_options = []
if platform.system() == "Windows":
	if platform.processor().split(' ')[0]=='Intel64':
		compiler_options = ['-DAVX_AVX2']
elif platform.system() == "Linux":
	if platform.machine()=='x86_64':
		compiler_options = ['-std=c++11', '-DAVX_AVX2', '-march=native', '-O3']
	elif platform.machine()=='armv7l':
		compiler_options = ['-std=c++11', '-DARM_NEON', '-mfpu=neon', '-mcpu=''cortex-a72''',  '-mtune=''cortex-a72''', '-O3']
	
		
module = Extension('racnnlib',
                    sources = ['./libs/py_racnnlib.cpp', './libs/racnn.cpp'],
                    include_dirs = [],
                    libraries = [],
                    library_dirs = ['/usr/local/lib'],                    
                    extra_compile_args=compiler_options)
 
setup(name = 'racnnlib',
      version = '1.0',
      description = 'racnn lib tools',
      ext_modules = [module])
