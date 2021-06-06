BLAS=-I /usr/include/mkl -L /usr/lib/x86_64-linux-gnu/mkl -lblas
CXX=g++
CXXFLAGS=-std=c++11 -shared -O3 -Wall -Wextra -Werror -fPIC -march=native -DUSE_ARMA
PYINCLUDE:=`python3-config --includes`
PYIEXTERN:=-Iextern/pybind11/include
PYSUFFIX:=`python3-config --extension-suffix`
Stock_data=Stock_data$(shell python3-config --extension-suffix)

# UNAME_S := $(shell uname -s)
# NUMPY_PATH := $(shell python3 -c "import numpy;print(numpy.get_include())")
# ifeq ($(UNAME_S),Darwin)
# MKLROOT ?= /opt/intel/mkl
# MKLEXT ?= a
# CXXFLAGS :=
# endif

# ifeq ($(UNAME_S),Linux)
# #MKLROOT ?= ${HOME}/opt/conda
# MKLROOT ?= /home/kctsai/miniconda3/envs/demo
# MKLEXT ?= so
# CXXFLAGS := -Wl,--no-as-needed
# endif

# MKLLINKLINE := \
# 	${MKLROOT}/lib/libmkl_intel_lp64.${MKLEXT} \
# 	${MKLROOT}/lib/libmkl_sequential.${MKLEXT} \
# 	${MKLROOT}/lib/libmkl_core.${MKLEXT} \
# 	${MKLROOT}/lib/libmkl_avx2.${MKLEXT} \
# 	${MKLROOT}/lib/libmkl_def.${MKLEXT} \


# CXXFLAGS := ${CXXFLAGS} \
# 	-I${MKLROOT}/include \
# 	${MKLLINKLINE}

.PHONY: default
all: Stock_data VAR


Stock_data: Stock_data.hpp
	${CXX} ${CXXFLAGS} $(PYINCLUDE) $(PYIEXTERN) -L ./URT/lib ./stock_data_Wrapper.cpp $< -o $@$(PYSUFFIX) $(BLAS) -lURT

VAR: VAR.cpp
	$(CXX) $(CXXFLAGS) $(PYINCLUDE) $(PYIEXTERN) $< -o $@$(PYSUFFIX) $(BLAS)


.PHONY: clean
clean:
	rm -f main *.o *.so

# main: main.o Stock_data.o
# 	g++ -O3 -std=c++17 -march=native -DUSE_ARMA -L ./URT/lib main.o Stock_data.o -o main -lURT -Wall -Wextra -Werror -fPIC  

# .PHONY: run
# run: main
# 	LD_LIBRARY_PAT=./URT/lib/:$LD_LIBRARY_PATH ./main


# # -march=native -DUSE_ARMA -o run -L ./URT/lib ./URT/examples/example2.cpp -lURT
# main.o: main.cpp Stock_data.hpp
# 	g++ -c main.cpp -o main.o -std=c++17

# Stock_data.o: Stock_data.cpp Stock_data.hpp
# 	g++ -c Stock_data.cpp -o Stock_data.o -std=c++17




