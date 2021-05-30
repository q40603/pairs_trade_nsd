BLAS=-I /usr/include/mkl -L /usr/lib/x86_64-linux-gnu/mkl -lblas
CXX=g++
CXXFLAGS=-std=c++11 -shared -O3 -Wall -Wextra -Werror -fPIC 
PYINCLUDE:=`python3-config --includes`
PYIEXTERN:=-Iextern/pybind11/include
PYSUFFIX:=`python3-config --extension-suffix`


main: main.o Stock_data.o
	g++ -O3 -std=c++17 -march=native -DUSE_ARMA -L ./URT/lib main.o Stock_data.o -o main -lURT -Wall -Wextra -Werror -fPIC  

run: main
	LD_LIBRARY_PAT=./URT/lib/:$LD_LIBRARY_PATH ./main

clean:
	rm -f main *.o


# -march=native -DUSE_ARMA -o run -L ./URT/lib ./URT/examples/example2.cpp -lURT
main.o: main.cpp Stock_data.hpp
	g++ -c main.cpp -o main.o -std=c++17

Stock_data.o: Stock_data.cpp Stock_data.hpp
	g++ -c Stock_data.cpp -o Stock_data.o -std=c++17

VAR: VAR.cpp
	$(CXX) $(CXXFLAGS) $(PYINCLUDE) $(PYIEXTERN) $< -o $@$(PYSUFFIX) $(BLAS)

