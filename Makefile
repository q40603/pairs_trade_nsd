main: main.o Stock_data.o
	g++ -O3 -std=c++17 -march=native -DUSE_ARMA -L ./URT/lib main.o Stock_data.o -o main -lURT -Wall -Wextra -Werror -fPIC  

run: main
	./main

clean:
	rm -f main *.o


# -march=native -DUSE_ARMA -o run -L ./URT/lib ./URT/examples/example2.cpp -lURT
main.o: main.cpp Stock_data.hpp
	g++ -c main.cpp -o main.o -std=c++17

Stock_data.o: Stock_data.cpp Stock_data.hpp
	g++ -c Stock_data.cpp -o Stock_data.o -std=c++17
