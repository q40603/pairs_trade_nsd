main: main.o Stock_data.o
	g++ main.o Stock_data.o -o main -std=c++11 -O3 -Wall -Wextra -Werror -fPIC  

run: main
	./main

clean:
	rm -f main *.o

main.o: main.cpp Stock_data.hpp
	g++ -c main.cpp -o main.o -std=c++11

Stock_data.o: Stock_data.cpp Stock_data.hpp
	g++ -c Stock_data.cpp -o Stock_data.o -std=c++11