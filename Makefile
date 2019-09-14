main: *.cpp
	mpicxx -Wall -Wpedantic -Werror --std=c++11 -o main *.cpp
