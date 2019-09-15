main: *.cpp
	mpicxx -Wall -Wuninitialized -Wpedantic -Wformat-security -Wignored-qualifiers -Winit-self \
                -Wswitch-default -Wshadow -Wpointer-arith \
                -Wtype-limits -Wempty-body -Wlogical-op \
                -Wctor-dtor-privacy -Wno-reorder\
                -Wnon-virtual-dtor -Wstrict-null-sentinel  \
                -Woverloaded-virtual -Wsign-promo -Wextra -pedantic -Werror --std=c++11 -O3 -o main *.cpp
