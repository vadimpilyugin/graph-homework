main: *.cpp
	mpicxx -Wall -Wuninitialized -pedantic -Wformat-security -Wignored-qualifiers -Winit-self \
                -Wswitch-default -Wshadow -Wpointer-arith \
                -Wtype-limits -Wempty-body -Wlogical-op \
                -Wctor-dtor-privacy -Wno-reorder\
                -Wnon-virtual-dtor -Wstrict-null-sentinel  \
                -Woverloaded-virtual -Wsign-promo -Wextra -Werror -std=c++0x -O5 -o main *.cpp
