# choose your compiler, e.g. gcc/clang
# example override to clang: make CC=clang
CC = g++

# the most basic way of building that is most likely to work on most systems
.PHONY: all
all: src/mamba.cpp
	$(CC) -O3 -o build/mamba src/mamba.cpp -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./mamba out/model.bin -n 3
.PHONY: debug
debug: src/mamba.cpp
	$(CC) -g -o build/mamba src/mamba.cpp -lm

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: fast
fast: src/mamba.cpp
	$(CC) -Ofast -o build/mamba src/mamba.cpp -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./mamba out/model.bin
.PHONY: omp
omp: src/mamba.cpp
	$(CC) -Ofast -fopenmp -march=native src/mamba.cpp -lm  -o build/mamba

.PHONY: win64
win64:
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o build/mamba.exe -I. mamba.c win.c

# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: gnu
gnu:
	$(CC) -Ofast -std=gnu11 -o build/mamba src/mamba.cpp -lm

.PHONY: ompgnu
ompgnu:
	$(CC) -Ofast -fopenmp -std=gnu11 src/mamba.cpp  -lm  -o build/mamba

# run all tests
.PHONY: test
test:
	pytest

# run only tests for mamba.c C implementation (is a bit faster if only C code changed)
.PHONY: testc
testc:
	pytest -k runc

# run the C tests, without touching pytest / python
# to increase verbosity level run e.g. as `make testcc VERBOSITY=1`
VERBOSITY ?= 0
.PHONY: testcc
testcc:
	$(CC) -DVERBOSITY=$(VERBOSITY) -O3 -o testc test.c -lm
	./testc

.PHONY: clean
clean:
	rm -f build/mamba
