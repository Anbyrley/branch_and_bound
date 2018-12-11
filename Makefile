#===General Variables===#
CC=gcc
CFLAGS=-Wall -Wextra -g3

all: makeAll

makeAll: makeHelper makeRandom makeStats makeAlgebra makeDictionary makeMachine makeBB makeMain
	$(CC) $(CFLAGS) helper.o random.o stats.o algebra.o dictionary.o machine_learning.o branch_and_bound.o main.o -o branch_and_bound -ldl -lm -lblas -llapack -lpthread 

makeMain: main.c
	$(CC) $(CFLAGS) -c main.c -o main.o 

makeBB: branch_and_bound.c branch_and_bound.h
	$(CC) $(CFLAGS) -c branch_and_bound.c -o branch_and_bound.o 

makeMachine: machine_learning.c machine_learning.h
	$(CC) $(CFLAGS) -c machine_learning.c -o machine_learning.o 

makeDictionary: dictionary.c dictionary.h
	$(CC) $(CFLAGS) -c dictionary.c -o dictionary.o 

makeAlgebra: linear_algebra.c linear_algebra.h
	$(CC) $(CFLAGS) -c linear_algebra.c -o algebra.o 

makeStats: stats.c stats.h
	$(CC) $(CFLAGS) -c stats.c -o stats.o 

makeRandom: random_numbers.c random_numbers.h
	$(CC) $(CFLAGS) -c random_numbers.c -o random.o 

makeHelper: helper.c helper.h macros.h
	$(CC) $(CFLAGS) -c helper.c -o helper.o


.PHONY: clean

clean:
	rm -f *~ $(ODIR)/*.o $(IDIR)/*~
