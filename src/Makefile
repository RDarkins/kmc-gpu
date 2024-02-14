CC=nvcc
NVCC=nvcc
CPPFLAGS=-x cu -O3 -std=c++17
INCLUDES=
LIBS=-lm -lstdc++
OBJS=main.o
PROG=kmc

# Targets

all : $(OBJS)
	$(NVCC) -o $(PROG) $(OBJS) $(LIBS)

%.o : %.cpp
	$(NVCC) $(CPPFLAGS) -dc -c -o $@ $< $(INCLUDES) $(LIBS)

.PHONY : clean
clean :
	find . -type f \( -name '*.o' -o -name '*~' \) -delete
	rm -f $(PROG)
