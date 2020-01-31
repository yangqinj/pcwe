ifeq ($(shell uname -s), Darwin)
	CC = clang
else
	CC = gcc
	CFLAGS += -pthread -lm
endif

CFLAGS += -O2 -std=c99

all: pcwe.c
	${CC} pcwe.c ${CFLAGS} -o pcwe
clean:
	rm pcwe


