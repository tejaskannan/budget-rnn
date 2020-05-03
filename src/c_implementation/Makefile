CCBIN=/usr/bin/gcc
CC=$(CCBIN) -Wall -Wextra -std=c99 -pedantic -g -Wno-unused-variable

CCBIN=/usr/bin/gcc
CC=$(CCBIN) -Wall -Wextra -std=c99 -pedantic -g -Wno-unused-variable

default: model

model: matrix.h memory.h params.h sample_model.h tensor.h utils.h utils.c activations.h memory.c tensor.c layers.h models.h models.c main.c
	$(CC) -o rnnmodel memory.c activations.c tensor.c utils.c layers.c models.c main.c -lm

matrix_test: math/matrix.h memory.h math/fixed_point_ops.h math/matrix_ops.h tests/matrix_tests.h memory.c math/fixed_point_ops.c math/matrix_ops.c tests/matrix_tests.c
	$(CC) -o matrix_test memory.c math/fixed_point_ops.c math/matrix_ops.c tests/matrix_tests.c

fixed_point_test: tests/fixed_point_tests.h tests/fixed_point_tests.c math/fixed_point_ops.h math/fixed_point_ops.c
	$(CC) -o fixed_point_tests math/fixed_point_ops.c tests/fixed_point_tests.c

memory_test: tests/memory_tests.h tests/memory_tests.c memory.h memory.c
	$(CC) -o memory_tests memory.c tests/memory_tests.c

clean: rnnmodel
	rm -f *.o rnnmodel *.dSYM
