CCBIN=/usr/bin/gcc
CC=$(CCBIN) -Wall -Wextra -std=c99 -pedantic -g -Wno-unused-variable

CCBIN=/usr/bin/gcc
CC=$(CCBIN) -Wall -Wextra -std=c99 -pedantic -g -Wno-unused-variable

default: model

model: utils/seeds.h utils/utils.h utils/string_utils.h math/matrix.h memory.h math/fixed_point_ops.h math/hashing.h math/matrix_ops.h layers/cells.h layers/layers.h neural_network.h main.h memory.c utils/string_utils.c math/fixed_point_ops.c math/hashing.c math/matrix_ops.c layers/layers.c neural_network.c main.c
	$(CC) -o model memory.c utils/string_utils.c math/fixed_point_ops.c math/hashing.c math/matrix_ops.c layers/layers.c neural_network.c main.c

matrix_test: utils/utils.h utils/string_utils.h math/matrix.h memory.h math/fixed_point_ops.h math/hashing.h math/matrix_ops.h tests/matrix_tests.h memory.c utils/string_utils.c math/fixed_point_ops.c math/hashing.c math/matrix_ops.c tests/matrix_tests.c
	$(CC) -o matrix_test memory.c utils/string_utils.c math/fixed_point_ops.c math/hashing.c math/matrix_ops.c tests/matrix_tests.c

fixed_point_test: utils/utils.h tests/fixed_point_tests.h tests/fixed_point_tests.c math/fixed_point_ops.h math/fixed_point_ops.c
	$(CC) -o fixed_point_tests math/fixed_point_ops.c tests/fixed_point_tests.c

memory_test: utils/utils.h tests/memory_tests.h tests/memory_tests.c memory.h memory.c
	$(CC) -o memory_tests memory.c tests/memory_tests.c

hashing_test: math/hashing.h tests/hashing_tests.h math/hashing.c tests/hashing_tests.c
	$(CC) -o hashing_tests math/hashing.c tests/hashing_tests.c

clean: rnnmodel
	rm -f *.o model *.dSYM
