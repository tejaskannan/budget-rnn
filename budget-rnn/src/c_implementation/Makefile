CCBIN=/usr/bin/gcc
CC=$(CCBIN) -Wall -Wextra -std=c99 -pedantic -g -Wno-unused-variable -fsanitize=address

default: model

model: utils/utils.h utils/neural_network_utils.h math/matrix.h neural_network_parameters.h math/fixed_point_ops.h math/matrix_ops.h controller/interpolation.h controller/budget_distribution.h controller/pid_control.h layers/cells.h layers/layers.h neural_network.h main.h math/fixed_point_ops.c math/matrix_ops.c controller/interpolation.c controller/budget_distribution.c controller/pid_control.c layers/layers.c neural_network.c main.c
	$(CC) -o model math/fixed_point_ops.c math/matrix_ops.c controller/interpolation.c controller/budget_distribution.c controller/pid_control.c layers/layers.c neural_network.c main.c

matrix_test: utils/utils.h math/matrix.h neural_network_parameters.h math/fixed_point_ops.h math/matrix_ops.h tests/matrix_tests.h math/fixed_point_ops.c math/matrix_ops.c tests/matrix_tests.c
	$(CC) -o matrix_test math/fixed_point_ops.c math/matrix_ops.c tests/matrix_tests.c

fixed_point_test: utils/utils.h tests/fixed_point_tests.h tests/fixed_point_tests.c math/fixed_point_ops.h math/fixed_point_ops.c
	$(CC) -o fixed_point_tests math/fixed_point_ops.c tests/fixed_point_tests.c

clean: model
	rm -f *.o model fixed_point_tests matrix_test *.dSYM
