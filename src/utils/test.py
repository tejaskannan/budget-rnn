import tensorflow as tf

from tfutils import expand_to_matrix


with tf.Session() as sess:
    x = tf.constant([-3, -2, -1, -0.5], dtype=tf.float32)
    size = 4
    mat_dims = (4, 4)

    mat = expand_to_matrix(x, size=size, matrix_dims=mat_dims, name='transform')

    print(sess.run(mat))
