import numpy as np       # load numpy library
import tensorflow as tf  # load tensorflow library


##### Problem 2 #####
A = tf.convert_to_tensor([[1, 2],
                          [3, 4]])
# tf.convert_to_tensor converts our Python list
# into a tensorflow Tensor, which we can do operations
# on. We can't do operations on a native Python list.
v = tf.convert_to_tensor([[1],
                          [2]])

print("2A:", 2 * A)             # <-- [[2, 4], [6, 8]]
print("v + v:", v + v)          # <-- [[2], [4]]
print("A^T:", tf.transpose(A))  # <-- [[1, 3], [2, 4]]
print("A * A:", A * A)          # <-- [[1, 4], [9, 16]]
# notice how when we print tensorflow Tensor, they give us
# three things: the underlying data, the shape, and the 
# datatype (i.e., integer, 32-bit precision float, 
# 64-bit precision float, string, etc.)

print("<v, v>:", tf.reduce_sum(tf.square(v)))  # <-- 5 
# tf.reduce_sum computes the sum of the elements
# here, we are taking advantage of the fact that the dot
# product is equal to sum(v * v) = sum(v^2). You might 
# have written different code, but it should produce the 
# same result.

##### Problem 3 #####
matrix_a = np.load("a.npy")
matrix_b = np.load("b.npy")

result = matrix_a @ matrix_b
np.save("p3.npy", result)

