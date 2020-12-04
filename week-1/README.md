Please complete the following questions by the next meeting:

1. For matrix A = [[1, 2], [3, 4]] and vector v = [1, 2]^T, calculate by hand:
   ->  2A
   ->  v + v
   ->  A ⊙  A
   ->  A^T
   ->  \<v, v\>

2. Verify your results obtained for the Sample Problems using TensorFlow and Python. HINT: create a Python list, then convert that to a tensorflow `Tensor` object using `tf.convert_to_tensor`. Note that matrix multiplication (`AB` in math) can be calculated using  `A @ B`, and the Hadamard product can be calculated by `A * B`.  Documentation can be found at https://www.tensorflow.org/api_docs.

3. Download the data files attached. Read the data into the appropriate sized matrices using `numpy`. Multiply the two matrices, and write the result to a file called `p3.npz`. HINT: see `np.load` documentation for help loading .npz data into python (https://numpy.org/doc/stable/reference/generated/numpy.load.html). For writing data, see `np.save` documentation (https://numpy.org/doc/stable/reference/generated/numpy.save.html).

Good luck! If you have any issues, reach out to us on Discord or email. Have a great weekend! 😊
