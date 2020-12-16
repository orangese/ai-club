# Week 2 Practice Problems

Please complete the following questions by the next meeting:

1. The derivative `f'(x)` of a monomial `f'(x) = ax ** n` is given by the formula `f'(x) = nax ** (n-1)`, where `**` is the exponent operation. Additionally, the derivative of a polynomial is given by the sum of the derivatives of each of its terms. Given these two rules, calculate by hand the derivatives of the below functions. Then, compute the value of the derivatives at the x-value `x = 1`.
	* `f(x) = 2x ** 3 - 5x ** 4 + 3`
	* `g(x) = x ** -1 + 3x`
	
	EXAMPLE: the derivative of `h(x) = x ** 2 - 5x` is `h'(x) = 2x - 5`

2. When we want to calculate derivatives in TensorFlow, we use a special class called `GradientTape`. This will automatically record any
 operations (add, subtract, multiply, exponent, etc) that we use- and will automatically and correctly differentiate all of them. We can use it via the following syntax:

	```python

	import tensorflow as tf

	x = tf.Variable(3.0)

	with tf.GradientTape() as tape:
	    # do operations here
	    y = x ** 2 + 3

	print(tape.gradient(y, x))
	# will output the derivative of `x^2 + 3`, evaluated at the x-value `x = 3`

	``` 

	Using this syntax, verify that the derivatives you calculated by hand in Problem 1 are correct, by using the value `x = 1` in the `GradientTape`. See https://www.tensorflow.org/guide/autodiff for more about automatic differentiation with `GradientTape`.
