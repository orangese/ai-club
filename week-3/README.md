# Week 3 Practice Problems

Please complete the following questions by the next meeting:

1. A very commonly used loss function (recall that loss functions are used to assess a model's accuracy) is the L2 loss function. It is named as such as its computation is derived from the L2 norm, which we discussed in our lecture on linear algebra from week 1. For a refresher, the L2 norm of a vector is simply defined as `sqrt(x_1 ** 2 + x_2 ** 2 + ... + x_n ** 2)`, where `**` is the exponent operation and `x` is some vector. L2 loss is defined as the square of this quantity when `x = y - y_hat` where `y` is a vector of true values of a given data set and `y_hat` is a vector of a given model's predictions on that data set, leaving us with `x_1 ** 2 + x_2 ** 2 + ... + x_n ** 2`. The larger this sum, the farther the model's predictions deviated from the true values of the data set, and thus the higher its loss. Calculate the L2 loss for two vectors `y = [4, 3, 6]` and `y_hat = [5, 5, 5]`.

	EXAMPLE: the L2 norm of `y = [1, 2, 3]` and `y_hat = [3, 2, 1]` would be `8`

2. Luckily for us, Tensorflow handles the computation of loss functions, providing us a variety of different loss function classes at our disposal in the tensorflow.keras.losses directlry. A complete list of available loss functions can be found at this link: https://www.tensorflow.org/api_docs/python/tf/keras/losses, though most of these you do not need to understand yet. To use a lost function, one simply instantiates an instance of the given loss function, and passes in the appropriate parameters. For example, the Hinge Loss Class takes in two parameters, `y and y_hat`, and is used as below:
	
	```python



	import tensorflow as tf

	y = tf.convert_to_tensor([1, 1, 1])
	y_hat = tf.convert_to_tensor([2, 2, 2])

	hinge_loss = tf.keras.losses.Hinge()
	print("Hinge Loss:", hinge_loss(y, y_hat))


	``` 

Compare the MeanSquaredError loss function output with MeanAbsoluteError on the same data from question 1 (`y = [4, 3, 6], y_hat = [5, 5, 5]`). MeanSquaredError is given by the expression `(x_1 ** 2 + x_2 ** 2 + ... + x_n ** 2) / n`, whereas MeanAbsoluteError is given by the expression `(|x_1| + |x_2| + ... + |x_n|) / n`. Which one is larger? Why?

3. What would be a potential pitfall of a loss function that was simply the sum of the differences between a given `y` and `y_hat`? In other words, a loss function given by `x_1 + x_2 + ... + x_n` where `x = y - y_hat`?


If you are stuck, message us!