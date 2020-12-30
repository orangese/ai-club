import tensorflow as tf  # load tensorflow library


##### Problem 1 #####
y = tf.convert_to_tensor([4, 3, 6])
y_hat = tf.convert_to_tensor([5, 5, 5])
# Recall from week 1 that we use convert_to_tensor
# to convert python lists into tensor form that is 
# usable in the context of other tensorflow operations

print("L2 norm:", tf.reduce_sum(tf.square(y - y_hat)))

# The square function is self-explanatory- reduce_sum
# simply returns the sum of all elements of a given 
# vector



##### Problem 2 #####
y = tf.convert_to_tensor([4, 3, 6])
y_hat = tf.convert_to_tensor([5, 5, 5])

MSE = tf.keras.losses.MeanSquaredError()
print("Mean Squared Error:", MSE(y, y_hat))

MAE = tf.keras.losses.MeanAbsoluteError()
print("Mean Absolute Error:", MAE(y, y_hat))

# MSE is larger because the terms in its summation
# are all second order whereas the same terms are 
# only first order in MAE



##### Problem 3 #####
# This loss function does not guaruntee that all terms are positive. This leads to the possibility
# that the error terms cancel each other out. A simple example of this would be when y = [1, 1] and 
# y_hat = [0, 2]. In this case, y - y_hat = [1, -1], but the hypotehtical error using a basic summation
# loss function would be 0, which is clearly not the case. 