import tensorflow as tf
import tensorflow_datasets as tfds

from util import predict_and_disp


#### Create multilayer perceptron
x_input = tf.keras.layers.Input((784,))  # input is 784 x 1 vector
x = tf.keras.layers.Dense(100, activation="sigmoid")(x_input)
x = tf.keras.layers.Dense(100, activation="sigmoid")(x)
x_output = tf.keras.layers.Dense(10, activation="softmax")(x)
# output is 10-D vector, representing the probs of each class

mlp = tf.keras.Model(inputs=[x_input], outputs=[x_output])


#### Load data
buffer = 1024    # see tf.data.shuffle size docs for explanation of this
batch_size = 32  # how many examples to process at once

ds = tfds.load("mnist", split="train", shuffle_files=True)
ds = ds.shuffle(buffer).batch(batch_size)  # shuffle files and batch them together
                                           # batching helps training go faster
ds = ds.prefetch(tf.data.experimental.AUTOTUNE)  # this helps dataloading go faster

#### Train loop
epochs = 5  # number of epochs
lr = 1e-4   # learning rate

for epoch in range(epochs):
    print(f"#### EPOCH {epoch + 1}")

    for step_num, example_batch in enumerate(ds):
        x_batch = example_batch["image"]
        # the shape of `x_batch` is (batch_size, height, width, 1)`
        # so we reshape to (batch_size, height * width)`
        x_batch = tf.reshape(x_batch, (batch_size, -1))
        # -1 means fill in the remaining dimension

        x_batch = tf.cast(x_batch, tf.float32) / 255.
        # normalize so each element of input is between 0 and 1

        y_batch = example_batch["label"]
        # y_batch is a batch of indexes, and we want to convert to a 
        # distribution over the classes. see the docs for one_hot for 
        # more information
        y_batch = tf.one_hot(y_batch, depth=10)

        #### Predict on inputs and calculate loss
        with tf.GradientTape() as tape:
            preds = mlp(x_batch, training=True)
            loss = tf.keras.losses.categorical_crossentropy(y_batch, preds)

        #### Calculate gradients wrt loss and apply them using grad descent
        grads = tape.gradient(loss, mlp.trainable_weights)
        for grad, param in zip(grads, mlp.trainable_weights):
            # w <- w - lr * grad
            param.assign_sub(lr * grad)

        #### Print current loss every 100 steps
        if step_num % 100 == 0:
            print(f"... STEP {step_num}. Current training loss: {float(tf.reduce_mean(loss))}")

    #### Display and predict
    predict_and_disp(ds, mlp)

