import tensorflow as tf

def predict_and_disp(ds, model):
    batch = next(iter(ds))

    img = batch["image"][0]
    label = batch["label"][0]

    img_p = tf.reshape(img, (1, -1))
    pred = tf.argmax(model.predict(img_p), axis=-1)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            c = "-" if img[i, j] < 50 else "@"
            print(c, end=" ")
        print()

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"PREDICTED: {pred[0]}")
    print(f"LABEL: {label}")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

