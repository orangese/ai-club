import tensorflow as tf

def predict_and_disp(ds, model):
    batch = next(iter(ds))

    img = batch["image"][0]
    label = batch["label"][0]
    img_ = img[None,:,:,:]
    print(model.predict(img_)[0])
    pred = tf.argmax(model.predict(img_)[0])

    img = tf.reshape(img, (28, 28))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            c = "-" if img[i, j] < 50 else "@"
            print(c, end=" ")
        print()

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"PREDICTED: {pred}")
    print(f"LABEL: {label}")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

