import tensorflow as tf
import numpy as np
import prepare_image
from scipy.special import expit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from PIL import Image


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def tf_accuracy(predict_data):
    test_label = prepare_image.load_images(data_type="test", image_type="label", classification=4,
                                            dataset="small")
    predict_label = np.load(predict_data)
    X = tf.constant(predict_label, dtype=tf.float32, shape=test_label.shape)
    Y = tf.constant(test_label, dtype=tf.float32, shape=test_label.shape)
    X_sigmoid = tf.nn.sigmoid(X)
    X_softmax = tf.nn.softmax(X_sigmoid, axis=1)
    sess = tf.InteractiveSession()

    verify = tf.cast(tf.equal(tf.argmax(X_softmax, axis=1), tf.argmax(Y, axis=1)), dtype=tf.float32)
    accuracy = tf.reduce_mean(verify)

    b_p = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(X_softmax, axis=1), 3), dtype=tf.float32))
    b_t = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(Y, axis=1), 3), dtype=tf.float32))

    a_p = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(X_softmax, axis=1), 0), dtype=tf.float32))
    a_t = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(Y, axis=1), 0), dtype=tf.float32))

    o_p = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(X_softmax, axis=1), 1), dtype=tf.float32))
    o_t = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(Y, axis=1), 1), dtype=tf.float32))

    v_p = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(X_softmax, axis=1), 2), dtype=tf.float32))
    v_t = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(Y, axis=1), 2), dtype=tf.float32))
    
    print("Accuracy of Background is {}".format(sess.run(1 - abs(b_t-b_p)/b_t)))
    print("Accuracy of Arteries is {}".format(sess.run(1 - abs(a_t - a_p) / a_t)))
    print("Accuracy of Veins is {}".format(sess.run(1 - abs(v_t - v_p) / v_t)))
    print("Accuracy of Background is {}".format(sess.run(1 - abs(o_t - o_p) / o_t)))
    print("Overall accuracy is {}".format(sess.run(accuracy)))


def np_f1score(predict_data):
    test_label = prepare_image.load_images(data_type="test", image_type="label", classification=4,
                                            dataset="small")
    predict_label = np.load(predict_data)
    predict_label = softmax(expit(predict_label.transpose(1, 2, 3, 0))).transpose(3, 0, 1, 2)
    predict = np.argmax(predict_label, axis=1).reshape(11 * 565 * 565)
    test = np.argmax(test_label, axis=1).reshape(11 * 565 * 565)
    f1_s = f1_score(predict, test, average=None)
    f1_s_w = f1_score(predict, test, average='weighted')
    precision_s = precision_score(predict, test, average=None)
    precision_s_w = precision_score(predict, test, average='weighted')
    recall_s = recall_score(predict, test, average=None)
    recall_s_w = recall_score(predict, test, average='weighted')
    print("Precision of A,O,V,B is {}".format(precision_s))
    print("Overall Precision is {}".format(precision_s_w))
    print("Recall of A,O,V,B is {}".format(recall_s))
    print("Overall Recall is {}".format(recall_s_w))
    print("F1-Score of A,O,V,B {}".format(f1_s))
    print("Overall F-1 Score is {}".format(f1_s_w))


def tf_image_reconstruct():
    predict_label = np.load('cache/test_predict2_class_4.npy')
    # X = tf.placeholder(dtype=tf.float32, shape=predict_label.shape)
    X = tf.constant(predict_label, dtype=tf.float32, shape=predict_label.shape)
    sess = tf.InteractiveSession()
    for i in range(len(predict_label)):
        image_data = tf.transpose(tf.nn.softmax(tf.nn.sigmoid(X[i]), axis=0),(1, 2, 0))
        max_image_data = tf.reduce_max(image_data, axis=2)
        a = tf.equal(max_image_data, image_data[..., 0])
        o = tf.equal(max_image_data, image_data[..., 1])
        v = tf.equal(max_image_data, image_data[..., 2])
        image_r = tf.stack((a, o, v), axis=2)
        print(image_r.shape)
        np_image = np.where(image_r.eval(), 255, 0)
        image = Image.fromarray(np.uint8(np_image), mode='RGB')

    sess.run(image_r)


# tf_image_reconstruct()
tf_accuracy(predict_data='cache/test_predict2_class_4_relu.npy')
np_f1score(predict_data='cache/test_predict2_class_4_relu.npy')



'''
https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras?noredirect=1&lq=1
import tensorflow as tf

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)

class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, tag):
        super().__init__() 
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):
        # Load image
        img = data.astronaut()
        # Do something to the image
        img =(255 * skimage.util.random_noise(img)).astype('uint8')

        image = make_image(img)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
        writer = tf.summary.FileWriter('./logs')
        writer.add_summary(summary, epoch)
        writer.close()

        return

tbi_callback = TensorBoardImage('Image Example')
'''
