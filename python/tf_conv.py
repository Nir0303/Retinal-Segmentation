import tensorflow as tf
import numpy as np
import prepare_image
from PIL import Image

def tf_accuracy():
    test_label = prepare_image.load_images(data_type="test", image_type="label", classification=4,
                                                 dataset="small")
    predict_label = np.load('cache/test_predict2_class_4.npy')
    X = tf.placeholder(dtype=tf.float32, shape=test_label.shape)
    Y = tf.placeholder(dtype=tf.float32, shape=test_label.shape)
    X_sigmoid = tf.nn.sigmoid(X)
    X_softmax = tf.nn.softmax(X_sigmoid, axis=0)
    sess = tf.InteractiveSession()
    verify = tf.cast(tf.equal(tf.argmax(X_softmax, axis=1), tf.argmax(Y, axis=1)),dtype=tf.float32)
    accuracy = tf.reduce_mean(verify)
    print(sess.run(accuracy, feed_dict={X: predict_label, Y: test_label}))
    
def image_reconstruct():
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
# image_reconstruct()
tf_accuracy()


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
        img = (255 * skimage.util.random_noise(img)).astype('uint8')

        image = make_image(img)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
        writer = tf.summary.FileWriter('./logs')
        writer.add_summary(summary, epoch)
        writer.close()

        return

tbi_callback = TensorBoardImage('Image Example')
'''