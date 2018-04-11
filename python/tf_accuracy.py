import tensorflow as tf
import numpy as np
import prepare_image

test_label = prepare_image.load_images(data_type="test", image_type="label", classification=4,
                                             dataset="small")
predict_label = np.load('cache/test_predict2_class_4.npy')
X = tf.placeholder(dtype=tf.float32, shape=test_label.shape)
Y = tf.placeholder(dtype=tf.float32, shape=test_label.shape)
X_sigmoid = tf.nn.sigmoid(X)
X_softmax = tf.nn.softmax(X_sigmoid, axis=1)
sess = tf.InteractiveSession()
verify = tf.cast(tf.equal(tf.argmax(X_softmax, axis=1), tf.argmax(Y, axis=1)),dtype=tf.float32)
accuracy = tf.reduce_mean(verify)
print(sess.run(accuracy, feed_dict={X: predict_label, Y: test_label}))