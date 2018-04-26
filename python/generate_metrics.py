import tensorflow as tf
import numpy as np
import prepare_image
from scipy.special import expit
from sklearn.metrics import f1_score, accuracy_score, precision_score, \
    recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def tf_accuracy(predict_data, test_data):
    X = tf.constant(predict_data, dtype=tf.float32, shape=test_data.shape)
    Y = tf.constant(test_data, dtype=tf.float32, shape=test_data.shape)
    sess = tf.InteractiveSession()

    verify = tf.cast(tf.equal(tf.argmax(X, axis=1), tf.argmax(Y, axis=1)), dtype=tf.float32)
    accuracy = tf.reduce_mean(verify)

    b_p = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(X, axis=1), 3), dtype=tf.float32))
    b_t = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(Y, axis=1), 3), dtype=tf.float32))

    a_p = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(X, axis=1), 0), dtype=tf.float32))
    a_t = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(Y, axis=1), 0), dtype=tf.float32))

    o_p = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(X, axis=1), 1), dtype=tf.float32))
    o_t = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(Y, axis=1), 1), dtype=tf.float32))

    v_p = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(X, axis=1), 2), dtype=tf.float32))
    v_t = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(Y, axis=1), 2), dtype=tf.float32))
    
    print("Accuracy of Background is {}".format(sess.run(1 - abs(b_t-b_p)/b_t)))
    print("Accuracy of Arteries is {}".format(sess.run(1 - abs(a_t - a_p) / a_t)))
    print("Accuracy of Veins is {}".format(sess.run(1 - abs(v_t - v_p) / v_t)))
    print("Accuracy of Overlap is {}".format(sess.run(1 - abs(o_t - o_p) / o_t)))
    print("Overall accuracy is {}".format(sess.run(accuracy)))


def np_f1score(predict_data, test_data):
    predict_data = softmax(expit(predict_data.transpose(1, 2, 3, 0))).transpose(3, 0, 1, 2)
    print(np.argmax(test_data, axis=1).shape)
    predict = np.argmax(predict_data, axis=1).reshape(10 * 565 * 565)
    test = np.argmax(test_data, axis=1).reshape(10 * 565 * 565)
    f1_s = f1_score(test, predict, average=None)
    f1_s_w = f1_score(test, predict, average='weighted')
    precision_s = precision_score(test, predict, average=None)
    precision_s_w = precision_score(test, predict, average='weighted')
    recall_s = recall_score(test, predict, average=None)
    recall_s_w = recall_score(test, predict, average='weighted')
    conf_matrix = confusion_matrix(test, predict)
    """
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, ['a','o','v','b'], rotation=45)
    plt.yticks(tick_marks, ['a','o','v','b'])
    plt.show()
    """
    print("Precision of A,O,V,B is {}".format(precision_s))
    print("Overall Precision is {}".format(precision_s_w))
    print("Recall of A,O,V,B is {}".format(recall_s))
    print("Overall Recall is {}".format(recall_s_w))
    print("F1-Score of A,O,V,B {}".format(f1_s))
    print("Overall F-1 Score is {}".format(f1_s_w))
    print("Confusion matrix is {}".format(conf_matrix))

if __name__ == "__main__":
    predict_data = np.load('cache/test_predict2_class_4_dev2_relu.npy')
    test_data = prepare_image.load_images(data_type="test", image_type="label", classification=4,
                                            dataset="small")
    tf_accuracy(predict_data=predict_data, test_data= test_data)
    np_f1score(predict_data=predict_data, test_data= test_data)



