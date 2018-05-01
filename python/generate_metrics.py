import tensorflow as tf
import numpy as np
import itertools
import prepare_image
from scipy.special import expit
from sklearn.metrics import f1_score, accuracy_score, precision_score, \
    recall_score, roc_auc_score, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import scikitplot as skplt
import sklearn
from PIL import Image
from sklearn.metrics import precision_recall_curve


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def tf_accuracy(predict_data, test_data):
    predict = tf.constant(predict_data, dtype=tf.float32, shape=test_data.shape)
    test = tf.constant(test_data, dtype=tf.float32, shape=test_data.shape)
    sess = tf.InteractiveSession()

    verify = tf.cast(tf.equal(tf.argmax(predict, axis=1), tf.argmax(test, axis=1)), dtype=tf.float32)
    accuracy = tf.reduce_mean(verify)

    b_p = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predict, axis=1), 3), dtype=tf.float32))
    b_t = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(test, axis=1), 3), dtype=tf.float32))

    a_p = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predict, axis=1), 0), dtype=tf.float32))
    a_t = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(test, axis=1), 0), dtype=tf.float32))

    o_p = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predict, axis=1), 1), dtype=tf.float32))
    o_t = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(test, axis=1), 1), dtype=tf.float32))

    v_p = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predict, axis=1), 2), dtype=tf.float32))
    v_t = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(test, axis=1), 2), dtype=tf.float32))
    
    print("Accuracy of Background is {}".format(sess.run(1 - abs(b_t-b_p)/b_t)))
    print("Accuracy of Arteries is {}".format(sess.run(1 - abs(a_t - a_p) / a_t)))
    print("Accuracy of Veins is {}".format(sess.run(1 - abs(v_t - v_p) / v_t)))
    print("Accuracy of Overlap is {}".format(sess.run(1 - abs(o_t - o_p) / o_t)))
    print("Overall accuracy is {}".format(sess.run(accuracy)))


def np_f1score(predict_data, test_data):
    print(len(predict_data))
    predict = np.argmax(predict_data, axis=1).reshape(len(predict_data) * 565 * 565)
    test = np.argmax(test_data, axis=1).reshape(len(predict_data) * 565 * 565)
    f1_s = f1_score(test, predict, average=None)
    f1_s_w = f1_score(test, predict, average='weighted')
    precision_s = precision_score(test, predict, average=None)
    precision_s_w = precision_score(test, predict, average='weighted')
    recall_s = recall_score(test, predict, average=None)
    recall_s_w = recall_score(test, predict, average='weighted')
    print("Precision of A,O,V,B is {}".format(precision_s))
    print("Overall Precision is {}".format(precision_s_w))
    print("Recall of A,O,V,B is {}".format(recall_s))
    print("Overall Recall is {}".format(recall_s_w))
    print("F1-Score of A,O,V,B {}".format(f1_s))
    print("Overall F-1 Score is {}".format(f1_s_w))


def plot_confusion_matrix(predict_data, test_data):
    predict = np.argmax(predict_data, axis=1).reshape(len(predict_data) * 565 * 565)
    test = np.argmax(test_data, axis=1).reshape(len(predict_data) * 565 * 565)
    conf_matrix = confusion_matrix(test, predict)
    print("Confusion matrix is {}".format(conf_matrix))
    plt.imshow(conf_matrix, interpolation='bessel', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, ['arteries', 'overlap', 'veins', 'background'], rotation=45)
    plt.yticks(tick_marks, ['arteries', 'overlap', 'veins', 'background'])

    thresh = conf_matrix.max() / 4.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, conf_matrix[i, j], horizontalalignment="center",
                  color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("data/document_images/confusion_matrix.png")


def plot_precision_recall(predict_data, test_data):
    precision = dict()
    recall = dict()
    average_precision = dict()
    print(test_data.shape)
    for i in range(4):
        precision[i], recall[i], _ = precision_recall_curve(test_data[i, ...],
                                                             predict_data[i, ...])
        average_precision[i] = average_precision_score(test_data[i, ...], predict_data[i, ...])

if __name__ == "__main__":
    # predict_data = np.load('cache/test_predict2_class_4_relu.npy')
    predict_data = np.load('cache/test_predict2_class_4_dev2_relu.npy')
    # predict_data = np.load ('cache/test_predict2_class_4_soft_relu.npy')
    test_data = prepare_image.load_images(data_type="test", image_type="label", classification=4,
                                            dataset="small")
    tf_accuracy(predict_data=predict_data, test_data= test_data)
    np_f1score(predict_data=predict_data, test_data= test_data)
    plot_confusion_matrix(predict_data=predict_data, test_data= test_data)
    """
    # plot_precision_recall(predict_data=predict_data, test_data= test_data)
    print(test_data.shape)
    predict = np.argmax(predict_data, axis=1).reshape(len(predict_data) * 565 * 565)
    test = np.argmax(test_data, axis=1).reshape(len(predict_data) * 565 * 565)
    print(sklearn.metrics.roc_curve(test,predict,pos_label=4))
    print(sklearn.metrics.precision_recall_curve(test, predict, pos_label=4))
    """



