def transform_conv_weight(W):
    # for non FC layers, do this because Keras does convolution vs Caffe correlation
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = np.rot90(W[i, j], 2)
    return W

def transform_fc_weight(W):
    return W.T

# load weights
CAFFE_WEIGHTS_DIR = "/home/nir0303/working/RetinaSegmentation/model/"

W_conv1 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv1.npy")))
b_conv1 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv1.npy"))

W_conv2 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv2.npy")))
b_conv2 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv2.npy"))

W_conv3 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv3.npy")))
b_conv3 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv3.npy"))

W_conv4 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv4.npy")))
b_conv4 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv4.npy"))

W_conv5 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv5.npy")))
b_conv5 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv5.npy"))

W_fc6 = transform_fc_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_fc6.npy")))
b_fc6 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_fc6.npy"))

W_fc7 = transform_fc_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_fc7.npy")))
b_fc7 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_fc7.npy"))

W_fc8 = transform_fc_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_fc8.npy")))
b_fc8 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_fc8.npy"))