from __future__ import division, print_function
import caffe
import numpy as np
import os

DATA_DIR = "/path/to/my/data"
OUTPUT_DIR = os.getcwd()




MODEL_PROTO = '/home/niranjan/working/RetinaSegmentation/model/train.prototxt'
MODEL_WEIGHTS = '/home/niranjan/working/RetinaSegmentation/model/train_start.caffemodel'


caffe.set_mode_cpu()
net = caffe.Net(MODEL_PROTO, MODEL_WEIGHTS, caffe.TEST)

# layer names and output shapes
print(net.blobs)
"""
for layer_name, blob in net.blobs.iteritems():
    print(layer_name, blob.data.shape)
"""

"""
for i in net.blobs:
    print(i, net.blobs[i].data.shape)
"""

# write out weight matrices and bias vectors
for k, v in net.params.items():
    print(k, v[0].data.shape, v[1].data.shape)
    # np.save(os.path.join(OUTPUT_DIR, "W_{:s}.npy".format(k)), v[0].data)
    # np.save(os.path.join(OUTPUT_DIR, "b_{:s}.npy".format(k)), v[1].data)

exit()
# write out mean image
blob = caffe.proto.caffe_pb2.BlobProto()
with open(MEAN_IMAGE, 'rb') as fmean:
    mean_data = fmean.read()
blob.ParseFromString(mean_data)
mu = np.array(caffe.io.blobproto_to_array(blob))
print("Mean image:", mu.shape)
np.save(os.path.join(OUTPUT_DIR, "mean_image.npy"), mu)