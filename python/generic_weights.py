import h5py
import json
from keras.models import Sequential, Model, model_from_json

file = h5py.File("cache/keras_crop_model_weights.h5")
with open("cache/model.json") as f:
    model = model_from_json(json.dumps(json.load(f)))


# print(dir(file))
# print(file.get('conv1_1'))
# print(list(file.attrs.items()))
"""
print(list(file.keys()))

#model.set_weights(file[])
print(model.layers)
print(model.layers[2].name)
print(dir(model.layers[2]))
"""

for layer in model.layers:
    print(layer.name)
    layer.set_weights(layer.get_weights())
    """
    print(layer.name)
    if file.get(layer.name, None):
        print(dir(file.get(layer.name).values()))
        print(file.get(layer.name))
        # layer.set_weights(list(file.get(layer.name, None)))
    """