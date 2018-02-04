from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input, concatenate, merge, Add
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras import backend as K
from keras.utils import plot_model

if __name__ == "__main__":
    input_shape = (1, 584, 565)
    K.set_image_data_format("channels_first")
    """
    model = Sequential()
    model.add(Conv2D(64,kernel_size=( 3, 3),activation='relu',input_shape="input_shape"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',name="conv_12"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(128, k ernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    plot_model(model, to_file='model.png')
    """

    data_input = Input(shape=input_shape, name="data_input", dtype="float32")

    conv1_1 = Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv1_1")(data_input)
    conv1_2 = Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv1_2")(conv1_1)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pool1')(conv1_2)

    conv2_1 = Conv2D(128, kernel_size=(3, 3), activation='relu', name="conv2_1")(max_pool1)
    conv2_2 = Conv2D(128, kernel_size=(3, 3), activation='relu', name="conv2_2")(conv2_1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2_2)

    conv3_1 = Conv2D(256, kernel_size=(3, 3), activation='relu', name="conv3_1")(max_pool2)
    conv3_2 = Conv2D(256, kernel_size=(3, 3), activation='relu', name="conv3_2")(conv3_1)
    max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3_2)

    conv4_1 = Conv2D(512, kernel_size=(3, 3), activation='relu', name="conv4_1")(max_pool3)
    conv4_2 = Conv2D(512, kernel_size=(3, 3), activation='relu', name="conv4_2")(conv4_1)
    conv4_3 = Conv2D(512, kernel_size=(3, 3), activation='relu', name="conv4_3")(conv4_2)

    conv1_2_16 = Conv2D(16, kernel_size=(3, 3), name="conv1_2_16")(conv1_2)
    conv2_2_16 = Conv2D(16, kernel_size=(3, 3), name="conv2_2_16")(conv2_2)
    conv3_2_16 = Conv2D(16, kernel_size=(3, 3), name="conv3_2_16")(conv3_2)
    conv4_3_16 = Conv2D(16, kernel_size=(3, 3), name="conv4_3_16")(conv4_3)

    upsample2_ = Conv2DTranspose(16, kernel_size=(4, 4), strides=(2, 2), padding="VALID", name="upsample2_")(conv2_2_16)

    upsample4_ = Conv2DTranspose(16, kernel_size=(8, 8), strides=(4, 4),padding="VALID", name="upsample4_")(conv3_2_16)
    # crop3 = Cropping2D(name="crop2")([data_input, upsample4_])

    upsample4_ = ZeroPadding2D((9, 9))(upsample4_)

    upsample8_ = Conv2DTranspose(16, kernel_size=(16, 16), strides=(8, 8),padding="VALID", name="upsample8_")(conv4_3_16)
    # crop3 = Cropping2D(name="crop2")([data_input, upsample8_])
    upsample8_ = ZeroPadding2D((37, 35))(upsample8_)

    #concat_layer = concatenate([upsample2_, upsample4_, upsample8_], name="concat_layer")
    #weighting_av = Conv2D(3, kernel_size=(1, 1), name="weighting_av")(concat_layer)

    # loss_layer = K.nn.sparse_softmax_cross_entropy_with_logits(name='loss_layer')(weighting_av)

    model = Model(inputs=[data_input], outputs=[upsample8_,upsample4_,upsample2_])

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    plot_model(model, to_file='model.png')

    print(model.get_config())
    print(model.summary())

    print("Hello")
