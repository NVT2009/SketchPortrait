from keras.models import Model
from keras.layers import Input, ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout

def vgg_face(shape, weights_path=None):
    image_tensor = Input(shape=shape)

    pad1_1 = ZeroPadding2D(padding=(1,1)) (image_tensor)
    conv1_1 = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(pad1_1)
    pad1_2 = ZeroPadding2D(padding=(1,1))(conv1_1)
    conv1_2 = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(pad1_2)
    pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1_2)

    pad2_1 = ZeroPadding2D(padding=(1,1))(pool1)
    conv2_1 = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(pad2_1)
    pad2_2 = ZeroPadding2D(padding=(1, 1))(conv2_1)
    conv2_2 = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(pad2_2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2)

    pad3_1 = ZeroPadding2D(padding=(1, 1))(pool2)
    conv3_1 = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(pad3_1)
    pad3_2 = ZeroPadding2D(padding=(1, 1))(conv3_1)
    conv3_1 = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(pad3_2)
    pad3_3 = ZeroPadding2D(padding=(1, 1))(conv3_1)
    conv3_3 = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(pad3_3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3_3)

    pad4_1 = ZeroPadding2D(padding=(1, 1))(pool3)
    conv4_1 = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(pad4_1)
    pad4_2 = ZeroPadding2D(padding=(1, 1))(conv4_1)
    conv4_1 = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(pad4_2)
    pad4_3 = ZeroPadding2D(padding=(1, 1))(conv4_1)
    conv4_3 = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(pad4_3)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_3)

    pad5_1 = ZeroPadding2D(padding=(1, 1))(pool4)
    conv5_1 = Convolution2D(512, 3, 3, activation='relu', name='conv5_1')(pad5_1)
    pad5_2 = ZeroPadding2D(padding=(1, 1))(conv5_1)
    conv5_1 = Convolution2D(512, 3, 3, activation='relu', name='conv5_2')(pad5_2)
    pad5_3 = ZeroPadding2D(padding=(1, 1))(conv5_1)
    conv5_3 = Convolution2D(512, 3, 3, activation='relu', name='conv5_3')(pad5_3)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5_3)

    flat = Flatten()(pool5)
    fc6 = Dense(4096, activation='relu', name='fc6')(flat)
    fc6_drop = Dropout(0.4)(fc6)
    fc7 = Dense(4096, activation='relu',name='fc7')(fc6_drop)
    fc7_drop = Dropout(0.4)(fc7)
    out = Dense(1000, activation='softmax', name='fc8')(fc7_drop)

    model = Model(input=image_tensor, output=out)

    if weights_path:
        model.load_weights(weights_path)
    return model

if __name__ == '__main__':
    img = (256,256,3)
    output = vgg_face(img)
    print(output.summary())