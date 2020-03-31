from keras.layers import *
from keras.models import Model,Sequential
from keras.regularizers import l2


class MiniException:

    @staticmethod
    def build(height,width,channels,classes):
        img_input = Input(shape=(height,width,channels))
        x = Conv2D(8,(3,3),strides=(1,1),kernel_regularizer=l2(l=0.01),use_bias=False)(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(8,(3,3),strides=(1,1),kernel_regularizer=l2(l=0.01),use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        #module1
        residual = Conv2D(16,(1,1),strides=(2,2),kernel_regularizer=l2(l=0.01),use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(16,(3,3),padding = 'same', kernel_regularizer=l2(l=0.01),use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(16,(3,3),padding='same',kernel_regularizer=l2(l=0.01),use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = add([x,residual])

        #module2
        residual = Conv2D(32,(1,1),strides=(2,2),kernel_regularizer=l2(l=0.01),use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(32,(3,3),padding = 'same', kernel_regularizer=l2(l=0.01),use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(32,(3,3),padding='same',kernel_regularizer=l2(l=0.01),use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = add([x,residual])

        #module3
        residual = Conv2D(64,(1,1),strides=(2,2),kernel_regularizer=l2(l=0.01),use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(64,(3,3),padding = 'same', kernel_regularizer=l2(l=0.01),use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(64,(3,3),padding='same',kernel_regularizer=l2(l=0.01),use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = add([x,residual])

        #module4
        residual = Conv2D(128,(1,1),strides=(2,2),kernel_regularizer=l2(l=0.01),use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(128,(3,3),padding = 'same', kernel_regularizer=l2(l=0.01),use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(128,(3,3),padding='same',kernel_regularizer=l2(l=0.01),use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = add([x,residual])

        x = Conv2D(classes, (3, 3), padding='same')(x)
        x = GlobalAveragePooling2D()(x)
        output = Activation('softmax',name='predictions')(x)
 
        model = Model(img_input, output)
        model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
        return model

