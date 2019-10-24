import keras
import keras.backend as K
import keras.layers as KL
import keras.models as KM

def vgg_conv_core(input_tensor, filter_num, stage_num, block_num):

    x = KL.Conv2D(filter_num, (3, 3), padding='same', name='block%d_conv%d'%(stage_num, block_num))(input_tensor)
    x = KL.BatchNormalization(name='block%d_bn%d'%(stage_num, block_num))(x)
    x = KL.Activation('relu')(x)
    return x

def vgg_conv_block(input_tensor, filter_num, block_count, stage):
    
    x = vgg_conv_core(input_tensor, filter_num, stage, 1)

    for i in range(block_count-1):
        x = vgg_conv_core(x, filter_num, stage, 2 + i)
        
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block%d_pool'%stage)(x)

    return x

def vgg_graph(classes=80, architecture = "vgg16"):

    base_size = 224

    input_shape = (base_size, base_size, 3)

    MAX_BC = {"vgg16": 3, "vgg19": 4}[architecture]

    img_input = KL.Input(input_shape, dtype="float32")
 
    # Block 1
    x = vgg_conv_block(img_input, filter_num = 64, block_count = 2, stage = 1)
    # Block 2
    x = vgg_conv_block(x, filter_num = 128, block_count = 2, stage = 2)
    # Block 3
    x = vgg_conv_block(x, filter_num = 256, block_count = MAX_BC, stage = 3)
    # Block 4
    x = vgg_conv_block(x, filter_num = 512, block_count = MAX_BC, stage = 4)
    # Block 5
    x = vgg_conv_block(x, filter_num = 512, block_count = MAX_BC, stage = 5)

    x = KL.Flatten(name='flatten')(x)
    x = KL.Dense(4096, name='fc1')(x)
    x = KL.BatchNormalization(name='fc1_bn')(x)
    x = KL.Activation('relu')(x)
    x = KL.Dropout(0.5)(x)

    x = KL.Dense(4096, name='fc2')(x)
    x = KL.BatchNormalization(name='fc2_bn')(x)
    x = KL.Activation('relu')(x)
    x = KL.Dropout(0.5)(x)

    x = KL.Dense(classes, activation='softmax', name='predictions')(x)

     # Create model.
    model = KM.Model(img_input, x, name='subnetvgg')

    return model, base_size


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: kernel size of middle conv layer at main path
        filters: list of filters of 3 conv layer at main path
        stage: current stage label, used for generating layer names
        block: current block label, used for generating layer names
    """
    flt_size1, flt_size2, flt_size3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(flt_size1, (1, 1), strides=strides,
                  name=conv_name_base + '2a')(input_tensor)
    x = KL.BatchNormalization(name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(flt_size2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b')(x)
    x = KL.BatchNormalization(name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(flt_size3, (1, 1), name=conv_name_base +
                  '2c')(x)
    x = KL.BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2D(flt_size3, (1, 1), strides=strides,
                         name=conv_name_base + '1')(input_tensor)
    shortcut = KL.BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names        
    """
    flt_size1, flt_size2, flt_size3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'


    x = KL.Conv2D(flt_size1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = KL.BatchNormalization(name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(flt_size2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b')(x)
    x = KL.BatchNormalization(name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(flt_size3, (1, 1), name=conv_name_base + '2c')(x)
    x = KL.BatchNormalization(name=bn_name_base + '2c')(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(classes = 80, architecture="resnet50"):
    """Build a ResNet graph.
        architecture: can be resnet50 or resnet101              
    """

    base_size = 256
    input_shape = (base_size, base_size, 3)
    input_image = KL.Input(input_shape, dtype="float32")
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = KL.BatchNormalization(name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # Stage 3

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))
    
    # Stage 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    x = KL.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x = KL.Flatten()(x)
    x = KL.Dense(classes, activation='softmax')(x)

    model = KM.Model(input_image, x, name='resnet')
    return model, base_size


