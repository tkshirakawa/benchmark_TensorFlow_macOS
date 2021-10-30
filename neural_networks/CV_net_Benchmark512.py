'''
    Copyright (c) 2021, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com

    Released under the MIT license.
    https://opensource.org/licenses/mit-license.php
'''


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, LeakyReLU, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, SeparableConv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout




####################################################################################################
####    Descriptions and definitions
####################################################################################################

def Model_Name(): return 'CV-net BENCHMARK'

def Model_Description(): return 'Neural network model for A.I.Segmentation plugin\n\
                            CV-net BENCHMARK is a neural network model for a benchmark test\n\
                            Copyright (c) 2021, Takashi Shirakawa\n\
                            URL: https://compositecreatures.jimdofree.com/a-i-segmentation/'

def Batch_Size(): return 16

def Input_Shape(): return (512, 512, 1)

def Class_Number(): return 1

def Learning_Rate_Parameters():
    LR_params = {'formula'          : ['poly', 0.25, 2],            # Learning rate formula calculates LR at points of epochs - ['poly', base_lr, number_of_epochs] is available
                 'graph'            : [[0,0.0], [1,0.0]],           # Learning rate graph defines LR at points of epochs - [[epoch_1, LR_1], [epoch_2, LR_2], ... [epoch_last, LR_last]]
                 'step'             : [0.1, 5.0],                   # Multiplying values to LR - will be applied when mIoU is [NOT improved, improved]
                 'limit'            : [0.01, 1.0] ,                 # Limitation of LR multiplier - when [NOT improved, improved]
                 'patience'         : [100, 100],                   # Patience counts before applying step for LR - when [NOT improved, improved]
                 'stop_count'       : 50 }                          # Define a count number before early stopping
    return LR_params




####################################################################################################
####    Main neural network
####################################################################################################

def SynapticNeuronUnit(neuro_potential, filter_size, kernel_size, CRP, d_rate):

    if CRP[1] == 'UpSampling': neuro_potential = UpSampling2D(interpolation='bilinear')(neuro_potential)


    # Main neural potential
    if CRP[0] == 'Normal':
        neuro_potential = Conv2D(   filters             = filter_size,
                                    kernel_size         = kernel_size,
                                    padding             = CRP[2],
                                    kernel_initializer  = 'he_uniform',
                                    use_bias            = False)(neuro_potential)

    elif CRP[0] == 'Transpose':
        neuro_potential = Conv2DTranspose(  filters             = filter_size,
                                            kernel_size         = kernel_size,
                                            padding             = CRP[2],
                                            kernel_initializer  = 'he_uniform',
                                            use_bias            = False)(neuro_potential)

    elif CRP[0] == 'Separable':
        neuro_potential = SeparableConv2D(  filters                 = filter_size,
                                            kernel_size             = kernel_size,
                                            padding                 = CRP[2],
                                            depthwise_initializer   = 'he_uniform',
                                            pointwise_initializer   = 'he_uniform',
                                            use_bias                = False)(neuro_potential)

    elif CRP[0] == 'Atrous':
        neuro_potential = Conv2D(   filters             = filter_size,
                                    kernel_size         = kernel_size,
                                    strides             = 2,
                                    padding             = CRP[2],
                                    kernel_initializer  = 'he_uniform',
                                    use_bias            = False)(neuro_potential)
        neuro_potential = ZeroPadding2D(padding=((1, 0), (1, 0)))(neuro_potential)

    else:
        neuro_potential = None      # Will be error

    neuro_potential = BatchNormalization(momentum=0.95)(neuro_potential)
    neuro_potential = LeakyReLU(alpha=0.2)(neuro_potential)


    # Output potential to axons
    if CRP[1] == 'MaxPooling': neuro_potential = MaxPooling2D()(neuro_potential)


    return Dropout(rate=d_rate)(neuro_potential)




'''
    Build and return Keras model

    Input/output images are grayscale = 1 channel per pixel.
    Type of the pixels is float normalized between 0.0 to 1.0.
    (Please note that the pixel values are NOT 8-bit unsigned char ranging between 0 to 255)

    Dimensions
    OpenCV : HEIGHT x WIDTH
    Keras  : HEIGHT x WIDTH x CHANNEL
'''
def Build_Model():

    # Convolution methods for SynapticNeuronUnit layers
    # _ENC = ('Normal', 'MaxPooling', 'same')
    _ENC = ('Atrous', 'None', 'valid')
    _NNv = ('Normal', 'None', 'valid')
    _SNs = ('Separable', 'None', 'same')
    _SNv = ('Separable', 'None', 'valid')
    _TUs = ('Transpose', 'UpSampling', 'same')


    # Do not change "name='input'", because the name is used to identify the input layer in A.I.Segmentation.
    inputs = Input(shape=(512, 512, 1), name='input')


    # Cerate stimulation from inputs
    stimulation = Conv2D(filters=8, kernel_size=3, padding='same', kernel_initializer='glorot_uniform', use_bias=False)(inputs)    # 512
    stimulation = BatchNormalization(momentum=0.95)(stimulation)
    stimulation = Activation('sigmoid')(stimulation)                # Skip connection


    # Main neural network
    # def SynapticNeuronUnit(dendrites, filter_size, kernel_size, CRP, d_rate)
    enc_potential_01  = SynapticNeuronUnit(stimulation,         8,  3, _ENC, 0.25)     # 512 -> 256
    enc_potential_01A = SynapticNeuronUnit(enc_potential_01,    8,  1, _NNv, 0.50)
    enc_potential_01B = SynapticNeuronUnit(enc_potential_01,    8,  5, _SNs, 0.50)
    enc_potential_01  = Concatenate()([enc_potential_01A, enc_potential_01B])          # Skip connection

    enc_potential_02  = SynapticNeuronUnit(enc_potential_01,   16,  3, _ENC, 0.25)     # 256 -> 128
    enc_potential_02A = SynapticNeuronUnit(enc_potential_02,   16,  1, _NNv, 0.50)
    enc_potential_02B = SynapticNeuronUnit(enc_potential_02,   16,  5, _SNs, 0.50)
    enc_potential_02  = Concatenate()([enc_potential_02A, enc_potential_02B])          # Skip connection

    enc_potential_03  = SynapticNeuronUnit(enc_potential_02,   32,  3, _ENC, 0.25)     # 128 -> 64
    enc_potential_03A = SynapticNeuronUnit(enc_potential_03,   32,  1, _NNv, 0.50)
    enc_potential_03B = SynapticNeuronUnit(enc_potential_03,   32,  5, _SNs, 0.50)
    enc_potential_03  = Concatenate()([enc_potential_03A, enc_potential_03B])          # Skip connection

    enc_potential_04  = SynapticNeuronUnit(enc_potential_03,  128,  3, _ENC, 0.25)      # 64 -> 32
    enc_potential_04A = SynapticNeuronUnit(enc_potential_04,  128,  1, _NNv, 0.50)
    enc_potential_04B = SynapticNeuronUnit(enc_potential_04,  128,  5, _SNs, 0.50)
    enc_potential_04  = Concatenate()([enc_potential_04A, enc_potential_04B])          # Skip connection

    enc_potential_05  = SynapticNeuronUnit(enc_potential_04,  512,  3, _ENC, 0.25)      # 32 -> 16
    enc_potential_05A = SynapticNeuronUnit(enc_potential_05,  512,  1, _NNv, 0.50)
    enc_potential_05B = SynapticNeuronUnit(enc_potential_05,  512,  5, _SNs, 0.50)
    enc_potential_05  = Concatenate()([enc_potential_05A, enc_potential_05B])          # Skip connection

    deep_potential    = SynapticNeuronUnit(enc_potential_05, 1024,  5, _SNv, 0.25)      # 16 -> 12
    deep_potential    = SynapticNeuronUnit(deep_potential,   2048,  5, _SNv, 0.25)      # 12 -> 8

    dec_potential_05  = SynapticNeuronUnit(deep_potential,    512,  3, _TUs, 0.25)      # 8 -> 16
    dec_potential_05  = Concatenate()([dec_potential_05, enc_potential_05])

    dec_potential_04  = SynapticNeuronUnit(dec_potential_05,  384,  3, _TUs, 0.25)      # 16 -> 32
    dec_potential_04  = Concatenate()([dec_potential_04, enc_potential_04])

    dec_potential_03  = SynapticNeuronUnit(dec_potential_04,  160,  3, _TUs, 0.25)      # 32 -> 64
    dec_potential_03  = Concatenate()([dec_potential_03, enc_potential_03])

    dec_potential_02  = SynapticNeuronUnit(dec_potential_03,   56,  3, _TUs, 0.25)     # 64 -> 128
    dec_potential_02  = Concatenate()([dec_potential_02, enc_potential_02])

    dec_potential_01  = SynapticNeuronUnit(dec_potential_02,   24,  3, _TUs, 0.25)     # 128 -> 256
    dec_potential_01  = Concatenate()([dec_potential_01, enc_potential_01])

    axon_potential    = SynapticNeuronUnit(dec_potential_01,   24,  3, _TUs, 0.25)     # 256 -> 512
    axon_potential    = Concatenate()([axon_potential, stimulation])


    # The vision from synaptic neurons
    vision = Conv2D(filters=32, kernel_size=3, kernel_initializer='he_uniform', use_bias=False)(axon_potential)
    vision = BatchNormalization(momentum=0.95)(vision)
    vision = LeakyReLU(alpha=0.2)(vision)

    vision = Conv2DTranspose(filters=16, kernel_size=3, kernel_initializer='he_uniform', use_bias=False)(vision)
    vision = BatchNormalization(momentum=0.95)(vision)
    vision = LeakyReLU(alpha=0.2)(vision)

    vision = Concatenate()([vision, inputs])


    # Do not change "name='output'", because the name is used to identify the output layer in A.I.Segmentation.
    outputs = Conv2D(filters=1, kernel_size=1, kernel_initializer='glorot_uniform')(vision)
    outputs = Activation('sigmoid', name='output')(outputs)


    # Generate model
    return Model(inputs=inputs, outputs=outputs)



