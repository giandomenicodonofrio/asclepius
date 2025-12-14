from reference.attention_with_context import AttentionWithContext

import keras
from keras.layers import Bidirectional, LeakyReLU, BatchNormalization
from keras.layers import Dense, Dropout, Input, GRU, Convolution1D

def get_reference_architecture(n_block = 5):
    
    seq = [Input(shape=(72000,12), dtype='float32', name='main_input'),

    Convolution1D(12, 3, padding='same'),
    LeakyReLU(alpha=0.3),
    Convolution1D(12, 3, padding='same'),
    LeakyReLU(alpha=0.3),
    Convolution1D(12, 24, strides = 2, padding='same'),
    LeakyReLU(alpha=0.3),
    Dropout(0.2),


    Convolution1D(12, 3, padding='same'),
    LeakyReLU(alpha=0.3),
    Convolution1D(12, 3, padding='same'),
    LeakyReLU(alpha=0.3),
    Convolution1D(12, 24, strides = 2, padding='same'),
    LeakyReLU(alpha=0.3),
    Dropout(0.2),


    Convolution1D(12, 3, padding='same'),
    LeakyReLU(alpha=0.3),
    Convolution1D(12, 3, padding='same'),
    LeakyReLU(alpha=0.3),
    Convolution1D(12, 24, strides = 2, padding='same'),
    LeakyReLU(alpha=0.3),
    Dropout(0.2),


    Convolution1D(12, 3, padding='same'),
    LeakyReLU(alpha=0.3),
    Convolution1D(12, 3, padding='same'),
    LeakyReLU(alpha=0.3),
    Convolution1D(12, 24, strides = 2, padding='same'),
    LeakyReLU(alpha=0.3),
    Dropout(0.2),


    Convolution1D(12, 3, padding='same'),
    LeakyReLU(alpha=0.3),
    Convolution1D(12, 3, padding='same'),
    LeakyReLU(alpha=0.3),
    Convolution1D(12, 48, strides = 2, padding='same'),
    LeakyReLU(alpha=0.3),
    Dropout(0.2),


    Bidirectional(GRU(12, input_shape=(2250,12),return_sequences=True,return_state=False)),
    LeakyReLU(alpha=0.3),
    Dropout(0.2),
    AttentionWithContext(),
    BatchNormalization(),
    LeakyReLU(alpha=0.3),
    Dropout(0.2),
    Dense(9,activation='sigmoid')]

    return keras.Sequential(seq)

def compile_architecture(arc, learning_rate=0.001, epsilon=1e-07):
    arc.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon),
        loss="categorical_crossentropy",
        metrics=["accuracy"])
    return arc

def get_compiled_architecture(learning_rate=0.001, epsilon=1e-07, n_block=5):
    arc = get_reference_architecture(n_block)
    arc.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon),
      loss="categorical_crossentropy",
      metrics=["accuracy"])
    return arc