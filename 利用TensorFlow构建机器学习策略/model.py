import tensorflow as tf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers

class Model:
    def __init__(self, FACTORS_NAME, INPUT_SHAPE):
        super().__init__()

        self.INPUT_SHAPE = INPUT_SHAPE
        #self.FACTORS_NUM = FACTORS_NUM
        self.FACTORS_NAME = FACTORS_NAME
        self.FACTORS_LENGTH = len(FACTORS_NAME)
        
        

    def build_lstm(self):
        input_layers = []
        features = []
        for fi in self.FACTORS_NAME:
            il = tf.keras.Input(shape=(self.INPUT_SHAPE, 1), name=fi)
            lstm = layers.LSTM(64)(il)
           
            input_layers.append(il)
            features.append(lstm)

        x = layers.concatenate(features)
        dense_0 = layers.Dense(128, activation='sigmoid')(x)
        dense_1 = layers.Dense(64, activation='sigmoid')(dense_0)
        reward_pred = layers.Dense(1, activation='sigmoid', name='priority')(dense_1)

        #department_pred = layers.Dense(num_departments, activation='softmax', name='department')(x)

        self.model = tf.keras.Model(inputs=input_layers,
                                    outputs=[reward_pred])

        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),
                           loss=['mean_squared_logarithmic_error'],
                           loss_weights=[0.2])

        self.model.summary()
        self.show_model_graph()

    def show_model_graph(self):

        tf.keras.utils.plot_model(self.model, './model_graph/multi_input_and_output_model.png', show_shapes=True)

    def train(self, train_data, test_data):
        self.model.fit(train_data,
                       verbose=1,
                       validation_data = test_data,
                       validation_steps = 500,
                       #validation_data=validation_data,
                       steps_per_epoch=2000,
                       epochs=100)

    def test(self, test_data, show_fig=False):

        result = self.model.predict(test_data[0])

        if show_fig:
            plt.plot(result)
            plt.plot(test_data[1]) 
            plt.savefig()

        return result

if __name__ == "__main__":
    INPUT_SHAPE = 128
    FACTOR_INFO = [{name: 'barBoll'}]
    
    model = Model(INPUT_SHAPE, FACTOR_INFO)