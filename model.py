import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class SingingVoiceSeparationModel:
    def __init__(self, num_features=513, num_hidden_units=256, num_rnn_layer=3):
        self.num_features = num_features
        self.num_hidden_units = num_hidden_units
        self.num_rnn_layer = num_rnn_layer
        self.model = self.build_model()
    
    def build_model(self):
        inputs = layers.Input(shape=(None, self.num_features))
        x = inputs

        # Recurrent layers
        for _ in range(self.num_rnn_layer):
            x = layers.Bidirectional(layers.LSTM(self.num_hidden_units, return_sequences=True))(x)
        
        # Two separate outputs for singing voice and accompaniment
        output_src1 = layers.Dense(self.num_features, activation='relu', name="vocals_output")(x)
        output_src2 = layers.Dense(self.num_features, activation='relu', name="accompaniment_output")(x)
        
        model = models.Model(inputs, [output_src1, output_src2])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, X_train, Y_train_vocals, Y_train_accompaniment, batch_size=32, epochs=50):
        self.model.fit(X_train, [Y_train_vocals, Y_train_accompaniment], batch_size=batch_size, epochs=epochs)
    
    def predict(self, stft_mono_magnitude):
        predictions = self.model.predict(stft_mono_magnitude)
        
        if isinstance(predictions, np.ndarray):
            raise ValueError(f"Expected model to return two outputs, but got a single output of shape {predictions.shape}")
        
        if isinstance(predictions, list) and len(predictions) == 2:
            return predictions[0], predictions[1]
        else:
            raise ValueError(f"Unexpected model output type: {type(predictions)}")
    
    def save_model(self, path='model.h5'):
        self.model.save(path)
    
    def load_model(self, path='model.h5'):
        self.model = models.load_model(path)
