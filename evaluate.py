import tensorflow as tf

class SVSRNN:
    def __init__(self, num_features, num_rnn_layer, num_hidden_units, tensorboard_directory=None, clear_tensorboard=False):
        self.num_features = num_features
        self.num_rnn_layer = num_rnn_layer
        self.num_hidden_units = num_hidden_units  # This should be a list of integers, e.g., [256, 256, 256]
        self.tensorboard_directory = tensorboard_directory
        self.clear_tensorboard = clear_tensorboard
        
        # Build the model
        self.model = self.build_model()

    def build_model(self):
        input_tensor = tf.keras.layers.Input(shape=(None, self.num_features))
        x = input_tensor
        
        # Stacking LSTM layers with hidden units defined in the list
        for units in self.num_hidden_units:
            x = tf.keras.layers.LSTM(units, return_sequences=True)(x)
        
        # Adding a final Dense layer to output the predictions
        x = tf.keras.layers.Dense(self.num_features)(x)
        
        # Create the Keras model
        model = tf.keras.Model(inputs=input_tensor, outputs=x)
        
        # Compile the model (using a sample optimizer and loss function, modify as necessary)
        model.compile(optimizer='adam', loss='mse')
        
        return model

def generate_demo():
    # Example values, replace these with actual ones
    num_features = 1025  # This should be num_fft // 2 + 1
    num_rnn_layer = 3  # Number of RNN layers you want
    num_hidden_units = [256, 256, 256]  # A list of hidden units for each LSTM layer
    tensorboard_directory = './tensorboard_logs'  # Directory for tensorboard logs
    clear_tensorboard = True  # Whether to clear tensorboard logs before starting
    
    # Initialize and build the model
    model = SVSRNN(num_features=num_features, 
                   num_rnn_layer=num_rnn_layer, 
                   num_hidden_units=num_hidden_units, 
                   tensorboard_directory=tensorboard_directory, 
                   clear_tensorboard=clear_tensorboard)
    
    # Display model summary
    model.model.summary()

if __name__ == "__main__":
    generate_demo()
