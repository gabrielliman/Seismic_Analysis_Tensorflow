import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def simple_cnn(tam_entrada=(50,50,1), num_classes=6, dropout_rate=0.3):
    # Define the model
    model = models.Sequential()

    # First block
    model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(tam_entrada)))
    model.add(layers.LocallyConnected2D(64, kernel_size=(3, 3), activation='relu'))  # Local normalization
    model.add(layers.MaxPooling2D((2, 2)))

    # Second block
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.LocallyConnected2D(128, kernel_size=(3, 3), activation='relu'))  # Local normalization
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the output before fully connected layers
    model.add(layers.Flatten())

    # Fully connected layers with dropout and weight decay
    model.add(layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(1e-3)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(1e-3)))
    model.add(layers.Dropout(dropout_rate))

    # Output layer
    model.add(layers.Dense(num_classes, activation='linear'))  # Adjust number_of_classes accordingly
    return model




def simple_cnn_article2(tam_entrada, num_classes):
    model = models.Sequential()
    
    # Convolutional layers
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=tam_entrada))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten layer
    model.add(layers.Flatten())
    
    # Dense layers
    model.add(layers.Dense(1024, activation='relu'))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='linear'))
    
    return model