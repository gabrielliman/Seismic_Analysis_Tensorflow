import tensorflow as tf
from tensorflow.keras import layers, models

def simple_cnn(tam_entrada=(50,50,1), num_classes=6):
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

    # Fully connected layers
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(2048, activation='relu'))

    # Output layer
    model.add(layers.Dense(num_classes, activation='linear'))  # Adjust number_of_classes accordingly
    return model

# # Compile the model
# model.compile(optimizer='adam',
#             loss='sparse_categorical_crossentropy',  # Use appropriate loss function
#             metrics=['accuracy'])

# # Display the model summary
# model.summary()