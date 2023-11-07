from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten, Activation
from keras.datasets import mnist
from keras.utils import np_utils
import tensorflow as tf
from keras import backend as K
from utils.datapreparation import my_division_data

NUM_CLASSES = 10

def get_input_datasets(use_bfloat16=False):
    """Downloads the MNIST dataset and creates train and eval dataset objects.

    Args:
      use_bfloat16: Boolean to determine if input should be cast to bfloat16

    Returns:
      Train dataset, eval dataset and input shape.

    """
    # input image dimensions
    img_rows, img_cols = 28, 28
    cast_dtype = tf.bfloat16 if use_bfloat16 else tf.float32

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if tf.keras.backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    # train dataset
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.repeat()
    train_ds = train_ds.map(lambda x, y: (tf.cast(x, cast_dtype), y))
    train_ds = train_ds.batch(64, drop_remainder=True)

    # eval dataset
    eval_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    eval_ds = eval_ds.repeat()
    eval_ds = eval_ds.map(lambda x, y: (tf.cast(x, cast_dtype), y))
    eval_ds = eval_ds.batch(64, drop_remainder=True)

    return train_ds, eval_ds, input_shape


def get_model(input_shape, dropout2_rate=0.5):
   
    # input image dimensions
    img_rows, img_cols = 28, 28

    
    # Define a CNN model to recognize MNIST.
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, name="conv2d_1"))
    model.add(Conv2D(64, (3, 3), activation='relu', name="conv2d_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_1"))
    model.add(Dropout(0.25, name="dropout_1"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(128, activation='relu', name="dense_1"))
    model.add(Dropout(dropout2_rate, name="dropout_2"))
    model.add(Dense(NUM_CLASSES, activation='softmax', name="dense_2"))
    
    return model


if __name__=='__main__':
    train_ds, eval_ds, input_shape = get_input_datasets()