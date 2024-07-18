from keras_unet_collection import models
def Unet_3plus(tam_entrada, n_filters, classes):

    model = models.unet_3plus_2d(
        input_size=tam_entrada, n_labels=classes, filter_num_down=n_filters,
        filter_num_skip='auto', filter_num_aggregate='auto',
        stack_num_down=2, stack_num_up=2, activation='ReLU', output_activation='Softmax',
        batch_norm=True, pool='max', unpool=False, deep_supervision=True, name='unet3plus'
    )

    model._name = "unet3plus"

    return model