from keras_unet_collection import models

def Unet(tam_entrada, num_filtros, classes):
    model=models.unet_2d(
        input_size=tam_entrada, filter_num=num_filtros, n_labels=classes,
        stack_num_down=2, stack_num_up=2, activation="ReLU", output_activation="Softmax",
        batch_norm=True, pool="max", #unpool="False",
        name='unet')

    model._name='unet'
    return model