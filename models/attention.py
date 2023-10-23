from keras_unet_collection import models


def Attention_unet(tam_entrada, num_filtros, classes):
    model = models.att_unet_2d(
        input_size=tam_entrada, filter_num=num_filtros, n_labels=classes,
        stack_num_down=5, stack_num_up=4, activation='ReLU',
        atten_activation='ReLU', attention='add', output_activation='Softmax',
        batch_norm=True, pool=False, unpool='bilinear', name='attunet'
    )

    model._name = "attunet"

    return model