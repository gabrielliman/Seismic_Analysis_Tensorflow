from pyseismic_local.date_time_functions import *
from pyseismic_local.AI_SeismicFaciesClassification import *
import time
import numpy as np
import segyio

################################################################################
# Definir parâmetros
patch_rows = 992  # Ajustar conforme necessário
patch_cols = 192  # Ajustar conforme necessário
stride = [32, 32]  # Ajustar conforme necessário

data_save = 1
Y_channels = 1
################################################################################
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Usar apenas CPU para manipulação de dados, se necessário
print('CPU is used.')

################################################################################
# Diretório para salvar dados de validação
module_data_dir = '/scratch/nunes/data_parihaka_article_mydivision'
if not os.path.exists(module_data_dir):
    os.mkdir(module_data_dir)

validation_data_dir = module_data_dir + '/' + 'validation_data_' + str(int(patch_rows)) + 'x' + str(
    int(patch_cols)) + '_Ychannels' + str(Y_channels)
if not os.path.exists(validation_data_dir):
    os.mkdir(validation_data_dir)
################################################################################
raw_data_X_ID = ['/scratch/nunes/seismic/parihaka_seismic.sgy']
raw_data_Y_ID = ['/scratch/nunes/seismic/parihaka_labels.sgy']
patch_stride = stride
################################################################################

if __name__ == '__main__':
    dataset_number = len(raw_data_X_ID)
    counter = 0
    print(patch_stride)
    start_time = time.time()
    for i in range(dataset_number):
        counter_per_data = 0
        print('dataID ' + str(i + 1) + '/' + str(dataset_number) + ': ' + raw_data_X_ID[i])
        print('patch_stride ' + str(patch_stride[i]))
        ########################################################################
        # Ler cubo de dados
        X = segyio.tools.cube(raw_data_X_ID[i])
        Y = segyio.tools.cube(raw_data_Y_ID[i])
        print('X.shape:' + str(X.shape))
        print('Y.shape:' + str(Y.shape))
        X = X.transpose((2, 1, 0)).astype('float32')
        Y = Y.transpose((2, 1, 0)).astype('float32')
        print('X.shape:' + str(X.shape))
        print('Y.shape:' + str(Y.shape))
        
        # Ajuste para representar as faixas especificadas
        if i == 0:
            X = X[:, 622:702, :510]  # [:,622:702,:510]
            Y = Y[:, 622:702, :510]  # [:,622:702,:510]
        elif i == 1:
            X = X[:, :622, 430:510]  # [:,:622,430:510]
            Y = Y[:, :622, 430:510]  # [:,:622,430:510]
        
        print('X.shape ajustado:' + str(X.shape))
        print('Y.shape ajustado:' + str(Y.shape))
        ########################################################################
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
        Y = np.reshape(Y, (Y.shape[0], Y.shape[1], Y.shape[2], Y_channels))
        print('X.shape:' + str(X.shape))
        print('Y.shape:' + str(Y.shape))
        if i == 0:
            X = X.transpose((1,0, 2, 3))
            Y = Y.transpose((1,0, 2, 3))
        else:
            X = X.transpose((1, 0, 2, 3))
            Y = Y.transpose((1, 0, 2, 3))
        print('X.shape:' + str(X.shape))
        print('Y.shape:' + str(Y.shape))


        X = data2patches(X, patch_rows, patch_cols, patch_stride[i], patch_stride[i])
        print('X.shape:' + str(X.shape))
        Y = data2patches(Y, patch_rows, patch_cols, patch_stride[i], patch_stride[i])
        print('Y.shape:' + str(Y.shape))
        sample_number_temp = X.shape[0]
        if data_save:
            for j in range(sample_number_temp):
                current_data_name_ID = counter + j + 1
                X_temp = np.squeeze(X[j])
                Y_temp = np.squeeze(Y[j])
                X_temp = X_temp.transpose((1, 0))
                if Y_channels > 1:
                    Y_temp = Y_temp.transpose((1, 0, 2))
                else:
                    Y_temp = Y_temp.transpose((1, 0))

                X_temp.astype('float32').tofile(
                    validation_data_dir + '/' + format(current_data_name_ID, '011d') + 'X.bin')
                Y_temp.astype('float32').tofile(
                    validation_data_dir + '/' + format(current_data_name_ID, '011d') + 'Y.bin')

                if current_data_name_ID % 500 == 0 or j == sample_number_temp - 1:
                    print('Writing ' + format(current_data_name_ID, '011d') + ' done!')
                del X_temp, Y_temp
        counter_per_data = counter_per_data + sample_number_temp
        counter = counter + sample_number_temp
        print('counter_per_data ' + str(counter_per_data))
        print('counter ' + str(counter))
        del sample_number_temp
        ##################################################################################
        elapsed_time = time.time() - start_time
        elapsed_time_str = time2HMS(elapsed_time=elapsed_time)
        print('                 ')
        del X, Y
