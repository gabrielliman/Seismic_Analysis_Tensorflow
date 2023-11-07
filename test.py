import numpy as np
from utils.datapreparation import my_division_data
import matplotlib.pyplot as plt
  
if __name__ == '__main__':
    stride1=16
    stridetrain2=16
    strideval2=16
    stridetest2=16
    train_image,train_label, test_image, test_label, val_image, val_label=my_division_data(shape=(992,192), stridetrain=(stride1,stridetrain2), strideval=(stride1,strideval2), stridetest=(stride1,stridetest2))
    print(train_label.shape)
    print(test_label.shape)
    print(val_label.shape)
    plt.imshow(train_label[0])
    plt.colorbar()
    plt.show()
    plt.savefig('test.png')