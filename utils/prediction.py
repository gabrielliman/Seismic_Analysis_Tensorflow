import numpy as np
import tensorflow as tf
from tabulate import tabulate


def seisfacies_predict(model, test_image): 
    if model.name=="unet3plus":
        predictions=[]
        #se
        for image in range(0,test_image.shape[0], 100):
            limit=np.min([test_image.shape[0], (image+100)])
            prediction=model.predict(test_image[image:limit])[-1]
            prediction=np.argmax(prediction, axis=3)
            if image==0:
                predictions=prediction
            else:
                predictions=np.append(predictions, prediction, axis=0)
        return predictions
    else:
        predictions=[]
        size=1000 #lower this if you have an memory overload in test
        for image in range(0,test_image.shape[0], size):
            limit=np.min([test_image.shape[0], (image+size)])
            prediction=model.predict(test_image[image:limit])
            prediction=np.argmax(prediction, axis=3)
            if image==0:
                predictions=prediction
            else:
                predictions=np.append(predictions, prediction, axis=0)
        return predictions
    
    
def calculate_accuracy(model, test_image, test_label):
    if model.name=="unet3plus":
        return model.evaluate(test_image,test_label)[-1]
    return model.evaluate(test_image, test_label)[1]

def calculate_micro_f1_score(true_positives, false_positives, false_negatives):
    """
    Calculate the micro F1 score from true positives, false positives, and false negatives.

    Args:
        true_positives (int): Total true positives across all classes.
        false_positives (int): Total false positives across all classes.
        false_negatives (int): Total false negatives across all classes.

    Returns:
        float: The micro F1 score.
    """
    micro_precision = true_positives / (true_positives + false_positives)
    micro_recall = true_positives / (true_positives + false_negatives)
    micro_f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

    return micro_f1_score

def calculate_class_info(model, test_image, test_label, num_classes, predicted_label):
    """
    Calculate the pixel-wise accuracy, precision, and recall for each class of the model on the given test data.
    """
    # Initialize counters for true positives, false positives, and false negatives for each class
    true_positives = [0] * num_classes
    false_positives = [0] * num_classes
    false_negatives = [0] * num_classes


    for class_idx in range(num_classes):
        true_positive_mask = (predicted_label == class_idx) & (test_label == class_idx)
        false_positive_mask = (predicted_label == class_idx) & (test_label != class_idx)
        false_negative_mask = (predicted_label != class_idx) & (test_label == class_idx)

        true_positives[class_idx] += true_positive_mask.sum().item()
        false_positives[class_idx] += false_positive_mask.sum().item()
        false_negatives[class_idx] += false_negative_mask.sum().item()

    # Calculate pixel-wise accuracy, precision, and recall for each class
    accuracy_by_class = {}
    for class_idx in range(num_classes):
        total_pixels = true_positives[class_idx] + false_positives[class_idx] + false_negatives[class_idx]
        if true_positives[class_idx]!=0:
            accuracy_by_class[class_idx] = true_positives[class_idx] / total_pixels
            precision = true_positives[class_idx] / (true_positives[class_idx] + false_positives[class_idx])
            recall = true_positives[class_idx] / (true_positives[class_idx] + false_negatives[class_idx])
        else:
            accuracy_by_class[class_idx]=0
            precision=0
            recall=0
        accuracy_by_class[class_idx] = [
            accuracy_by_class[class_idx],
            precision,
            recall
        ]

     # Calculate total true positives, false positives, and false negatives across all classes
    total_true_positives = sum(true_positives)
    total_false_positives = sum(false_positives)
    total_false_negatives = sum(false_negatives)

    # Calculate micro F1 score
    micro_f1_score = calculate_micro_f1_score(total_true_positives, total_false_positives, total_false_negatives)

    return accuracy_by_class, micro_f1_score

def calculate_macro_f1_score(class_info, num_classes=6):
    class_f1 = list(range(0,num_classes))
    for class_idx in class_info:
        precision = class_info[class_idx][1]
        recall = class_info[class_idx][2]
        if(precision+recall!=0):
            class_f1[class_idx] = 2 * (precision * recall) / (precision + recall)
        else:
            class_f1[class_idx]=0

    # Calculate the macro F1 score as the unweighted average of per-class F1 scores
    macro_f1_score = np.mean(list(class_f1))

    return macro_f1_score, class_f1

def make_prediction(name,folder, model, test_image, test_label, num_classes=6):

    predicted_label = seisfacies_predict(model,test_image)
    if len(test_image.shape)>3:
        class_info, micro_f1=calculate_class_info(model, tf.squeeze(test_image).numpy(), tf.squeeze(test_label).numpy(), num_classes, predicted_label)
    else:
        class_info, micro_f1=calculate_class_info(model, test_image, test_label, num_classes, predicted_label)
    macro_f1, class_f1=calculate_macro_f1_score(class_info, num_classes)

    data=[]
    for i in range(len(class_info)):
        data.append(["Classe "+str(i)] + class_info[i]+[class_f1[i]])

  
    #define header names
    col_names = ["Classe","Accuracy", "Precision", 'Recall', 'F1 score']
    f = open("results/"+folder+"/tables/table_"+name+".txt", "w")
    f.write(tabulate(data, headers=col_names, tablefmt="fancy_grid",floatfmt=".4f"))
    texto="\nMacro F1 "+ str(macro_f1) + "\nMicro F1 " + str(micro_f1)
    f.write(texto)
    # results=model.evaluate(test_image,test_label)
    # if model.name=="unet3plus":
    #         results_str="\nTest loss " + str(results[5])+ "\nTest acc " + str(results[-1])
    # else:
    #     results_str="\nTest loss " + str(results[0])+ "\nTest acc " + str(results[1])
    # f.write(results_str)
    f.close()

