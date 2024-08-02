import numpy as np

def IoUs_calculator(label, pred, num_classes):
    # Calculate IoUs for each mini-batch
    # Finally, you should exclude 'zero'
    label = label.flatten()
    pred  = pred.flatten()
    assert label.shape == pred.shape

    # We should calculate intersection, union each!
    intersection, union = np.zeros(num_classes), np.zeros(num_classes)
    for i in range(num_classes):
        class_i_occur_label = (label == i).sum()
        class_i_occur_pred  = (pred  == i).sum()
        class_i_occur_both  = ((label == i) & (pred == i)).sum()

        intersection[i] = class_i_occur_both
        union[i]        = class_i_occur_label + class_i_occur_pred - class_i_occur_both

    return intersection, union