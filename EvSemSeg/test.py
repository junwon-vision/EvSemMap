import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.metrics.ece_calculator import optimized_ece_with_bin

def test(dataset, model, args) :
    test_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16, shuffle=False, drop_last=False)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of test images = %d' % dataset_size)

    model.cuda()
    model.eval()
    
    total_acc, total_ece = 0, 0
    accum_isCorrect, accum_certainty = None, None
    accum_Intersections, accum_Unions = None, None
    with torch.no_grad():	
        for i, data in enumerate(tqdm(test_loader), start=0):
            img, lbl = data
            img, lbl = img.cuda(), lbl.cuda()
            mini_batch_size = img.shape[0]
            # acc, ece, ece2, ece3, ece4, ece5, ece6 = model.evaluate_uncertainty_measure(img, lbl)
            acc, intersection, union, isCorrect, certainty = model.evaluate_uncertainty_measure(img, lbl)
            
            # 1. Accuracy
            total_acc  += acc * mini_batch_size
            # 2. mIoU
            if (accum_Intersections is None) and (accum_Unions is None):
                accum_Intersections, accum_Unions = intersection, union
            else:
                accum_Intersections += intersection
                accum_Unions        += union
            # 3. ECE
            if (accum_isCorrect is None) and (accum_certainty is None):
                accum_isCorrect, accum_certainty = isCorrect, certainty
            else:
                accum_isCorrect = torch.concat((accum_isCorrect, isCorrect))
                accum_certainty = torch.concat((accum_certainty, certainty))

    # Calculate Test-Dataset's entire mIoU
    # print(accum_Intersections)
    # print(accum_Unions)
    accum_Intersections = accum_Intersections[ accum_Unions > 0.0 ]
    accum_Unions        = accum_Unions[ accum_Unions > 0.0 ]
    print(accum_Intersections)
    print(accum_Unions)

    total_IoUs = accum_Intersections / accum_Unions # print(total_IoUs)
    print(f"------- Class-wise mIoUs ------------------------")
    print(total_IoUs)
    total_mIoU = np.average(total_IoUs[total_IoUs > 0.0])

    # Calculate Test-Dataset's entire ECE
    fig_base_name = os.path.join(args.log_dir, args.remark, 'bin10_reliability_diagram')
    ECE1, MCE1 = optimized_ece_with_bin(accum_isCorrect, accum_certainty, bin=10, withMCE=True, base_name=fig_base_name)
    fig_base_name = os.path.join(args.log_dir, args.remark, 'bin20_reliability_diagram')
    ECE2, MCE2 = optimized_ece_with_bin(accum_isCorrect, accum_certainty, bin=20, withMCE=True, base_name=fig_base_name)
    print(f"-------- Test Results ----------------------------")
    print(f"Data_size  : {dataset_size}")
    print(f"Accuracy   : {total_acc / dataset_size : .4f} ")
    print(f"mIoU       : {total_mIoU               : .4f} ")
    print(f"ECE Bin  10: {ECE1 : .4f} | MCE: {MCE1:.4f}")
    print(f"ECE Bin  20: {ECE2 : .4f} | MCE: {MCE2:.4f}")