import os
import torch
import torchvision
from torch.utils.data import DataLoader
from datasets.loader_util import inv_transform

def validation(img_save_dir, train_dataset, model, args, epoch) :
    val_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=16, shuffle=False, drop_last=True)
    print(f'The number of validation images = {len(train_dataset)}')

    model.cuda()
    model.eval()

    with torch.no_grad() :	
        for i, data in enumerate(val_loader, start=0):  # inner loop within one epoch
            img, lbl = data
            img = img.cuda()

            for j in range(args.batch_size) :
                
                # 1. save image
                cls, unc_map = model.inference(img[j:j+1], uncMap=True)

                cls_result = train_dataset.label_transform(cls[0])
                cls_gt = train_dataset.label_transform(lbl[j].squeeze(0))
                orig_img = inv_transform(img[j])

                # concat_img = torch.cat([orig_img, cls_result, cls_gt], dim=2)
                # torchvision.utils.save_image(concat_img, os.path.join(img_save_dir, "e{}_{}.png".format(epoch, args.batch_size * i + j)))\
                
                # 2-1. Uncertainty map
                unc_map = unc_map.repeat((3, 1, 1))

                # pdb.set_trace()
                # 2-2. Error Map
                err_map = torch.zeros((3, cls_gt.shape[1], cls_gt.shape[2])).cuda()
                err_map[0][(cls_gt != cls_result).any(axis=0)] = 255
                
                concat_unc_img = torch.cat([orig_img, unc_map, err_map, cls_result, cls_gt], dim=2)
                torchvision.utils.save_image(concat_unc_img, os.path.join(img_save_dir, "merged_e{}_{}.png".format(epoch, args.batch_size * i + j)))