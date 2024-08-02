import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
import torchvision.transforms.functional as TF

from datasets.loader_util import inv_transform
from argparser import folder_check_and_create

from datasets.loader_rellis import rellis_split_dict

def continuous_exaggeration(unc, power=0.3):
    """
    Apply a continuous exaggeration to values in a [0, 1] array.

    Parameters:
    - unc: The input array with values in [0, 1].
    - power: The power to which each element is raised. Values <1 will exaggerate the array continuously.

    Returns:
    - A transformed array with exaggerated values, maintaining the original order.
    """
    # Ensure unc is a numpy array for element-wise operations
    if isinstance(unc, torch.Tensor):
        unc = unc.numpy()

    # Apply the power function for continuous exaggeration
    exaggerated_unc = np.power(unc, power)

    min_val, max_val = exaggerated_unc.min(), exaggerated_unc.max()
    normalized_unc = (exaggerated_unc - min_val) / (max_val - min_val)

    return normalized_unc

def prep_function(OUTPUT_DIR, DATASET_DIR, IMG_POSTFIX, Sequence,
                  label_transform, model, ds_factor, cut_car, only_vis=True):
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        for seq in Sequence:
            # Directory Preparation
            seq_root = os.path.join(DATASET_DIR, seq, IMG_POSTFIX)
            out_root_npy = os.path.join(OUTPUT_DIR, f'01_inferenced_npy', seq)
            out_root_vis = os.path.join(OUTPUT_DIR, f'02_inferenced_vis', seq)

            if not os.path.isdir(seq_root):
                print(f"IMAGE NOT FOUND: {seq_root}")
                return
            
            folder_check_and_create(out_root_vis)
            if not only_vis:
                folder_check_and_create(out_root_npy)
            
            # Make a image lists
            img_files = np.sort(os.listdir(seq_root))
            img_files = sorted([os.path.join(seq_root, f) for f in img_files if f.endswith(".jpg")])

            for img_file in img_files:
                # Format image for Neural Network. (FROM Data Loader Code)
                image = Image.open(img_file)
                img = TF.to_tensor(image)
                img = TF.normalize(img, mean=[103.939/255, 116.779/255, 123.68/255], std=[0.229, 0.224, 0.225])
                
                img = img[:, :img.shape[1] * 5 // 6, :] if cut_car == True else img
                img = img.unsqueeze(0).cuda()

                if ds_factor is not None and ds_factor != 1:
                    ds_size = (int(img.shape[2] // ds_factor), int(img.shape[3] // ds_factor))
                    img = torch.nn.functional.interpolate(img, size=ds_size, mode='bilinear')
                
                prob, label, unc = model.inference_prob(img, visualization=True)
                # print(img.shape, logit.shape) # torch.Size([1, 3, 1200, 1920]) torch.Size([1, 13, 1200, 1920])

                if not only_vis:
                    np.save(os.path.join(out_root_npy, os.path.basename(img_file)[:-4]), prob.detach().cpu().squeeze(0).numpy().astype(np.float16))

                # Visualization
                cls_result = label_transform(label[0])
                orig_img = inv_transform(img).squeeze(0)

                # Apply a colormap (e.g., 'jet') to the uncertainty values
                unc_colormap = plt.get_cmap('cividis')
                unc_color = unc_colormap(continuous_exaggeration(unc.squeeze().cpu().numpy()))  # This will be a (H, W, 4) array
                unc_color_rgb = unc_color[..., :3]
                unc_color_tensor = torch.from_numpy(unc_color_rgb).permute(2, 0, 1).float()

                # pdb.set_trace()
                if not only_vis:
                    concat_unc_img = torch.cat([orig_img, unc_color_tensor.cuda(), cls_result], dim=2)
                    torchvision.utils.save_image(concat_unc_img, os.path.join(out_root_vis, os.path.basename(img_file)))
                else:
                    torchvision.utils.save_image(cls_result, os.path.join(out_root_vis, f"{os.path.basename(img_file)[:-4]}_PRED.jpg"))
                    torchvision.utils.save_image(unc_color_tensor, os.path.join(out_root_vis, f"{os.path.basename(img_file)[:-4]}_UNC.jpg"))
                    torchvision.utils.save_image(orig_img, os.path.join(out_root_vis, f"{os.path.basename(img_file)[:-4]}_RGB.jpg"))

RELLIS_ROOT = '/data/Rellis-3D'

def preparation(model, dataset, data_subset, remark, label_transform, prep_only_vis):
    if dataset == 'rellis':
        OUTPUT_ROOT = '/kjyoung/convertedRellis'
        DATASET_ROOT = RELLIS_ROOT
        IMG_POSTFIX = 'pylon_camera_node'
        ds_factor = 2.0 # ds_factor = 2.2
        cut_car = False

        # FULL_SEQ = ['00000', '00001', '00002', '00003', '00004']
        Sequence = rellis_split_dict[data_subset]
    else:
        raise NotImplementedError("Not Implemented @ prep_mapping")

    OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, remark)
    # in OUT_REMARK directory
    # 01_inferenced_npy: 2D Seg Model Inferenced (C * H * W) Numpy Files
    # 02_inferenced_vis: 2D Seg Model Inferenced Visualizations
    # 03_nby3pK_npy: 3D Points Projected & Merged (N * (3 + C)) Numpy Files
    # 04_vtk: 3D Point Projected & Merged VTK Visualizations
    # 05_pVec_pcd: PCD converted files - Each point has 13 double fields for describing class probabilities!

    prep_function(OUTPUT_ROOT, DATASET_ROOT, IMG_POSTFIX, Sequence,
                  label_transform, model, ds_factor, cut_car, prep_only_vis)
