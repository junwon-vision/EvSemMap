import random, time
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Custom Parts
from datasets.loader_rugd import RUGDLoader
from datasets.loader_rellis import Rellis3DLoader

from models.unc_seg_models import deeplabv3 as DeeplabV3Evidential
from models.seg_models import deeplabv3 as DeeplabV3Vanilla

from argparser import MODEL_TYPE, NOT_SPECIFIED, prepare_argparser, folder_check_and_config_save

# Pipelines
from train import train
from val import validation
from test import test
from prep import preparation

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # set random seed for all gpus

### Preparation Jobs ########################
# Merged version of (_get_class_num, _get_void_index).
# This function returns (#class, the index of the void class) for the specified dataset setting.
# Return] It returns the values from the `Original Dataset (The dataset on which the model is trained)`
def get_info_dataset(remap_version, dataset, with_void):
    n_class, idx_void = None, None

    if remap_version != -1: # Remapped Dataset!
        if remap_version == 3:
            n_class, idx_void = 9, 0
    else:                   # Original Dataset!
        if dataset == 'rugd':
            n_class, idx_void = 25, 0
        elif dataset == 'rellis':
            n_class, idx_void = 20, 0
    
    if n_class == None:
        raise NotImplementedError("NoT a Valid DataSet!")
    
    if not with_void:
        idx_void = None
    
    return n_class, idx_void


def setup_dataset(args):
    '''
        ============ DATASET GUIDE ===========================================
        DATASET                 | PHASE                     | SUBSET
        rugd                    | train, val, test          | train, test, val
        rellis                  | train, val, test, prep    | 0, 1, 2, 3, 4, -0, -1, -2, -3, -4
    '''
    # _setup_dataset: THE DATASET USED TO TRAIN THE MODEL
    def _setup_dataset(dataset, data_subset, remap_version, partial_val):
        if dataset == 'rugd':
            return RUGDLoader(data_subset, partial_val=partial_val, remap_version=remap_version)
        elif dataset == 'rellis':
            return Rellis3DLoader(data_subset, partial_val=partial_val, ds_factor=2.0, remap_version=remap_version)
        else:
            raise Exception('DATASET - Not Implemented')
    
    # (cross_inf dataset,   training dataset) * cross_inf dataset can be None
    if args.cross_inference != NOT_SPECIFIED and args.phase != 'prep':
            return (
                _setup_dataset(args.cross_inference, args.cross_subset, args.remap_version, args.partial_val), 
                _setup_dataset(args.dataset, args.data_subset, args.remap_version, None)
            )
    else:
            return (
                None,
                _setup_dataset(args.dataset, args.data_subset, args.remap_version, args.partial_val), 
            )


def setup_model(args, writer, n_class, void_index):

    if args.model == MODEL_TYPE[0]:
        model = DeeplabV3Vanilla(writer=writer, n_classes = n_class, void_index=void_index)
    elif args.model == MODEL_TYPE[1]:
        unc_arg = {}
        unc_arg['evd_type'] = args.evd_type
        unc_arg['unc_act'] = args.unc_act
        unc_arg['unc_type'] = args.unc_type
        unc_arg['kl_strength'] = args.kl_strength
        unc_arg['ohem'] = args.ohem if (args.ohem < 1.0 and args.ohem > 0.0) else None

        model = DeeplabV3Evidential(writer=writer, n_classes = n_class, unc_args=unc_arg, void_index=void_index)
    else:
        raise Exception("MODEL - Not Implemented!")
    
    epoch_start = 1
    if args.load != NOT_SPECIFIED:
        model.load_state_dict(torch.load(args.load)['network'])
        epoch_start = torch.load(args.load)['epoch'] + 1

    return model, epoch_start


now = datetime.now()
start = time.time()
if __name__ == '__main__':
    # 1. Parameter processing & Validity Checks
    args = prepare_argparser()

    #### check arg's validity and directories / prepare directories / save config to config.txt
    SAVE_DIR, IMG_SAVE_DIR = folder_check_and_config_save(args)

    # 2. Setup writer, dataset, model objects ###################################################################################################################
    writer = SummaryWriter(log_dir=f"logs_tb/{args.remark}_{now.month}.{now.day}_{now.hour}-{now.minute}") if args.phase == 'train' else None
    n_class, idx_void = get_info_dataset(remap_version=args.remap_version, dataset=args.dataset, with_void=args.with_void)
    
    # dataset: dataset where we will load the (image, label) pairs.
    # [if inference on other dataset] train_dataset: original dataset the loaded model used to train.
    cross_inf_dataset, train_dataset = setup_dataset(args) 
    model, epoch_start = setup_model(args, writer, n_class, idx_void)

    '''
        train, preparation  : only the train_dataset should be specified
        validation, test    : cross_inf_dataset can be specified
    '''
    if args.phase == 'val':
        validation(IMG_SAVE_DIR, cross_inf_dataset if cross_inf_dataset else train_dataset, model, args, epoch_start)
    elif args.phase == 'test':
        test(train_dataset, model, args) # Test only on the same dataset (data_subset can be different)
    elif args.phase == 'train':
        train(SAVE_DIR, train_dataset, model, epoch_start, writer, args)
        writer.close()
    elif args.phase == 'prep':
        preparation(model, args.cross_inference, args.cross_subset, args.remark, label_transform=train_dataset.label_transform, prep_only_vis=args.prep_only_vis)
    else:
        raise Exception("PHASE - Not Implemented")
    
    end = time.time()
    print(f"\n\n Done:: {end - start:.5f} sec")