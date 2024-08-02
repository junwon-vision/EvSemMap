import os
import argparse

# --log_dir ./ckpts (default)
# --save_freq 10 (default)
common_params_for_train = f'--evd_type edl \
   --unc_act exp \
   --unc_type log \
   --kl_strength 0.5 \
   --ohem -1.0 '
############################################# TRAIN #############################################
rellisv3_train = f'CUDA_VISIBLE_DEVICES=2 python main.py \
   --n_epoch 100 \
   --batch_size 24 \
   --l_rate 2e-4 \
   --model evidential \
   --dataset rellis_-4 \
   --remap_version 3 \
   --phase train \
   --remark rellisv3_edl_train-4_temp \
   {common_params_for_train}\
   --with_void False'
########################################### VALIDATION ###########################################
rellisv3_val = f'CUDA_VISIBLE_DEVICES=2 python main.py \
   --n_epoch 100 \
   --batch_size 24 \
   --l_rate 2e-4 \
   --model evidential \
   --dataset rellis_-4 \
   --remap_version 3 \
   --phase val \
   --remark rellisv3_edl_train-4 \
   --load /kjyoung/EvSemMapCode/EvSemSeg/ckpts/rellisv3_edl_train-4/100.pth \
   --partial_val 100 \
   {common_params_for_train}\
   --with_void False'
rellisv3_val_holdout = f'CUDA_VISIBLE_DEVICES=2 python main.py \
   --n_epoch 100 \
   --batch_size 24 \
   --l_rate 2e-4 \
   --model evidential \
   --dataset rellis_4 \
   --remap_version 3 \
   --phase val \
   --remark rellisv3_edl_train-4 \
   --load /kjyoung/EvSemMapCode/EvSemSeg/ckpts/rellisv3_edl_train-4/100.pth \
   --partial_val 100 \
   {common_params_for_train}\
   --with_void False'
rellisv3_val_cross = f'CUDA_VISIBLE_DEVICES=2 python main.py \
   --n_epoch 100 \
   --batch_size 24 \
   --l_rate 2e-4 \
   --model evidential \
   --dataset rellis_4 \
   --cross_inference DIFFERENT_DATASET(Offroad dataset would not be published) \
   --remap_version 3 \
   --phase val \
   --remark rellisv3_edl_train-4 \
   --load /kjyoung/EvSemMapCode/EvSemSeg/ckpts/rellisv3_edl_train-4/100.pth \
   --partial_val 100 \
   {common_params_for_train}\
   --with_void False'

########################################### TEST ###########################################
rellisv3_test_holdout = f'CUDA_VISIBLE_DEVICES=2 python main.py \
   --n_epoch 100 \
   --batch_size 24 \
   --l_rate 2e-4 \
   --model evidential \
   --dataset rellis_4 \
   --remap_version 3 \
   --phase test \
   --remark rellisv3_edl_train-4 \
   --load /kjyoung/EvSemMapCode/EvSemSeg/ckpts/rellisv3_edl_train-4/100.pth \
   {common_params_for_train}\
   --with_void False'

########################################### PREP ###########################################
rellisv3_prep = f'CUDA_VISIBLE_DEVICES=2 python main.py \
   --batch_size 24 \
   --model evidential \
   --dataset rellis_4 \
   --cross_inference rellis_4 \
   --remap_version 3 \
   --phase prep \
   --remark rellisv3_edl_train-4 \
   {common_params_for_train}\
   --load /kjyoung/EvSemMapCode/EvSemSeg/ckpts/rellisv3_edl_train-4/100.pth \
   --with_void False'

def main():
    parser = argparse.ArgumentParser(description="Execute Commands")
    parser.add_argument('--mode', type=str, help='Mode (Pipeline) to execute')
    
    args = parser.parse_args()
    
    if args.mode == 'ex-train':
        os.system(rellisv3_train)
    
    elif args.mode == 'ex-val':
        os.system(rellisv3_val)
    elif args.mode == 'ex-val-holdout':
        os.system(rellisv3_val_holdout)
    elif args.mode == 'ex-val-cross':
        os.system(rellisv3_val_cross)

    elif args.mode == 'ex-test-holdout':
        os.system(rellisv3_test_holdout)

    elif args.mode == 'ex-prep-rellis':
        os.system(rellisv3_prep)
    
    else:
        raise Exception('MODE - Not Implemented')

if __name__ == "__main__":
    main()