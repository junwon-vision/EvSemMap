## Projection Suite
- Given 2D Semantic Segmentation Results, 3D LiDAR Points, Pose information, The scripts in this directory carry out the projection task.
- `universal_utils/universal_utils.py`: the utility functions which are used in both `Rellis` and `Changwon` dataset (which is not published).

### Rellis
`python rellis_acc_inference.py {remark} {sequence} {binning_num}`
#### Example Scripts for Projection of Semantic Segmentation Results
- `python rellis_acc_inference.py rellisv3_edl_train-4 00004 30`