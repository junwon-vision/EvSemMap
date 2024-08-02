## Semantic Mapping Project (ROS)
- `catkin_make` at this directory to compile Semantic Mapping Project written in `ROS1-noetic`
- Reading & Parsing `.npy` files from Projected 3D Point Clouds to yield `.pcd` format
    - `roslaunch evsemmap pcd_conversion.launch`
- Construct Semantic Map!
    - `roslaunch evsemmap mapping.launch`
- Visualize the results!
    - `roslaunch evsemmap map_reader.launch`
- **CAUTION**: You have to modify the parameters in launch files properly depending on your directory structure!

### Rellis
#### Example Scripts for Building Semantic Maps
- `roslaunch evsemmap pcd_conversion.launch`
- `roslaunch evsemmap mapping.launch dataset:=deploy_rellisv3_4_1-30 method:=dempster result_name:=/workspace/deployTest/`