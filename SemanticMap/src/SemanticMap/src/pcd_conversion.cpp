#include <string>
#include <iostream>
#include <ros/ros.h>
#include "npy.hpp"
#include "pcd_extension.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

using namespace std;

void convert_npyClass_pcdXYZVector(string path, string out_path){
    // 1. Read .npy file to make 2d std::vector (N * K)
    vector<vector<double>> points;
    npy::npy_data<double> d = npy::read_npy<double>(path);
    std::vector<double> data = d.data;
    std::vector<unsigned long> shape = d.shape; // It should be (N, 3 + K)

    points.resize(shape[0]);
    for(int i = 0; i < shape[0]; i++){
        points[i].resize(shape[1]);
        for(int j = 0; j < shape[1]; j++){
            points[i][j] = data[i*shape[1] + j];
        }
    }

    // points variable has (N, 3+K) information in 2D vector.
    // 2. `points` 2D-Vector to PointXYZL
    pcl::PointCloud<VKJYPointXYZVector> cloud;
    cloud.resize(shape[0]);

    int i = 0;
    for (auto& point: cloud){
        save_std_vector_PointXYZVector(&point, points[i]);
        i++;
    }

    pcl::io::savePCDFileASCII (out_path, cloud);
    std::cerr << "Saved " << cloud.size() << " data points to " << out_path << " " << std::endl;
}

void convert_npyClass_pcdXYZL(string path, string out_path){
    // 1. Read .npy file to make 2d std::vector (N * K)
    vector<vector<double>> points;
    npy::npy_data<double> d = npy::read_npy<double>(path);
    std::vector<double> data = d.data;
    std::vector<unsigned long> shape = d.shape; // It should be (N, K)

    cout << "Current Shape : " << shape[0] << ", " << shape[1] << endl;
    points.resize(shape[0]);
    for(int i = 0; i < shape[0]; i++){
        points[i].resize(4);

        // 2-1. XYZ
        points[i][0] = data[i* shape[1] + 0 ];
        points[i][1] = data[i* shape[1] + 1 ];
        points[i][2] = data[i* shape[1] + 2 ];

        double max_value = -100.0;
        int max_index    = -100;

        for(int j = 3; j < shape[1]; j++){
            double current_value = data[i* shape[1] + j];
            if(current_value > max_value){
                max_value = current_value;
                max_index = j-3;
            }
        }

        points[i][3] = max_index;
    }

    // 2. `points` 2D-Vector to PointXYZL
    pcl::PointCloud<pcl::PointXYZL> cloud;
    cloud.resize(shape[0]);

    int i = 0;
    for (auto& point: cloud){
        auto row = points[i];
        point.x = row[0];
        point.y = row[1];
        point.z = row[2];
        point.label = row[3];

        i++;
    }

    pcl::io::savePCDFileASCII (out_path, cloud);
    std::cerr << "Saved " << cloud.size() << " data points to " << out_path << " " << std::endl;
}

int main(int argc, char **argv){
    ros::init(argc, argv, "pcd_conversion");
    ros::NodeHandle nh("~");

    // 0. Parameters
    bool convert_mode_to_my_extension = true;
    string out_path {""};
    string dir {""};
    int scan_num = 130;
    int scan_start = 1;

    nh.param<bool>("convert_mode_to_my_extension", convert_mode_to_my_extension, convert_mode_to_my_extension);
    nh.param<int>("scan_num", scan_num, scan_num);
    nh.param<int>("scan_start", scan_start, scan_start);
    nh.param<string>("dir", dir, dir);
    nh.param<string>("out_path", out_path, out_path);

    ros::Time start = ros::Time::now();
    // 1. npy4 to XYZL pcd
    for(int file_id = scan_start; file_id <= scan_num; file_id++){
        if(convert_mode_to_my_extension)
            convert_npyClass_pcdXYZVector(dir + "merged" + to_string(file_id) + ".npy", out_path + "merged" + to_string(file_id) + ".pcd");
        else
            convert_npyClass_pcdXYZL(dir + "merged" + to_string(file_id) + ".npy", out_path + "xyzl" + to_string(file_id) + ".pcd");
    }

    ros::Time end = ros::Time::now();
    ROS_INFO_STREAM("Vector Practice finished in " << (end - start).toSec() << "s");
    ros::spin();
    return 0;
}