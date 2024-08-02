#define PCL_NO_PRECOMPILE

#include <pcl/pcl_base.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include "pcd_extension.hpp"

void zero_initialize_PointXYZVector(VKJYPointXYZVector *point){
    for(int i=0; i<NUM_CLASS_PCD; i++){
        point->cVec[i] = 0.0;
    }
}

void free_initialize_PointXYZVector(VKJYPointXYZVector *point){
    point->cVec[0] = 1.0;
    for(int i=1; i<NUM_CLASS_PCD; i++){
        point->cVec[i] = 0.0;
    }
}

void get_from_PointXYZVector(const VKJYPointXYZVector *point, vsemantic_bki::point3f &s_point, std::vector<double> &prob_vector){
    s_point.x() = point->x;
    s_point.y() = point->y;
    s_point.z() = point->z;

    prob_vector.resize(NUM_CLASS_PCD);
    for(int i=0; i<NUM_CLASS_PCD; i++){
        prob_vector[i] = point->cVec[i];
    }
}

void save_std_vector_PointXYZVector(VKJYPointXYZVector *point, std::vector<double> vector){
    if(vector.size() != NUM_CLASS_PCD + 3){
        std::cout << "ERROR! vector size should be ${NUM_CLASS_PCD} + 3!" << std::endl;  
    }else{
        point->x = vector[0]; point->y = vector[1]; point->z = vector[2];

        for(int i=0; i<NUM_CLASS_PCD; i++){
            point->cVec[i] = vector[3+i];
        }
    }
}
void save_std_vector_PointXYZVector(VKJYPointXYZVector *point, float x, float y, float z, std::vector<double> vector){
    if(vector.size() != NUM_CLASS_PCD){
        std::cout << "ERROR! vector size should be ${NUM_CLASS_PCD}!" << std::endl;  
    }else{
        point->x = x; point->y = y; point->z = z;

        for(int i=0; i<NUM_CLASS_PCD; i++){
            point->cVec[i] = vector[i];
        }
    }
}

void print_PointXYZVector(const VKJYPointXYZVector *point){
    std::cout << "(" << point->x << ", " << point->y << ", " << point->z << ")";
    std::cout << " - probability [";
    
    for(int i=0; i<NUM_CLASS_PCD - 1; i++){
        std::cout << point->cVec[i] << ", ";
    }
    std::cout << point->cVec[NUM_CLASS_PCD - 1] << "]\n";
}
void print_PointXYZVector(const vsemantic_bki::point3f p, const std::vector<double> prob_vector){
    if(prob_vector.size() != NUM_CLASS_PCD){
        std::cout << "ERROR:print_PointXYZVector - size: " << std::to_string(prob_vector.size()) << std::endl;
        return;
    }
    std::cout << "(" << p.x() << ", " << p.y() << ", " << p.z() << ")";
    std::cout << " - probability [";

    for(int i=0; i< NUM_CLASS_PCD; i++){
        std::cout << prob_vector[i];
        if(i+1 != NUM_CLASS_PCD)
            std::cout << ", ";
    }
    std::cout << "]\n";
}
