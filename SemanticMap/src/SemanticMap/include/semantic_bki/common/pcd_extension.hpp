#ifndef VKJY_PCD_
#define VKJY_PCD_

#define PCL_NO_PRECOMPILE
#include <pcl/pcl_base.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include "vpoint3f.h"

#ifdef VKJYPCD20
#define NUM_CLASS_PCD 20
#endif
#ifdef VKJYPCD13
#define NUM_CLASS_PCD 13
#endif
#ifdef VKJYPCD9
#define NUM_CLASS_PCD 9
#endif

struct EIGEN_ALIGN16 VKJYPointXYZVector    // enforce SSE padding for correct memory alignment
{
    PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
    double cVec[NUM_CLASS_PCD];
    PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
};
POINT_CLOUD_REGISTER_POINT_STRUCT (VKJYPointXYZVector,           // here we assume a XYZ + "test" (as fields)
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (double[NUM_CLASS_PCD], cVec, cVec)
)

void zero_initialize_PointXYZVector(VKJYPointXYZVector *point);
void free_initialize_PointXYZVector(VKJYPointXYZVector *point);

void get_from_PointXYZVector(const VKJYPointXYZVector *point, vsemantic_bki::point3f &s_point, std::vector<double> &prob_vector);

void save_std_vector_PointXYZVector(VKJYPointXYZVector *point, std::vector<double> vector);
void save_std_vector_PointXYZVector(VKJYPointXYZVector *point, float x, float y, float z, std::vector<double> vector);

void print_PointXYZVector(const VKJYPointXYZVector *point);

void print_PointXYZVector(const vsemantic_bki::point3f p, const std::vector<double> prob_vector);

#endif