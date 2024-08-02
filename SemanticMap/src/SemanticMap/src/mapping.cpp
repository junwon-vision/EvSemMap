#include <string>
#include <iostream>
#include <sstream>
#include <ros/ros.h>
#include <chrono>
#include <thread>
#include "vbkioctomap.h"
#include "vmarkerarray_pub.h"
#include "npy.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

using namespace std;

// UTILIZATION FUNCTIONS
// int get_int_mapping_media(const string&) : Parameter Parsing
int get_int_mapping_media(const string& mapping_media){
    if(mapping_media.compare("onehot") == 0){
        return MAP_MEDIA_ONEHOT;
    }else if(mapping_media.compare("alpha") == 0) {
        return MAP_MEDIA_ALPHA;
    }else if(mapping_media.compare("prob") == 0) {
        return MAP_MEDIA_PROB;
    }else{
        cout << "You Specified Wrong mapping_media [onehot, alpha, prob]\n";
        return -1;
    }
};
// int get_int_unc_policy(const string&) : Parameter Parsing
int get_int_unc_policy(const string& uncertainty_strategy){
    if(uncertainty_strategy.compare("none") == 0){
        return MAP_UNC_POLICY_NONE;
    }else if(uncertainty_strategy.compare("thres") == 0) {
        return MAP_UNC_POLICY_THRES;
    }else if(uncertainty_strategy.compare("adlen") == 0) {
        return MAP_UNC_POLICY_ADLEN;
    }else if(uncertainty_strategy.compare("adaptive") == 0){
        return MAP_UNC_POLICY_ADAPTIVE_KERNEL_LENGTH;
    }else{
        cout << "You Specified Wrong uncertainty_strategy [none, weight, thres, all, adaptive]\n";
        return -1;
    }
};

void save_map_txtfile(const vsemantic_bki::SemanticBKIOctoMap* map, const string result_name, const stringstream &headerBuffer, int num_class, bool isDempster = false){
    float max_var = std::numeric_limits<float>::min();
    float min_var = std::numeric_limits<float>::max();
    // SAVE Map
    ofstream result_file;
    result_file.open(result_name + ".map");

    // Write Header
    result_file << headerBuffer.str();    

    for (auto it = map->begin_leaf(); it != map->end_leaf(); ++it) {
        if (it.get_node().get_state() == vsemantic_bki::State::OCCUPIED) {
            vsemantic_bki::point3f p = it.get_loc();
            int semantics = it.get_node().get_semantics();

            // double uncertainty = 0.0f;
            // if(int_mapping_media != MAP_MEDIA_ONEHOT)
            //     uncertainty = it.get_node().get_uncertainty();

            std::vector<float> vars(num_class);
            if(isDempster)
                it.get_node().get_dempster_vars(vars);
            else
                it.get_node().get_vars(vars);
            // m_pub_csm.insert_point3d_semantics(p.x(), p.y(), p.z(), it.get_size(), semantics, mapping_mode);

            result_file << p.x() << " " << p.y() << " " << p.z() << " " << it.get_size() << " " << semantics << " " << vars[semantics] << "\n";
            // if(int_mapping_media != MAP_MEDIA_ONEHOT)
            //     result_file << p.x() << " " << p.y() << " " << p.z() << " " << it.get_size() << " " << semantics << " " << vars[semantics] << " " << uncertainty << "\n";
            // else

            if (vars[semantics] > max_var)
		        max_var = vars[semantics];
            if (vars[semantics] < min_var){
                min_var = vars[semantics];
                cout << "min_var updated! " << min_var << endl;
            }
        }
    }
    result_file.close();
    cout << "max_var" << max_var << "\n" << "min_var" << min_var << "\n";
};


int main(int argc, char **argv){
    ros::init(argc, argv, "mapping");
    ros::NodeHandle nh("~");

    // 0. Parameters
    string dir, prefix, result_name;
    string mapping_strategy, mapping_media, uncertainty_strategy;
    double resolution, ds_resolution, max_range;
    int num_class, block_depth;

    // 1. Hyper-Parameters
    int scan_start, scan_stride, scan_num, mapping_mode, int_mapping_media, int_unc_policy, ds_multiplier;
    double unc_thres; // unc_thres = 0.7 => Remove top 30% uncertain data

    // 2. Default-Parameters
    double sf2 = 1.0;
    double ell = 1.0;
    float prior = 1.0f;
    double free_resolution = 4;

    // Parameter Parsing ----------------------------
    { 
        bool parameterPassed;
        parameterPassed = nh.hasParam("max_range") && nh.hasParam("resolution") && nh.hasParam("ds_resolution") && nh.hasParam("block_depth") && nh.hasParam("num_class");
        parameterPassed = parameterPassed && nh.hasParam("scan_start") && nh.hasParam("scan_stride") && nh.hasParam("scan_num") && nh.hasParam("dir") && nh.hasParam("result_name");
        parameterPassed = parameterPassed && nh.hasParam("mapping_mode") && nh.hasParam("mapping_strategy") && nh.hasParam("mapping_media") && nh.hasParam("uncertainty_strategy") && nh.hasParam("ell");

        if(!parameterPassed){
            ROS_ERROR("There are something missing parameters...");
            return -1;
        }

        // Basic Parameters
        nh.param<double>("max_range", max_range, max_range);
        nh.param<double>("resolution", resolution, resolution);
        nh.param<double>("ds_resolution", ds_resolution, ds_resolution);
        nh.param<int>("block_depth", block_depth, block_depth);
        nh.param<int>("num_class", num_class, num_class);

        // Map Input Parameters
        nh.param<int>("scan_start", scan_start, scan_start);
        nh.param<int>("scan_stride", scan_stride, scan_stride);
        nh.param<int>("scan_num", scan_num, scan_num);
        nh.param<string>("dir", dir, dir);
        nh.param<string>("result_name", result_name, result_name);
        nh.param<string>("prefix", prefix, prefix);

        // BKI Mapping Parameters
        nh.param<double>("ell", ell, ell);
        nh.param<int>("mapping_mode", mapping_mode, mapping_mode);
        nh.param<string>("mapping_strategy", mapping_strategy, mapping_strategy);
        nh.param<string>("mapping_media", mapping_media, mapping_media);
        nh.param<string>("uncertainty_strategy", uncertainty_strategy, uncertainty_strategy);
        nh.param<double>("unc_thres", unc_thres, unc_thres);                                                                            // unc_thres = 0.7 => Remove top 30% uncertain data
        nh.param<int>("ds_multiplier", ds_multiplier, ds_multiplier);                                                                   // ds_multiplier = 10 => Ours downsampling to have 10 points within certain grid.
        nh.param<float>("prior", prior, prior);

        int_mapping_media = get_int_mapping_media(mapping_media);
        int_unc_policy    = get_int_unc_policy(uncertainty_strategy);
        if(mapping_strategy.compare("csm") == 0){
            block_depth = 1;
        }
        if(int_unc_policy == MAP_UNC_POLICY_ADAPTIVE_KERNEL_LENGTH){
            if(!nh.hasParam("unc_thres") || !nh.hasParam("ds_multiplier")){
                ROS_ERROR("There are something missing parameters{ unc_thres, ds_multiplier }...");
                return -1;
            }
        }
    }

    vsemantic_bki::SemanticBKIOctoMap* map = new vsemantic_bki::SemanticBKIOctoMap(resolution, (unsigned short)block_depth, num_class, sf2, ell, prior, 0.0f, 0.0f, 0.0f);
    ros::Time start = ros::Time::now();

    // 2. Consturcts Maps - Server Offline Mapping Mode
    for(int i=scan_start; i<= scan_num; i+=scan_stride){
        pcl::PointCloud<VKJYPointXYZVector> loadedCloud;
        pcl::io::loadPCDFile(dir + prefix + to_string(i) + ".pcd", loadedCloud);
        cout << "working on " << dir + prefix + to_string(i) + ".pcd" << endl;
        vsemantic_bki::point3f origin(0.0, 0.0, 0.0);

        if(mapping_strategy.compare("csm") == 0){
            map->insert_pointcloud_csm(loadedCloud, origin, ds_resolution, free_resolution, max_range, int_mapping_media, int_unc_policy);            

        }else if(mapping_strategy.compare("bki") == 0 || mapping_strategy.compare("dempster") == 0){
            map->insert_pointcloud(loadedCloud, origin, ds_resolution, free_resolution, max_range, int_mapping_media, int_unc_policy, unc_thres, ds_multiplier, mapping_strategy.compare("dempster") == 0);

        }else{
            cout << "You Specified Wrong Method!\n";
            return -1;
        }
        ROS_INFO_STREAM("Scan " << i << " done");
    }

    // Save to Map ---------------------------------
    stringstream headerBuffer;
    headerBuffer << "#mapping_strategy mapping_media uncertainty_strategy dir prefix block_depth sf2 ell prior resolution num_class free_resolution ds_resolution scan_start scan_stride scan_num max_range mapping_mode\n#";
    headerBuffer << mapping_strategy << " " << mapping_media << " " << uncertainty_strategy << " "
                << dir << " " << prefix << " " << block_depth << " " << sf2 << " " << ell << " " << prior << " " 
                << resolution << " " << num_class << " " << free_resolution << " "
                << ds_resolution << " " << scan_start << " " << scan_stride << " " << scan_num << " " << max_range << " " << mapping_mode << "\n";
    save_map_txtfile(map, result_name, headerBuffer, num_class, mapping_strategy.compare("dempster") == 0);

    ROS_INFO_STREAM("The identifier was.. " + result_name); 
    // ---------------------------------------------
    
    delete map;
 
    // All Jobs Done!
    ros::Time end = ros::Time::now();
    ROS_INFO_STREAM("Map Finished in " << (end - start).toSec() << "s");

    ros::Duration duration(60.0); // Duration in seconds

    while (ros::ok()) {
        // Check if the desired duration has elapsed
        if ((ros::Time::now() - end) >= duration) {
            ROS_INFO_STREAM("Time is up! Shutting down...");
            ros::shutdown();
        }

        ros::spinOnce();
    }
    return 0;
}