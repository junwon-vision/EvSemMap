#include <string>
#include <iostream>
#include <ros/ros.h>
#include "vmarkerarray_pub.h"
#include "npy.hpp"

using namespace std;

void publish_map(string path, vsemantic_bki::MarkerArrayPub &m_pub_csm, int mapping_mode){
    // Publish Map
    std::ifstream file(path);
    std::string str; 
    while (std::getline(file, str))
    {
        if(str[0] == '#'){ // Header
            continue;
        }else{
            stringstream ss(str);
            string s;
            vector<string> result;
            while(getline(ss, s, ' '))
                result.push_back(s);
            
            m_pub_csm.insert_point3d_semantics(stof(result[0]), stof(result[1]), stof(result[2]), stof(result[3]), stoi(result[4]), mapping_mode);
        }
    }
}

void publish_varmap(string path, vsemantic_bki::MarkerArrayPub &m_pub_csm){
    // Publish Map
    float max_var = std::numeric_limits<float>::min();
    float min_var = std::numeric_limits<float>::max();

    std::ifstream file(path);
    std::string str;
    int header_counter = 0;
    while (std::getline(file, str))
    {
        if(str[0] == '#'){ // Header
            if(header_counter == 1){
                char * pch = strtok (&str[1], " ");
                min_var = stof(pch);
                pch = strtok (NULL, " ");
                max_var = stof(pch);
            }
            header_counter += 1;
        }else{
            stringstream ss(str);
            string s;
            vector<string> result;
            while(getline(ss, s, ' '))
                result.push_back(s);

            m_pub_csm.insert_point3d_variance(stof(result[0]), stof(result[1]), stof(result[2]), min_var, max_var,  stof(result[3]), stof(result[4]));
        }
    }
    assert(header_counter == 2);
}

void publish_varmap_bymap(string path, vsemantic_bki::MarkerArrayPub &m_pub_csm){
    // Publish Map
    float max_var = std::numeric_limits<float>::min();
    float min_var = std::numeric_limits<float>::max();

    int DIRICHLET_VARIANCE = 5;
    int TARGET_IDX = DIRICHLET_VARIANCE;

    std::ifstream file(path);
    std::string str; 
    while (std::getline(file, str))
    {
        if(str[0] == '#'){ // Header
            continue;
        }else{
            stringstream ss(str);
            string s;
            vector<string> result;
            while(getline(ss, s, ' '))
                result.push_back(s);
            
            if(result.size() < TARGET_IDX+1)
                return;

            if(stof(result[TARGET_IDX]) > max_var)
                max_var = stof(result[TARGET_IDX]);
            if(stof(result[TARGET_IDX]) < min_var)
                min_var = stof(result[TARGET_IDX]);
        }
    }
    std::cout << max_var << "\n";
    std::cout << min_var << "\n";

    std::ifstream file2(path);
    while (std::getline(file2, str))
    {
        if(str[0] == '#'){ // Header
            continue;
        }else{
            stringstream ss(str);
            string s;
            vector<string> result;
            while(getline(ss, s, ' '))
                result.push_back(s);
            
            m_pub_csm.insert_point3d_variance(stof(result[0]), stof(result[1]), stof(result[2]), min_var >= 0.0 ? min_var : 0.0, max_var, stof(result[3]), stof(result[TARGET_IDX]) > 0.0 ? stof(result[TARGET_IDX]) : 0.0);
            // m_pub_csm.insert_point3d_variance(stof(result[0]), stof(result[1]), stof(result[2]), min_var, max_var, stof(result[3]), stof(result[TARGET_IDX]));
            // m_pub_csm.insert_point3d_variance(stof(result[0]), stof(result[1]), stof(result[2]), 0.0, 1.0, stof(result[3]), stof(result[TARGET_IDX]) > 0.0 ? stof(result[TARGET_IDX]) : 0.0);
        }
    }
}

int main(int argc, char **argv){
    ros::init(argc, argv, "map_reader");
    ros::NodeHandle nh("~");

    // 0. Parameters
    string map_path1 {""};
    string varmap_path1 {""};
    int mapping_mode = 13;

    // 1. Mapping Parameters
    double resolution = 0.0;

    nh.param<double>("resolution", resolution, resolution);

    nh.param<string>("map_path1", map_path1, map_path1);
    nh.param<string>("varmap_path1", varmap_path1, varmap_path1);
    
    nh.param<int>("mapping_mode", mapping_mode, mapping_mode);
    
    vsemantic_bki::MarkerArrayPub m_pub_csm1(nh, "/map1", resolution);
    vsemantic_bki::MarkerArrayPub v_pub_csm1(nh, "/varmap1", resolution);
    if(map_path1.compare("") != 0){
        publish_map(map_path1, m_pub_csm1, mapping_mode);
        m_pub_csm1.publish();
        publish_varmap_bymap(map_path1, v_pub_csm1);
        v_pub_csm1.publish();
    }
    
    ros::spin();
    return 0;
}