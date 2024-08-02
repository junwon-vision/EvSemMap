#include <algorithm>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>
#include "pcd_extension.hpp"
#include <pcl/filters/voxel_grid.h>

#include "vbkioctomap.h"
#include "vbki.h"

using std::vector;

#include <iostream>

namespace vsemantic_bki {

    SemanticBKIOctoMap::SemanticBKIOctoMap() : SemanticBKIOctoMap(0.1f, // resolution
                                        4, // block_depth
                                        3,  // num_class
                                        1.0, // sf2
                                        1.0, // ell
                                        1.0f, // prior
                                        1.0f, // var_thresh
                                        0.3f, // free_thresh
                                        0.7f // occupied_thresh
                                    ) { }

    SemanticBKIOctoMap::SemanticBKIOctoMap(float resolution,
                        unsigned short block_depth,
                        int num_class,
                        float sf2,
                        float ell,
                        float prior,
                        float var_thresh,
                        float free_thresh,
                        float occupied_thresh)
            : resolution(resolution), block_depth(block_depth),
              block_size((float) pow(2, block_depth - 1) * resolution) {
        Block::resolution = resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
        Block::index_map = init_index_map(Block::key_loc_map, block_depth);
        
        // Bug fixed
        Block::cell_num = static_cast<unsigned short>(round(Block::size / Block::resolution));
        std::cout << "block::resolution: " << Block::resolution << std::endl;
        std::cout << "block::size: " << Block::size << std::endl;
        std::cout << "block::cell_num: " << Block::cell_num << std::endl;
        
        SemanticOcTree::max_depth = block_depth;

        SemanticOcTreeNode::num_class = num_class;
        SemanticOcTreeNode::sf2 = sf2;
        SemanticOcTreeNode::ell = ell;
        SemanticOcTreeNode::prior = prior;
        SemanticOcTreeNode::var_thresh = var_thresh;
        SemanticOcTreeNode::free_thresh = free_thresh;
        SemanticOcTreeNode::occupied_thresh = occupied_thresh;
    }

    SemanticBKIOctoMap::~SemanticBKIOctoMap() {
        for (auto it = block_arr.begin(); it != block_arr.end(); ++it) {
            if (it->second != nullptr) {
                delete it->second;
            }
        }
    }

    void SemanticBKIOctoMap::set_resolution(float resolution) {
        this->resolution = resolution;
        Block::resolution = resolution;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void SemanticBKIOctoMap::set_block_depth(unsigned short max_depth) {
        this->block_depth = max_depth;
        SemanticOcTree::max_depth = max_depth;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void SemanticBKIOctoMap::insert_pointcloud_csm(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_res, float max_range, int int_mapping_media, int int_unc_policy) {

        ////////// Preparation //////////////////////////
        GPPointCloud xy;
        get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy);

        // If pointcloud after max_range filtering is empty, no need to do anything
        if (xy.size() == 0) {
            return;
        }

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks);

        for (auto it = xy.cbegin(); it != xy.cend(); ++it) {
            float p[] = {it->first.x(), it->first.y(), it->first.z()};
            rtree.Insert(p, p, const_cast<GPPointType *>(&*it));
        }
        ////////// Training /////////////////////////////
        vector<BlockHashKey> test_blocks;
        std::unordered_map<BlockHashKey, SemanticBKI3f *> bgk_arr;
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < blocks.size(); ++i) {
            BlockHashKey key = blocks[i];
            ExtendedBlock eblock = get_extended_block(key);
            if (has_gp_points_in_bbox(eblock))
#ifdef OPENMP
#pragma omp critical
#endif
            {
                test_blocks.push_back(key);
            };

            GPPointCloud block_xy;
            get_gp_points_in_bbox(key, block_xy);
            if (block_xy.size() < 1)
                continue;

            vector<float> block_x, block_y;
            for (auto it = block_xy.cbegin(); it != block_xy.cend(); ++it) {
                block_x.push_back(it->first.x());
                block_x.push_back(it->first.y());
                block_x.push_back(it->first.z());

                for(auto prob : it->second){
                    block_y.push_back(prob);
                }
            }

            SemanticBKI3f *bgk = new SemanticBKI3f(SemanticOcTreeNode::num_class, SemanticOcTreeNode::sf2, SemanticOcTreeNode::ell);
            bgk->train(block_x, block_y);
#ifdef OPENMP
#pragma omp critical
#endif
            {
                bgk_arr.emplace(key, bgk);
            };
        }

        ////////// Prediction ///////////////////////////
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < test_blocks.size(); ++i) {
            BlockHashKey key = test_blocks[i];
#ifdef OPENMP
#pragma omp critical
#endif
            {
                if (block_arr.find(key) == block_arr.end())
                    block_arr.emplace(key, new Block(hash_key_to_block(key)));
            };
            Block *block = block_arr[key];
            vector<float> xs;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it) {
                point3f p = block->get_loc(leaf_it);
                xs.push_back(p.x());
                xs.push_back(p.y());
                xs.push_back(p.z());
            }
            //std::cout << "xs size: "<<xs.size() << std::endl;

            // For counting sensor model
            auto bgk = bgk_arr.find(key);
            if (bgk == bgk_arr.end())
              continue;

            vector<vector<float>> ybars;

            bgk->second->predict_csm(xs, ybars, int_mapping_media, int_unc_policy);

            int j = 0;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it, ++j) {
                SemanticOcTreeNode &node = leaf_it.get_node();

                // Only need to update if kernel density total kernel density est > 0
                node.update(ybars[j]);
            }

        }

        ////////// Cleaning /////////////////////////////
        for (auto it = bgk_arr.begin(); it != bgk_arr.end(); ++it)
            delete it->second;

        rtree.RemoveAll();
    }

#ifdef INSERT_POINTCLOUD_DIRECT_END
#endif
    void SemanticBKIOctoMap::get_unc_threshold(const GPPointCloud &cloud, float &thres1, float &thres2, double unc_thres) const {
        // thres2 : not used
        std::vector<float> uncertainties;
        uncertainties.resize(cloud.size());
        int i = 0;
        for (auto it = cloud.cbegin(); it != cloud.cend(); ++it, ++i) {
            double sum_alpha = 0.0;
            for(int i =0; i < it->second.size(); i++){
                sum_alpha += it->second[i];
            }
            uncertainties[i] = (double) (it->second.size()) / sum_alpha;
        }
        
        std::sort(uncertainties.begin(), uncertainties.end());

        // top (unc_thres)% index
        int uncIndex = static_cast<int>(uncertainties.size() * unc_thres);
        thres1 = uncertainties[uncIndex]; 
    }

    void SemanticBKIOctoMap::downsample_by_uncertainty(const GPPointCloud &origin, GPPointCloud &newpc, GPPointCloud &newpc_middle, float uncThres1, float uncThres2) {
        newpc.clear();
        newpc_middle.clear();
            
        for (auto it = origin.cbegin(); it != origin.cend(); ++it) {
            double sum_alpha = 0.0;
            for(int i =0; i < it->second.size(); i++){
                // std::cout << it->second[i] << ", ";
                sum_alpha += it->second[i];
            }
            double unc = (it->second.size()) / sum_alpha;
            if ( unc <= uncThres1 ){
                point3f p;
                std::vector<double> probs;

                p.x() = it->first.x();
                p.y() = it->first.y();
                p.z() = it->first.z();

                probs.resize(it->second.size());
                for(int i=0; i< it->second.size(); i++){
                    probs[i] = it->second[i];
                }

                newpc.emplace_back(p, probs);
            }else if ( unc <= uncThres2) {
                point3f p;
                std::vector<double> probs;

                p.x() = it->first.x();
                p.y() = it->first.y();
                p.z() = it->first.z();

                probs.resize(it->second.size());
                for(int i=0; i< it->second.size(); i++){
                    probs[i] = it->second[i];
                }

                newpc_middle.emplace_back(p, probs);
            }
        }
    }

    void SemanticBKIOctoMap::insert_pointcloud(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_res, float max_range, int int_mapping_media, int int_unc_policy,
                                      double unc_thres, int ds_multiplier, bool isDempster) {
        ////////// Preparation //////////////////////////
        if (int_mapping_media == MAP_MEDIA_ONEHOT) {
            GPPointCloud xy1;
            get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy1);

            // If pointcloud after max_range filtering is empty, no need to do anything
            if (xy1.size() == 0) {
                return;
            }
            insert_pointcloud_worker(xy1, int_mapping_media, int_unc_policy, 1.0);
            return;
        }
        GPPointCloud xy1;
        get_training_data(cloud, origin, -1, free_res, max_range, xy1);

        // If pointcloud after max_range filtering is empty, no need to do anything
        if (xy1.size() == 0) {
            return;
        }

        float top70percentUnc, top30percentUnc;
        get_unc_threshold(xy1, top70percentUnc, top30percentUnc, unc_thres);
        // std::cout << "UncThres" << top70percentUnc << ", " << top30percentUnc << "\n";
        
        GPPointCloud xy, xy_ds, xy_middle, xy_middle_ds;

        if(int_unc_policy != MAP_UNC_POLICY_THRES && int_unc_policy != MAP_UNC_POLICY_ADAPTIVE_KERNEL_LENGTH){
            top70percentUnc = 1.0;
        }
        downsample_by_uncertainty(xy1, xy, xy_middle, top70percentUnc, 1.0); // 7:3 partitioning
        downsample(xy, xy_ds, ds_resolution, ds_multiplier);
        std::cout<< xy.size() << "=>" << xy_ds.size() << std::endl;
        insert_pointcloud_worker(xy_ds, int_mapping_media, int_unc_policy, top70percentUnc, isDempster);
    }

    void SemanticBKIOctoMap::insert_pointcloud_worker(const GPPointCloud &xy, int int_mapping_media, int int_unc_policy, float uncThres, bool isDempster) {
        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max); // Compute bounding box from the pointcloud.

        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks);

        for (auto it = xy.cbegin(); it != xy.cend(); ++it) {
            float p[] = {it->first.x(), it->first.y(), it->first.z()};
            rtree.Insert(p, p, const_cast<GPPointType *>(&*it));
        }
        ////////// Training /////////////////////////////
        vector<BlockHashKey> test_blocks;
        std::unordered_map<BlockHashKey, SemanticBKI3f *> bgk_arr;

#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < blocks.size(); ++i) {
            BlockHashKey key = blocks[i];
            ExtendedBlock eblock = get_extended_block(key);
            if (has_gp_points_in_bbox(key))
#ifdef OPENMP
#pragma omp critical
#endif
            {
                test_blocks.push_back(key);
            };

            GPPointCloud block_xy;
            get_gp_points_in_bbox(key, block_xy);
            if (block_xy.size() < 1)
                continue;

            vector<float> block_x, block_y;
            GPPointCloud eblock_xy;
            get_gp_points_in_bbox(eblock, eblock_xy);
            for (auto it = eblock_xy.cbegin(); it != eblock_xy.cend(); ++it) {
                block_x.push_back(it->first.x()); block_x.push_back(it->first.y()); block_x.push_back(it->first.z());
                
                int probi = 0;
                int max_idx = -1;
                float max_value = -1.0f;
                for(auto prob : it->second){
                    block_y.push_back(prob);
                    if ( max_value < prob ){
                        max_idx = probi;
                        max_value = prob;
                    }  
                }
            }
            SemanticBKI3f *bgk = new SemanticBKI3f(SemanticOcTreeNode::num_class, SemanticOcTreeNode::sf2, SemanticOcTreeNode::ell);
            bgk->train(block_x, block_y);
#ifdef OPENMP
#pragma omp critical
#endif
            {
                bgk_arr.emplace(key, bgk);
            };
        }

        ////////// Prediction ///////////////////////////
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < test_blocks.size(); ++i) {
            BlockHashKey key = test_blocks[i];
#ifdef OPENMP
#pragma omp critical
#endif
            {
                if (block_arr.find(key) == block_arr.end())
                    block_arr.emplace(key, new Block(hash_key_to_block(key)));
            };
            Block *block = block_arr[key];
            vector<float> xs;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it) {
                point3f p = block->get_loc(leaf_it);
                xs.push_back(p.x()); xs.push_back(p.y()); xs.push_back(p.z());
            }

            ExtendedBlock eblock = block->get_extended_block();
            for (auto block_it = eblock.cbegin(); block_it != eblock.cend(); ++block_it) {
                auto bgk = bgk_arr.find(*block_it);
                if (bgk == bgk_arr.end())
                    continue;

                vector<vector<float>> ybars;
                double uncertainty;
                int data_number;
                if(isDempster){
                    bgk->second->predict_direct_vector(xs, ybars, uncertainty, data_number);
                }else{
                    bgk->second->predict_bki(xs, ybars, int_mapping_media, int_unc_policy, uncertainty, data_number, uncThres);
                }

                int j = 0;
                for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it, ++j) {
                    SemanticOcTreeNode &node = leaf_it.get_node();

                    if(isDempster)
                        node.update_direct(ybars[j], uncertainty, data_number);
                    else
                        node.update(ybars[j], uncertainty, data_number);
                }
            }
        }
        ////////// Cleaning /////////////////////////////
        for (auto it = bgk_arr.begin(); it != bgk_arr.end(); ++it)
            delete it->second;

        rtree.RemoveAll();
    }

    void SemanticBKIOctoMap::get_training_data(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_resolution, float max_range, GPPointCloud &xy) const {
        PCLPointCloud sampled_hits;
        downsample(cloud, sampled_hits, ds_resolution, max_range);

        PCLPointCloud frees;
        frees.height = 1;
        frees.width = 0;
        xy.clear();
        for (auto it = sampled_hits.begin(); it != sampled_hits.end(); ++it) {
            point3f p;
            std::vector<double> probs;

            get_from_PointXYZVector(dynamic_cast<VKJYPointXYZVector*>(&*it), p, probs);

            if (max_range > 0) {
                double l = (p - origin).norm();
                if (l > max_range)
                    continue;
            }
            
            xy.emplace_back(p, probs);
        }
    }

    // Custom Down Sample
    int64_t SemanticBKIOctoMap::coord_to_hash_key(float x, float y, float z) const {
        return (int64_t(x / (double) Block::size + 524288.5) << 40) |
               (int64_t(y / (double) Block::size + 524288.5) << 20) |
               (int64_t(z / (double) Block::size + 524288.5));
    }
    void SemanticBKIOctoMap::coord_to_voxel_center_coord(point3f &p, point3f &c, point3f &lim_min, point3f &lim_max, float ds_resolution) const {
        float x_stride = (lim_max.x() - lim_min.x()) / ds_resolution;
        float y_stride = (lim_max.y() - lim_min.y()) / ds_resolution;
        float z_stride = (lim_max.z() - lim_min.z()) / ds_resolution;

        bool x_set = false;
        bool y_set = false;
        bool z_set = false;
        for(int i = (int)((p.x() - lim_min.x()) / ds_resolution) - 1; i <= (int)x_stride; i++){
            float cur_x_min = (lim_min.x() + (float)i * ds_resolution);
            float next_x_min = (lim_min.x() + (float)(i+1) * ds_resolution);
            if ( p.x() - cur_x_min < next_x_min - p.x()){
                c.x() = cur_x_min;
                x_set = true;
                break;
            }
        }
        if (x_set == false)
            c.x() = lim_max.x();

        for(int i = (int)((p.y() - lim_min.y()) / ds_resolution) - 1; i <= (int)y_stride; i++){
            float cur_y_min = (lim_min.y() + (float)i * ds_resolution);
            float next_y_min = (lim_min.y() + (float)(i+1) * ds_resolution);
            if ( p.y() - cur_y_min < next_y_min - p.y()){
                c.y() = cur_y_min;
                y_set = true;
                break;
            }
        }
        if (y_set == false)
            c.y() = lim_max.y();

        for(int i = (int)((p.z() - lim_min.z()) / ds_resolution) - 1; i <= (int)z_stride; i++){
            float cur_z_min = (lim_min.z() + (float)i * ds_resolution);
            float next_z_min = (lim_min.z() + (float)(i+1) * ds_resolution);
            if ( p.z() - cur_z_min < next_z_min - p.z()){
                c.z() = cur_z_min;
                z_set = true;
                break;
            }
        }
        if (z_set == false)
            c.z() = lim_max.z();
    }

    void SemanticBKIOctoMap::downsample(const PCLPointCloud &in, PCLPointCloud &out, float ds_resolution, float max_range, int ds_multiplier) const {
        if (ds_resolution <= 0) {
            out = in;
            return;
        }
        std::unordered_map<int64_t, int> ds_checkmap;
        
        GPPointCloud xy;
        xy.clear();
        for (auto it = in.begin(); it != in.end(); ++it) {
            point3f p;
            std::vector<double> probs;

            get_from_PointXYZVector(const_cast<VKJYPointXYZVector*>(&*it), p, probs);

            if (max_range > 0) {
                double l = (p - point3f(0.0f, 0.0f, 0.0f)).norm();
                if (l > max_range)
                    continue;
            }
            
            xy.emplace_back(p, probs);
        }
        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

        for (auto it = in.begin(); it != in.end(); ++it) {
            point3f p, c;
            std::vector<double> probs;

            get_from_PointXYZVector(const_cast<VKJYPointXYZVector*>(&*it), p, probs);
            coord_to_voxel_center_coord(p, c, lim_min, lim_max, ds_resolution);
            int64_t key = coord_to_hash_key(c.x(), c.y(), c.z());

            if (ds_checkmap.find(key) == ds_checkmap.end() || ds_checkmap[key] <= ds_multiplier){
                if (ds_checkmap.find(key) == ds_checkmap.end())
                    ds_checkmap.emplace(key, 1);
                else    
                    ds_checkmap[key] += 1;

                VKJYPointXYZVector point;
                save_std_vector_PointXYZVector(&point, p.x(), p.y(), p.z(), probs);
                out.push_back(point);
                // out.width++;
            }else{
                // Already In.
            }
        }
        // VoxelGrid Filtering Removes the Vector Information.
    }
    void SemanticBKIOctoMap::downsample(const GPPointCloud &in, GPPointCloud &out, float ds_resolution, int ds_multiplier) const {
        if (ds_resolution <= 0) {
            out = in;
            return;
        }
        std::unordered_map<int64_t, int> ds_checkmap;
        
        GPPointCloud xy;
        xy.clear();
        out.clear();
        for (auto it = in.begin(); it != in.end(); ++it) {
            point3f p;
            p.x() = it->first.x();
            p.y() = it->first.y();
            p.z() = it->first.z();
            std::vector<double> probs;

            xy.emplace_back(p, probs);
        }
        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

        for (auto it = in.cbegin(); it != in.cend(); ++it) {
            point3f p, c;
            p.x() = it->first.x(); p.y() = it->first.y(); p.z() = it->first.z();

            coord_to_voxel_center_coord(p, c, lim_min, lim_max, ds_resolution);
            int64_t key = coord_to_hash_key(c.x(), c.y(), c.z());
            if (ds_checkmap.find(key) == ds_checkmap.end() || ds_checkmap[key] <= ds_multiplier){
                if (ds_checkmap.find(key) == ds_checkmap.end())
                    ds_checkmap.emplace(key, 1);
                else    
                    ds_checkmap[key] += 1;

                std::vector<double> probs;
                probs.resize(it->second.size());
                for(int i=0; i< it->second.size(); i++){
                    probs[i] = it->second[i];
                }
                out.emplace_back(p, probs);
            }else {
                // Already In.
            }
        }
        // VoxelGrid Filtering Removes the Vector Information.
    }

    /*
     * Compute bounding box of pointcloud
     * Precondition: cloud non-empty
     */
    void SemanticBKIOctoMap::bbox(const GPPointCloud &cloud, point3f &lim_min, point3f &lim_max) const {
        assert(cloud.size() > 0);
        vector<float> x, y, z;
        for (auto it = cloud.cbegin(); it != cloud.cend(); ++it) {
            x.push_back(it->first.x()); y.push_back(it->first.y()); z.push_back(it->first.z());
        }

        auto xlim = std::minmax_element(x.cbegin(), x.cend());
        auto ylim = std::minmax_element(y.cbegin(), y.cend());
        auto zlim = std::minmax_element(z.cbegin(), z.cend());

        lim_min.x() = *xlim.first; lim_min.y() = *ylim.first; lim_min.z() = *zlim.first;
 
        lim_max.x() = *xlim.second; lim_max.y() = *ylim.second; lim_max.z() = *zlim.second;
    }

    void SemanticBKIOctoMap::get_blocks_in_bbox(const point3f &lim_min, const point3f &lim_max, vector<BlockHashKey> &blocks) const {
        for (float x = lim_min.x() - block_size; x <= lim_max.x() + 2 * block_size; x += block_size) {
            for (float y = lim_min.y() - block_size; y <= lim_max.y() + 2 * block_size; y += block_size) {
                for (float z = lim_min.z() - block_size; z <= lim_max.z() + 2 * block_size; z += block_size) {
                    blocks.push_back(block_to_hash_key(x, y, z));
                }
            }
        }
    }

    int SemanticBKIOctoMap::get_gp_points_in_bbox(const BlockHashKey &key, GPPointCloud &out) {
        point3f half_size(block_size / 2.0f, block_size / 2.0f, block_size / 2.0);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return get_gp_points_in_bbox(lim_min, lim_max, out);
    }

    int SemanticBKIOctoMap::has_gp_points_in_bbox(const BlockHashKey &key) {
        point3f half_size(block_size / 2.0f, block_size / 2.0f, block_size / 2.0);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return has_gp_points_in_bbox(lim_min, lim_max);
    }

    int SemanticBKIOctoMap::get_gp_points_in_bbox(const point3f &lim_min, const point3f &lim_max, GPPointCloud &out) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, SemanticBKIOctoMap::search_callback, static_cast<void *>(&out));
    }

    int SemanticBKIOctoMap::has_gp_points_in_bbox(const point3f &lim_min, const point3f &lim_max) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, SemanticBKIOctoMap::count_callback, NULL);
    }

    bool SemanticBKIOctoMap::count_callback(GPPointType *p, void *arg) {
        return false;
    }

    bool SemanticBKIOctoMap::search_callback(GPPointType *p, void *arg) {
        GPPointCloud *out = static_cast<GPPointCloud *>(arg);
        out->push_back(*p);
        return true;
    }

    int SemanticBKIOctoMap::has_gp_points_in_bbox(const ExtendedBlock &block) {
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            if (has_gp_points_in_bbox(*it) > 0)
                return 1;
        }
        return 0;
    }

    int SemanticBKIOctoMap::get_gp_points_in_bbox(const ExtendedBlock &block, GPPointCloud &out) {
        int n = 0;
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            n += get_gp_points_in_bbox(*it, out);
        }
        return n;
    }

    Block *SemanticBKIOctoMap::search(BlockHashKey key) const {
        auto block = block_arr.find(key);
        if (block == block_arr.end()) {
            return nullptr;
        } else {
            return block->second;
        }
    }

    SemanticOcTreeNode SemanticBKIOctoMap::search(point3f p) const {
        Block *block = search(block_to_hash_key(p));
        if (block == nullptr) {
          return SemanticOcTreeNode();
        } else {
          return SemanticOcTreeNode(block->search(p));
        }
    }

    SemanticOcTreeNode SemanticBKIOctoMap::search(float x, float y, float z) const {
        return search(point3f(x, y, z));
    }
}
