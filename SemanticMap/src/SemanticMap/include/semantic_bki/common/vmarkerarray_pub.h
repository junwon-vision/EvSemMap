#include <pcl_ros/point_cloud.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/ColorRGBA.h>

#include <cmath>
#include <string>

namespace vsemantic_bki {

    double interpolate( double val, double y0, double x0, double y1, double x1 ) {
        return (val-x0)*(y1-y0)/(x1-x0) + y0;
    }

    double base( double val ) {
        if ( val <= -0.75 ) return 0;
        else if ( val <= -0.25 ) return interpolate( val, 0.0, -0.75, 1.0, -0.25 );
        else if ( val <= 0.25 ) return 1.0;
        else if ( val <= 0.75 ) return interpolate( val, 1.0, 0.25, 0.0, 0.75 );
        else return 0.0;
    }

    double red( double gray ) {
        return base( gray - 0.5 );
    }
    double green( double gray ) {
        return base( gray );
    }
    double blue( double gray ) {
        return base( gray + 0.5 );
    }
    
    std_msgs::ColorRGBA JetMapColor(float gray) {
      std_msgs::ColorRGBA color;
      color.a = 1.0;

      color.r = red(gray);
      color.g = green(gray);
      color.b = blue(gray);
      return color;
    }

    std_msgs::ColorRGBA SemanticMapColor(int c) {
      std_msgs::ColorRGBA color;
      color.a = 1.0;

      switch (c) {
        case 1:
          color.r = 1;
          color.g = 0;
          color.b = 0;
          break;
        case 2:
          color.r = 70.0/255;
          color.g = 130.0/255;
          color.b = 180.0/255;
          break;
        case 3:
          color.r = 218.0/255;
          color.g = 112.0/255;
          color.b = 214.0/255;
          break;
        default:
          color.r = 1;
          color.g = 1;
          color.b = 1;
          break;
      }

      return color;
    }

    std_msgs::ColorRGBA RUGDRemapV3C9MapColor(int c) {
      std_msgs::ColorRGBA color;
      color.a = 1.0;

      switch (c) {
        case 0:
          color.r = 0.0;
          color.g = 0.0;
          color.b = 0.0;
          break;
        case 1:
          color.r = 196.0 / 255.0;
          color.g = 1.0;
          color.b = 1.0;
          break;
        case 2:
          color.r = 0.0;
          color.g = 0.0;
          color.b = 1.0;
          break;
        case 3:
          color.r = 204.0 / 255.0;
          color.g = 153.0 / 255.0;
          color.b = 1.0;
          break;
        case 4:
          color.r = 255.0 / 255.0;
          color.g = 255.0 / 255.0;
          color.b = 0.0 / 255.0;
          break;
        case 5:
          color.r = 1.0;
          color.g = 153.0 / 255.0;
          color.b = 204.0 / 255.0;
          break;
        case 6:
          color.r = 156.0 / 255.0;
          color.g = 76.0 / 255.0;
          color.b = 0.0;
          break;
        case 7:
          color.r = 111.0 / 255.0;
          color.g = 1.0;
          color.b = 74.0 / 255.0;
          break;
        case 8:
          color.r = 0.0;
          color.g = 102.0 / 255.0;
          color.b = 0.0;
          break;
        default:
          color.r = 1.0;
          color.g = 1.0;
          color.b = 1.0;
          break;
      }

      return color;
    }

    std_msgs::ColorRGBA RellisMapColor(int c) {
      std_msgs::ColorRGBA color;
      color.a = 1.0;

      switch (c) {
        case 0:
          color.r = 0.0;           color.g = 0.0;           color.b = 0.0;           break;
        case 1:
          color.r = 108.0 / 255.0; color.g = 64.0 / 255.0;  color.b = 20.0 / 255.0;  break;
        case 2:
          color.r = 0.0;           color.g = 102.0 / 255.0; color.b = 0.0;           break;
        case 3:
          color.r = 0.0;           color.g = 1.0;           color.b = 0.0;           break;
        case 4:
          color.r = 0.0;           color.g = 153.0 / 255.0; color.b = 153.0 / 255.0; break;
        case 5:
          color.r = 0.0;           color.g = 128.0 / 255.0; color.b = 1.0;           break;
        case 6:
          color.r = 0.0;           color.g = 0.0;           color.b = 1.0;           break;
        case 7:
          color.r = 1.0;           color.g = 1.0;           color.b = 0.0;           break;
        case 8:
          color.r = 1.0;           color.g = 0.0;           color.b = 127.0 / 255.0; break;
        case 9:
          color.r = 64.0 / 255.0;  color.g = 64.0 / 255.0;  color.b = 64.0 / 255.0;  break;
        case 10:
          color.r = 1.0;           color.g = 0.0;           color.b = 0.0;           break;
        case 11:
          color.r = 102.0 / 255.0; color.g = 0.0;           color.b = 0.0;           break;
        case 12:
          color.r = 204.0 / 255.0; color.g = 153.0 / 255.0; color.b = 1.0;           break;
        case 13:
          color.r = 102.0 / 255.0; color.g = 0.0;           color.b = 204.0 / 255.0; break;
        case 14:
          color.r = 1.0;           color.g = 153.0 / 255.0; color.b = 204.0 / 255.0; break;
        case 15:
          color.r = 170.0 / 255.0; color.g = 170.0 / 255.0; color.b = 170.0 / 255.0; break;
        case 16:
          color.r = 41.0 / 255.0;  color.g = 121.0 / 255.0; color.b = 1.0;           break;
        case 17:
          color.r = 134.0 / 255.0; color.g = 1.0;           color.b = 239.0 / 255.0; break;
        case 18:
          color.r = 99.0 / 255.0;  color.g = 66.0 / 255.0;  color.b = 34.0 / 255.0;  break;
        case 19:
          color.r = 110.0 / 255.0; color.g = 22.0 / 255.0;  color.b = 138.0 / 255.0; break;
        default:
          color.r = 1.0;           color.g = 1.0;           color.b = 1.0;           break;
      }

      return color;
    }

    class MarkerArrayPub {
        typedef pcl::PointXYZ PointType;
        typedef pcl::PointCloud<PointType> PointCloud;
    public:
        MarkerArrayPub(ros::NodeHandle nh, std::string topic, float resolution) : nh(nh),
                                                                                  msg(new visualization_msgs::MarkerArray),
                                                                                  topic(topic),
                                                                                  resolution(resolution),
                                                                                  markerarray_frame_id("map") {
            pub = nh.advertise<visualization_msgs::MarkerArray>(topic, 1, true);

            msg->markers.resize(1);
            for (int i = 0; i < 1; ++i) {
                msg->markers[i].header.frame_id = markerarray_frame_id;
                msg->markers[i].ns = "map";
                msg->markers[i].id = i;
                msg->markers[i].type = visualization_msgs::Marker::CUBE_LIST;
                msg->markers[i].scale.x = resolution * pow(2, 1);
                msg->markers[i].scale.y = resolution * pow(2, 1);
                msg->markers[i].scale.z = resolution * pow(2, 1);
                std_msgs::ColorRGBA color;
                color.r = 0.0;
                color.g = 0.0;
                color.b = 1.0;
                color.a = 1.0;
                msg->markers[i].color = color;
            }
        }

        void clear_map(float size) {
          int depth = 0;

          msg->markers[depth].points.clear();
          msg->markers[depth].colors.clear();
        }

        void insert_point3d_semantics(float x, float y, float z, float size, int c, int dataset) {
            if((dataset == 13 && c == 3) || (dataset == 9 && c == 1) || (dataset == 9 && c == 0) || (dataset == 102 && c == 6) || (dataset == 101 && c == 0))
              return; // Let's ignore sky voxels.
            geometry_msgs::Point center;
            center.x = x;
            center.y = y;
            center.z = z;

            int depth = 0;

            msg->markers[depth].points.push_back(center);
            switch (dataset) {
              case 9:
                msg->markers[depth].colors.push_back(RUGDRemapV3C9MapColor(c));
                break;
              case 102:
                msg->markers[depth].colors.push_back(RellisMapColor(c));
                break;
              default:
                msg->markers[depth].colors.push_back(SemanticMapColor(c));
            }
        }

        void insert_point3d_variance(float x, float y, float z, float min_v, float max_v, float size, float var) {
            geometry_msgs::Point center;
            center.x = x;
            center.y = y;
            center.z = z;

            int depth = 0;

            float middle = (max_v + min_v) / 2;
            var = (var - middle) / (middle - min_v);
            //std::cout << var << std::endl; 
            msg->markers[depth].points.push_back(center);
            msg->markers[depth].colors.push_back(JetMapColor(var));

        }

        void clear() {
            for (int i = 0; i < 10; ++i) {
                msg->markers[i].points.clear();
                msg->markers[i].colors.clear();
            }
        }

        void publish() const {
            msg->markers[0].header.stamp = ros::Time::now();
            pub.publish(*msg);
            ros::spinOnce();
        }

    private:
        ros::NodeHandle nh;
        ros::Publisher pub;
        visualization_msgs::MarkerArray::Ptr msg;
        std::string markerarray_frame_id;
        std::string topic;
        float resolution;
    };

}
