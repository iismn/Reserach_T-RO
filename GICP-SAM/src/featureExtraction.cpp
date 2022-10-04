#include "utility.h"
#include "agv_global_module/cloud_info.h"

#ifdef CUDA_FOUND
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

struct smoothness_t{
    float value;
    size_t ind;
};

struct by_value{
    bool operator()(smoothness_t const &left, smoothness_t const &right) {
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:

    ros::Subscriber subLaserCloudInfo;

    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;
    ros::Publisher pubFusedPoints;
    ros::Publisher pubRoadPoints;
    ros::Publisher pubBuildPoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud_L;
    pcl::PointCloud<PointType>::Ptr extractedCloud_M;
    pcl::PointCloud<PointType>::Ptr extractedCloud_R;
    pcl::PointCloud<PointType>::Ptr extractedCloud;

    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;
    pcl::PointCloud<PointType>::Ptr fusedCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    agv_global_module::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;

    fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ> vgicp_cuda;


    FeatureExtraction()
    {

        // ROS : Subscriber Setting (Syncronize LiDAR)
        // ROS : Subscriber Setting (Syncronize LiDAR)
        if(MULTI_LIDAR == 3){
            cout << MULTI_LIDAR << endl;
          typedef agv_global_module::cloud_info PointCloudMsgT;
          typedef message_filters::sync_policies::ApproximateTime<PointCloudMsgT, PointCloudMsgT, PointCloudMsgT> SyncPolicy;
          message_filters::Subscriber<PointCloudMsgT> *sub1_, *sub2_, *sub3_;
          message_filters::Synchronizer<SyncPolicy>* sync_;
          sub1_ = new message_filters::Subscriber<PointCloudMsgT>(nh, "RETRIEVAL_MODULE/GLOBAL/deskew/cloud_info_L", 1);
          sub2_ = new message_filters::Subscriber<PointCloudMsgT>(nh, "RETRIEVAL_MODULE/GLOBAL/deskew/cloud_info_M", 1);
          sub3_ = new message_filters::Subscriber<PointCloudMsgT>(nh, "RETRIEVAL_MODULE/GLOBAL/deskew/cloud_info_R", 1);
          sync_ = new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), *sub1_, *sub2_, *sub3_);
          sync_->registerCallback(boost::bind(&FeatureExtraction::laserCloudInfoHandler_MultiLiDAR, this, _1, _2, _3));
        }else if(MULTI_LIDAR == 2){
            cout << MULTI_LIDAR << endl;
          typedef agv_global_module::cloud_info PointCloudMsgT;
          typedef message_filters::sync_policies::ApproximateTime<PointCloudMsgT, PointCloudMsgT> SyncPolicy;
          message_filters::Subscriber<PointCloudMsgT> *sub1_, *sub2_;
          message_filters::Synchronizer<SyncPolicy>* sync_;
          sub1_ = new message_filters::Subscriber<PointCloudMsgT>(nh, "RETRIEVAL_MODULE/GLOBAL/deskew/cloud_info_L", 1);
          sub2_ = new message_filters::Subscriber<PointCloudMsgT>(nh, "RETRIEVAL_MODULE/GLOBAL/deskew/cloud_info_R", 1);
          sync_ = new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), *sub1_, *sub2_);
          sync_->registerCallback(boost::bind(&FeatureExtraction::laserCloudInfoHandler_DoubleLiDAR, this, _1, _2));
        }else{
            cout << MULTI_LIDAR << endl;
          subLaserCloudInfo = nh.subscribe<agv_global_module::cloud_info>("RETRIEVAL_MODULE/GLOBAL/deskew/cloud_info_M", 1, &FeatureExtraction::laserCloudInfoHandler_SingleLiDAR, this, ros::TransportHints().tcpNoDelay());
        }

        pubLaserCloudInfo = nh.advertise<agv_global_module::cloud_info> ("RETRIEVAL_MODULE/GLOBAL/feature/cloud_info", 1);
        // pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("RETRIEVAL_MODULE/GLOBAL/feature/cloud_corner", 1);
        // pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("RETRIEVAL_MODULE/GLOBAL/feature/cloud_surface", 1);
        pubFusedPoints = nh.advertise<sensor_msgs::PointCloud2>("RETRIEVAL_MODULE/GLOBAL/feature/cloud_fused", 1);

        pubRoadPoints = nh.advertise<sensor_msgs::PointCloud2>("RETRIEVAL_MODULE/GLOBAL/feature/cloud_road", 1);
        pubBuildPoints = nh.advertise<sensor_msgs::PointCloud2>("RETRIEVAL_MODULE/GLOBAL/feature/cloud_build", 1);

        initializationValue();
    }

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        downSizeFilter.setLeafSize(odometryLeafSize, odometryLeafSize, odometryLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());

        extractedCloud_L.reset(new pcl::PointCloud<PointType>());
        extractedCloud_M.reset(new pcl::PointCloud<PointType>());
        extractedCloud_R.reset(new pcl::PointCloud<PointType>());

        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());
        fusedCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];

        vgicp_cuda.setResolution(1.0);

    }

    void laserCloudInfoHandler_MultiLiDAR(const agv_global_module::cloud_infoConstPtr& msgIn_L, const agv_global_module::cloud_infoConstPtr& msgIn_M, const agv_global_module::cloud_infoConstPtr& msgIn_R)
    {

        cloudInfo = *msgIn_L; // new cloud info
        // cloudHeader = msgIn_L->header; // new cloud header
        pcl::fromROSMsg(msgIn_L->cloud_deskewed, *extractedCloud_L); // new cloud for extraction

        cloudInfo = *msgIn_M; // new cloud info
        cloudHeader = msgIn_M->header; // new cloud header
        pcl::fromROSMsg(msgIn_M->cloud_deskewed, *extractedCloud_M); // new cloud for extraction

        cloudInfo = *msgIn_R; // new cloud info
        // cloudHeader = msgIn_R->header; // new cloud header
        pcl::fromROSMsg(msgIn_R->cloud_deskewed, *extractedCloud_R); // new cloud for extraction
        //
        // if(ONLINE_CALIBRATION == 1){
        //   Eigen::Matrix4f transform_left_SE3;
        //   transform_left_SE3.setIdentity();
        //   Eigen::Matrix4f transform_right_SE3;
        //   transform_right_SE3.setIdentity();
        //
        //   Eigen::Matrix4f transform_online_calibration_L;
        //   Eigen::Matrix4f transform_online_calibration_R;
        //
        //   pcl::PointCloud<pcl::PointXYZ>::Ptr extractedCloud_M_(new pcl::PointCloud<pcl::PointXYZ>);
        //   pcl::PointCloud<pcl::PointXYZ>::Ptr extractedCloud_L_(new pcl::PointCloud<pcl::PointXYZ>);
        //   pcl::PointCloud<pcl::PointXYZ>::Ptr extractedCloud_R_(new pcl::PointCloud<pcl::PointXYZ>);
        //
        //   extractedCloud_M_->points.resize(extractedCloud_M->size());
        //   for (size_t i = 0; i < extractedCloud_M->points.size(); i++) {
        //       extractedCloud_M_->points[i].x = extractedCloud_M->points[i].x;
        //       extractedCloud_M_->points[i].y = extractedCloud_M->points[i].y;
        //       extractedCloud_M_->points[i].z = extractedCloud_M->points[i].z;
        //   }
        //   extractedCloud_L_->points.resize(extractedCloud_L->size());
        //   for (size_t i = 0; i < extractedCloud_L->points.size(); i++) {
        //       extractedCloud_L_->points[i].x = extractedCloud_L->points[i].x;
        //       extractedCloud_L_->points[i].y = extractedCloud_L->points[i].y;
        //       extractedCloud_L_->points[i].z = extractedCloud_L->points[i].z;
        //   }
        //   extractedCloud_R_->points.resize(extractedCloud_R->size());
        //   for (size_t i = 0; i < extractedCloud_R->points.size(); i++) {
        //       extractedCloud_R_->points[i].x = extractedCloud_R->points[i].x;
        //       extractedCloud_R_->points[i].y = extractedCloud_R->points[i].y;
        //       extractedCloud_R_->points[i].z = extractedCloud_R->points[i].z;
        //   }
        //
        //
        //   vgicp_cuda.setInputTarget(extractedCloud_M_);
        //   vgicp_cuda.setInputSource(extractedCloud_L_);
        //   vgicp_cuda.align(*extractedCloud_L_);
        //   transform_online_calibration_L = vgicp_cuda.getFinalTransformation();
        //
        //   vgicp_cuda.setInputSource(extractedCloud_R_);
        //   vgicp_cuda.align(*extractedCloud_R_);
        //   transform_online_calibration_R = vgicp_cuda.getFinalTransformation();
        //
        //   pcl::transformPointCloud(*extractedCloud_L, *extractedCloud_L, transform_online_calibration_L);
        //   pcl::transformPointCloud(*extractedCloud_R, *extractedCloud_R, transform_online_calibration_R);
        // }

        fusedCloud->clear();

        *fusedCloud += *extractedCloud_L;
        *fusedCloud += *extractedCloud_M;
        *fusedCloud += *extractedCloud_R;

        publishFeatureCloud();

    }

    void laserCloudInfoHandler_DoubleLiDAR(const agv_global_module::cloud_infoConstPtr& msgIn_L, const agv_global_module::cloud_infoConstPtr& msgIn_R)
    {

        cloudInfo = *msgIn_L; // new cloud info
        cloudHeader = msgIn_L->header; // new cloud header
        pcl::fromROSMsg(msgIn_L->cloud_deskewed, *extractedCloud_L); // new cloud for extraction

        cloudInfo = *msgIn_R; // new cloud info
        // cloudHeader = msgIn_R->header; // new cloud header
        pcl::fromROSMsg(msgIn_R->cloud_deskewed, *extractedCloud_R); // new cloud for extraction

        // if(ONLINE_CALIBRATION == 1){
        //   Eigen::Matrix4f transform_left_SE3;
        //   transform_left_SE3.setIdentity();
        //   Eigen::Matrix4f transform_right_SE3;
        //   transform_right_SE3.setIdentity();
        //
        //   Eigen::Matrix4f transform_online_calibration_L;
        //   Eigen::Matrix4f transform_online_calibration_R;
        //   ndt_.setInputTarget(extractedCloud_M);
        //   ndt_.setInputSource(extractedCloud_L);
        //   ndt_.align(transform_left_SE3);
        //   transform_online_calibration_L = ndt_.getFinalTransformation();
        //
        //   ndt_.setInputSource(extractedCloud_R);
        //   ndt_.align(transform_right_SE3);
        //   transform_online_calibration_R = ndt_.getFinalTransformation();
        //
        //   pcl::transformPointCloud(*extractedCloud_L, *extractedCloud_L, transform_online_calibration_L);
        //   pcl::transformPointCloud(*extractedCloud_R, *extractedCloud_R, transform_online_calibration_R);
        // }

        fusedCloud->clear();

        *fusedCloud += *extractedCloud_L;
        *fusedCloud += *extractedCloud_R;

        publishFeatureCloud();

    }

    void laserCloudInfoHandler_SingleLiDAR(const agv_global_module::cloud_infoConstPtr& msgIn_M)
    {
        cloudInfo = *msgIn_M; // new cloud info
        cloudHeader = msgIn_M->header; // new cloud header
        pcl::fromROSMsg(msgIn_M->cloud_deskewed, *extractedCloud_M); // new cloud for extraction

        Eigen::Affine3f transform_top;
        pcl::getTransformation(xT,yT,zT,rollT*M_PI/180,pitchT*M_PI/180,yawT*M_PI/180,transform_top);
        Eigen::Matrix4f transform_top_SE3 = transform_top.matrix ();

        pcl::transformPointCloud(*extractedCloud_M, *extractedCloud_M, transform_top_SE3);

        fusedCloud->clear();
        *fusedCloud += *extractedCloud_M;

        publishFeatureCloud();
    }

    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointRange.clear();
    }

    void publishFeatureCloud()
    {
        // free cloud info memory
        // freeCloudInfoMemory();
        // save newly extracted features
        pcl::PointCloud<PointType>::Ptr roadCloud (new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr buildCloud (new pcl::PointCloud<PointType>);

        pcl::PassThrough<PointType> ROAD_SEG;
        ROAD_SEG.setInputCloud (fusedCloud);
        ROAD_SEG.setFilterFieldName ("z");
        // ROAD_SEG.setFilterLimits (-3, -1.72); // {PHAROS}
        ROAD_SEG.setFilterLimits (-3, 0.2); // {IRAP, MulRan}
        ROAD_SEG.filter (*roadCloud);

        pcl::PassThrough<PointType> BUILD_SEG;
        BUILD_SEG.setInputCloud (fusedCloud);
        BUILD_SEG.setFilterFieldName ("z");
        BUILD_SEG.setFilterLimits (3, 50);
        BUILD_SEG.filter (*buildCloud);
        
        cloudInfo.cloud_road = publishCloud(&pubRoadPoints, roadCloud, cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_build = publishCloud(&pubBuildPoints, buildCloud, cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_deskewed = publishCloud(&pubFusedPoints, fusedCloud, cloudHeader.stamp, lidarFrame);

        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "agv_global_module");

    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> [AGV Global Module] : Feature Extraction\033[0m");

    ros::spin();

    return 0;
}
