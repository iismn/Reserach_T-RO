#!/usr/bin/env python3
# [RAW] : PKG Dependency
from __future__ import print_function
import rospy
from std_msgs.msg import String
from rospy.numpy_msg import numpy_msg

# SYSTEM
import argparse
from math import log10, ceil
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ
import random, shutil, json, time

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.onnx
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import netvlad

# THB
import cv2
import pcl
import h5py
import time
import math
import faiss
import numpy as np
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from collections import OrderedDict
from skimage.transform import resize
from tensorboardX import SummaryWriter
from cv_bridge import CvBridge, CvBridgeError
import filterpy.monte_carlo as MCL
import open3d as o3d
import open3d.core as o3c

# ROS
import ros_numpy
import message_filters
from nav_msgs.msg import Odometry as ROS_Odometry
from nav_msgs.msg import Path as ROS_Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image as ROS_Image
from sensor_msgs.msg import PointCloud2 as ROS_PCL
from std_msgs.msg import Int32,Int32MultiArray,MultiArrayLayout,MultiArrayDimension
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
# [RAW] : PARSER Argument
class Localization_PARAM():
    def __init__(self):
        self.mode =                      'test'
        self.batchSize =                 4
        self.cacheBatchSize =            24
        self.cacheRefreshRate =          1000
        self.nEpochs =                   30
        self.start_epoch =               0
        self.nGPU =                      1
        self.optim =                     'ADAM'
        self.lr =                        0.0001
        self.lrStep =                    5
        self.lrGamma =                   0.5
        self.weightDecay =               0.001
        self.momentum=                   0.9
        self.nocuda =                    False
        self.threads =                   8
        self.seed =                      123
        self.savePath =                  'checkpoints'
        self.ckpt =                      'best'
        self.evalEvery =                 1
        self.patience =                  10
        self.dataset =                   'urban'
        self.arch =                      'resnet18'
        self.vladv2 =                    False
        self.pooling =                   'netvlad'
        self.num_clusters =              64
        self.margin =                    0.1
        self.split =                     'val'
        self.fromscratch =               True

        # HARD CODED -> [Future Works] : Change with ROS Param
        self.dataPath =                  '/home/iismn/WorkSpace/CU11_DL/ROS/src/RESEARCH_PACK/OSM_NetVLAD/src/data/'
        self.runsPath =                  '/home/iismn/WorkSpace/CU11_DL/ROS/src/RESEARCH_PACK/OSM_NetVLAD/src/runs/'
        self.cachePath =                 '/home/iismn/WorkSpace/CU11_DL/ROS/src/RESEARCH_PACK/OSM_NetVLAD/src/cache/'
        self.resume =                    '/home/iismn/WorkSpace/CU11_DL/ROS/src/RESEARCH_PACK/OSM_NetVLAD/src/runs/14K_NetVLAD'

        self.MAP_range =                 rospy.get_param('AGV_Local_Module/ACC_DIST', 80)
        self.MAP_save =                  rospy.get_param('AGV_Local_Module/SAVE_DIST', 80)

class Localization_MAIN():
    def __init__(self):
        # [PyTorch] Load Deep-Learning Network ------------------------------------------
        # [PyTorch] A : Load Paramter
        print('[PyTorch] >> START')
        opt = Localization_PARAM()
        restore_var = ['lr', 'lrStep', 'lrGamma', 'weightDecay', 'momentum',
            'runsPath', 'savePath', 'arch', 'num_clusters', 'pooling', 'optim',
            'margin', 'seed', 'patience']

        # [PyTorch] B : Load DB Lib
        if opt.dataset.lower() == 'urban':
            import urban as dataset

        # [PyTorch] C : Load CUDA GPU
        print('[PyTorch] CUDA Init')
        cuda = not opt.nocuda
        if cuda and not torch.cuda.is_available():
            raise Exception("[PyTorch] GPU ERROR : Run with --nocuda")
        self.device = torch.device("cuda" if cuda else "cpu")

        # [PyTorch] D : Load Random Seed
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        if cuda:
            torch.cuda.manual_seed(opt.seed)

        # [PyTorch] D : Load Main Dataset
        print('[PyTorch] Load DB')
        self.whole_test_set = dataset.get_whole_test_set(onlyDB=True)

        # [PyTorch] E : Load ResNet+NetVLAD Model
        print('[PyTorch] Load NetWork : ResNet18')
        pretrained = not opt.fromscratch
        if opt.arch.lower() == 'resnet18':
            encoder_dim = 512
            encoder = models.resnet18(pretrained=True)
            # capture only feature part and remove last relu and maxpool
            layers = list(encoder.children())[:-2]

            if pretrained:
                # if using pretrained then only train conv5_1, conv5_2, and conv5_3
                for l in layers[:-3]:
                    for p in l.parameters():
                        p.requires_grad = False

            encoder = nn.Sequential(*layers)

        # [PyTorch] E-1 : Module Generate - Resnet Encoder
        self.model = nn.Module()
        self.model.add_module('encoder', encoder)
        print('[PyTorch] Load NetWork : NetVLAD')

        # [PyTorch] E-2 : Module Generate - NetVLAD Pooling
        if opt.mode.lower() != 'cluster':
            if opt.pooling.lower() == 'netvlad':
                net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2)
                if not opt.resume:
                    if opt.mode.lower() == 'train':
                        initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + train_set.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')
                    else:
                        initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + whole_test_set.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')

                    if not exists(initcache):
                        raise FileNotFoundError('Could not find clusters, please run with --mode=cluster before proceeding')

                    with h5py.File(initcache, mode='r') as h5:
                        clsts = h5.get("centroids")[...]
                        traindescs = h5.get("descriptors")[...]
                        net_vlad.init_params(clsts, traindescs)
                        del clsts, traindescs
                self.model.add_module('pool', net_vlad)
            else:
                raise ValueError("[PyTorch] PTH ERROR : Unknown Pooling")

        # [PyTorch] E-3 : Module Generate - GPU Parallel
        isParallel = False
        if opt.nGPU > 1 and torch.cuda.device_count() > 1:
            self.model.encoder = nn.DataParallel(self.model.encoder)
            if opt.mode.lower() != 'cluster':
                self.model.pool = nn.DataParallel(self.model.pool)
            isParallel = True
        if not opt.resume:
            self.model = self.model.to(self.device)

        # [PyTorch] E-4 : Module Generate - Main NetWork Load
        if opt.resume:
            if opt.ckpt.lower() == 'latest':
                resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
            elif opt.ckpt.lower() == 'best':
                resume_ckpt = join(opt.resume, 'checkpoints', 'model_best.pth.tar')

            if isfile(resume_ckpt):
                print("[PyTorch] Load Weight : '{}'".format(resume_ckpt))
                checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
                opt.start_epoch = checkpoint['epoch']
                best_metric = checkpoint['best_score']

                # ONLY FOR MODULE WITH Multi-GPU
                if opt.nGPU == 1:
                    checkpoint_T = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)['state_dict']
                    for key in list(checkpoint_T.keys()):
                        if '.module.' in key:
                            checkpoint_T[key.replace('.module.', '.')] = checkpoint_T[key]
                            del checkpoint_T[key]
                    self.model.load_state_dict(checkpoint_T)
                # ONLY FOR MODULE WITH Single-GPU
                if opt.nGPU > 1:
                    model.load_state_dict(checkpoint['state_dict'])

                self.model = self.model.to(self.device)
                if opt.mode == 'train':
                    optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print("=> no checkpoint found at '{}'".format(resume_ckpt))

        # [PyTorch] F. TEST DB Ready
        test_data_loader = DataLoader(dataset=self.whole_test_set,
                    num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False,
                    pin_memory=False)
        self.model.eval()
        with torch.no_grad():
            print('[PyTorch] Extract DB Feature')
            self.pool_size = encoder_dim
            if opt.pooling.lower() == 'netvlad': self.pool_size *= opt.num_clusters
            dbFeat_ALL = np.empty((len(self.whole_test_set), self.pool_size))

            for iteration, (input, indices) in enumerate(test_data_loader, 1):
                input = input.to(self.device)
                image_encoding = self.model.encoder(input)
                vlad_encoding = self.model.pool(image_encoding)
                dbFeat_ALL[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()

                del input, image_encoding, vlad_encoding
        del test_data_loader
        self.dbFeat = dbFeat_ALL.astype('float32')

        # [PyTorch] G. TEST DB Matcher Faiss
        print('[PyTorch] FAISS')
        self.faiss_index = faiss.IndexFlatL2(self.pool_size)
        self.faiss_index.add(self.dbFeat)
        print('[PyTorch] >> DONE')
        self.input_transform = Localization_SUB.input_transform()

        # [ROS] ROS Set Initializeing -------------------------------------------------
        print('[  ROS  ] >> START')

        # [ROS] Topic Name Param Server
        self.pos_topic = rospy.get_param('~pub_pos_topic', '/RETRIEVAL_MODULE/DL/UTM')
        self.gps_topic = rospy.get_param('~sub_init_Pos_topic', '/RETRIEVAL_MODULE/LOCAL/MAP/Init')

        self.gtPath_topic = rospy.get_param('~pub_GTPath_topic', '/RETRIEVAL_MODULE/DL/Path/GTUTM')
        self.dlPath_topic = rospy.get_param('~pub_DLPath_topic', '/RETRIEVAL_MODULE/DL/Path/DLUTM')

        self.MCL_topic = rospy.get_param('~pub_debug_topic', '/RETRIEVAL_MODULE/DL/MAP/MCL')
        self.MCL_P_topic = rospy.get_param('~pub_debug_topic', '/RETRIEVAL_MODULE/DL/MAP/MCL2')
        self.TPL_P_topic = rospy.get_param('~pub_debug_topic', '/RETRIEVAL_MODULE/DL/MAP/TPL')
        self.map_3D_topic = rospy.get_param('~sub_map_topic', '/RETRIEVAL_MODULE/LOCAL/MAP/Local')
        self.bld_3D_topic = rospy.get_param('~sub_mapt_topic', '/RETRIEVAL_MODULE/LOCAL/MAP/Build_F')
        self.map_POS_topic = rospy.get_param('~sub_pos_topic', '/RETRIEVAL_MODULE/LOCAL/MAP/Pos')
        self.mcl_POS_topic = rospy.get_param('~sub_pos_topic', '/RETRIEVAL_MODULE/LOCAL/MAP/MCL/Input')         # SE2 Between Pos

        self.pred_IDX_topic = rospy.get_param('~sub_pos_topic', '/RETRIEVAL_MODULE/DL/POS/Index')


        # [ROS] PUB NetVLAD Topic
        self.pose_pub = rospy.Publisher(self.pos_topic, ROS_Odometry, queue_size=50)
        self.debug_pub = rospy.Publisher(self.MCL_topic, ROS_PCL, queue_size=50)
        self.debug_pub2 = rospy.Publisher(self.MCL_P_topic, ROS_PCL, queue_size=50)

        self.debug_pub3 = rospy.Publisher(self.TPL_P_topic, ROS_Image, queue_size=50)

        self.debug_pub4 = rospy.Publisher(self.gtPath_topic, ROS_Path, queue_size=50)
        self.debug_pub5 = rospy.Publisher(self.dlPath_topic, ROS_Path, queue_size=50)
        # [ROS] SUB LocalMap Module Topic
        self.init_pos_sub = rospy.Subscriber(self.gps_topic, ROS_Odometry, self.initposCB, queue_size = 1, buff_size = 10)
        self.rtk_pos_sub = rospy.Subscriber('/ublox/fix/odom', ROS_Odometry, self.rtkposCB, queue_size = 1, buff_size = 10)

        # [ROS] CB Syncronize A / RTRV
        self.local_bld_sub = message_filters.Subscriber(self.bld_3D_topic, ROS_PCL)
        self.local_map_sub = message_filters.Subscriber(self.map_3D_topic, ROS_PCL)
        self.local_pos_sub = message_filters.Subscriber(self.map_POS_topic, ROS_Odometry)
        self.mcl_pos_sub = message_filters.Subscriber(self.mcl_POS_topic, ROS_Odometry)
        ts = message_filters.ApproximateTimeSynchronizer([self.local_bld_sub, self.local_map_sub, self.local_pos_sub, self.mcl_pos_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.locmapCB)

        # [ROS] INIT parameters
        self.initialized = False
        self.initializedTPL = False
        self.initializedALL = 0
        self.initializedPRD = 0

        self.overallIter = 0
        self.overallSum = 0

        self.Vehicle_Pos_Yaw = 0
        self.Vehicle_Pos_X = 0
        self.Vehicle_Pos_Y = 0
        self.Vehicle_Pos_IDX = 0

        self.init_Pos = np.array([0,0])
        self.init_RTK = np.array([0,0])
        self.init_RTK_Header = 0
        self.init_UTM = np.array([0,0])
        self.init_ODM = np.array([0,0])

        self.weight_old = 1
        self.correct_at_n = np.zeros(len([1,5,10,20,30,40,50,60,70,80]))

        self.UTM_2_ODOM_OLD = 0
        self.UTM_PCL_Prev = np.empty(0)
        self.UTM_DB = np.asarray(self.whole_test_set.dbStruct.utmDb)
        self.UTM_Q = np.asarray(self.whole_test_set.dbStruct.utmQ)

        self.RSTL_UTM_DLC = np.empty((0,3))
        self.Localization_SUBF = Localization_SUB()
        self.range = localizer_PARM.MAP_range/2
        self.res = self.range*2/700+0.002

        self.bridge = CvBridge()
        self.GT_Path = ROS_Path()
        self.GT_Path.header.frame_id = "UTM"
        self.DL_Path = ROS_Path()
        self.DL_Path.header.frame_id = "UTM"

        print('[  ROS  ] >> SPIN')
        rospy.spin()

    def locmapCB(self, BLD_Data, PCL_Data, POS_Data, MCL_Data):

        print('PCL IN')
        self.Vehicle_Pos_X = POS_Data.pose.pose.position.x
        self.Vehicle_Pos_Y = POS_Data.pose.pose.position.y
        self.Vehicle_Pos_Yaw = POS_Data.pose.pose.position.z
        self.Vehicle_Pos_IDX = POS_Data.pose.pose.orientation.x

        # [ROS] Image Call Back : Convert Image to PyTorch Tensor
        input_img_Q = Localization_SUB.image_transform(PCL_Data, self.Vehicle_Pos_X, self.Vehicle_Pos_Y)

        # [ROS] MCL
        img_PyTorch = self.input_transform(input_img_Q)
        PRED_Data, DESC_Data = self.descriptorCB(img_PyTorch)

        predictions = self.predMCL(MCL_Data, DESC_Data)

        # [ROS] TMPL SQDIFF
        if self.initializedTPL:
            npPCL_B = self.pclnpCB(BLD_Data, self.Vehicle_Pos_X, self.Vehicle_Pos_Y)
            input_img_Building = self.Localization_SUBF.pcl_transform(npPCL_B, self.Vehicle_Pos_X, self.Vehicle_Pos_Y, False)
            input_img_QT = cv2.cvtColor(input_img_Building, cv2.COLOR_BGR2GRAY)
            UTM_GT_STK = np.empty([5,2])
            UTM_GT_IDX = 0

            for idex in predictions[:5]:
                input_DB = self.whole_test_set.images[idex]
                TF_G2D, dst = self.pclcvCB(input_DB, input_img_Building)
                TF_GTD = self.whole_test_set.dbStruct.utmDb[idex]
                UTM_GT = - TF_G2D + TF_GTD
                UTM_GT_STK[UTM_GT_IDX,:] = UTM_GT
                UTM_GT_IDX += 1

                imgMsg = self.bridge.cv2_to_imgmsg(dst, encoding='mono8')
                imgMsg.header.stamp = PCL_Data.header.stamp
                self.debug_pub3.publish(imgMsg)

            UTM_GT_OUT = self.MDL_FLT(UTM_GT_STK,0.5)
            UTM_U = np.array([self.Vehicle_Pos_X, self.Vehicle_Pos_Y])

            if self.initializedALL == 1 and self.initializedPRD == 1:
                # UTM_GT_OUT = UTM_GT_OUT - self.init_UTM
                print('GT-ODOM DIFF : ', np.linalg.norm((UTM_GT_OUT - self.init_UTM + self.init_ODM) - (UTM_U)))
                print('GT-UTM DIFF : ', np.linalg.norm(UTM_GT_OUT - self.init_UTM))

                UTM_2_ODOM = np.linalg.norm((UTM_GT_OUT - self.init_UTM + self.init_ODM) - (UTM_U))
                UTM_2_UTM = np.linalg.norm(UTM_GT_OUT - self.init_UTM)
                if UTM_2_ODOM > 10 or UTM_2_UTM < 0.5*localizer_PARM.MAP_save or UTM_2_UTM > 1.5*localizer_PARM.MAP_save:
                    self.init_UTM = UTM_GT_OUT
                    self.init_ODM = UTM_U               # RESET
                elif UTM_2_ODOM < 3 and UTM_2_UTM > 0.5*localizer_PARM.MAP_save and UTM_2_UTM < 1.5*localizer_PARM.MAP_save:
                    self.UTM_2_ODOM_OLD = (UTM_GT_OUT - self.init_UTM) + self.init_ODM
                    self.initializedPRD = 2             # FIX INIT

            elif self.initializedALL == 1 and self.initializedPRD == 0:
                self.init_UTM = UTM_GT_OUT
                self.init_ODM = UTM_U
                self.initializedPRD = 1                 # INIT
            elif self.initializedALL == 1 and self.initializedPRD == 2:

                UTM_2_ODOM = np.linalg.norm((UTM_GT_OUT - self.init_UTM + self.init_ODM) - (UTM_U))
                UTM_Pub = (UTM_GT_OUT - self.init_UTM) + self.init_ODM - (self.init_ODM + self.init_Pos - self.init_UTM)
                print('Pub')
                if UTM_2_ODOM < 5:
                    self.utmPB(UTM_Pub)

                DL_Odom = self.init_UTM - self.init_ODM + UTM_U
                pose = PoseStamped()
                pose.header.stamp = rospy.Time.from_sec(POS_Data.pose.pose.orientation.y)
                pose.pose.position.x = DL_Odom[0]
                pose.pose.position.y = DL_Odom[1]
                self.DL_Path.poses.append(pose)
                self.debug_pub4.publish(self.DL_Path)
                print(DL_Odom)
            # self.RSTL_UTM_DLC = np.append(self.RSTL_UTM_DLC,np.array([[POS_Data.pose.pose.orientation.y, UTM_GT_OUT[0], UTM_GT_OUT[1]]]),axis=0)
            # np.save('/home/iismn/WorkSpace/DLC.npy', self.RSTL_UTM_DLC)

        # PUB PATH
        GT_Odom = self.init_RTK + self.init_Pos
        pose = PoseStamped()
        pose.header.stamp = self.init_RTK_Header
        pose.pose.position.x = GT_Odom[0]
        pose.pose.position.y = GT_Odom[1]
        print(GT_Odom)
        self.GT_Path.poses.append(pose)
        self.debug_pub5.publish(self.GT_Path)

    def pclnpCB(self, PCL_Data, Vehicle_Pos_X, Vehicle_Pos_Y):
        LocalMAP = ros_numpy.numpify(PCL_Data)
        # CONVERT PointCloud2 to NP.
        LocalMAP_Pts=np.zeros((LocalMAP.shape[0],4))
        LocalMAP_Pts[:,0]=LocalMAP['x']
        LocalMAP_Pts[:,1]=LocalMAP['y']
        LocalMAP_Pts[:,2]=LocalMAP['z']
        LocalMAP_Pts[:,3]=LocalMAP['intensity']
        ff = np.logical_and((LocalMAP_Pts[:,0] > Vehicle_Pos_X-self.range), (LocalMAP_Pts[:,0] < Vehicle_Pos_X+self.range))
        ss = np.logical_and((LocalMAP_Pts[:,1] > Vehicle_Pos_Y-self.range), (LocalMAP_Pts[:,1] < Vehicle_Pos_Y+self.range))
        tt = np.logical_and((LocalMAP_Pts[:,2] > 1), (LocalMAP_Pts[:,2] < 50))

        indicesB = np.argwhere(np.logical_and(tt,np.logical_and(ff,ss))).flatten()
        npPCL_B    = LocalMAP_Pts[indicesB,0:4]

        return npPCL_B

    def pclcvCB(self, input_DB, input_img_Building):
        Build_idx = np.all(cv2.imread(input_DB) == [0,255,255], axis=-1)
        input_img_DBT = np.zeros([700,700,3],dtype = np.uint8)
        input_img_DBT[Build_idx] = [255,255,255]
        input_img_DBT = cv2.GaussianBlur(input_img_DBT, (0, 0), 0.3)
        input_img_DBT = cv2.cvtColor(input_img_DBT, cv2.COLOR_BGR2GRAY)
        input_img_DBT[np.nonzero(input_img_DBT)] = 255
        input_img_DBT = input_img_DBT.astype(np.uint8)

        input_img_QT = cv2.cvtColor(input_img_Building, cv2.COLOR_BGR2GRAY)
        input_img_QT = cv2.GaussianBlur(input_img_QT, (0, 0), 0.1)
        input_img_QT = np.pad(input_img_QT, ((100,100),(100,100)), 'constant', constant_values=0)

        res = cv2.matchTemplate(input_img_QT,input_img_DBT,cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + 700, top_left[1] + 700)

        FS_img = np.zeros((900,900), dtype=np.uint8)
        cv2.rectangle(FS_img,top_left, bottom_right, 255, 1)
        cv2.rectangle(FS_img,(100,100), (800,800), 255, 1)
        ST_X = np.max([top_left[1],0])
        ST_Y = np.max([top_left[0],0])

        END_X = np.min([top_left[1] + 700,900])
        END_Y = np.min([top_left[0] + 700,900])

        UTM_G2D_Y = (ST_X - 100)*self.res
        UTM_G2D_X = (ST_Y - 100)*self.res

        FS_img[ST_X:END_X, ST_Y:END_Y] = input_img_DBT[0:END_X-ST_X, 0:END_Y-ST_Y]
        dst = cv2.addWeighted( input_img_QT, 0.5, FS_img, 0.5, 0.0);
        # cv2.imshow('Match Result', dst)
        # cv2.waitKey(100)

        TF_G2D = np.asarray([UTM_G2D_X, -UTM_G2D_Y])

        return TF_G2D, dst

    def descriptorCB(self, input_img_Torch):
        input_img_Torch = input_img_Torch.unsqueeze(0)
        input_img_Torch = input_img_Torch.to(self.device)

        image_encoding = self.model.encoder(input_img_Torch)
        vlad_encoding = self.model.pool(image_encoding)
        vlad_encoding = vlad_encoding.detach().cpu().numpy()
        distances, predictions = self.faiss_index.search(vlad_encoding.astype('float32'), 30)
        return predictions, vlad_encoding

    def predPB(self, data_in):
        msg = Int32MultiArray()
        msg.data = data_in
        self.pred_pub.publish(msg)

    def utmPB(self, UTM_GT_OUT):
        msg = ROS_Odometry()

        msg.pose.pose.position.x = UTM_GT_OUT[0]
        msg.pose.pose.position.y = UTM_GT_OUT[1]
        msg.pose.pose.orientation.x = self.Vehicle_Pos_IDX

        self.pose_pub.publish(msg)

    def predMCL(self, U_input, U_desc_input):

        if not self.initialized:
            # [MCL] Initialize Particles
            _, DB_Pred = self.faiss_index.search(U_desc_input.astype('float32'), 80)
            self.UTM_PCL_Prev = self.whole_test_set.dbStruct.utmDb[DB_Pred,:]
            self.UTM_PCL_Prev = np.c_[self.UTM_PCL_Prev[0], np.ones((np.asarray(self.UTM_PCL_Prev).shape[1],1))*self.Vehicle_Pos_Yaw]
            self.initialized = True
            self.weight_old = np.ones((1,np.asarray(self.UTM_PCL_Prev).shape[0]))/np.sum(np.ones((1,np.asarray(self.UTM_PCL_Prev).shape[0])))
            self.weight_old = self.weight_old[0]

        else:
            self.initializedTPL = True
            # [MCL] Prediction
            Pos2D_input = np.array([U_input.pose.pose.position.x, U_input.pose.pose.position.y, U_input.pose.pose.position.z])        # Input [X Y Yaw]
            randXY = (np.random.rand(np.asarray(self.UTM_PCL_Prev).shape[0],1)-0.5)*5
            randTh = (np.random.rand(np.asarray(self.UTM_PCL_Prev).shape[0],1)-0.5)
            rand2D = np.ones((np.asarray(self.UTM_PCL_Prev).shape[0],1))*Pos2D_input + np.c_[randXY[:], randXY[:], randTh[:]]

            # [MCL] Weight Update
            UTM_PCL_Curr = self.MDL_2D(self.UTM_PCL_Prev,-rand2D)
            Loss_Idx = []
            for U_input_Idx in UTM_PCL_Curr[:,0:2]:
                Loss_MCL = np.sum(np.abs((self.UTM_DB-np.ones((self.UTM_DB.shape[0],2))*U_input_Idx))**2,axis=-1)**(1./2)
                Loss_Idx.append(np.argmin(Loss_MCL))

            Loss_Idx = np.array(Loss_Idx)
            VLAD_Distance = np.sum(np.abs((self.dbFeat[Loss_Idx]-U_desc_input))**2,axis=-1)**(1./2)

            # [MCL] Resample
            Weight = self.weight_old * ((np.max(VLAD_Distance) - VLAD_Distance) / (np.max(VLAD_Distance) -  np.min(VLAD_Distance)))
            Weight = Weight/np.sum(Weight)
            print('Neff : ', 1./np.sum(np.square(Weight)))
            Neff = 1./np.sum(np.square(Weight))
            if Neff < 60:
                DB_Pred = np.array([MCL.stratified_resample(Weight)])
                Loss_Idx = Loss_Idx[DB_Pred]
                if self.initializedPRD != 2:
                    self.initializedALL = 0
                    self.initializedPRD = 0
            else:
                self.initializedALL = 1
                Loss_Idx = np.array([Loss_Idx])

            # [MCL] Segment Top-10
            # EXTRACT TOP-K
            self.faiss_index_MCL = faiss.IndexFlatL2(self.pool_size)
            self.faiss_index_MCL.add(self.dbFeat[Loss_Idx[0]])
            _, DB_PredT = self.faiss_index_MCL.search(U_desc_input.astype('float32'), 10)
            DB_PredTX = Loss_Idx[:,DB_PredT[0]]
            # DB_PredTX = DB_PredTX.astype(int)

            # # FIND NEAREST QUERY
            # UTM_Curr = np.array([self.Vehicle_Pos_X, self.Vehicle_Pos_Y])
            # # print(UTM_Curr)
            # UTM_Curr = UTM_Curr + self.init_Pos
            # UTM_Curr_Func = np.sum(np.abs((self.UTM_Q-np.ones((self.UTM_Q.shape[0],2))*UTM_Curr))**2,axis=-1)**(1./2)
            # UTM_Curr_Idx = np.argmin(UTM_Curr_Func)
            #
            # # FIND QUERY TO GT
            # self.overallIter += 1
            # GT = self.whole_test_set.getPositives()
            # for itc,ntc in enumerate([1,5,10,20,30,40,50,60,70,80]):
            #     if np.any(np.in1d(DB_PredTX[:ntc], GT[UTM_Curr_Idx])):
            #         self.correct_at_n[itc:] += 1
            #         # print(itc)
            #         break

            # [MCL] MCL Publish
            # print('Recall : ', self.correct_at_n/self.overallIter)

            self.MCL_prvPB(Loss_Idx, U_input)
            self.MCL_prdPB(Loss_Idx, U_input, rand2D)

            self.UTM_PCL_Prev = self.whole_test_set.dbStruct.utmDb[Loss_Idx.astype(int),:]
            self.UTM_PCL_Prev = np.c_[self.UTM_PCL_Prev[0], np.ones((np.asarray(self.UTM_PCL_Prev).shape[1],1))*self.Vehicle_Pos_Yaw]

            return DB_PredTX[0]

    def initposCB(self, posInput):
        Vehicle_Pos_X = posInput.pose.pose.position.x
        Vehicle_Pos_Y = posInput.pose.pose.position.y
        Pos_Init = np.array([Vehicle_Pos_X, Vehicle_Pos_Y])
        self.init_Pos = Pos_Init

    def rtkposCB(self, posInput):
        Pos_Init = np.array([posInput.pose.pose.position.x, posInput.pose.pose.position.y])
        self.init_RTK = Pos_Init
        self.init_RTK_Header = posInput.header.stamp





    def MCL_prvPB(self, Loss_Idx, U_input):
        UTM_PCL_Curr = self.whole_test_set.dbStruct.utmDb[Loss_Idx,:]
        UTM_PCL_Curr_PCL = UTM_PCL_Curr[0][:,0:2] - self.init_Pos
        ROS_MCL = np.zeros(Loss_Idx.shape[1], dtype=[
          ('x', np.float32),
          ('y', np.float32),
          ('z', np.float32),
        ])
        ROS_MCL['x'] = UTM_PCL_Curr_PCL[:, 0]
        ROS_MCL['y'] = UTM_PCL_Curr_PCL[:, 1]
        ROS_MCL_Msg = ros_numpy.msgify(ROS_PCL, ROS_MCL)
        ROS_MCL_Msg.header = U_input.header
        self.debug_pub.publish(ROS_MCL_Msg)

    def MCL_prdPB(self, Loss_Idx, U_input, rand2D):
        UTM_PCL_Curr = self.whole_test_set.dbStruct.utmDb[Loss_Idx,:]
        UTM_PCL_Curr = np.c_[UTM_PCL_Curr[0], np.ones((np.asarray(UTM_PCL_Curr).shape[1],1))*self.Vehicle_Pos_Yaw]
        UTM_PCL_Curr = self.MDL_2D(UTM_PCL_Curr,-rand2D)
        UTM_PCL_Curr_PCL = UTM_PCL_Curr[:,0:2] - self.init_Pos
        ROS_MCL = np.zeros(Loss_Idx.shape[1], dtype=[
          ('x', np.float32),
          ('y', np.float32),
          ('z', np.float32),
        ])
        ROS_MCL['x'] = UTM_PCL_Curr_PCL[:, 0]
        ROS_MCL['y'] = UTM_PCL_Curr_PCL[:, 1]
        ROS_MCL_Msg = ros_numpy.msgify(ROS_PCL, ROS_MCL)
        ROS_MCL_Msg.header = U_input.header
        self.debug_pub2.publish(ROS_MCL_Msg)

    def MDL_2D(self, UTM_PCL, U_input):

        X_ij = UTM_PCL[:,0]           # X
        Y_ij = UTM_PCL[:,1]           # Y
        Th_ij = UTM_PCL[:,2]          # Z

        X_jk = U_input[:,0]
        Y_jk = U_input[:,1]
        Th_jk = U_input[:,2]

        Pos_2D_RSLT_X = X_jk*np.cos(Th_ij) - Y_jk*np.sin(Th_ij) +X_ij;
        Pos_2D_RSLT_Y = X_jk*np.sin(Th_ij) + Y_jk*np.cos(Th_ij) +Y_ij;
        Pos_2D_RSLT_Z = Th_ij + Th_jk;

        UTM_PCL_Curr = np.c_[Pos_2D_RSLT_X, Pos_2D_RSLT_Y, Pos_2D_RSLT_Z]

        return UTM_PCL_Curr

    def MDL_FLT(self, UTM_GT, threshold):

        SHP = np.unique(UTM_GT,axis=0)
        if SHP.shape[0] > 3:
            median = np.median(UTM_GT, axis=0)
            diff = np.sum((UTM_GT - median) ** 2, axis=-1)
            diff = np.sqrt(diff)
            med_abs_deviation = np.median(diff)
            # scale constant 0.6745
            modified_z_score = 0.6745 * diff / med_abs_deviation
            # print(modified_z_score)
            UTM_GT = UTM_GT[modified_z_score > threshold]
            UTM_GT = np.mean(UTM_GT, axis=0)
            # print(UTM_GT)
            return UTM_GT

        else:
            UTM_GT = np.mean(UTM_GT, axis=0)
            return UTM_GT

class Localization_SUB():
    def __init__(self):
        self.range = localizer_PARM.MAP_range/2
        self.res = self.range*2/700+0.001 #100M / 700 Pixels

    def input_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    def image_transform(PCL_Data, Vehicle_Pos_X, Vehicle_Pos_Y):
        LocalMAP = ros_numpy.numpify(PCL_Data)

        LocalMAP_Points=np.zeros((LocalMAP.shape[0],4))
        LocalMAP_Points[:,0]=LocalMAP['x']
        LocalMAP_Points[:,1]=LocalMAP['y']
        LocalMAP_Points[:,2]=LocalMAP['z']
        LocalMAP_Points[:,3]=LocalMAP['intensity']

        MAP_range = localizer_PARM.MAP_range/2
        MAP_resolution = MAP_range*2/700+0.0005 #100M / 700 Pixels

        ff = np.logical_and((LocalMAP_Points[:,0] > Vehicle_Pos_X-MAP_range), (LocalMAP_Points[:,0] < Vehicle_Pos_X+MAP_range))
        ss = np.logical_and((LocalMAP_Points[:,1] > Vehicle_Pos_Y-MAP_range), (LocalMAP_Points[:,1] < Vehicle_Pos_Y+MAP_range))
        # tt = np.logical_and((LocalMAP_Points[:,2] > 0.5), (LocalMAP_Points[:,2] < 50))
        indices = np.argwhere(np.logical_and(ff,ss)).flatten()

        IMG_x = (-LocalMAP_Points[indices,1]/MAP_resolution).astype(np.int32)
        IMG_y = (LocalMAP_Points[indices,0]/MAP_resolution).astype(np.int32)
        IMG_x += int(np.floor((Vehicle_Pos_Y+MAP_range)/MAP_resolution))
        IMG_y -= int(np.floor((Vehicle_Pos_X-MAP_range)/MAP_resolution))
        IMG_i = LocalMAP_Points[indices,3].astype(np.int32)

        # FILL PIXEL VALUES IN IMAGE ARRAY
        x_max = int((MAP_range*2)/MAP_resolution)
        y_max = int((MAP_range*2)/MAP_resolution)
        input_img = np.zeros([700, 700,3], dtype=np.uint8)

        HSV_Table = cm.hsv(range(256))

        input_img[IMG_x-1, IMG_y-1] = HSV_Table[IMG_i,0:3]*255

        # opencv_image=cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
        # cv2.imshow("PIL2OpenCV",opencv_image)
        # cv2.waitKey(0)

        return input_img

    def pcl_transform(self, npPCL, Vehicle_Pos_X, Vehicle_Pos_Y, Building):

        if Building == True:
            FLT_S = self.pcl_normal(npPCL, Vehicle_Pos_X, Vehicle_Pos_Y)
            FLT_S = FLT_S[0]
            input_imgBF = self.pcl_imgTF(FLT_S)
            return input_imgBF

        else:

            npPCL[:,0] = npPCL[:,0] - Vehicle_Pos_X
            npPCL[:,1] = npPCL[:,1] - Vehicle_Pos_Y
            input_img = self.pcl_imgTF(npPCL)

            return input_img

    def pcl_normal(self, PCL_IN, Vehicle_Pos_X, Vehicle_Pos_Y):
        PCL_IN[:,0] = PCL_IN[:,0] - Vehicle_Pos_X
        PCL_IN[:,1] = PCL_IN[:,1] - Vehicle_Pos_Y

        minX = np.min(PCL_IN[:,0])
        maxX = np.max(PCL_IN[:,0])
        minY = np.min(PCL_IN[:,1])
        maxY = np.max(PCL_IN[:,1])

        SubMAP_SIZE = (maxX-minX)/10


        FLT_S = np.array([[[0,0,0,0]]])

        for i in range(10):
            for j in range(10):
                idx_X = np.logical_and((PCL_IN[:,0] >= minX+i*SubMAP_SIZE), (PCL_IN[:,0] < minX+(i+1)*SubMAP_SIZE))
                idx_Y = np.logical_and((PCL_IN[:,1] >= minY+j*SubMAP_SIZE), (PCL_IN[:,1] < minY+(j+1)*SubMAP_SIZE))
                idx_T = np.argwhere(np.logical_and(idx_X,idx_Y)).flatten()
                if idx_T.shape[0] > 200:
                    sub_npPCL = PCL_IN[idx_T,:]
                    PCL_NormF = o3d.geometry.PointCloud()
                    PCL_NormF.points = o3d.utility.Vector3dVector(sub_npPCL[:,0:3])
                    # PCL_NormF = PCL_NormF.voxel_down_sample(voxel_size=0.5)
                    PCL_NormF.estimate_normals( search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
                    norm = np.asarray(PCL_NormF.normals)
                    idx_F = np.where(np.absolute(norm[:,2]) < 0.04)
                    FLT_S = np.append(FLT_S, sub_npPCL[idx_F,:], axis=1)


        # PCL_NormF = o3d.geometry.PointCloud()
        # PCL_NormF.points = o3d.utility.Vector3dVector(PCL_IN[:,0:3])
        #
        # PCL_NormF.estimate_normals(
        #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=200))
        #
        # norm = np.asarray(PCL_NormF.normals)
        # idx_F = np.where(np.absolute(norm[:,2]) < 0.04)
        #
        # FLT_S = PCL_IN[idx_F,:]

        return FLT_S

    def pcl_imgTF(self, PCL_IN):
        IMG_x = (-PCL_IN[:,1]/self.res).astype(np.int32)
        IMG_y = (PCL_IN[:,0]/self.res).astype(np.int32)
        IMG_x += int(np.floor((self.range)/self.res))
        IMG_y -= int(np.floor((self.range)/self.res))
        IMG_i = PCL_IN[:,3].astype(np.int32)


        # FILL PIXEL VALUES IN IMAGE ARRAY
        x_max = int((self.range*2)/self.res)
        y_max = int((self.range*2)/self.res)
        input_img = np.zeros([700, 700,3], dtype=np.uint8)

        HSV_Table = cm.hsv(range(256))
        input_img[IMG_x, IMG_y] = HSV_Table[IMG_i,0:3]*255

        return input_img


if __name__ == '__main__':
    rospy.init_node("DL_Module_MCL")
    localizer_PARM = Localization_PARAM()
    localizer_SUB = Localization_SUB()
    localizer_DL = Localization_MAIN()
