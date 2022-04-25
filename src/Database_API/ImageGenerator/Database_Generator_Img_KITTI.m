%% Deep Learning / DB GENERATOR
% iismn@kaist.ac.kr
% KAIST IRiS Lab.
% Autonomouse Vehicle Team
%
% DB Generator with PCD + LocalMap Point DataBase
% DB for Siamese OSM-NetVLAD DeepLearning Network
% Research : Autonomous Driving without High-Definition Detailed Prior-Map
%
% Copyright 2021.9.15

clc; clear;
map = pcread('GlobalMapRGB.pcd');
trajectory = pcread('trajectory.pcd');

%%
% clc; clear; 
SAVE_Path_2D = '/home/iismn/WorkSpace/Research/KITTI_06_Dataset/DeepLearningDB/2D/';
SAVE_Path_3D = '/home/iismn/WorkSpace/Research/KITTI_06_Dataset/DeepLearningDB/3D/';
SAVE_Path_POS = '/home/iismn/WorkSpace/Research/KITTI_06_Dataset/DeepLearningDB/POS_3D/';

%% B. ROS BAG to START POINT
COORD = [49.0534930479 8.39721998765];
Lat = COORD(1);
Long =  COORD(2);
[LOCAL_Coordinate.StartPoint.UTM_X, LOCAL_Coordinate.StartPoint.UTM_Y, ~] = deg2utm(Lat, Long);

%% C. 2D NGII IMG CONVERTER
% C1. Plot 2D Digital Map 

rawB = fread(fopen('kitti05_buildings.geojson'),inf);
rawB = char(rawB');
geojsonValue_BUILD = jsondecode(rawB);

rawR = fread(fopen('kitti05_roads.geojson'),inf);
rawR = char(rawR');
geojsonValue_ROAD = jsondecode(rawR);
%% Parameter Setting
%----------------------------------------------------------
res = 1000;
DISTMAP = 50;
Output.Size = DISTMAP;
map_Size = DISTMAP;
%----------------------------------------------------------
%% Trajectory Interpolation
% DB.List_UTM_3D = dir('**/POS_3D');
% Iteration = size(DB.List_UTM_3D,1);
% FList_UTM_2D = [];
% Interpolated_Trajectory = [];
% isFirst = 0;

[Interpolated_Trajectory_X, Interpolated_Trajectory_Y, ~] = deg2utm(kitti06osmpose(2,:), kitti06osmpose(1,:));
Interpolated_Trajectory = [Interpolated_Trajectory_X, Interpolated_Trajectory_Y];


%% POS Saving
Iteration = size(trajectory.Location,1);

if Iteration >= 5
    for i = 1:Iteration

        LOCAL_Coordinate.X_FT = double(trajectory.Location(i,1));
        LOCAL_Coordinate.Y_FT = double(trajectory.Location(i,2));

        % A3. Convert Local UTM to Global UTM WGS 84
        LOCAL_Coordinate.LocalPoint.UTM_X_FT = LOCAL_Coordinate.X_FT + LOCAL_Coordinate.StartPoint.UTM_X;
        LOCAL_Coordinate.LocalPoint.UTM_Y_FT = LOCAL_Coordinate.Y_FT + LOCAL_Coordinate.StartPoint.UTM_Y;
        
        Fname = sprintf('%08d',i);
        fileID = fopen(strcat(SAVE_Path_POS,Fname,'.txt'),'w');
        fprintf(fileID,'%f64 %f64',[LOCAL_Coordinate.LocalPoint.UTM_X_FT LOCAL_Coordinate.LocalPoint.UTM_Y_FT]);
        fclose(fileID);


    end

end


%% IMG Saving
blankimage = ones(res,res,3);
blankimage(:,:,:) = 0;
Output.Img = blankimage;


Iteration = size(Interpolated_Trajectory,1);

if Iteration >= 5
    parfor (i = 1:Iteration)
        Fname = sprintf('%08d',i);
        %% A. DATA LOADER
        % A1. Load PCL / Coordinate TXT 
        X_FT = double(Interpolated_Trajectory(i,1));
        Y_FT = double(Interpolated_Trajectory(i,2));

        % A3. Convert Local UTM to Global UTM WGS 84
        UTM_X_FT = X_FT;
        UTM_Y_FT = Y_FT;

        %% B. 3D PCL IMG CONVERTER
        % B1. Road Segmentation - Initial Filter with Plane Fit
        maxX = UTM_X_FT + DISTMAP;
        minX = UTM_X_FT - DISTMAP;
        maxY = UTM_Y_FT + DISTMAP;
        minY = UTM_Y_FT - DISTMAP;
       
        %%
        blankimage = ones(res,res,3);
        blankimage(:,:,:) = 0;
        [~,~,blankimage]=plot_NGII(geojsonValue_BUILD, 'BUILDING', maxX , minX, maxY, minY, blankimage, map_Size);
        [~,~,blankimage]=plot_NGII(geojsonValue_ROAD, 'ROAD', maxX , minX, maxY, minY, blankimage, map_Size);
        
        imwrite(blankimage,strcat(SAVE_Path_2D,Fname,'.png'))
        i
        fileID = fopen(strcat(SAVE_Path_POS,Fname,'.txt'),'w');
        fprintf(fileID,'%f64 %f64',[UTM_X_FT UTM_Y_FT]);
        fclose(fileID);


    end

end

%% PCL Saving
%----------------------------------------------------------
%%
res = 700;
blankimage = ones(res,res,3);
map_Build_FULL = [double(Buildmap.Location)];
map_Road_FULL = [double(Roadmap.Location)];
tic
parfor (i = 1: size(trajectory.Location,1))
    Fname = sprintf('%08d',i);
    X = trajectory.Location(i,1);
    Y = trajectory.Location(i,2);
    [row, ~, ~] = find(map_Build_FULL(:,1) < X+map_Size & map_Build_FULL(:,1) > X-map_Size & map_Build_FULL(:,2) < Y+map_Size & map_Build_FULL(:,2) > Y-map_Size);
    build_R= map_Build_FULL(row,:);
    build_u = round((build_R(:,1)-X+map_Size)*res/(map_Size*2));
    build_v = round((build_R(:,2)-Y+map_Size)*res/(map_Size*2));

    %%
    [row, ~, ~] = find(map_Road_FULL(:,1) < X+map_Size & map_Road_FULL(:,1) > X-map_Size & map_Road_FULL(:,2) < Y+map_Size & map_Road_FULL(:,2) > Y-map_Size);
    road_R= map_Road_FULL(row,:);
    road_u = round((road_R(:,1)-X+map_Size)*res/(map_Size*2));
    road_v = round((road_R(:,2)-Y+map_Size)*res/(map_Size*2));
    
    %%
    blankimage = zeros(res,res,3);
    blankimage = insertShape(blankimage, 'Circle', [build_u res-build_v ones(size(build_u,1),1)*0.05],'LineWidth',1, 'Color', 'yellow');
    blankimage = insertShape(blankimage, 'Circle', [road_u res-road_v ones(size(road_u,1),1)*0.05],'LineWidth',1, 'Color', 'red');
    blankimage = imresize(blankimage,[700 700]);
    imshow(blankimage)
    
    imwrite(blankimage,strcat(SAVE_Path_3D,Fname,'.png'))
end
toc

%% PCL Saving (RGB MAP)
%----------------------------------------------------------
%%
res = 700;
blankimage = ones(res,res,3);
map_FULL = [double(map.Location) double(map.Color)];
map_Size = 50;
tic
parfor (i = 1: size(trajectory.Location,1))
    Fname = sprintf('%08d',i);
    X = trajectory.Location(i,1);
    Y = trajectory.Location(i,2);
    [row, ~, ~] = find(map_FULL(:,1) < X+map_Size & map_FULL(:,1) > X-map_Size & map_FULL(:,2) < Y+map_Size & map_FULL(:,2) > Y-map_Size);
    temp = map_FULL(row,:);
    build_R=find(ismember(temp(:,4:6),[255 200 0],'row'));
    build_u = round((temp(build_R,1)-X+map_Size)*res/(map_Size*2));
    build_v = round((temp(build_R,2)-Y+map_Size)*res/(map_Size*2));

    %%
    road_R=find(ismember(temp(:,4:6),[255 0 255],'row'));
    road_u = round((temp(road_R,1)-X+map_Size)*res/(map_Size*2));
    road_v = round((temp(road_R,2)-Y+map_Size)*res/(map_Size*2));
    
    %%
    blankimage = zeros(res,res,3);
    blankimage = insertShape(blankimage, 'Circle', [build_u res-build_v ones(size(build_u,1),1)*0.05],'LineWidth',1, 'Color', 'yellow');
    blankimage = insertShape(blankimage, 'Circle', [road_u res-road_v ones(size(road_u,1),1)*0.05],'LineWidth',1, 'Color', 'red');
    blankimage = imresize(blankimage,[700 700]);
%     imshow(blankimage)
    
    imwrite(blankimage,strcat(SAVE_Path_3D,Fname,'.png'))
end
toc




    




