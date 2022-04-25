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
% map = pcread('GlobalMap.pcd');
% Buildmap = pcread('BuildMap.pcd');
% Roadmap = pcread('RoadMap.pcd');
trajectory = pcread('trajectory.pcd');

%%
% clc; clear; 
SAVE_Path_2D = '/home/iismn/WorkSpace/Research/ComplexUrban_33/DeepLearningDB/2D/';
SAVE_Path_3D = '/home/iismn/WorkSpace/Research/ComplexUrban_33/DeepLearningDB/3D/';
SAVE_Path_POS = '/home/iismn/WorkSpace/Research/ComplexUrban_33/DeepLearningDB/POS_2D/';

%% B. ROS BAG to START POINT
COORD = [37.52233301	126.938584561667];
Lat = COORD(1);
Long =  COORD(2);
[UTM_ST_X,UTM_ST_Y, ~] = deg2utm(Lat, Long);

%% C. 2D NGII IMG CONVERTER
% C1. Plot 2D Digital Map 
rawB = fread(fopen('ComplexUrban_33_Build.geojson'),inf);
rawB = char(rawB');
geojsonValue_BUILD = jsondecode(rawB);

rawR = fread(fopen('ComplexUrban_33_Road.geojson'),inf);
rawR = char(rawR');
geojsonValue_ROAD = jsondecode(rawR);
%% Image Saving
%----------------------------------------------------------
res = 1000;
DISTMAP = 100;
Output.Size = DISTMAP;
map_Size = DISTMAP;
%----------------------------------------------------------
%% Trajectory Interpolation
Iteration = size(trajectory.Location,1);

Interpolated_Trajectory = [];
isFirst = 0;
for i = 1:Iteration-1
    
    StartPoint = [trajectory.Location(i,1), trajectory.Location(i,2)];
    EndPoint = [trajectory.Location(i+1,1), trajectory.Location(i+1,2)];
    
    Intrp_X = linspace(StartPoint(1), EndPoint(1),2);
    Intrp_Y = linspace(StartPoint(2), EndPoint(2),2);
    
    if isFirst == 0
        Interpolated_Trajectory = [Interpolated_Trajectory ; [Intrp_X' Intrp_Y']];
        isFirst = 1;
    else
        Interpolated_Trajectory = [Interpolated_Trajectory ; [Intrp_X(2:end)' Intrp_Y(2:end)']];
    end
    
    
end


%%
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
        UTM_X_FT = X_FT + UTM_ST_X;
        UTM_Y_FT = Y_FT + UTM_ST_Y;

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
for (i = 1: size(trajectory.Location,1))
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




    




