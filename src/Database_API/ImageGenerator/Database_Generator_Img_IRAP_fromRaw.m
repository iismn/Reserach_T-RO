%% Convert LAS - PCD
clc; clear;
Data_Path = '/home/iismn/WorkSpace_SMB/WSL_LINUX/All_DB/ComplexUrban/07/urban07/sensor_data/';
SAVE_Path_2D = '/home/iismn/WorkSpace/Research/ComplexUrban_07/DeepLearningDB/2D/';
SAVE_Path_POS = '/home/iismn/WorkSpace/Research/ComplexUrban_07/DeepLearningDB/POS_2D/';

% Read GlobalPose
GT_TBL = readtable(strcat(Data_Path,'global_pose.csv'),'Format','%s %s %s %s %s %s %s %s %s %s %s %s %s');
GT_ = table2cell(GT_TBL);
GT_STR = string(GT_);

%% Convert String to Num
GT_Pos = [];
isFirst = 0;
AFCompt_STAMP = 0;
BFCompt_STAMP = 0;
for i = 1:size(GT_STR,1)
    trans = [sscanf(GT_STR(i,5), '%lf') sscanf(GT_STR(i,9), '%lf') 0];
    GT_Pos = [GT_Pos; sscanf(GT_STR(i,2), '%lf'), sscanf(GT_STR(i,3), '%lf'), sscanf(GT_STR(i,4), '%lf'), ...
                    sscanf(GT_STR(i,5), '%lf'), sscanf(GT_STR(i,6), '%lf'), sscanf(GT_STR(i,7), '%lf'), ...
                    sscanf(GT_STR(i,8), '%lf'), sscanf(GT_STR(i,9), '%lf'), sscanf(GT_STR(i,10), '%lf'), ...
                    sscanf(GT_STR(i,11), '%lf'), sscanf(GT_STR(i,12), '%lf'), sscanf(GT_STR(i,13), '%lf')];
    SO_3 = [sscanf(GT_STR(i,2), '%lf') sscanf(GT_STR(i,3), '%lf') sscanf(GT_STR(i,4), '%lf'); 
            sscanf(GT_STR(i,6), '%lf'), sscanf(GT_STR(i,7), '%lf') sscanf(GT_STR(i,8), '%lf');
            sscanf(GT_STR(i,10), '%lf'), sscanf(GT_STR(i,11), '%lf'), sscanf(GT_STR(i,12), '%lf')];
    Quat = rotm2quat(SO_3);
    STAMP = insertBefore(GT_STR(i,1),11,".");
    Compt_STAMP = sscanf(GT_STR(i,1),'%ld');
    AFCompt_STAMP = Compt_STAMP;

    if isFirst == 0
        TUM_Format = {STAMP trans(1) trans(2) trans(3) Quat(1) Quat(2) Quat(3) Quat(4)};
        isFirst = 1;
        BFCompt_STAMP = Compt_STAMP;
    end
    % # timestamp tx ty tz qx qy qz qw
    if AFCompt_STAMP - BFCompt_STAMP < 300000000
        TUM_Format(end+1,:) = {STAMP trans(1) trans(2) trans(3) Quat(1) Quat(2) Quat(3) Quat(4)};
    end
    BFCompt_STAMP = AFCompt_STAMP;

end

trajectory.Location = [GT_Pos(:,4) GT_Pos(:,8)];
%%
F_Path = ('/home/iismn/WorkSpace_SMB/WSL_LINUX/Research_SW/IEEE_T-RO/Result/Ref_CPLX05_Cpm_CPLX07/RE_Raw/');
writecell(TUM_Format,strcat(F_Path,'TUM_F_GT.csv'));
writecell(TUM_Format,strcat(F_Path,'TUM_F_GT.txt'),'Delimiter',' ');


%% C. 2D NGII IMG CONVERTER
% C1. Plot 2D Digital Map 
geojson_Folder = '/home/iismn/WorkSpace/Research/ComplexUrban_07/GeoJSON/';
rawB = fread(fopen(strcat(geojson_Folder,'ComplexUrban_07_Build.geojson')),inf);
rawB = char(rawB');
geojsonValue_BUILD = jsondecode(rawB);

rawR = fread(fopen(strcat(geojson_Folder,'ComplexUrban_07_Road.geojson')),inf);
rawR = char(rawR');
geojsonValue_ROAD = jsondecode(rawR);
%% Image Saving
%----------------------------------------------------------
res = 1000;
DISTMAP = 80;
Output.Size = DISTMAP;
map_Size = DISTMAP;
%----------------------------------------------------------
%% Trajectory Interpolation
Val_SplitRatio = 10;
Interpolated_Trajectory = [];
for i = 1:round(size(trajectory.Location,1)/Val_SplitRatio)
    
    iter = round(i*Val_SplitRatio);
    
    if iter >= size(trajectory.Location,1)
         continue
    end
    
    Interpolated_Trajectory = [Interpolated_Trajectory; trajectory.Location(iter,:)];
    
end

% Interpolated_Trajectory = trajectory.Location;
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
        UTM_X_FT = X_FT;
        UTM_Y_FT = Y_FT;

%         UTM_X_FT = X_FT + UTM_ST_X;
%         UTM_Y_FT = Y_FT + UTM_ST_Y;

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

%% Debugging
UTM_Pos_Deg = [];
for i = 1:size(Interpolated_Trajectory,1)
    [Lat, Long] = utm2deg(Interpolated_Trajectory(i,1), Interpolated_Trajectory(i,2),'52 S');
    UTM_Pos_Deg = [UTM_Pos_Deg; Lat Long];
end

figure(2)
webmap('World Imagery')

wmlimits([min(UTM_Pos_Deg(:,1)) max(UTM_Pos_Deg(:,1))],[min(UTM_Pos_Deg(:,2)) max(UTM_Pos_Deg(:,2))])
wmline(UTM_Pos_Deg(:,1),UTM_Pos_Deg(:,2),'Color','yellow')

