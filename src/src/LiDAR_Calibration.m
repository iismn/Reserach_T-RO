rosinit('192.168.50.3')
%%
OS1_128 = rossubscriber('/os_cloud_node/points');
VLP_16L = rossubscriber('/vlp_l/velodyne_points');
VLP_16R = rossubscriber('/vlp_r/velodyne_points');

%%
T_PCL = OS1_128.LatestMessage;
L_PCL = VLP_16L.LatestMessage;
R_PCL = VLP_16R.LatestMessage;


T_PCL_Matlab = pointCloud(readXYZ(T_PCL));
L_PCL_Matlab = pointCloud(readXYZ(L_PCL));
R_PCL_Matlab = pointCloud(readXYZ(R_PCL));

%%
close all;
roi = [5 15 -15 15 -15 15];
% roi = [-30 30 -30 30 -30 30];
indices = findPointsInROI(T_PCL_Matlab,roi);
T_PCL_Matlab_FLT = select(T_PCL_Matlab,indices);
indices = findPointsInROI(L_PCL_Matlab,roi);
L_PCL_Matlab_FLT = select(L_PCL_Matlab,indices);
indices = findPointsInROI(R_PCL_Matlab,roi);
R_PCL_Matlab_FLT = select(R_PCL_Matlab,indices);

% plot3(T_PCL_Matlab_FLT.Location(:,1),T_PCL_Matlab_FLT.Location(:,2),T_PCL_Matlab_FLT.Location(:,3),'.','Color','r')
% hold on; grid on; axis equal
% plot3(L_PCL_Matlab_FLT.Location(:,1),L_PCL_Matlab_FLT.Location(:,2),L_PCL_Matlab_FLT.Location(:,3),'.','Color','g')
% plot3(R_PCL_Matlab_FLT.Location(:,1),R_PCL_Matlab_FLT.Location(:,2),R_PCL_Matlab_FLT.Location(:,3),'.','Color','b')

%%
[tformL,~,temp] = pcregistericp(L_PCL_Matlab_FLT,T_PCL_Matlab_FLT,'Metric','pointToPlane','InlierRatio',0.8,'MaxIterations',200)
L_PCL_Matlab_FLT_MV = pctransform(L_PCL_Matlab_FLT,tformL);

[tformR,~,temp] = pcregistericp(R_PCL_Matlab_FLT,T_PCL_Matlab_FLT,'Metric','pointToPlane','InlierRatio',0.8,'MaxIterations',200)
R_PCL_Matlab_FLT_MV = pctransform(R_PCL_Matlab_FLT,tformR);


%%
gridSize = 100;
gridStep = 0.3;

% tform = pcregistercorr(movingCorrected,fixedCorrected,gridSize,gridStep);
[tformL,~,temp] = pcregisterndt(L_PCL_Matlab_FLT,T_PCL_Matlab_FLT,gridStep,'InitialTransform',tfromL_TOT2)
L_PCL_Matlab_FLT_MV = pctransform(L_PCL_Matlab_FLT,tformL);

[tformR,~,temp] = pcregisterndt(R_PCL_Matlab_FLT,T_PCL_Matlab_FLT,gridStep,'InitialTransform',tfromR_TOT2)
R_PCL_Matlab_FLT_MV = pctransform(R_PCL_Matlab_FLT,tformR);

%%
L_PCL_Matlab_FLT_DS = pcdownsample(L_PCL_Matlab_FLT,'gridAverage',0.2);
R_PCL_Matlab_FLT_DS = pcdownsample(R_PCL_Matlab_FLT,'gridAverage',0.2);
T_PCL_Matlab_FLT_DS = pcdownsample(T_PCL_Matlab_FLT,'gridAverage',0.2);
tformL = pcregisterndt(L_PCL_Matlab_FLT_DS,T_PCL_Matlab_FLT_DS,gridStep,'InitialTransform',tfromL_TOT2)
tformR = pcregisterndt(R_PCL_Matlab_FLT_DS,T_PCL_Matlab_FLT_DS,gridStep,'InitialTransform',tfromR_TOT2)
L_PCL_Matlab_FLT_MV = pctransform(L_PCL_Matlab_FLT_DS,tformL);
R_PCL_Matlab_FLT_MV = pctransform(R_PCL_Matlab_FLT_DS,tformR);

%%
close all
plot3(T_PCL_Matlab_FLT.Location(:,1),T_PCL_Matlab_FLT.Location(:,2),T_PCL_Matlab_FLT.Location(:,3),'.','Color','r')
hold on; grid on; axis equal
plot3(L_PCL_Matlab_FLT_MV.Location(:,1),L_PCL_Matlab_FLT_MV.Location(:,2),L_PCL_Matlab_FLT_MV.Location(:,3),'.','Color','g')
plot3(R_PCL_Matlab_FLT_MV.Location(:,1),R_PCL_Matlab_FLT_MV.Location(:,2),R_PCL_Matlab_FLT_MV.Location(:,3),'.','Color','b')

%%

L_PCL_Matlab_FLT_MV = pctransform(L_PCL_Matlab,tformL);
R_PCL_Matlab_FLT_MV = pctransform(R_PCL_Matlab,tformR);

roi = [-50 50 -50 50 -50 50];
indices = findPointsInROI(T_PCL_Matlab,roi);
T_PCL_Matlab_FLT_MV = select(T_PCL_Matlab,indices);
indices = findPointsInROI(L_PCL_Matlab_FLT_MV,roi);
L_PCL_Matlab_FLT_MV = select(L_PCL_Matlab_FLT_MV,indices);
indices = findPointsInROI(R_PCL_Matlab_FLT_MV,roi);
R_PCL_Matlab_FLT_MV = select(R_PCL_Matlab_FLT_MV,indices);


close all
pcshow(T_PCL_Matlab_FLT_MV)
hold on
pcshow(L_PCL_Matlab_FLT_MV)
pcshow(R_PCL_Matlab_FLT_MV)
