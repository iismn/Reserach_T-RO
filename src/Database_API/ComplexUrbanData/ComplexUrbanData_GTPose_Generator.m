%% KAIST IRIS Lab
% dbStruct Generator for PyTorch Siamse-NetVLAD
%
% Autonomous Vehicle Localization without Prior High-Definition Map
% Transaction of Robotics (T-RO) Supplementary Material
%
% iismn@kaist.ac.kr
% Sangmin Lee
% KAIST IRiS Lab. Autnomous Vehicle Team PHAROS

%% A. Read Table for Generating Global Pose Data
clc; clear;

% Read DataStamp
DT_STAMP_TBL = readtable('data_stamp.csv','Format','%s %s');
DT_STAMP = table2cell(DT_STAMP_TBL);
DT_STAMP_STR = string(DT_STAMP);
% Read GlobalPose
GT_STAMP_TBL = readtable('global_pose.csv','Format','%s %s %s %s %s %s %s %s %s %s %s %s %s');
GT_STAMP = table2cell(GT_STAMP_TBL(:,1));
GT_STAMP_STR = string(GT_STAMP);


%% B. Edit Ground Truth Stamp
FUSED_STAMP = cell(size(DT_STAMP,1)+size(GT_STAMP,1),3);
GT_ID = cell(size(GT_STAMP,1),1);

for i = 1:size(GT_STAMP,1)
    GT_ID(i) = {"GT"};
end

DT_STAMP_STR = cellstr(DT_STAMP_STR(:,1));
GT_STAMP_STR = cellstr(GT_STAMP_STR(:,1));

%% C. GENERATING NEW DATA
FUSED_STAMP(1:size(DT_STAMP,1),1) = DT_STAMP_STR;
FUSED_STAMP(1:size(DT_STAMP,1),2) = DT_STAMP(:,1);
FUSED_STAMP(1:size(DT_STAMP,1),3) = DT_STAMP(:,2);
FUSED_STAMP(size(DT_STAMP,1)+1:end,1) = GT_STAMP_STR;
FUSED_STAMP(size(DT_STAMP,1)+1:end,2) = GT_STAMP(:,1);
FUSED_STAMP(size(DT_STAMP,1)+1:end,3) = GT_ID(:,1);

%% D. Sort Data and Save GT Fused with Table
FUSED_STAMP = sortrows(FUSED_STAMP,1);
FNL_STAMP = cell(size(DT_STAMP,1)+size(GT_STAMP,1),2);
FNL_STAMP(:,1) = FUSED_STAMP(:,1);
FNL_STAMP(:,2) = FUSED_STAMP(:,3);
writetable(cell2table(FNL_STAMP),'data_stamp_fused.csv','WriteVariableNames',false)

