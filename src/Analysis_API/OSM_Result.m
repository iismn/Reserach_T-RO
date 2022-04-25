clc; clear;
GT_OSM = readtable('/home/iismn/WorkSpace/Research/ComplexUrban_07/OSMDesc_Result/ComplexUrban_07_osm_descriptor.csv');
osm_descriptor = table2array(GT_OSM);

GT_POS = readtable('/home/iismn/WorkSpace/Research/ComplexUrban_07/OSMDesc_Result/ComplexUrban_07_osm_pose.csv');
osm_pos = table2array(GT_POS)';

PCL_OSM = readtable('/home/iismn/WorkSpace/Research/ComplexUrban_07/OSMDesc_Result/descriptor.csv');
pcl_descriptor = table2array(PCL_OSM);
pcl_descriptor = pcl_descriptor(2:end,:);

PCL_POS = dir('/home/iismn/WorkSpace/Research/ComplexUrban_07/OSMDesc_DATA/PCL_RAW/POS');
pcl_pos = [];
for i = 1:size(PCL_POS,1)
    
    % Pass Through Unknown File Name
    if PCL_POS(i).bytes == 0
        continue
    end
    %-----------------------------------------------------------------------------
    F_Name = PCL_POS(i).name;
    F_Path = PCL_POS(i).folder;
    Full_Path = strcat(F_Path,'/',F_Name);
    
    LOCAL_Coordinate_IO = textscan(fopen(Full_Path),'%f64 %f64 %f64 %f64 %f64 %f64');
    pcl_pos = [pcl_pos cell2mat(LOCAL_Coordinate_IO)'];
    
    fclose all;

end

INIT_POS = '/home/iismn/WorkSpace/Research/ComplexUrban_07/DeepLearningDB/POS_2D/00000001.txt';
init_pos = cell2mat(textscan(fopen(INIT_POS),'%f64 %f64'));

%% Recall Calculation
pcl_pos = pcl_pos';
pcl_pos = pcl_pos(:,1:2);
pcl_pos = pcl_pos + init_pos;
Recall_ = [];
%%
for Recall_N = [1 5 10 20 30 40 50 60 70 80]
    Matched_Idx = 0;
    for i = 1:size(pcl_descriptor,1)

        L1=sum(abs(osm_descriptor-pcl_descriptor(i,:)),2);
        [~, idxMin] = mink(L1,Recall_N);
%         [result_x, result_y] = deg2utm(osm_pos(idxMin,2),osm_pos(idxMin,1));
%         result_pos = [result_x, result_y];
        
        result_pos = [osm_pos(idxMin,1),osm_pos(idxMin,2)]; 
        gt_pos = pcl_pos(i,:);
        
        dist_pos = sqrt(sum((result_pos - gt_pos).^2,2));
        
        if sum(dist_pos<25)>0
            Matched_Idx = Matched_Idx+1;
        end
    end
    
    Recall_N_Result = Matched_Idx/size(pcl_descriptor,1);
    Recall_ = [Recall_; Recall_N_Result];
    Recall_N_Result
    Recall_N
end

%% 

Result_Map = [1,5,10,20,30,40,50,60,70,80];
Recall = [Result_Map', Recall_];


%%
F = figure(2)
hold on
f=fit(Recall(:,1),Recall(:,2),'smoothingspline','SmoothingParam',1.0);
Top_K = 1:0.01:80;
Recall_K = f(Top_K);

hold on
AX = gca;
% plot(Top_K,Recall_K,'LineWidth',3,'Color',[1 0.5 0])
plot(Recall(:,1),Recall(:,2),'LineWidth',3,'Color',[1 0.5 0])
hold on; grid on;
plot(Recall(:,1),Recall(:,2),'x','LineWidth',3,'Color',[1 0.5 0],'MarkerSize',20)
ylim([0,1])
xlim([0,80])
title('Recall@K Curve - KITTI06','FontSize',25,'FontName','Arial')
ylabel('Recall Accuracy at Top-K','FontSize',20,'FontName','Arial')
xlabel('Top-K','FontSize',20,'FontName','Arial')
AX.XAxis.FontSize = 20;
AX.YAxis.FontSize = 20;