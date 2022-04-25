%% KAIST IRIS Lab
% Recall Plotting Service
%
% Autonomous Vehicle Localization without Prior High-Definition Map
% Transaction of Robotics (T-RO) Supplementary Material
%
% iismn@kaist.ac.kr
% Sangmin Lee
% KAIST IRiS Lab. Autnomous Vehicle Team PHAROS

%% A. Load Database
clc; clear
load('/home/iismn/WorkSpace/Database/Urban_25K/MAT/Urban_25K_URBAN10.mat')
Result = readNPY('/home/iismn/WorkSpace/Research/ComplexUrban_10/Result/Numpy/UTM.npy');
cmap = flipud(colormap(parula(10)));
Result_Accuracy = [];
Result_Map = [1,5,10,20,30,40,50,60,70,80];

% figure(1)
% clf

for i = 1:size(Result,1)
        
    UTM = dbStruct.utmQ(:,Result(i,1)+1);
    ACC = Result(i,2);
    IDX = find(Result_Map==ACC);
%     plot(UTM(1), UTM(2),'o','MarkerFaceColor',cmap(IDX,:),'MarkerEdgeColor',cmap(IDX,:))
%     hold on
%     axis equal; grid on
    Result_Accuracy = [Result_Accuracy; ACC];
end
%%
Result_Recall = []
% RECALL Calculate
for i = 1:10
    Indx = find(Result_Accuracy==Result_Map(i));
    Indx = size(Indx,1);
    Result_Recall(end+1) = Indx
    
    
end

Result_Recall = cumsum(Result_Recall)/dbStruct.numQueries;

Recall = [Result_Map' Result_Recall']

%%

% F = figure(3)
hold on
f=fit(Recall(:,1),Recall(:,2),'smoothingspline','SmoothingParam',1.0);
Top_K = 1:0.01:80;
Recall_K = f(Top_K);

hold on
AX = gca;
% plot(Top_K,Recall_K,'LineWidth',3,'Color',[1 0.5 0])
plot(Recall(:,1),Recall(:,2),'LineWidth',3,'Color',[0 0.5 0])
hold on; grid on;
plot(Recall(:,1),Recall(:,2),'x','LineWidth',3,'Color',[0 0.5 0],'MarkerSize',20)
ylim([0,1])
xlim([0,80])
title('Recall@K Curve - Jukdong','FontSize',25,'FontName','Arial')
ylabel('Recall Accuracy at Top-K','FontSize',20,'FontName','Arial')
xlabel('Top-K','FontSize',20,'FontName','Arial')
AX.XAxis.FontSize = 20;
AX.YAxis.FontSize = 20;
%%
close all;
load('/home/iismn/WorkSpace/CU11_DL/PyTorch/NetVLAD+OSM/pytorch-NetVlad/datasets/LSM3/MAT/OSM_Test_DCC.mat')
PRED = readNPY('PRED.npy');
UTM = readNPY('UTM.npy');

for i =1:size(PRED,1)
    idx = UTM(i,1)+1;
    idxp = PRED(i,:)+1;
    qimage = imread(dbStruct.qImageFns{idx});
    
    dbimage1 = imread(dbStruct.dbImageFns{idxp(1)});
    dbimage2 = imread(dbStruct.dbImageFns{idxp(2)});
    dbimage3 = imread(dbStruct.dbImageFns{idxp(3)});
    dbimage4 = imread(dbStruct.dbImageFns{idxp(4)});
    dbimage5 = imread(dbStruct.dbImageFns{idxp(5)});
    dbimage6 = imread(dbStruct.dbImageFns{idxp(6)});
    dbimage7 = imread(dbStruct.dbImageFns{idxp(7)});
    dbimage8 = imread(dbStruct.dbImageFns{idxp(8)});
    
    if UTM(i,2) >= 5
        UTM(i,2)
        figure('Renderer', 'painters', 'Position', [1200 1100 2500 400])
        t = tiledlayout(1,9)
        % Tile 1
        nexttile
        imshow(qimage)
        % Tile 1
        nexttile
        imshow(dbimage1)
        % Tile 2
        nexttile
        imshow(dbimage2)
        % Tile 3
        nexttile
        imshow(dbimage3)
        % Tile 4
        nexttile
        imshow(dbimage4)        
        % Tile 1
        nexttile
        imshow(dbimage5)
        % Tile 2
        nexttile
        imshow(dbimage6)
        % Tile 3
        nexttile
        imshow(dbimage7)
        % Tile 4
        nexttile
        imshow(dbimage8)        
        ax=gca;
        temp = 1
        close all;

    end
































end
