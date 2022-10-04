%% KAIST IRIS Lab
% dbStruct Generator for PyTorch Siamse-NetVLAD
%
% Autonomous Vehicle Localization without Prior High-Definition Map
% Transaction of Robotics (T-RO) Supplementary Material
%
% iismn@kaist.ac.kr
% Sangmin Lee
% KAIST IRiS Lab. Autnomous Vehicle Team PHAROS

%% DB GEN
clc; clear; 
DB.List_2D = dir('**/2D');
DB.List_3D = dir('**/3D');
DB.List_UTM_2D = dir('**/POS_2D');
DB.List_UTM_3D = dir('**/POS_3D');

% Parameter
DatabaseName = 'DB_Test/';
DatabaseSort = 'Test';
DatabaseMatName = 'Urban_25K_SEJONG';
imgResize = 0;
Val_SplitRatio = 1.5;
Test_SplitRatio = 2.16;

%% B. Adding File List to Database

% Filter Different Count of Database
% if size(DB.List_2D,1) ~= size(DB.List_3D,1) || size(DB.List_2D,1) ~= size(DB.List_UTM,1)
%     fprintf('SIZE NOT EQUAL')
% end

% Initialize Parameter Database
FList_2D  ={};
FList_3D  ={};
FList_UTM_2D = [];
FList_UTM_3D = [];

for i = 1:size(DB.List_2D,1)
    
    % Pass Through Unknown File Name
    if DB.List_2D(i).bytes == 0
        continue
    end
    %-----------------------------------------------------------------------------
    % File List 2D Generator
    F_Name = DB.List_2D(i).name;
    F_Path = DB.List_2D(i).folder;
    F_Path_Split = split(F_Path,DatabaseName);
    F_Path_Split = F_Path_Split{2};

    if imgResize == 1
        Full_Path = strcat(F_Path,'/',F_Name);
        TEMP_IMG = imresize(imread(Full_Path), [710 710]);
        r = centerCropWindow2d(size(TEMP_IMG),[700 700]);
        TEMP_IMG = imcrop(TEMP_IMG,r);
        imwrite(TEMP_IMG,Full_Path)
    end

    Full_Path = strcat(F_Path_Split,'/',F_Name);   % For Separate Test
    FList_2D{end+1} = Full_Path;

    %-----------------------------------------------------------------------------
    % UTM 2D
    F_Name = DB.List_UTM_2D(i).name;
    F_Path = DB.List_UTM_2D(i).folder;
    Full_Path = strcat(F_Path,'/',F_Name);
    
    LOCAL_Coordinate_IO = textscan(fopen(Full_Path),'%f64 %f64');
    FList_UTM_2D = [FList_UTM_2D cell2mat(LOCAL_Coordinate_IO)'];
    
    fclose all;

end

for i = 1:size(DB.List_3D,1)
    
    % Pass Through Unknown File Name
    if DB.List_3D(i).bytes == 0
        continue
    end

    %-----------------------------------------------------------------------------
    % File List 3D Generator
    F_Name = DB.List_3D(i).name;
    F_Path = DB.List_3D(i).folder;
    F_Path_Split = split(F_Path,DatabaseName);
    F_Path_Split = F_Path_Split{2};

    if imgResize == 1
        Full_Path = strcat(F_Path,'/',F_Name);
        TEMP_IMG = imresize(imread(Full_Path), [710 710]);
        r = centerCropWindow2d(size(TEMP_IMG),[700 700]);
        TEMP_IMG = imcrop(TEMP_IMG,r);
        imwrite(TEMP_IMG,Full_Path)
    end

    Full_Path = strcat(F_Path_Split,'/',F_Name);   % For Separate Test
    FList_3D{end+1} = Full_Path;
    
    %-----------------------------------------------------------------------------
    % File List UTM 3D Generator
    F_Name = DB.List_UTM_3D(i).name;
    F_Path = DB.List_UTM_3D(i).folder;
    Full_Path = strcat(F_Path,'/',F_Name);
    
    LOCAL_Coordinate_IO = textscan(fopen(Full_Path),'%f64 %f64');
    LOCAL_Coordinate_IO = cell2mat(LOCAL_Coordinate_IO);
    FList_UTM_3D = [FList_UTM_3D; LOCAL_Coordinate_IO];
    
    fclose all;
end


FList_2D = FList_2D';
FList_3D = FList_3D';
FList_UTM_3D = FList_UTM_3D';

%% C. Generating Validation Set
% Initialize Parameter Database
FList_2D_Train  ={};
FList_3D_Train  ={};
FList_UTM_Train = [];

for i = 1:round(size(FList_2D,1)/Val_SplitRatio)
    
    iter = round(i*Val_SplitRatio);
    
    if iter >= size(FList_2D,1)
         continue
    end
    
    FList_2D_Train{end+1} = FList_2D{iter};
    FList_3D_Train{end+1} = FList_3D{iter};
    FList_UTM_Train = [FList_UTM_Train FList_UTM(:,iter)];
    
end

% Step Back for Processing
pause(2)
fprintf('[DB Geneator] : Validation Generating Done\n')
FList_2D_Train = FList_2D_Train';
FList_3D_Train = FList_3D_Train';

%% C. Generating Test Set
% Initialize Parameter Database
FList_2D_Test  ={};
FList_3D_Test ={};
FList_UTM_Test = [];

for i = 1:round(size(FList_3D_Train,1))
    
    iter = i;
    
    if iter >= size(FList_3D,1)
         continue
    end
    
    FList_2D_Test{end+1} = FList_2D{iter};
    FList_3D_Test{end+1} = FList_3D{iter};
    FList_UTM_Test = [FList_UTM_Test FList_UTM_3D(:,iter)];
    
end

pause(2)
fprintf('[DB Geneator] : Test Generating Done\n')
FList_2D_Test = FList_2D_Test';
FList_3D_Test = FList_3D_Test';

%% C-2. Generating Test Set (For Odometry Eval)
% Initialize Parameter Database
FList_2D_Test  ={};
FList_3D_Test ={};
FList_UTM_Test = [];

for i = 1:round(size(FList_2D,1))
    
    iter = i;
    
    if iter >= size(FList_3D,1)
         continue
    end
    
    FList_2D_Test{end+1} = FList_2D{iter};
    FList_3D_Test{end+1} = FList_3D{iter};
    FList_UTM_Test = [FList_UTM_Test FList_UTM_3D(:,iter)];
    
end

pause(2)
fprintf('[DB Geneator] : Test Generating Done\n')
FList_2D_Test = FList_2D_Test';
FList_3D_Test = FList_3D_Test';

%% D. MAT Generating for PyTorch NetVLAD Core

if strcmp(DatabaseSort,'Train')
    % dbStruct Save - Train
    dbStruct.whichSet = 'train';
    dbStruct.dbImageFns = FList_2D;
    dbStruct.utmDb = FList_UTM_2D;
    dbStruct.qImageFns = FList_3D_Train;
    dbStruct.utmQ = FList_UTM_Train;
    dbStruct.numImages = size(dbStruct.dbImageFns,1);
    dbStruct.numQueries = size(dbStruct.qImageFns,1);
    dbStruct.posDistThr = 10;
    dbStruct.posDistSqThr=100;
    dbStruct.nonTrivPosDistSqThr = 800;
    save('Urban_25K_Train.mat','dbStruct');

elseif strcmp(DatabaseSort,'Test')
    % dbStruct Save - Test
    dbStruct.whichSet = 'test';
    dbStruct.dbImageFns = FList_2D;
    dbStruct.utmDb = FList_UTM_2D;
    dbStruct.qImageFns = FList_3D_Test;
    dbStruct.utmQ = FList_UTM_Test;
    dbStruct.numImages = size(dbStruct.dbImageFns,1);
    dbStruct.numQueries = size(dbStruct.qImageFns,1);
    dbStruct.posDistThr = 25;
    dbStruct.posDistSqThr = 625;
    dbStruct.nonTrivPosDistSqThr = 800;
    
    save(strcat(DatabaseMatName,'.mat'),'dbStruct');
end
