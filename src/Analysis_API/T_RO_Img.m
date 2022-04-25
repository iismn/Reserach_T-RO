rawB = fread(fopen('DCC_NGII_BUILD2.geojson'),inf);
rawB = char(rawB');
geojsonValue_BUILD = jsondecode(rawB);
rawR = fread(fopen('DCC_NGII_ROAD.geojson'),inf);
rawR = char(rawR');
geojsonValue_ROAD = jsondecode(rawR);

%%
MAP_All = pcread('GlobalMap.pcd');
MAP_Odom = pcread('trajectory.pcd');
gridStep = 2.0;
MAP_All_Temp = pcdownsample(MAP_All,'gridAverage',gridStep);

%%
pcshow(MAP_All_Temp)
hold on; grid on; axis equal;
plot3(MAP_Odom.Location(:,1), MAP_Odom.Location(:,2),MAP_Odom.Location(:,3)-200,'y','LineWidth',5)
plot3(MAP_Odom.Location(:,1), MAP_Odom.Location(:,2),MAP_Odom.Location(:,3)-400,'g','LineWidth',3)
plot3(MAP_Odom.Location(:,1), MAP_Odom.Location(:,2),MAP_Odom.Location(:,3),'r','LineWidth',3)

%%
ngii_Plot_3D(geojsonValue_BUILD,'BUILDING')

latitude=36.3749136;
longitude=127.3906068;
[UTM_X, UTM_Y, ~] = deg2utm(latitude,longitude);
Init_UTM = [UTM_X, UTM_Y];