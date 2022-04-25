% You should change save path at the last line.
clc;
clear;
close all;

%% Load data
pc_path = '/home/iismn/WorkSpace/Research/ComplexUrban_07/OSMDesc_DATA/PCL_RAW/PCD';
pose_path = '/home/iismn/WorkSpace/Research/ComplexUrban_07/OSMDesc_DATA/PCL_RAW/POS';
save_path = '/home/iismn/WorkSpace/Research/ComplexUrban_07/OSMDesc_Result/descriptor.csv';
lidar_rotinv_save_path = '/home/iismn/WorkSpace/Research/ComplexUrban_07/OSMDesc_Result/rotinv_descriptor.csv';

files = dir(pc_path);
files(1:2) = [];
files = {files(:).name};

pose_files = dir(pose_path);
pose_files(1:2) = [];
pose_files = {pose_files(:).name};

%% Make descriptors
num_bin = 360;
sensor_range = 50;
bin_size = 5;

descriptor_list = [];
for i = 1:length(files)
    if rem(i,100) == 0
        disp([num2str(i), '/', num2str(length(files))])
    end
    filename = files{i};
    current_path = fullfile(pc_path, filename);
    
    pose_filename = pose_files{i};
    pose_current_path = fullfile(pose_path, pose_filename);

    pc = pcread(current_path);
    pose = dlmread(pose_current_path);

%     eul = pose(4:6);
    eul = [pose(6), pose(5), pose(4)];
    rotm = eul2rotm(eul);
    
    tform = rigid3d(rotm', [0 0 0]);
    pc = pctransform(pc, tform);
    roi = [-inf inf -inf inf 0.5 inf];
    indices = findPointsInROI(pc,roi);
    pc = select(pc,indices);
    pc = pc.Location;
    
    tan_list = atan2(pc(:,2), pc(:,1));
    tan_list = tan_list + (tan_list < 0)*2*pi;
    
    descriptor = [];
    for j = 1:num_bin
        target_angle_orig = (j-1)*2*pi/num_bin;
        
        target_angle = pi;
        tan_list_temp = tan_list + pi - target_angle_orig;
        tan_list_temp = tan_list_temp + (tan_list_temp < 0)*2*pi;
        tan_list_temp = tan_list_temp - (tan_list_temp > 2*pi)*2*pi;
        target_points = pc(tan_list_temp > target_angle - pi/num_bin & tan_list_temp < target_angle + pi/num_bin, :);
        if isempty(target_points)
            descriptor = [descriptor, 0];
        else
            if min(sqrt(target_points(:,1).^2 + target_points(:,2).^2)) > sensor_range
                descriptor = [descriptor, 0];
            else
                descriptor = [descriptor, min(sqrt(target_points(:,1).^2 + target_points(:,2).^2))];
            end
        end
    end
    descriptor_list = [descriptor_list; descriptor];
end

csvwrite(save_path, descriptor_list);
% 
% pc_descriptor_rot_inv = zeros(length(descriptor_list), sensor_range/bin_size);
% for i = 1:length(descriptor_list)
%     range_angle_descriptor = zeros(num_bin, sensor_range/bin_size);
% 
%     pc_descriptor = descriptor_list(i,:);
%     for j = 1:num_bin
%         if pc_descriptor(j) ~= 0
%             range_angle_descriptor(j,ceil(pc_descriptor(j)/bin_size)) = 1;
%         end
%     end
%     rot_inv_descriptor = sum(range_angle_descriptor);
% 
%     pc_descriptor_rot_inv(i,:) = rot_inv_descriptor;
% end
% 
% csvwrite(lidar_rotinv_save_path, pc_descriptor_rot_inv);
% 
% disp("rotation invariant descriptor finished.")