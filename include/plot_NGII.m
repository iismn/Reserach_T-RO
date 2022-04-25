function [geojsonCoordinate_X, geojsonCoordinate_Y,Img] = plot_NGII(geojsonValue, Case, maxX, minX, maxY, minY, Img, MapSize)
%figure = ngii_Plot( 'geoJson File', LimitArea, FigureNumber, Case)
%  iismn@kaist.ac.kr
%  KAIST IRiS Lab.
%  Autonomouse Vehicle Team
%
%  DB Generator with PCD + LocalMap Point DataBase
%  DB for Siamese OSM-NetVLAD DeepLearning Network
%  Research : Autonomous Driving without High-Definition Detailed Prior-Map
%
%  Copyright 2021.9.15

%% A. READ GEO-JSON FILE
resolution = size(Img,1);
mapsize = MapSize;
%% B. PARAMETER SETTING
geojsonCoordinate_XY = 0;
temp_DB = [];
temp_Build = [];
temp_Build_Cell = {};
over_size = 100;
%% C. PLOTTING FIGURE
hold on
switch Case
    case 'BUILDING'
        for i = 1:length(geojsonValue.features)
            geojsonCoordinate_Multi_XY = geojsonValue.features(i).geometry.coordinates;

            if iscell(geojsonCoordinate_Multi_XY)
                
                try
                    geojsonCoordinate_XY = reshape((geojsonCoordinate_Multi_XY{1}), [], 2);
                catch
                    warning('PASS')
                end
                
                if iscell(geojsonCoordinate_XY)
                    geojsonCoordinate_XY = reshape((geojsonCoordinate_XY{1}), [], 2);
                end
            else
                if iscell(geojsonCoordinate_XY)
                    geojsonCoordinate_XY = geojsonCoordinate_XY{1,1};
                    geojsonCoordinate_XY = reshape(geojsonCoordinate_Multi_XY, [], 2);
                else
                    geojsonCoordinate_XY = reshape(geojsonCoordinate_Multi_XY, [], 2);
                end
            end
            
            if iscell(geojsonCoordinate_XY)
                
            else
                [geojsonCoordinate_X, geojsonCoordinate_Y,~] = deg2utm(geojsonCoordinate_XY(:,2), geojsonCoordinate_XY(:,1));     
            end
            
            
            
            for j = 1:length(geojsonCoordinate_X)
                if geojsonCoordinate_X(rem(j-1,length(geojsonCoordinate_X))+1) > minX-over_size && geojsonCoordinate_X(rem(j-1,length(geojsonCoordinate_X))+1) < maxX+over_size && ...
                    geojsonCoordinate_Y(rem(j-1,length(geojsonCoordinate_Y))+1) > minY-over_size && geojsonCoordinate_Y(rem(j-1,length(geojsonCoordinate_Y))+1) < maxY+over_size
                    
                    x_start = geojsonCoordinate_X(rem(j-1,length(geojsonCoordinate_X))+1) - minX+over_size;
                    x_end = geojsonCoordinate_X(rem(j+1-1,length(geojsonCoordinate_X))+1) - minX+over_size;
                    y_start = geojsonCoordinate_Y(rem(j-1,length(geojsonCoordinate_Y))+1) - minY+over_size;
                    y_end = geojsonCoordinate_Y(rem(j+1-1,length(geojsonCoordinate_Y))+1) - minY+over_size;
                    
                    temp = [x_start (over_size+mapsize)*2-y_start x_end (over_size+mapsize)*2-y_end]*(resolution/((over_size+mapsize)*2));
                    temp_DB = [temp_DB; temp];
                    
                    
                    
                end
            end
        end
        Img = insertShape(Img, 'Line', temp_DB,'LineWidth',4);
        
    case 'ROAD'
        for i = 1:length(geojsonValue.features)

            geojsonCoordinate_Multi_XY = geojsonValue.features(i).geometry.coordinates;

            if iscell(geojsonCoordinate_Multi_XY)
                geojsonCoordinate_XY = reshape((geojsonCoordinate_Multi_XY{1}), [], 2);
                if iscell(geojsonCoordinate_XY)
                    geojsonCoordinate_XY = reshape((geojsonCoordinate_XY{1}), [], 2);
                end
            else
                if iscell(geojsonCoordinate_XY)
                    geojsonCoordinate_XY = geojsonCoordinate_XY{1,1};
                    geojsonCoordinate_XY = reshape(geojsonCoordinate_Multi_XY, [], 2);
                else
                    geojsonCoordinate_XY = reshape(geojsonCoordinate_Multi_XY, [], 2);
                end
            end
            
            if iscell(geojsonCoordinate_XY)
                A = 1
            else
                [geojsonCoordinate_X, geojsonCoordinate_Y,temp] = deg2utm(geojsonCoordinate_XY(:,2), geojsonCoordinate_XY(:,1));     
            end
            
%             [geojsonCoordinate_X, geojsonCoordinate_Y] = deg2utm(geojsonCoordinate_XY(:,2), geojsonCoordinate_XY(:,1));
            
            
            for j = 1:length(geojsonCoordinate_X)
                if geojsonCoordinate_X(rem(j-1,length(geojsonCoordinate_X))+1) > minX-over_size && geojsonCoordinate_X(rem(j-1,length(geojsonCoordinate_X))+1) < maxX+over_size && ...
                    geojsonCoordinate_Y(rem(j-1,length(geojsonCoordinate_Y))+1) > minY-over_size && geojsonCoordinate_Y(rem(j-1,length(geojsonCoordinate_Y))+1) < maxY+over_size
                    
                    temp_Build = [(geojsonCoordinate_X - minX+over_size)'; (geojsonCoordinate_Y - minY+over_size)']*(resolution/((over_size+mapsize)*2));
                    temp_Build(2,:) = resolution-temp_Build(2,:);
                    temp_Build = reshape(temp_Build,[1,numel(temp_Build)]);
                    temp_Build_Cell{end+1} = temp_Build;
                        
                    if rem(j-1,length(geojsonCoordinate_X))+1 > rem(j+1-1,length(geojsonCoordinate_Y))+2 && rem(j+1-1,length(geojsonCoordinate_Y))+1 ==1
                        
                    else
                        x_start = geojsonCoordinate_X(rem(j-1,length(geojsonCoordinate_X))+1) - minX+over_size;
                        x_end = geojsonCoordinate_X(rem(j+1-1,length(geojsonCoordinate_X))+1) - minX+over_size;
                        y_start = geojsonCoordinate_Y(rem(j-1,length(geojsonCoordinate_Y))+1) - minY+over_size;
                        y_end = geojsonCoordinate_Y(rem(j+1-1,length(geojsonCoordinate_Y))+1) - minY+over_size;
                        
                        x = linspace(x_start, x_end, 500);
                        y = linspace(((over_size+mapsize)*2)-y_start, ((over_size+mapsize)*2)-y_end, 500);
                        r = 2;

                        Coord = [x' y' (ones(size(x))*r)']*(resolution/((over_size+mapsize)*2));
                        temp_DB = [temp_DB; Coord];

                    end
                        
                end

            end
        end
        
        % KITTI
        Img = insertShape(Img, 'Circle', temp_DB,'LineWidth',4, 'Color', 'red');
        
        % NGII
%         Img = insertShape(Img,'FilledPolygon',temp_Build_Cell,'Color','red'); 
        x_start = resolution/2-mapsize*(resolution/((over_size+mapsize)*2));
        y_start = resolution/2-mapsize*(resolution/((over_size+mapsize)*2));
        width = (mapsize*2)*(resolution/((over_size+mapsize)*2));
        height = (mapsize*2)*(resolution/((over_size+mapsize)*2));

        Img = imcrop(Img,[x_start y_start width height]);
        Img = imresize(Img,[700 700]);
        
    otherwise
        warning('Select Case ''BUILDING'' or ''ROAD''');
        
end