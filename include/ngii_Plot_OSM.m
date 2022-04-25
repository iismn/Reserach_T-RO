function [geojsonCoordinate_X, geojsonCoordinate_Y] = ngii_Plot_OSM(geojsonValue, Case, LOCAL_Map)
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


%% B. PARAMETER SETTING
geojsonCoordinate_XY = 0;

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
                A = 1
            else
                [geojsonCoordinate_X, geojsonCoordinate_Y,temp] = deg2utm(geojsonCoordinate_XY(:,2), geojsonCoordinate_XY(:,1));     
            end
            
            
            
            for j = 1:length(geojsonCoordinate_X)
                if geojsonCoordinate_X(rem(j-1,length(geojsonCoordinate_X))+1) > LOCAL_Map.minLength.X-100 && geojsonCoordinate_X(rem(j-1,length(geojsonCoordinate_X))+1) < LOCAL_Map.maxLength.X+100 && ...
                    geojsonCoordinate_Y(rem(j-1,length(geojsonCoordinate_Y))+1) > LOCAL_Map.minLength.Y-100 && geojsonCoordinate_Y(rem(j-1,length(geojsonCoordinate_Y))+1) < LOCAL_Map.maxLength.Y+100
                    
                    plot([geojsonCoordinate_X(rem(j-1,length(geojsonCoordinate_X))+1), geojsonCoordinate_X(rem(j+1-1,length(geojsonCoordinate_X))+1)], ...
                            [geojsonCoordinate_Y(rem(j-1,length(geojsonCoordinate_Y))+1), geojsonCoordinate_Y(rem(j+1-1,length(geojsonCoordinate_Y))+1)], 'yellow','LineWidth',2)
                        
                end
            end
        end
        
        axis equal
        set(gca,'Color','k')
        
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
                if geojsonCoordinate_X(rem(j-1,length(geojsonCoordinate_X))+1) > LOCAL_Map.minLength.X-500 && geojsonCoordinate_X(rem(j-1,length(geojsonCoordinate_X))+1) < LOCAL_Map.maxLength.X+500 && ...
                    geojsonCoordinate_Y(rem(j-1,length(geojsonCoordinate_Y))+1) > LOCAL_Map.minLength.Y-500 && geojsonCoordinate_Y(rem(j-1,length(geojsonCoordinate_Y))+1) < LOCAL_Map.maxLength.Y+500
                    
%                     fill(geojsonCoordinate_X,geojsonCoordinate_Y,'r')
%                     plot([geojsonCoordinate_X(rem(j-1,length(geojsonCoordinate_X))+1), geojsonCoordinate_X(rem(j+1-1,length(geojsonCoordinate_X))+1)], ...
%                             [geojsonCoordinate_Y(rem(j-1,length(geojsonCoordinate_Y))+1), geojsonCoordinate_Y(rem(j+1-1,length(geojsonCoordinate_Y))+1)], 'red')
                        
                    if rem(j-1,length(geojsonCoordinate_X))+1 > rem(j+1-1,length(geojsonCoordinate_Y))+2 && rem(j+1-1,length(geojsonCoordinate_Y))+1 ==1
                        
                    else
                        plot(linspace(geojsonCoordinate_X(rem(j-1,length(geojsonCoordinate_X))+1), geojsonCoordinate_X(rem(j+1-1,length(geojsonCoordinate_X))+1),1000), ...
                                linspace(geojsonCoordinate_Y(rem(j-1,length(geojsonCoordinate_Y))+1), geojsonCoordinate_Y(rem(j+1-1,length(geojsonCoordinate_Y))+1),1000),'.r' ,'MarkerSize',50)
                    end
                        
                end

            end
        end
        
        axis equal
        set(gca,'Color','k')
    otherwise
        warning('Select Case ''BUILDING'' or ''ROAD''');


end
        

