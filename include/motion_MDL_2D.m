function [X_ik_A, X_ik_B, X_ik_C] = motion_MDL_2D(X_ij,X_jk)
% 2D version of head2tail operation

% X_ik = zeros(3,1);

x_ij = X_ij(:,1); y_ij = X_ij(:,2); t_ij = X_ij(:,3);
x_jk = X_jk(1); y_jk = X_jk(2); t_jk = X_jk(3);

X_ik_A = x_jk*cos(t_ij) - y_jk*sin(t_ij) +x_ij;
X_ik_B = x_jk*sin(t_ij) + y_jk*cos(t_ij) +y_ij;
X_ik_C = t_ij + t_jk;
