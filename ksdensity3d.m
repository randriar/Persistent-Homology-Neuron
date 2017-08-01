function f = ksdensity3d(sample,gridx,gridy,gridz)
% Compute kernel density estimate in 3D.
%
% F = KSDENSITY3D(X,GRIDX,GRIDX2,GRIDX3,BW)
%  INPUTS:
%   sample - N-by-3 matrix: a list of N sample points (3-dim coordinates)
%   gridx - vector: a list of coordinates along the x-axis to consider
%   gridy - vector: a list of coordinates along the y-axis to consider
%   gridz - vector: a list of coordinates along the z-axis to consider
%
%  OUTPUT:
%   f - 3D-tensor: a tensor of probability density estimates evaluated at the 
%                  points in 3D-grid defined by GRIDX, GRIDY, and GRIDZ.
%                  The estimates are based on a normal kernel function, using 
%                  standard deviations (bandwidth) that are either given by BW.

bw = [85,85,85];

% Some basic size values.
[n,~] = size(sample);
nx = length(gridx);
ny = length(gridy);
nz = length(gridz);

% Pull apart the 3-dim mesh defined by GRIDX, GRIDY, and GRIDZ.
% Make copies for each data point in the sample.
[gridy,gridx,gridz] = meshgrid(gridy,gridx,gridz);
x = repmat(gridx, [1,1,1,n]);
y = repmat(gridy, [1,1,1,n]);
z = repmat(gridz, [1,1,1,n]);

% Record the coordinates for each data point in the sample as "mean values"
% for subsequent normal distributions.
mux(1,1,1,:) = sample(:,1);
mux = repmat(mux,[nx,ny,nz,1]);
muy(1,1,1,:) = sample(:,2);
muy = repmat(muy,[nx,ny,nz,1]);
muz(1,1,1,:) = sample(:,3);
muz = repmat(muz,[nx,ny,nz,1]);

% For each point in the in the 3D mesh, compute the contribution at that
% point coming from all normal probability distributions in each of the
% three coordinate directions centered around each sample point with standard
% deviation given by BW.
% Multiply these density values to obtain to total probability density at the 
% given mesh point.
f = sum((normpdf(x,mux,bw(1)) .* normpdf(y,muy,bw(2)) .* normpdf(z,muz,bw(3))), 4) / n;