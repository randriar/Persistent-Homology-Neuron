% ####*** Directions ***####
%   - Put the cells( with '.txt' extension) in the 'INPUT' folder
%   - Run the program
%   - Enter the number of spherical shells you want to use for the experiment
%   - Enter the dimension 

%   Revisited by Rindra Randriamanantena 
%   July 28, 2017
% ####*** *** *** *** ***####


% This script prepares the javaplex library for use.
clc; clear all; close all;
javaaddpath('./lib/javaplex.jar');
import edu.stanford.math.plex4.*;
javaaddpath('./lib/plex-viewer.jar');
import edu.stanford.math.plex_viewer.*;
cd './utility';
addpath(pwd);
cd '..';

disp('START...');
% Some parameters to set.
bound1=-500;bound2=500; % Upper/lower boundary values
split=200;              % radius where we split the spherical shells into two groups
l1=20;                  % Number of steps along x axis
l2=20;                  % Number of steps along y axis
l3=20;                  % Number of steps along z axis

% sigma=((bound2-bound1)/(2*l1))/sqrt(3);  % Standard deviation used to determine "thickness" of spherical shells.
                                               % ((bound2-bound1)/(2*l1)) is half the distance between consecutive sample
                                               % points along each axis.
sigma=17;
prompt2=' Enter the number of spherical shells you want to use for this experiment: '; 
kN= input(prompt2); % Number of spherical shells

% Create  directories where the CELLS will be saved
if exist('OUTPUTS','file')==7 % If the folder already exists, delete it
    rmdir('OUTPUTS','s');
end
mkdir('OUTPUTS'); % create a directory where the data about each cells will be saved

mkdir('OUTPUTS','Parameters'); % create a directory to save the parameters

mkdir('OUTPUTS','SphericalShells'); % create a directory to save the spherical shells

mkdir('OUTPUTS','CELLS'); % Create a folder called 'CELLS' where the data about each cell will be stored

% Some pre-computed Data structures we will need.
hgrid=bound1:(bound2-bound1)/l1:bound2; % Discrete points along x axis.
vgrid=bound1:(bound2-bound1)/l2:bound2; % Discrete points along y axis.
dgrid=bound1:(bound2-bound1)/l3:bound2; % Discrete points along z axis.
[ysample,xsample,zsample]=meshgrid(vgrid,hgrid,dgrid); % Mesh in 3D space (copies of hgrid, vgrid and dgrid).

U(:,1)=reshape(xsample,size(xsample,1)*size(xsample,2)*size(xsample,3),1); % Re-format, the mesh as a list of the
U(:,2)=reshape(ysample,size(ysample,1)*size(ysample,2)*size(ysample,3),1); %  coordinate-triples 
U(:,3)=reshape(zsample,size(zsample,1)*size(zsample,2)*size(zsample,3),1); %  for the vertices 

n1=length(hgrid);n2=length(vgrid);n3=length(dgrid); % The number of points in each directional axis.

% Enter the desired dimension
prompt3=' Enter the dimension of homology you want to compute for each neuron: '; 
dim= input(prompt3);
if(dim<0)
    disp('Invalid dimension, using 0 instead.')
    dim=0;
end
if(dim>2)
    disp('Invalid dimension, using 2 instead.')
    dim=2;
end

% E is a collection of tetrahedra filling the 3D mesh.
% That is, a list of quadruples of points (numbered top-to-bottom, 
%   then left-to-right, then front-to-back) which form tetrahedra.
% As appropriate, we record the tetrahedra, the faces, and the edges.
disp('Making the 3D mesh');
number_of_tetrahedra = l1*l2*l3*6;
tetrahedra=zeros(number_of_tetrahedra,4);

number_of_faces = l1*l2*l3*12 + l1*l2*2 + l1*l3*2 + l2*l3*2;
faces=zeros(number_of_faces,3);

number_of_edges = l1*l2*l3*7 + l1*l2*3 + l1*l3*3 + l2*l3*3 + l1 + l2 + l3;
edges=zeros(number_of_edges,2);

counter1 = 1; % counter for the tetrahedra
counter2 = 1; % counter for the triangular faces
counter3 = 1; % counter for the edges
for i=1:n3-1
    for j=1:n2-1
        for k=1:n1-1
            v = (i-1)*n1*n2+(j-1)*n1+k;
            
            if (dim==2)
                tetrahedra(counter1,:)=[v v+1 v+n1+1 v+n1*n2+n1+1]; % 6 terahedra that start with the same points v
                tetrahedra(counter1+1,:)=[v v+1 v+n1*n2+1 v+n1*n2+n1+1];
                tetrahedra(counter1+2,:)=[v v+n1 v+n1+1 v+n1*n2+n1+1];
                tetrahedra(counter1+3,:)=[v v+n1 v+n1*n2+n1 v+n1*n2+n1+1];
                tetrahedra(counter1+4,:)=[v v+n1*n2 v+n1*n2+1 v+n1*n2+n1+1];
                tetrahedra(counter1+5,:)=[v v+n1*n2 v+n1*n2+n1 v+n1*n2+n1+1];
                counter1 = counter1 + 6;
            end
            
            if (dim==1) || (dim==2)
                faces(counter2,:)=[v v+1 v+n1+1]; % 12 faces that start with the same point v
                faces(counter2+1,:)=[v v+1 v+n1*n2+1];
                faces(counter2+2,:)=[v v+1 v+n1*n2+n1+1];
                faces(counter2+3,:)=[v v+n1 v+n1+1];
                faces(counter2+4,:)=[v v+n1 v+n1*n2+n1];
                faces(counter2+5,:)=[v v+n1 v+n1*n2+n1+1];
                faces(counter2+6,:)=[v v+n1+1 v+n1*n2+n1+1];
                faces(counter2+7,:)=[v v+n1*n2 v+n1*n2+1];
                faces(counter2+8,:)=[v v+n1*n2 v+n1*n2+n1];
                faces(counter2+9,:)=[v v+n1*n2 v+n1*n2+n1+1];
                faces(counter2+10,:)=[v v+n1*n2+1 v+n1*n2+n1+1];
                faces(counter2+11,:)=[v v+n1*n2+n1 v+n1*n2+n1+1];
                counter2 = counter2 + 12;
            end
            
            edges(counter3,:)=[v v+1]; % 7 edges that start with the same point v
            edges(counter3+1,:)=[v v+n1];
            edges(counter3+2,:)=[v v+n1+1];
            edges(counter3+3,:)=[v v+n1*n2];
            edges(counter3+4,:)=[v v+n1*n2+1];
            edges(counter3+5,:)=[v v+n1*n2+n1];
            edges(counter3+6,:)=[v v+n1*n2+n1+1];
            counter3 = counter3 + 7;
        end
        % In the last page
        k = n1;
        v = (i-1)*n1*n2+(j-1)*n1+k;
        
        if (dim==1) || (dim==2)
            faces(counter2,:)=[v v+n1 v+n1*n2+n1];
            faces(counter2+1,:)=[v v+n1*n2 v+n1*n2+n1];
            counter2 = counter2 + 2;
        end
        
        edges(counter3,:)=[v v+n1];
        edges(counter3+1,:)=[v v+n1*n2];
        edges(counter3+2,:)=[v v+n1*n2+n1];
        counter3 = counter3 + 3;
    end
    % In the last column
    j = n2;
    for k=1:n1-1
        v = (i-1)*n1*n2+(j-1)*n1+k;
        
        if (dim==1) || (dim==2)
            faces(counter2,:)=[v v+1 v+n1*n2+1];
            faces(counter2+1,:)=[v v+n1*n2 v+n1*n2+1];
            counter2 = counter2 + 2;
        end
        
        edges(counter3,:)=[v v+1];
        edges(counter3+1,:)=[v v+n1*n2];
        edges(counter3+2,:)=[v v+n1*n2+1];
        counter3 = counter3 + 3;
    end
    
    k = n1;
    v = (i-1)*n1*n2+(j-1)*n1+k;
    
    edges(counter3,:)=[v v+n1*n2];
    counter3 = counter3 + 1;
end
% In the last row
i = n3;
for j=1:n2-1
    for k=1:n1-1
        v = (i-1)*n1*n2+(j-1)*n1+k;
        
        if (dim==1) || (dim==2)
            faces(counter2,:)=[v v+1 v+n1+1];
            faces(counter2+1,:)=[v v+n1 v+n1+1];
            counter2 = counter2 + 2;
        end
        
        edges(counter3,:)=[v v+1];
        edges(counter3+1,:)=[v v+n1];
        edges(counter3+2,:)=[v v+n1+1];
        counter3 = counter3 + 3;
    end
    k = n1;
    v = (i-1)*n1*n2+(j-1)*n1+k;
    
    edges(counter3,:)=[v v+n1];
    counter3 = counter3 + 1;
end
j = n2;
for k=1:n1-1
    v = (i-1)*n1*n2+(j-1)*n1+k;
    
    edges(counter3,:)=[v v+1];
    counter3 = counter3 + 1;
end
% ######****Finished making 3D mesh****###### %


% Saving the Parameters
save('OUTPUTS/Parameters/bound1','bound1'); % saving the lower bound
save('OUTPUTS/Parameters/bound2','bound2');% saving the upper bound
save('OUTPUTS/Parameters/l1','l1'); % saving l1
save('OUTPUTS/Parameters/l2','l2'); % saving l2
save('OUTPUTS/Parameters/l3','l3'); % saving l3
save('OUTPUTS/Parameters/sigma','sigma'); % saving sigma
save('OUTPUTS/Parameters/split','split'); % saving the split

% Saving the 3D mesh
if dim==2
    save('OUTPUTS/Parameters/tetrahedra','tetrahedra'); % save the tetrahedra
end
if (dim==1) || (dim==2)
    save('OUTPUTS/Parameters/faces','faces'); % save the faces
end
save('OUTPUTS/Parameters/edges','edges'); % always save the edges for any value value of dim

% Saving the Spherical Shells (matrices and lists) and the radii
disp('Saving the Spherical Shells');
mkdir('OUTPUTS/SphericalShells','SphericalShell_List');  % Create a directory to save the lists of spherical shells
mkdir('OUTPUTS/SphericalShells','SphericalShell_Matrix'); % Create a directory to save the matrices of spherical shells
radii=zeros(kN,1); % create a list of 0s to save the radius of each Spherical Shell
for t=1:kN
    if(t<=ceil(kN/2))
        rad = t*split/ceil(kN/2);
    else
        rad=split+((t-ceil(kN/2))*(bound2-split)/(kN-ceil(kN/2)));
    end
    radii(t)=rad; 
    SphericalShell=exp(-(sum((U-repmat([0 0 0],size(U,1),1)).^2,2).^0.5-rad).^2/(2*sigma^2)); % Spherical Shell kernel
    SphericalShell3D = reshape(SphericalShell,size(xsample)); % Reshape onto 3D mesh.
    disp(strcat('   Spherical Shell #',num2str(t)));
    save(strcat('OUTPUTS/SphericalShells/SphericalShell_List/','SphericalShell_List_#',num2str(t)),'SphericalShell'); % saving the list of the annulus kernel
    save(strcat('OUTPUTS/SphericalShells/SphericalShell_Matrix/','SphericalShell_Matrix_#',num2str(t)),'SphericalShell3D'); % saving the matrix of the annulus kernel
end
save('OUTPUTS/SphericalShells/Radii','radii'); % save the radius of the shephical shells

disp('Processing the Cells');
file=dir('INPUTS/*.txt'); % Take the files with '.txt' extension
for k=1:length(file) % For every cell
    disp(strcat('   Cell #',num2str(k)));
    filename=file(k).name;
    cell = textread(strcat('INPUTS/',filename),'','delimiter',','); % Read the cell
    filename=strtok(filename,'.'); % name of the cell without extension
    mkdir('OUTPUTS/CELLS',strcat('Cell#',num2str(k),'_',filename)); % Create a directory where the Cell's information(pictures,etc...) will be stored
    save(strcat('OUTPUTS/CELLS/','Cell#',num2str(k),'_',filename,'/Cell_Raw'),'cell'); % Saving the cell
    
    % Recenter at the "soma"
    disp('   Recentering CELLS');
    % Shift so that the "physical center of the cell" - AKA the "soma" is
    % at (0,0,0). Our standard Cells-format should include the soma
    % coordinates as the first coordinate-triple.
    cell = cell-repmat(cell(1,:),length(cell),1); % recenter the cell 
    
    % Save a scatter plot of the cells points comprising this neuron.
    fig0 = figure('visible', 'off'); % turn off the visibility of the figure
    scatter3(cell(:,1),cell(:,2),cell(:,3)); % plot the original cell in a 3D plan 
    print(strcat('OUTPUTS/CELLS/','Cell#',num2str(k),'_',filename,'/Cell_Plot'),'-dpdf'); % save the 3D image of the cell
    close(fig0);
        
    % Compute the Gaussian density estimator. See KSDENSITY3D for details.
    disp('   Computing KDE');
    KDE = ksdensity3d([cell(:,1),cell(:,2),cell(:,3)],hgrid,vgrid,dgrid); % compute KDE( more detail at 'ksdensity3d.m')
    save(strcat('OUTPUTS/CELLS/Cell#',num2str(k),'_',filename,'/Cell_KDE'),'KDE'); % saving the KDE
    
    disp('   Computing Spherical KDEs and Barcodes');
    % Compare with the KDE with spherical shell kernels:
    mkdir(strcat('OUTPUTS/CELLS/Cell#',num2str(k),'_',filename),'Sphere*KDE_Matrix');  % Create a directory to save the product of the sphere and KDE (matrices)
    mkdir(strcat('OUTPUTS/CELLS/Cell#',num2str(k),'_',filename),'Sphere*KDE_List'); % Create a directory to save the product of the spherical shells and KDE (lists)
    mkdir(strcat('OUTPUTS/CELLS/Cell#',num2str(k),'_',filename),'Barcodes'); % Create a directory to save the barcodes
    
    for t=1:kN % for each spherical shells
        disp(strcat('      Sphere #',num2str(t)))
        % Compute the spherical shell kernel.
        % rad
        % center at [0 0 0]
        % sigma gives 'thickness'
        rad =radii(t);
        SphericalShell=exp(-(sum((U-repmat([0 0 0],size(U,1),1)).^2,2).^0.5-rad).^2/(2*sigma^2));
        SphericalShell3D = reshape(SphericalShell,size(xsample)); % Reshape onto 3D mesh.
        
        % Multiply the image KDE by the spherical shell kernel.
        disp('      Computing Spherical KDE');
        product=reshape(reshape(KDE,size(SphericalShell)).*SphericalShell,size(KDE)); 
        save(strcat('OUTPUTS/CELLS/Cell#',num2str(k),'_',filename,'/Sphere*KDE_Matrix/SphericalShell*KDE_Matrix_', num2str(t)),'product'); % save the product as a matrix
        ZZ=-reshape(product,n1*n2*n3,1); % Negative function - switch superlevel set to sublevel set
        save(strcat('OUTPUTS/CELLS/Cell#',num2str(k),'_',filename,'/Sphere*KDE_List/SphericalShell*KDE_List_', num2str(t)),'ZZ'); % save the product as a list

        % The following is to compute persistent barcodes for each spherical shell.
        % Initialize an explicit chain complex.
        disp('      Creating Simplex');
        stream = api.Plex4.createExplicitSimplexStream(2); 
        % Add vertices to our explicit chain complex. 
            % Each vertex is given a 'birth time' according to the KDE*Sphere value at
            % that point.
        for i=1:length(U) % adding vertices
            stream.addVertex(i,ZZ(i));
        end
        
        if (dim==2) % form tetrahedra given 4 vertices
            for i=1:number_of_tetrahedra
                stream.addElement([tetrahedra(i,1) tetrahedra(i,2) tetrahedra(i,3) tetrahedra(i,4)],max([ZZ(tetrahedra(i,1)) ZZ(tetrahedra(i,2)) ZZ(tetrahedra(i,3)) ZZ(tetrahedra(i,4))]));
            end
        end
        
        if (dim==1) || (dim==2) % form triangular faces given 3 vertices
            for i=1:number_of_faces
                stream.addElement([faces(i,1) faces(i,2) faces(i,3)],max([ZZ(faces(i,1)),ZZ(faces(i,2)),ZZ(faces(i,3))]));
            end
        end
        
        % always form edges given 2 vertices
        for i=1:number_of_edges
            stream.addElement([edges(i,1) edges(i,2)],max([ZZ(edges(i,1)),ZZ(edges(i,2))]));
        end
        
        % Finalize the chain complex.
        stream.finalizeStream();
        
        disp('      Computing Barcode');
        % Compute homology in dimension "dim", and with coefficients in Z_2.
        persistence = api.Plex4.getModularSimplicialAlgorithm((dim+1), 2);
        % Compute persistence intervals.
        intervals = persistence.computeIntervals(stream);
        % Some options for displaying the barcodes.
        options.min_dimension=0;
        options.max_dimension=dim;
        options.min_filtration_value=min(ZZ);
        options.max_filtration_value=0;
        plot_barcodes(intervals,options); % Actually display the barcodes.
        print(strcat('OUTPUTS\CELLS\','Cell#',num2str(k),'_',filename,'\Barcodes\','Barcode_', num2str(t)),'-dpdf'); % save the barcodes in the corresponding folder of the cooresponding cell
        
        Interval0{k}{t}=intervals.getIntervalsAtDimension(0); % Record the 0-dim intervals for pairwise comparison.
        if (dim==1) || (dim==2)
            Interval1{k}{t}=intervals.getIntervalsAtDimension(1); % Record the 1-dim intervals for pairwise comparison.
        end
        if (dim==2)
            Interval2{k}{t}=intervals.getIntervalsAtDimension(2); % Record the 2-dim intervals for pairwise comparison.
        end
    end    
end
disp('Computing Pairwise Bottleneck Distances')

save('OUTPUTS/Interval0','Interval0'); % Save the interval0
if (dim==1) || (dim==2)
    save('OUTPUTS/Interval1','Interval1'); % Save the interval1
end
if (dim==2)
    save('OUTPUTS/Interval2','Interval2'); % Save the interval2
end

% Computing and Caving pairwise bottleneck distance
% create a multidimensional array of nb rows of nb column 
nb=length(file); % number of cells
BD0=zeros(nb,nb,kN); % creating kN array(s) of nb*nb matrices
BD1=zeros(nb,nb,kN); % creating kN array(s) of nb*nb matrices
BD2=zeros(nb,nb,kN); % creating kN array(s) of nb*nb matrices


% Compute the bootleneck distance between cell i and cell j in sphere t for the corresponding dimension
% If the sizes of the pairwise interval of the cells in a specific sphere where we are computing the distance for
%    are 0, we directly set the bootleneck distance to 0 to avoid error
for t=1:kN
    for i=1:nb-1
        for j=i+1:nb
            % checking the size of the pairwise interval of cell i and cell j in sphere t for dimension 0
            if size(Interval0{i}{t})==0 && size(Interval0{j}{t})==0 % If both are empty, set distance to 0.
                BD0(i,j,t)=0;
            else % If either is non-empty, compute distance using Bottleneck.
                BD0(i,j,t) = edu.stanford.math.plex4.bottleneck.BottleneckDistance.computeBottleneckDistance(Interval0{i}{t},Interval0{j}{t}); % compute the bottleneck distance between the cell i and cell j with the sherical shell t for dimension 0
            end
            if (dim==1) || (dim==2)
                % checking the size of the pairwise interval of cell i and cell j in sphere t for dimension 1
                if size(Interval1{i}{t})==0 && size(Interval1{j}{t})==0 % If both are empty, set distance to 0.
                    BD1(i,j,t)=0;
                else % If either is non-empty, compute distance using Bottleneck.
                    BD1(i,j,t) = edu.stanford.math.plex4.bottleneck.BottleneckDistance.computeBottleneckDistance(Interval1{i}{t},Interval1{j}{t}); % compute the bottleneck distance between the cell i and cell j with the sherical shell t for dimension 1 and 2
                end
            end
            if (dim==2)
                % checking the size of the pairwise interval of cell i and cell j in sphere t for dimension 2
                if size(Interval2{i}{t})==0 && size(Interval2{j}{t})==0 % If both are empty, set distance to 0.
                    BD2(i,j,t)=0;
                else % If either is non-empty, compute distance using Bottleneck.
                    BD2(i,j,t) = edu.stanford.math.plex4.bottleneck.BottleneckDistance.computeBottleneckDistance(Interval2{i}{t},Interval2{j}{t}); % compute the bottleneck distance between the cell i and cell j with the sherical shell t for dimension 2
                end
            end
        end
    end
end


% Create nb*nb matrices according the chosen dimension (dim);
% For every spherical shell, add the pairwise distances then take the
%   square root to get the real distances according to the chosen dimension.

% for any chosen dimension, always compute for dimension dim=0
BD0_Neuron=zeros(nb,nb); % creating a nb*nb matrix
for t=1:kN 
    BD0_Neuron=BD0_Neuron+BD0(:,:,t).^2;
end
BD0_Neuron=sqrt(BD0_Neuron); % pairwise distance
BD0_Neuron=BD0_Neuron+BD0_Neuron'; % BD0_Neuron is the final pairwise distance symmetric matrix
save('OUTPUTS/Pairwise-distances_0dim','BD0_Neuron'); %saving BD0_Neuron

% if the chosen dimension dim=1 or dim=2
if(dim==1 || dim==2)
    BD1_Neuron=zeros(nb,nb); % creating the nb*nb matrix
    for t=1:kN
        BD1_Neuron=BD1_Neuron+BD1(:,:,t).^2;
    end
    BD1_Neuron=sqrt(BD1_Neuron);
    BD1_Neuron=BD1_Neuron+BD1_Neuron'; % BD1_Neuron is the final pairwise distance symmetric matrix.
    save('OUTPUTS/Pairwise-distances_1dim','BD1_Neuron'); % saving BD1_Neuron
end

% if the chosen dimension dim=2
if(dim==2)
    BD2_Neuron=zeros(nb,nb); % creating the nb*nb matrix
    for t=1:kN
        BD2_Neuron=BD2_Neuron+BD2(:,:,t).^2;
    end
    BD2_Neuron=sqrt(BD2_Neuron);
    BD2_Neuron=BD2_Neuron+BD2_Neuron'; % BD2_Neuron is the final pairwise distance symmetric matrix.
    save('OUTPUTS/Pairwise-distances_2dim','BD2_Neuron'); % saving BD2_Neuron
end

disp('FINISHED!')