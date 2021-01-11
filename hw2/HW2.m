clc;
clear all;
img_array=cell(1,2);
img_array{1} = imread('skeleton_orig.bmp');
img_array{2} = imread('blurry_moon.tif');
img_array{1}=rgb2gray(img_array{1});

for i=1:2
    
    mask=[0,-1,0;-1,4,-1;0,-1,0];
    I_Lap = img_array{i}+convolution_spatial(img_array{i},mask);
    
    mask=[0,-1,0;-1,5,-1;0,-1,0];
    I_Unsharp = convolution_spatial(img_array{i},mask);
    
    mask=[0,-1,0;-1,5.5,-1;0,-1,0];
    I_High_Boost = convolution_spatial(img_array{i},mask);
    
    figure;
    sgtitle('Spatial domain');
    subplot(2,2,1),imshow(img_array{i});title('Original image');
    subplot(2,2,2),imshow(I_Lap);title('Laplacian operator');
    subplot(2,2,3),imshow(I_Unsharp);title('unsharp masking');
    subplot(2,2,4),imshow(I_High_Boost);title('high-boost filtering ');
end

for i=1:2
    
    mask=[0,-1,0;-1,4,-1;0,-1,0]; 
    I_Lap = img_array{i}+frequency(img_array{i},mask);

    mask=[0,-1,0;-1,5,-1;0,-1,0]; 
    I_Unsharp = frequency(img_array{i},mask);
    
    mask=[0,-1,0;-1,5.5,-1;0,-1,0]; 
    I_High_Boost =  frequency(img_array{i},mask);
    
    figure;
    sgtitle('frequency domain');
    subplot(2,2,1),imshow(img_array{i});title('Original image');
    subplot(2,2,2),imshow(I_Lap);title('Laplacian operator');
    subplot(2,2,3),imshow(I_Unsharp);title('unsharp masking');
    subplot(2,2,4),imshow(I_High_Boost);title('high-boost filtering ');
end

function  output = convolution_spatial(I,mask) 
[m,n]=size(I);
I=im2double(I);
output=zeros(size(I));
for i = 2 : m - 1
    for j = 2 : n - 1
        tmp=mask.*I(i-1:i+1,j-1:j+1);
        output(i-1:i+1,j-1:j+1)=sum(tmp(:));
    end
end
output=im2uint8(output);
end

function output = frequency(I,mask)
I = im2double(I);
[m n] = size(I);

% 1.
%影像乘上(-1)x+y,為了置中
for i=1:m
    for j=1:n
        I(i,j) = I(i,j)*(-1)^(i+j);
    end
end

% 2.
F = fft2(I);
% Frequency response 
H =  freqz2(mask,[m n]); 

% 3.和4. 
%G(u,v) = H(u,v)F(u,v)
G = H.*F;  

% 5.
%取Filtered image的實部
G = real(ifft2(G)); 

% 6.
%乘上(-1)x+y抵銷一開始對影像乘(-1)x+y
for i=1:m
    for j=1:n
        G(i,j) = G(i,j)*(-1)^(i+j);
    end
end

G = im2uint8(G);
output=G;
end