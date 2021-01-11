clc;
clear all;
% 初始化
img_array=cell(1,3);
img_array{1} = imread('image1.jpg');
img_array{2} = imread('image2.jpg');
img_array{3} = imread('image3.jpg');

for i=1:3

% Sobel
mask=[-1,-2,-1;0,0,0;1,2,1];
I_Sobel_X = convolution(img_array{i},3,mask);
mask=[1,0,-1;2,0,-2;1,0,-1];
I_Sobel_Y = convolution(img_array{i},3,mask);

I_Sobel=I_Sobel_X+I_Sobel_Y;

% LoG
mask=[0,0,-1,0,0;0,-1,-2,-1,0;-1,-2,16,-2,-1;0,-1,-2,-1,0;0,0,-1,0,0];
I_LoG = convolution(img_array{i},5,mask);

% 結果
figure;
sgtitle('edge detection');
subplot(1,3,1),imshow(img_array{i});title('Original image');
subplot(1,3,2),imshow(I_Sobel);title('Sobel');
subplot(1,3,3),imshow(I_LoG);title('LoG');
end


function  output = convolution(I,m_size,mask) 
[m,n]=size(I);
I=im2double(I);
output=zeros(size(I));
Start = ceil(m_size/2);
End = floor(m_size/2);
for i = Start : m - End
    for j = Start : n - End
        tmp=mask.*I(i-End:i+End,j-End:j+End);
        output(i-End:i+End,j-End:j+End)=sum(tmp(:));      
    end
end
output=im2uint8(output);
end

