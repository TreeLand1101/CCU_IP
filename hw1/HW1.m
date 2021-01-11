clc;
clear all; 

clc;
clear all; 
img_array=cell(1,3);
img_array{1} = imread('Cameraman.bmp');
img_array{2} = imread('Lena.bmp');
img_array{3} = imread('Peppers.bmp');

for i=1:3
    
    I_d = im2double(img_array{i});
    I_Power_Law = 2*(I_d.^1.5);
    
    I_Histogram = myhisteq(img_array{i});
   
    I_Sharpening = mylap(img_array{i});
 
    figure()
    subplot(2,2,1),imshow(img_array{i});title('Original image');
    subplot(2,2,2),imshow(I_Power_Law);title('power-law transformation');
    subplot(2,2,3),imshow(I_Histogram);title('histogram equalization');
    subplot(2,2,4),imshow(I_Sharpening);title('image sharpening using the Laplacian');
    figure()
    histogram(I_Histogram);title('histograms');
end

function  I_Histogram = myhisteq(I) 

%image size 
[r,c] = size(I); 
I_Histogram = uint8(zeros(r,c));

% pixels
n = r*c;

% 初始化
f = zeros(256,1);
cdf = zeros(256,1);
out = zeros(256,1);

%array index 1~256
%value 0~255

%計算每個value出現次數
for i = 1:r
    for j = 1:c
        value = I(i,j);
        f(value+1) = f(value+1)+1;
    end
end

%計算每個value出現的比例並加入cdf
%round(cdf(i)*L) => 將每個出現的 cdf 轉成對應的gray level
sum = 0;
L = 255;

for i = 1:256
    sum = sum + f(i);
    cdf(i) = sum/n;
    out(i) = round(cdf(i)*L);
end

%將對應的gray level取代原本的值
for i = 1:r
    for j = 1:c
        I_Histogram(i,j) = out(I(i,j)+1);
    end
end    
end

function  I_S = mylap(I) 

I_S = im2double(I);

% mask = [0,-1,0;-1,4,-1;0,-1,0];
%resp為原圖經過轉換的結果
[m,n] = size(I_S);
resp = I_S;
for i = 2:m-1
    for j = 2:n-1
        resp(i,j) = 4*I_S(i,j)-I_S(i+1,j)-I_S(i-1,j)-I_S(i,j+1)-I_S(i,j-1);
    end
end

%將mask處理的圖加上原圖
I_S = I_S + resp;

end