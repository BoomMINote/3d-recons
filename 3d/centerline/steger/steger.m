clc;
clear;

srcImage=imread('test.jpg');
srcImage=rgb2gray(srcImage);
srcImage1=srcImage;
sigma=3;
thresh= graythresh(srcImage);%otsu  
srcImage=double(srcImage);


[m,n]=size(srcImage);

ky=[-1,1];
kx=[-1;1];
kyy=[1,-2,1];
kxx=[1;-2;1];
kxy=[1,-1;-1,1];


gausFilter = fspecial('gaussian',17,sigma);   %高斯滤波
dstImage=imfilter(srcImage,gausFilter,'replicate');  

dx=imfilter(dstImage,kx);
dy=imfilter(dstImage,ky);
dxx=imfilter(dstImage,kxx);
dyy=imfilter(dstImage,kyy);
dxy=imfilter(dstImage,kxy);

hessian=zeros(2,2);
points=zeros(m*n,2);

for i=1:m
    for j=1:n
        if(srcImage(i,j)/255>thresh)
            hessian(1,1)=dxx(i,j);
            hessian(1,2)=dxy(i,j);
            hessian(2,1)=dxy(i,j);
            hessian(2,2)=dyy(i,j);
            [eigenvectors,eigenvalues]=eig(hessian);
            
            if(abs(eigenvalues(1,1))>= abs(eigenvalues(2,2)))
            nx=eigenvectors(1,1);
            ny=eigenvectors(2,1);
            fmax_dist=eigenvalues(1,1);
            else
            nx=eigenvectors(1,2);
            ny=eigenvectors(2,2);
            fmax_dist=eigenvalues(2,2);
                
            end
            
            t=-(nx*dx(i,j)+ny*dy(i,j))/(nx*nx*dxx(i,j)+2 * nx*ny*dxy(i,j)+ny*ny*dyy(i,j));
            
            if(abs(t*nx) <= 0.5 && abs(t*ny) <= 0.5)
                points((i-1)*m+j,:)=[ j+t*ny,i+t*nx];
            end
            
            
        end
        
    end
end

index = find(points(:,1)==0);
points(index,:) = [];
index = find(points(:,2)==0);
points(index,:) = [];

figure(1);
imshow(uint8(srcImage));
hold on
plot(points(:,1),points(:,2),'r.','MarkerSize',1)
hold off







