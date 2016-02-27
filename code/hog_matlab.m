n_pos_samples=10;
n_neg_samples=100;
training=zeros(n_pos_samples+n_neg_samples,15*7*36);
label=zeros(n_pos_samples+n_neg_samples,1);
cellSize=8;
idx=1;
%positive samples
i=1;
path='/home/gazzib/Downloads/honours/Tracking/Tracking dataset/positive samples1/';
while i<=n_pos_samples
    fname=strcat(path,strcat(int2str(i),'.jpg'));
    %im=single(imread(fname));
    %hog = vl_hog(im, cellSize, 'verbose', 'variant', 'dalaltriggs') ;
    im=imread(fname);
    hog=hog_feature_vector(im);
    descriptor=hog(:);
    training(idx,:)=descriptor(:);
    label(idx)=1;
    idx=idx+1;
    i=i+1;
end

%negative samples
i=1;
path='/home/gazzib/Downloads/honours/Tracking/Tracking dataset/negative samples/';
while i<=n_neg_samples
    fname=strcat(path,strcat(int2str(i),'.jpg'));
    %im=single(imread(fname));
    %hog = vl_hog(im, cellSize, 'verbose', 'variant', 'dalaltriggs') ;
    im=imread(fname);
    hog=hog_feature_vector(im);
    descriptor=hog(:);
    training(idx,:)=descriptor(:);
    label(idx)=-1;
    idx=idx+1;
    i=i+1;
end
n_pos_samples=10;
n_neg_samples=200;
pos=zeros(n_pos_samples,1);
[w b] =vl_svmtrain(training',label,0.01);
% path='/home/gazzib/Downloads/honours/Tracking/Tracking dataset/positive samples1/';
% i=1;
% while i<=n_pos_samples
%     fname=strcat(path,strcat(int2str(i),'.jpg'));
%     %im=single(imread(fname));
%     %hog = vl_hog(im, cellSize, 'verbose', 'variant', 'dalaltriggs') ;
%     im=imread(fname);
%     hog=hog_feature_vector(im);
%     descriptor=hog(:);
%     scores=w'*descriptor+b;
%     pos(i)=scores;
%     i=i+1;
% end
% 
% neg=zeros(n_neg_samples,1);
% path='/home/gazzib/Downloads/honours/Tracking/Tracking dataset/negative samples/';
% i=1;
% while i<=n_neg_samples
%     fname=strcat(path,strcat(int2str(i),'.jpg'));
%     %im=single(imread(fname));
%     im=imread(fname);
%     hog=hog_feature_vector(im);
%     %hog = vl_hog(im, cellSize, 'verbose', 'variant', 'dalaltriggs') ;
%     descriptor=hog(:);
%     scores=w'*descriptor+b;
%     neg(i)=scores;
%     i=i+1;
% end


im=imread('/home/gazzib/Downloads/honours/Tracking/Tracking dataset/dos1_all_frames_jpg/00001.jpg');
im=impyramid(im,'reduce');
im=impyramid(im,'reduce');
im=impyramid(im,'reduce');
counter=0;
arr=zeros(10000,4);
arr1=zeros(135,240)-inf;
tic
for i=1:size(im,1)-35
    for j=1:size(im,2)-17
         imx=im(i:i+35,j:j+17);
         imx=imresize(imx,[128,64]);
%          hog = vl_hog(single(imx), cellSize, 'verbose', 'variant', 'dalaltriggs') ;
        hog=hog_feature_vector(imx); 
        descriptor=hog(:);
         scores=w'*descriptor+b;
         arr1(i,j)=scores;
         if scores>-0.5 
            counter=counter+1;
            arr(counter,1)=j;
            arr(counter,2)=i;
            arr(counter,3)=j+17;
            arr(counter,4)=i+35;
         end
    end
end
toc
top=nms(arr,0.5);
figure,imshow(im);
for i=1:size(top,1)-1
    rectangle ('Position',[top(i,1) top(i,2) 17 35])
end