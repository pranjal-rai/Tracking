//pranjalr34
#include<bits/stdc++.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <opencv2/ml/ml.hpp>
#include "hog.h"
#include "/home/gazzib/vlfeat-0.9.20/vl/svm.h"
#include "/home/gazzib/vlfeat-0.9.20/vl/hog.h"
using namespace std;
using namespace cv;


int main()
{
    string str;
    int len=3780;
    vector <float> descriptor;
    descriptor=hog_descriptor(str);
    int n_neg_samples=100,n_pos_samples=10;
    int n=n_neg_samples+n_pos_samples;
    double training[len*(n_neg_samples+n_pos_samples)];
    double labels[(n_neg_samples+n_pos_samples)];
    string path="/home/gazzib/Downloads/honours/Tracking/Tracking dataset/positive samples/";
    int idx=0,i,j,idx1=0;
    vl_size numOrientations=9;
    for(i=1;i<=n_pos_samples;i++,idx++)
    {
        str=path+to_string(i)+".jpg";
        descriptor=hog_descriptor(str);
        VlHog * hog = vl_hog_new(VlHogVariantDalalTriggs, numOrientations, VL_FALSE) ;
        vl_hog_put_image(hog, image, height, width, numChannels, cellSize) ;
        hogWidth = vl_hog_get_width(hog) ;
        hogHeight = vl_hog_get_height(hog) ;
        hogDimenison = vl_hog_get_dimension(hog) ;
        hogArray = vl_malloc(hogWidth*hogHeight*hogDimension*sizeof(float)) ;
        vl_hog_extract(hog, hogArray);
        vl_hog_delete(hog);
        for(j=0;j<len;j++)
        {
            training[idx1]=descriptor[j];
            idx1++;
        }
        labels[idx]=1;
    }
    path="/home/gazzib/Downloads/honours/Tracking/Tracking dataset/negative samples/";
    for(i=1;i<=n_neg_samples;i++,idx++)
    {
        str=path+to_string(i)+".jpg";
        descriptor=hog_descriptor(str);
        for(j=0;j<len;j++)
        {
            training[idx1]=descriptor[j];
            idx1++;
        }
        labels[idx]=-1;
    }
    const double *model;
    double bias ;
    double lambda = 0.01;
    VlSvm * svm = vl_svm_new(VlSvmSolverSgd,training,len,n_pos_samples+n_neg_samples,labels,lambda);
    vl_svm_train(svm);
    model = vl_svm_get_model(svm) ;
    bias = vl_svm_get_bias(svm);
    
    
    
    path="/home/gazzib/Downloads/honours/Tracking/Tracking dataset/positive samples/";
    cout <<"********Positive Samples********"<<"\n";
    for(i=1;i<=10;i++,idx++)
    {
        str=path+to_string(i)+".jpg";
        descriptor=hog_descriptor(str);
        double ans=0;
        for(j=0;j<len;j++)
            ans+=descriptor[j]*model[j];
        ans+=bias;
        cout <<ans<<"\n";
    }
    path="/home/gazzib/Downloads/honours/Tracking/Tracking dataset/negative samples/";
    
    cout <<"********Negative Samples********"<<"\n";
    for(i=1;i<=100;i++,idx++)
    {
        str=path+to_string(i)+".jpg";
        descriptor=hog_descriptor(str);
        double ans=0;
        for(j=0;j<len;j++)
            ans+=descriptor[j]*model[j];
        ans+=bias;
        cout <<ans<<"\n";
    }
   
    
    
    
    
    
    
    return 0;
}
