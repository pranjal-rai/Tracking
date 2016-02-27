//pranjalr34


#include<bits/stdc++.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <opencv2/ml/ml.hpp>
#include "/home/gazzib/vlfeat-0.9.20/vl/svm.h"
#include "/home/gazzib/vlfeat-0.9.20/vl/hog.h"
using namespace std;
using namespace cv;

void FastIO(){ios_base::sync_with_stdio(0);cin.tie(NULL);cout.tie(NULL);cout.precision(15);}


vl_size numOrientations=9;
vl_size height=128;
vl_size width=64;
vl_size numChannels=1;
vl_size cellSize=8;


void cvtmat(Mat &mat,float image[])
{
    int i,j,l=0;
    for (int i=0;i<mat.rows;i++)
    {
        for (int j=0;j<mat.cols;j++)
        {
            image[l]=mat.at<unsigned char>(i,j);
            l++;
        }
    }
}

float *compute_descriptor(float image[])
{

    VlHog * hog = vl_hog_new(VlHogVariantDalalTriggs, numOrientations, VL_FALSE) ;
    vl_hog_put_image(hog, image, height, width, numChannels, cellSize) ;
    vl_size hogWidth = vl_hog_get_width(hog) ;
    vl_size hogHeight = vl_hog_get_height(hog) ;
    vl_size hogDimension = vl_hog_get_dimension(hog) ;
    float *hogArray = (float*)vl_malloc(hogWidth*hogHeight*hogDimension*sizeof(float)) ;
    vl_hog_extract(hog, hogArray);
    vl_hog_delete(hog);
    return hogArray;
}


int main()
{
    FastIO();
    string str;

    int dim=4608;
    int n_pos_samples=4;
    int n_neg_samples=50;
    int n=n_neg_samples+n_pos_samples;

    float image[width*height];  //width*height

    double training[dim*n];
    double labels[n];

    string path_pos_samples="/home/gazzib/Downloads/honours/Tracking/Tracking dataset/positive samples/";
    string path_neg_samples="/home/gazzib/Downloads/honours/Tracking/Tracking dataset/negative samples/";


    int idx=0,idx1=0,i,j,k,l;

    Mat mat;

    for(i=1;i<=n_pos_samples;i++,idx++)
    {
        str=path_pos_samples+to_string(i)+".jpg";
        mat=imread(str.c_str(),0);
        cvtmat(mat,image);    
        float *hogArray=compute_descriptor(image);
        for(j=0;j<dim;j++)
        {
            training[idx1]=hogArray[j];
            idx1++;
        }
        labels[idx]=1;
    }

    for(i=1;i<=n_neg_samples;i++,idx++)
    {
        str=path_neg_samples+to_string(i)+".jpg";
        mat=imread(str.c_str(),0);
        cvtmat(mat,image);
        float *hogArray=compute_descriptor(image);
        for(j=0;j<dim;j++)
        {
            training[idx1]=hogArray[j];
            idx1++;
        }
        labels[idx]=-1;
    }


    //SVM training
    const double *model;
    double bias ;
    double lambda = 0.01;
    VlSvm * svm = vl_svm_new(VlSvmSolverSgd,training,dim,n_pos_samples+n_neg_samples,labels,lambda);
    vl_svm_train(svm);
    model = vl_svm_get_model(svm) ;
    bias = vl_svm_get_bias(svm);


    /*
    cout <<"********Positive Samples********"<<"\n";
    for(i=1;i<=10;i++,idx++)
    {
        str=path_pos_samples+to_string(i)+".jpg";
        mat=imread(str.c_str(),0);
        cvtmat(mat,image);
        float *hogArray=compute_descriptor(image);
        double ans=0;
        for(j=0;j<dim;j++)
            ans+=hogArray[j]*model[j];
        ans+=bias;
        cout <<ans<<"\n";
    }


    cout <<"********Negative Samples********"<<"\n";
    for(i=1;i<=100;i++,idx++)
    {
        str=path_neg_samples+to_string(i)+".jpg";
        mat=imread(str.c_str(),0);
        cvtmat(mat,image);
        float *hogArray=compute_descriptor(image);
        double ans=0;
        for(j=0;j<dim;j++)
            ans+=hogArray[j]*model[j];
        ans+=bias;
        cout <<ans<<"\n";
    }   
    */

    string path="/home/gazzib/Downloads/honours/Tracking/Tracking dataset/dos1_all_frames_jpg/00001.jpg";
    mat=imread(path.c_str(),0);
    
    //cout <<mat.rows<<" "<<mat.cols<<"\n";
    //Size sz(64,128);
   // resize(mat,mat,sz,0,0);
   
    
    /* namedWindow("window",WINDOW_AUTOSIZE);
    imshow("window",mat);
     waitKey(0);*/
    
    int num_levels=1;
    int w=64,h=128;
    Mat submat;
    int counter=0;
    //cout <<(1LL*w)*h*mat.rows*mat.cols<<"\n";
    for(i=1;i<=num_levels;i++)
    {
        for(j=0;j+h<mat.rows;j++)
        {
            for(k=0;k+w<mat.cols;k++)
            {
                submat=mat(Rect(k,j,w,h));
                cvtmat(submat,image);
                float *hogArray=compute_descriptor(image);
                double ans=0;
                for(l=0;l<dim;l++)
                    ans+=hogArray[l]*model[l];
                ans+=bias;
                //cout <<ans<<"\n";
                if(ans>0)
                    cout <<j<<" "<<k<<"\n";
            }
        }
    }
    return 0;
}
