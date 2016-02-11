//pranjalr34
#include<bits/stdc++.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <opencv2/ml/ml.hpp>
#include "hog.h"
using namespace std;
using namespace cv;


int main()
{
    string str;
    int len=3780;
    vector <float> descriptor;
    descriptor=hog_descriptor(str);
    int n_neg_samples=10,n_pos_samples=8;
    int n=n_neg_samples+n_pos_samples;
    Mat training_mat(n,len,CV_32F);
    Mat labels(n,1,CV_32F);
    string path="/home/gazzib/Downloads/honours/project 2/code/Tracking dataset/positive samples/";
    int idx=0,i,j;
    for(i=1;i<=n_pos_samples;i++,idx++)
    {
        str=path+to_string(i)+".jpg";
        descriptor=hog_descriptor(str);
        for(j=0;j<len;j++)
            training_mat.at<float>(idx,j)=descriptor[j];
        labels.at<float>(idx,0)=1;
    }
    path="/home/gazzib/Downloads/honours/project 2/code/Tracking dataset/negative samples/";
    for(i=1;i<=n_neg_samples;i++,idx++)
    {
        str=path+to_string(i)+".jpg";
        descriptor=hog_descriptor(str);
        for(j=0;j<len;j++)
            training_mat.at<float>(idx,j)=descriptor[j];
        labels.at<float>(idx,0)=-1;
    }
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    
    CvSVM SVM;
    SVM.train(training_mat, labels, Mat(), Mat(), params);
    SVM.save("my_svm");
    
    //SVM.load("my_svm");
    
    
    path="/home/gazzib/Downloads/honours/project 2/code/Tracking dataset/positive samples/";
    Mat classify_mat(1,len,CV_32F);
    cout <<"********Positive Samples********"<<"\n";
    for(i=1;i<=n_pos_samples;i++,idx++)
    {
        str=path+to_string(i)+".jpg";
        descriptor=hog_descriptor(str);
        for(j=0;j<len;j++)
            classify_mat.at<float>(0,j)=descriptor[j];
        cout <<SVM.predict(classify_mat)<<"\n";
    }
    path="/home/gazzib/Downloads/honours/project 2/code/Tracking dataset/negative samples/";
    
    cout <<"********Negative Samples********"<<"\n";
    for(i=1;i<=n_neg_samples;i++,idx++)
    {
        str=path+to_string(i)+".jpg";
        descriptor=hog_descriptor(str);
        for(j=0;j<len;j++)
            classify_mat.at<float>(0,j)=descriptor[j];
        cout <<SVM.predict(classify_mat)<<"\n";
    }
   
    
    
    
    
    /* int c=SVM.get_support_vector_count();
    cout<<c<<"\n";
    
    Mat training_mat1(8,1,CV_32F);
    Mat labels1(8,1,CV_32F);
    training_mat1.at<float>(0,0)=1.0;
    training_mat1.at<float>(1,0)=2.0;
    training_mat1.at<float>(2,0)=3.0;
    training_mat1.at<float>(3,0)=4.0;
    training_mat1.at<float>(4,0)=5.0;
    training_mat1.at<float>(5,0)=6.0;
    training_mat1.at<float>(6,0)=7.0;
    training_mat1.at<float>(7,0)=8.0;
    labels1.at<float>(0,0)=1;
    labels1.at<float>(1,0)=1;
    labels1.at<float>(2,0)=1;
    labels1.at<float>(3,0)=-1;
    labels1.at<float>(4,0)=-1;
    labels1.at<float>(5,0)=-1;
    labels1.at<float>(6,0)=-1;
    labels1.at<float>(7,0)=-1;
    Mat classify_mat(1,1,CV_32F);
    SVM.train(training_mat1, labels1, Mat(), Mat(), params);
    classify_mat.at<float>(0,0)=3.6;
    cout <<SVM.predict(classify_mat)<<"\n";*/
    return 0;
}
