//pranjalr34


#include<bits/stdc++.h>
#include <cv.h>
#include <omp.h>
#include <cxcore.h>
#include <highgui.h>
#include <opencv2/ml/ml.hpp>
#include "/home/gazzib/vlfeat-0.9.20/vl/svm.h"
#include "/home/gazzib/vlfeat-0.9.20/vl/hog.h"
#define F first
#define S second
using namespace std;
using namespace cv;

void FastIO(){ios_base::sync_with_stdio(0);cin.tie(NULL);cout.tie(NULL);cout.precision(15);}


vl_size numOrientations=9;
vl_size height=128;
vl_size width=64;
vl_size numChannels=3;
vl_size cellSize=8;


VlHog * hog;
void cvtmat(Mat &mat,float image[])
{
    int i,j,l=0;
#pragma omp parallel default(shared)
    {
#pragma omp for collapse(2)
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                image[j + mat.cols*i + mat.cols*mat.rows*0] = mat.at<cv::Vec3b>(i, j)[0];
                image[j + mat.cols*i + mat.cols*mat.rows*1] = mat.at<cv::Vec3b>(i, j)[1];
                image[j + mat.cols*i + mat.cols*mat.rows*2] = mat.at<cv::Vec3b>(i, j)[2];
            }
        }
    }
}

float *compute_descriptor(float image[])
{
    vl_hog_put_image(hog, image, height, width, numChannels, cellSize) ;
/*  vl_size hogWidth = vl_hog_get_width(hog) ;
    vl_size hogHeight = vl_hog_get_height(hog) ;
    vl_size hogDimension = vl_hog_get_dimension(hog) ;
    cout <<hogWidth<<" "<<hogHeight<<" "<<hogDimension<<"\n";
    float *hogArray = (float*)vl_malloc(hogWidth*hogHeight*hogDimension*sizeof(float)) ;*/
    float *hogArray = (float*)vl_malloc(18432);
    vl_hog_extract(hog, hogArray);
//    vl_hog_delete(hog);
    return hogArray;
}

vector<vector<int> > nms(vector<vector<int> > &v,double overlap)
{
    vector<vector<int> > top;
    if(v.empty())
        return top;
    int len=v.size();
    vector<int> x1(len),y1(len),x2(len),y2(len);
    vector <int> xx1,xx2,yy1,yy2,vec;
    vector<pair<int,int> > s(len),s1;
    int i,j,last;
    for(i=0;i<len;i++)
    {
        x1[i]=v[i][0];
        y1[i]=v[i][1];
        x2[i]=v[i][2];
        y2[i]=v[i][3];
        s[i].F=v[i][3];
        s[i].S=i;
    }
    double area=(x2[0]-x1[0]+1)*(y2[0]-y1[0]+1),o,w,h;
    sort(s.begin(),s.end());
    while(!s.empty())
    {
        last=s.size()-1;
        i=s[last].S;
        vec.clear();
        vec.push_back(x1[i]);
        vec.push_back(y1[i]);
        vec.push_back(x2[i]);
        vec.push_back(y2[i]);
        top.push_back(vec);
        xx1.resize(last);
        yy1.resize(last);
        xx2.resize(last);
        yy2.resize(last);
        s.erase(s.begin()+last);
        for(j=0;j<=last-1;j++)
        {
            xx1[j]=max(x1[i],x1[s[j].S]);
            yy1[j]=max(y1[i],y1[s[j].S]);
            xx2[j]=min(x2[i],x2[s[j].S]);
            yy2[j]=min(y2[i],y2[s[j].S]);
            w=max(0.0,double(xx2[j]-xx1[j]+1));
            h=max(0.0,double(yy2[j]-yy1[j]+1));
            o=(w*h)/area;
            if(o<=overlap)
                s1.push_back(s[j]);
        }
        s.clear();
        for(j=0;j<s1.size();j++)
            s.push_back(s1[j]);
        s1.clear();
    }
    return top;
}


int main()
{
    FastIO();
    string str;

    int dim=4608;
    int n_pos_samples=10;
    int n_neg_samples=200;
    int n=n_neg_samples+n_pos_samples;

    float image[width*height*3];  //width*height

    double training[dim*n];
    double labels[n];

    string path_pos_samples="/home/gazzib/Downloads/honours/Tracking/Tracking dataset/positive samples1/";
    string path_neg_samples="/home/gazzib/Downloads/honours/Tracking/Tracking dataset/negative samples/";


    int idx=0,idx1=0,i,j,k,l;

    Mat mat;

    for(i=1;i<=n_pos_samples;i++,idx++)
    {
        str=path_pos_samples+to_string(i)+".jpg";
        mat=imread(str.c_str(),CV_LOAD_IMAGE_COLOR);
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
        mat=imread(str.c_str(),CV_LOAD_IMAGE_COLOR);
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


    
    /*   cout <<"********Positive Samples********"<<"\n";
       for(i=1;i<=10;i++,idx++)
       {
       str=path_pos_samples+to_string(i)+".jpg";
       mat=imread(str.c_str(),CV_LOAD_IMAGE_COLOR);
       cvtmat(mat,image);
       float *hogArray=compute_descriptor(image);
       double ans=0;
       for(j=0;j<dim;j++)
       ans+=hogArray[j]*model[j];
       ans+=bias;
       cout <<ans<<"\n";
       }


       cout <<"********Negative Samples********"<<"\n";
       for(i=1;i<=200;i++,idx++)
       {
       str=path_neg_samples+to_string(i)+".jpg";
       mat=imread(str.c_str(),CV_LOAD_IMAGE_COLOR);
       cvtmat(mat,image);
       float *hogArray=compute_descriptor(image);
       double ans=0;
       for(j=0;j<dim;j++)
       ans+=hogArray[j]*model[j];
       ans+=bias;
       cout <<ans<<"\n";
       }   
     */

    string path="/home/gazzib/Downloads/honours/Tracking/Tracking dataset/dos1_all_frames_jpg/00948.jpg";
    mat=imread(path.c_str(),CV_LOAD_IMAGE_COLOR);
    pyrDown(mat,mat);
    pyrDown(mat,mat);
    pyrDown(mat,mat);
    Mat newmat=mat.clone();
    int r=mat.rows;
    int c=mat.cols;
    int h=35,w=17;
    Mat submat=mat;
    int counter=0;
    Size sz(64,128);
    vector<int> vec;
    vector<vector<int> > v;
    int mxi = r-h;
    int mxj = c-w;
    //double ans;
    hog = vl_hog_new(VlHogVariantDalalTriggs, numOrientations, VL_FALSE) ;
    omp_set_num_threads(4);
#pragma omp parallel private(submat,vec) firstprivate(dim,bias,model,image,sz) default(shared)
    {
#pragma omp for collapse(2)
        for(int i=0;i<mxi;i++)
        {
            for(int j=0;j<mxj;j++)
            {
                Rect r(j,i,w,h);
                submat=mat(r).clone();
                resize(submat,submat,sz);
                cvtmat(submat,image);
                float* hogArray=compute_descriptor(image);
                double ans=0;
#pragma omp parallel for reduction(+:ans)
                for(int l=0;l<dim;l++)
                {
                    ans+=hogArray[l]*model[l];
                }
                ans+=bias;
                if(ans>1.9)
                {
                    vec.clear();
                    vec.push_back(j); 
                    vec.push_back(i); 
                    vec.push_back(j+w); 
                    vec.push_back(i+h); 
                    v.push_back(vec);
                }
            }
        }
    }
    v=nms(v,0.5);
    Point p1,p2;
    for(i=0;i<v.size();i++)
    {
        p1.x=v[i][0];
        p1.y=v[i][1];
        p2.x=v[i][2];
        p2.y=v[i][3];
        rectangle(newmat,p1,p2,0);
    }
    //namedWindow("results",WINDOW_NORMAL);
    //imshow("results",newmat);
    //waitKey(0);
    return 0;
}
