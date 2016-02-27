//pranjalr34

#include<bits/stdc++.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
using namespace std;
using namespace cv;

double eps=1e-7;

void normalize(vector <float> &vec)
{
    int len,m;
    len=vec.size();
    float val=norm(vec,NORM_L2);
    val=sqrt(val*val+eps);
    for(m=0;m<len;m++)
        vec[m]=vec[m]/val;
}


vector<float> hog_descriptor(Mat im)
{
    Mat Ix,Iy,Gdir,Gmag;
//    im=imread(str.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
    int rows=im.rows;
    int cols=im.cols;
    Ix=Mat::zeros(rows,cols,CV_32F);
    Iy=Mat::zeros(rows,cols,CV_32F);
    Gdir=Mat::zeros(rows,cols,CV_32F);
    Gmag=Mat::zeros(rows,cols,CV_32F);
    /*  namedWindow("InputImage",WINDOW_NORMAL);
        imshow("InputImage",im);
        waitKey(0);*/
    int i,j,k,l,p,q,x,y,m;
    for(i=0;i<rows;i++)
    {
        for(j=0;j<cols;j++)
        {
            if(j<cols-2)
                Ix.at<float>(i,j)=im.at<uchar>(i,j)-im.at<uchar>(i,j+2);
            else
                Ix.at<float>(i,j)=im.at<uchar>(i,j);
            if(i<rows-2)
                Iy.at<float>(i,j)=im.at<uchar>(i,j)-im.at<uchar>(i+2,j);
            else
                Iy.at<float>(i,j)=im.at<uchar>(i,j);
            Gdir.at<float>(i,j)=90+((atan(Ix.at<float>(i,j)/Iy.at<float>(i,j)))*180/M_PI);
            if(isnan(Gdir.at<float>(i,j))||Gdir.at<float>(i,j)<0)
                Gdir.at<float>(i,j)=0;
            Gmag.at<float>(i,j)=sqrt((pow(Iy.at<float>(i,j),2)+pow(Ix.at<float>(i,j),2)));
        }
    }
    int block_size=2;//2x2 cells per block
    int cell_size=8; //8x8 pixels per cell
    int idx1,idx2,len;
    float ang,mag;
    vector <float> bin,block,descriptor;
    for(i=0;i<=rows-2*cell_size;i=i+cell_size)
    {
        for(j=0;j<=cols-2*cell_size;j=j+cell_size)
        {
            block.clear();
            for(k=0;k<block_size;k++)
            {
                for(l=0;l<block_size;l++)
                {
                    bin.resize(9,0);
                    for(p=0;p<cell_size;p++)
                    {
                        x=i+k*cell_size+p;
                        for(q=0;q<cell_size;q++)
                        {
                            y=j+l*cell_size+q;
                            ang=Gdir.at<float>(x,y);
                            mag=Gmag.at<float>(x,y);
                            //binning
                            if(ang>10&&ang<=170)
                            {
                                idx1=(ang-10-eps)/20+1;
                                idx2=idx1+1;
                                bin[idx1-1]+=mag*((20*idx1+10-ang)/20);
                                bin[idx2-2]+=mag*((ang-(20*idx1-10))/20);
                            }
                            else
                            {
                                if(ang>=0&&ang<=10)
                                {
                                    bin[0]+=mag*(ang+10)/20;
                                    bin[8]+=mag*(10-ang)/20;
                                }
                                else if(ang>=170&&ang<=180)
                                {
                                    bin[0]+=mag*(190-ang)/20;
                                    bin[8]+=mag*(ang-170)/20;
                                }
                            }
                        }
                    }
                    block.insert(block.end(),bin.begin(),bin.end());
                }
            }
            normalize(block);
            descriptor.insert(descriptor.end(),block.begin(),block.end());
        }
    }
    len=descriptor.size();
    normalize(descriptor);
    //clipping
    for(m=0;m<len;m++)
    {
        if(descriptor[m]>0.2)
            descriptor[m]=0.2;
    }
    normalize(descriptor);
    return descriptor;
}
