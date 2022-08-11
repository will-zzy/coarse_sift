#include<opencv2/opencv.hpp>
#include <opencv2/videostab/motion_core.hpp>
#include<iostream>
#include<math.h>
#include <vector>
#include "mySift.h"
using namespace cv;
using namespace std;
#define SIGMA 1//��˹�˷���
#define S 2//��ֵ���������
#define K pow(2,(float)1/S)//�߶ȼ�����
#define OCT_NUM 5//��˹������octave��
#define LEVEL_NUM S+3//��˹��������ÿһ��octave�Ĳ���
#define PI 3.1415926535
MyPoint::MyPoint(int x, int y) {
	this->x = x;
	this->y = y;
}
pyramidLevel::pyramidLevel(int num, float sigma, Mat img) {
	this->num = num;
	this->sigma = sigma;
	this->LevelMat = img;
}
pyramidLevel::pyramidLevel() {}
pyramidOctave::pyramidOctave(int numO, float sigma, Mat m) {
	this->num = numO;
	Mat begin;
	GaussianBlur(m, begin, Size(3, 3), sigma, sigma);
	pyramidLevel mid(0, sigma, begin);
	this->PL.push_back(mid);
	for (int i = 1; i < LEVEL_NUM; i++) {
		Mat img;
		float sigma = K * this->PL[i - 1].sigma;
		GaussianBlur(this->PL[i - 1].LevelMat, img, Size(3, 3), sigma, sigma);//��˹ģ��
		pyramidLevel mid(i, sigma, img);
		this->PL.push_back(mid);
	}
}
pyramidOctave::pyramidOctave() {}
void sift::DoGStruct() {//����DOG������
	for (int i = 0; i < OCT_NUM; i++) {
		for (int j = 1; j < LEVEL_NUM; j++) {
			Mat img = this->octave[i].PL[j].LevelMat - this->octave[i].PL[j - 1].LevelMat;

			if (img.size() != this->octave[0].PL[0].LevelMat.size()) {
				Mat gray;
				for (int u = i; u > 0; u--) {
					int POW = pow(2, u - 1);
					pyrUp(img, gray, (this->octave[0].PL[0].LevelMat.size() / POW));//��Ҫ����i���ϲ�����������ܻ����λ�õ�ƫ��
					img = gray;
				}
				pyramidLevel midL(j - 1, this->octave[i].PL[j - 1].sigma, gray);
				this->DOG.push_back(midL);

			}
			else {
				pyramidLevel midL(j - 1, this->octave[i].PL[j - 1].sigma, img);
				this->DOG.push_back(midL);
			}
		}
	}
}
void sift::ExtractEval() {
	for (int i = 1; i < this->DOG.size() - 1; i++) {
		Mat img = this->DOG[i].LevelMat;
		for (int m = 1; m < img.rows - 1; m++) {
			for (int n = 1; n < img.cols - 1; n++) {
				float val = img.ptr<float>(m)[n];
				if (abs(val) < 0.3)continue;
				float f1 = this->DOG[i - 1].LevelMat.ptr<float>(m - 1)[n - 1];
				float f2 = this->DOG[i - 1].LevelMat.ptr<float>(m - 1)[n];
				float f3 = this->DOG[i - 1].LevelMat.ptr<float>(m - 1)[n + 1];

				float f4 = this->DOG[i - 1].LevelMat.ptr<float>(m)[n - 1];
				float f5 = this->DOG[i - 1].LevelMat.ptr<float>(m)[n];
				float f6 = this->DOG[i - 1].LevelMat.ptr<float>(m)[n + 1];

				float f7 = this->DOG[i - 1].LevelMat.ptr<float>(m + 1)[n - 1];
				float f8 = this->DOG[i - 1].LevelMat.ptr<float>(m + 1)[n];
				float f9 = this->DOG[i - 1].LevelMat.ptr<float>(m + 1)[n - 1];

				float f10 = this->DOG[i].LevelMat.ptr<float>(m - 1)[n - 1];
				float f11 = this->DOG[i].LevelMat.ptr<float>(m - 1)[n];
				float f12 = this->DOG[i].LevelMat.ptr<float>(m - 1)[n + 1];

				float f13 = this->DOG[i].LevelMat.ptr<float>(m)[n - 1];
				float f14 = this->DOG[i].LevelMat.ptr<float>(m)[n];
				float f15 = this->DOG[i].LevelMat.ptr<float>(m)[n + 1];

				float f16 = this->DOG[i].LevelMat.ptr<float>(m + 1)[n - 1];
				float f17 = this->DOG[i].LevelMat.ptr<float>(m + 1)[n];
				float f18 = this->DOG[i].LevelMat.ptr<float>(m + 1)[n - 1];

				float f19 = this->DOG[i + 1].LevelMat.ptr<float>(m - 1)[n - 1];
				float f20 = this->DOG[i + 1].LevelMat.ptr<float>(m - 1)[n];
				float f21 = this->DOG[i + 1].LevelMat.ptr<float>(m - 1)[n + 1];

				float f22 = this->DOG[i + 1].LevelMat.ptr<float>(m)[n - 1];
				float f23 = this->DOG[i + 1].LevelMat.ptr<float>(m)[n];
				float f24 = this->DOG[i + 1].LevelMat.ptr<float>(m)[n + 1];

				float f25 = this->DOG[i + 1].LevelMat.ptr<float>(m + 1)[n - 1];
				float f26 = this->DOG[i + 1].LevelMat.ptr<float>(m + 1)[n];
				float f27 = this->DOG[i + 1].LevelMat.ptr<float>(m + 1)[n - 1];
				if ((
					val > f10 &&
					val > f11 &&
					val > f12 &&
					val > f13 &&
					val > f15 &&
					val > f16 &&
					val > f17 &&
					val > f18 &&

					val > f1 &&
					val > f2 &&
					val > f3 &&
					val > f4 &&
					val > f5 &&
					val > f6 &&
					val > f7 &&
					val > f8 &&
					val > f9 &&

					val > f19 &&
					val > f20 &&
					val > f21 &&
					val > f22 &&
					val > f23 &&
					val > f24 &&
					val > f25 &&
					val > f26 &&
					val > f27	) || (

					val < f1 &&
					val < f2 &&
					val < f3 &&
					val < f4 &&
					val < f5 &&
					val < f6 &&
					val < f7 &&
					val < f8 &&

					val < f9 &&
					val < f10 &&
					val < f11 &&
					val < f12 &&
					val < f13 &&
					val < f15 &&
					val < f16 &&
					val < f17 &&
					val < f18 &&
						  
					val < f19 &&
					val < f20 &&
					val < f21 &&
					val < f22 &&
					val < f23 &&
					val < f24 &&
					val < f25 &&
					val < f26 &&
					val < f27)
					) {
					
					Mat H(3,3,CV_32F);
					Mat D(3,1,CV_32F);
					//��ɭ����
					H.ptr<float>(0)[0] = (f15 + f13 - 2 * f14);//dxx
					
					H.ptr<float>(0)[1] = (f10 + f18 - f12 - f16) / 4;//dxy
					H.ptr<float>(1)[0] = (f10 + f18 - f12 - f16) / 4;//dxy
					
					H.ptr<float>(0)[2] = (f26 + f2 - f8 - f20) / 4;//dxd
					H.ptr<float>(2)[0] = (f26 + f2 - f8 - f20) / 4;//dxd
					
					H.ptr<float>(1)[1] = (f17 + f11 - 2 * f14);//dyy
					
					H.ptr<float>(1)[2] = (f24 + f4 - f6 - f22) / 4;//dyd
					H.ptr<float>(2)[1] = (f24 + f4 - f6 - f22) / 4;//dyd
					
					H.ptr<float>(2)[2] = (f23 + f5 - 2 * f14);//ddd
					//�ݶ�
					D.ptr<float>(0)[0] = (f17 - f11) / 2;
					D.ptr<float>(1)[0] = (f15 - f13) / 2;
					D.ptr<float>(2)[0] = (f23 - f5) / 2;
					Mat xhat(3,1,CV_32F);
					int count = 0;
					int idxI = i;
					int idxM = m;
					int idxN = n;
					while (count < 10) {//����̩��չ����Ͼֲ����ߣ�����⾫ȷ������λ��
						float f1 = this->DOG[idxI - 1].LevelMat.ptr<float>(idxM - 1)[idxN - 1];
						float f2 = this->DOG[idxI - 1].LevelMat.ptr<float>(idxM - 1)[idxN];
						float f3 = this->DOG[idxI - 1].LevelMat.ptr<float>(idxM - 1)[idxN + 1];
											 
						float f4 = this->DOG[idxI - 1].LevelMat.ptr<float>(idxM)[idxN - 1];
						float f5 = this->DOG[idxI - 1].LevelMat.ptr<float>(idxM)[idxN];
						float f6 = this->DOG[idxI - 1].LevelMat.ptr<float>(idxM)[idxN + 1];
											 
						float f7 = this->DOG[idxI - 1].LevelMat.ptr<float>(idxM + 1)[idxN - 1];
						float f8 = this->DOG[idxI - 1].LevelMat.ptr<float>(idxM + 1)[idxN];
						float f9 = this->DOG[idxI - 1].LevelMat.ptr<float>(idxM + 1)[idxN - 1];

						float f10 = this->DOG[idxI].LevelMat.ptr<float>(idxM - 1)[idxN - 1];
						float f11 = this->DOG[idxI].LevelMat.ptr<float>(idxM - 1)[idxN];
						float f12 = this->DOG[idxI].LevelMat.ptr<float>(idxM - 1)[idxN + 1];
											  							
						float f13 = this->DOG[idxI].LevelMat.ptr<float>(idxM)[idxN - 1];
						float f14 = this->DOG[idxI].LevelMat.ptr<float>(idxM)[idxN];
						float f15 = this->DOG[idxI].LevelMat.ptr<float>(idxM)[idxN + 1];
											  							
						float f16 = this->DOG[idxI].LevelMat.ptr<float>(idxM + 1)[idxN - 1];
						float f17 = this->DOG[idxI].LevelMat.ptr<float>(idxM + 1)[idxN];
						float f18 = this->DOG[idxI].LevelMat.ptr<float>(idxM + 1)[idxN - 1];

						float f19 = this->DOG[idxI + 1].LevelMat.ptr<float>(idxM - 1)[idxN - 1];
						float f20 = this->DOG[idxI + 1].LevelMat.ptr<float>(idxM - 1)[idxN];
						float f21 = this->DOG[idxI + 1].LevelMat.ptr<float>(idxM - 1)[idxN + 1];
											 								
						float f22 = this->DOG[idxI + 1].LevelMat.ptr<float>(idxM)[idxN - 1];
						float f23 = this->DOG[idxI + 1].LevelMat.ptr<float>(idxM)[idxN];
						float f24 = this->DOG[idxI + 1].LevelMat.ptr<float>(idxM)[idxN + 1];
											 								
						float f25 = this->DOG[idxI + 1].LevelMat.ptr<float>(idxM + 1)[idxN - 1];
						float f26 = this->DOG[idxI + 1].LevelMat.ptr<float>(idxM + 1)[idxN];
						float f27 = this->DOG[idxI + 1].LevelMat.ptr<float>(idxM + 1)[idxN - 1];
						H.ptr<float>(0)[0] = (f15 + f13 - 2 * f14);//dxx

						H.ptr<float>(0)[1] = (f10 + f18 - f12 - f16) / 4;//dxy
						H.ptr<float>(1)[0] = (f10 + f18 - f12 - f16) / 4;//dxy

						H.ptr<float>(0)[2] = (f26 + f2 - f8 - f20) / 4;//dxd
						H.ptr<float>(2)[0] = (f26 + f2 - f8 - f20) / 4;//dxd

						H.ptr<float>(1)[1] = (f17 + f11 - 2 * f14);//dyy

						H.ptr<float>(1)[2] = (f24 + f4 - f6 - f22) / 4;//dyd
						H.ptr<float>(2)[1] = (f24 + f4 - f6 - f22) / 4;//dyd

						H.ptr<float>(2)[2] = (f23 + f5 - 2 * f14);//ddd
						//�ݶ�
						D.ptr<float>(0)[0] = -(f17 - f11) / 2;
						D.ptr<float>(1)[0] = -(f15 - f13) / 2;
						D.ptr<float>(2)[0] = -(f23 - f5) / 2;
						solve(H, D, xhat);
						if (abs(xhat.ptr<float>(0)[0]) > 0.5 && idxM + round(xhat.ptr<float>(0)[0]) < img.rows -2 && idxM + round(xhat.ptr<float>(0)[0]) > 2)
							idxM += floor(xhat.ptr<float>(0)[0]);
						if (abs(xhat.ptr<float>(1)[0]) > 0.5 && idxN + round(xhat.ptr<float>(1)[0]) < img.cols - 2 && idxN + round(xhat.ptr<float>(1)[0]) > 2)
							idxN+=floor(xhat.ptr<float>(1)[0]);
						//if (abs(xhat.ptr<float>(2)[0]) > 0.5 && idxI + round(xhat.ptr<float>(2)[0]) <= DOG.size() - 2 && idxI + round(xhat.ptr<float>(2)[0]) > 2)
						//	idxI+=floor(xhat.ptr<float>(2)[0]);//�߶����ǻ��������߽�
						count++;
					}
					MyPoint key(idxM, idxN);
					this->KPoint.push_back(key);
				}
			}
		}
	}
}
Mat GetGaussKernel(float sigma) {
	int dim = (int)max(3.0, 2.0 * 3.5 * sigma + 1.0);
	if (dim % 2 == 0)dim++;
	Mat ker(dim, dim, CV_32FC1);
	float s2 = sigma * sigma;
	int c = dim / 2;
	float m = 1.0 / (sqrt(2.0 * CV_PI) * sigma);
	for (int i = 0; i < (dim + 1) / 2; i++) {
		for (int j = 0; j < (dim + 1) / 2; j++) {
			float v = m * exp(-(1.0 * i * i + 1.0 * j * j) / (2.0 * s2));
			ker.ptr<float>(c + i)[c + j] = v;
			ker.ptr<float>(c - i)[c + j] = v;
			ker.ptr<float>(c + i)[c - j] = v;
			ker.ptr<float>(c - i)[c - j] = v;
		}
	}
	return ker;
}
Mat MyHist(Mat& amp,Mat& ori,Point2i p, Point2i Patch,int scale,float weight) {
//amp,ori,Patch���Ͻǵ�����������Patch��С�뷽��߶ȣ�����Patch�ڵķ���ֱ��ͼͳ��
	Mat LocalDes=Mat::zeros(scale,1,CV_32F);
	for (int i = p.x; i < p.x + Patch.x; i++) {
		for (int j = p.y; j < p.y + Patch.y; j++) {
			if (i<0 || j<0 || i>=ori.rows || j>=ori.cols)continue;//������������Ӱ��descriptor����
			//��ֵ����
			float oriTemp= (ori.ptr<float>(i)[j] + PI) * 8 / 2 / PI;
			int idx = floor(oriTemp);//�ѱ�ӳ�䵽[0,scale]
			float ratio1 = oriTemp - idx;
			float ratio2 = idx + 1 - oriTemp;
			if (idx == scale-1) {
				LocalDes.ptr<float>(idx)[0] += weight * amp.ptr<float>(i)[j]*ratio1;
				LocalDes.ptr<float>(0)[0] += weight * amp.ptr<float>(i)[j]*ratio2;
			}
			else if (idx==scale) {
				LocalDes.ptr<float>(0)[0] += weight * amp.ptr<float>(i)[j];
			}
			else {
				LocalDes.ptr<float>(idx)[0] += weight * amp.ptr<float>(i)[j]*ratio1;
				LocalDes.ptr<float>(idx+1)[0] += weight * amp.ptr<float>(i)[j]*ratio2;
			}
		}
	}
	return LocalDes;
}
void correctDir(Mat dir, int prim,int scale) {//����������
	Mat clip(prim, 1, CV_32F);
	for (int i = 0; i < 16; i++) {
		if (scale >= 0) {//������Ϊ��������λ
			dir(Rect(0, i * scale + scale - prim, 1, prim)).copyTo(clip);
			dir(Rect(0, i * scale, 1, scale - prim)).copyTo(dir(Rect(0, i * scale + prim, 1, scale - prim)));
			clip(Rect(0, 0, clip.cols, clip.rows)).copyTo(dir(Rect(0, i * scale, clip.cols, clip.rows)));
		}
		else {//������Ϊ��������λ
			dir(Rect(0, i * scale, 1, prim)).copyTo(clip);
			dir(Rect(0, i * scale + prim, 1, scale - prim)).copyTo(dir(Rect(0, i * scale, 1, scale - prim)));
			clip(Rect(0, 0, 1, prim)).copyTo(dir(Rect(0, i * scale + scale - prim, 1, prim)));
		}
	}
}
void sift::ExtractDescriptor() {//������ȡ�����ӿɲ������ڽ���������ȡ����������ԭͼ����ȡ?
	Mat GradX, GradY;
	Mat BlurImg;
	Mat ker = GetGaussKernel(1.0 * SIGMA);//�Ȳ���1.0sigma����ƽ��
	filter2D(this->SourceImg, BlurImg, CV_32F, ker);
	Mat GradKerX = (Mat_ <float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
	Mat GradKerY = (Mat_ <float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
	filter2D(BlurImg, GradX, CV_32F, GradKerX);
	filter2D(BlurImg, GradY, CV_32F, GradKerY);
	//�����ݶȺͷ���
	for (int i = 0; i < GradX.rows; i++) {
		for (int j = 0; j < GradX.cols; j++) {
			this->amp.ptr<float>(i)[j] = sqrt(pow(GradX.ptr<float>(i)[j], 2) + pow(GradY.ptr<float>(i)[j], 2));
			this->ori.ptr<float>(i)[j] = atan2(GradY.ptr<float>(i)[j], GradX.ptr<float>(i)[j]);	
		}
	}
	//ͳ�Ʒ���ֱ��ͼ
	int scale = 36;//Ԥ��ķ���߶�;
	Point2i Patch(4,4);
	//��������Ϊ4x4xscale��ÿ�����ڵ�ͳ��patch��СΪPatch.x X Patch.y
	for (int i = 0; i < this->KPoint.size(); i++) {
		Point p(this->KPoint[i].x, this->KPoint[i].y);//�Ը�����Ϊ���ĵ�
		for (int m = -2; m < 2; m++) {
			for (int n = -2; n < 2; n++) {
				//pΪÿ��Patch���������Ͻǵĵ�
				Point2i p(this->KPoint[i].x + m * Patch.x, this->KPoint[i].y + n * Patch.y);
				//���ڵ����ĵ�ľ��벻ͬ��Ȩ��Ҳ��ͬ
				//ƴ����Patch��descripor
				this->KPoint[i].descriptor.push_back(MyHist(this->amp, this->ori, p, Patch, scale, exp(-(abs(m) + abs(n)))));
			}
		}
		//������
		//��һ��
		double MaxValue;
		int MaxIdx[2];
		minMaxIdx(this->KPoint[i].descriptor,0,&MaxValue,0,MaxIdx);
		this->KPoint[i].PrimeDir = MaxIdx[0]%scale;//��ȡ������Ϊһ������ֵ
		this->KPoint[i].descriptor.convertTo(this->KPoint[i].descriptor, CV_32F, 1 / MaxValue);
		//��ȡ������ȷ����ֵ���ߵõ�������������
		//Ȼ��4x4xscale���ȵ������У�ÿһ��scal���乲��4x4�������ж�Ҫ����������ѭ����λ���н���
		int prim = this->KPoint[i].PrimeDir;
		if (prim != 0) {
			for (int kk = 0; kk < 16; kk++) {
				Mat clip(prim, 1, CV_32F);
				if (scale >= 0) {//������Ϊ��������λ
					this->KPoint[i].descriptor(Rect(0, kk * scale + scale - prim, 1, prim)).copyTo(clip);
					this->KPoint[i].descriptor(Rect(0, kk * scale, 1, scale - prim)).copyTo(this->KPoint[i].descriptor(Rect(0, kk * scale + prim, 1, scale - prim)));
					clip(Rect(0, 0, clip.cols, clip.rows)).copyTo(this->KPoint[i].descriptor(Rect(0, kk * scale, clip.cols, clip.rows)));
				}
				else {//������Ϊ��������λ
					this->KPoint[i].descriptor(Rect(0, kk * scale, 1, prim)).copyTo(clip);
					this->KPoint[i].descriptor(Rect(0, kk * scale + prim, 1, scale - prim)).copyTo(this->KPoint[i].descriptor(Rect(0, kk * scale, 1, scale - prim)));
					clip(Rect(0, 0, 1, prim)).copyTo(this->KPoint[i].descriptor(Rect(0, kk * scale + scale - prim, 1, prim)));
				}
			}
		}
	}
}
void sift::Display(string s) {
	namedWindow(s, CV_WINDOW_NORMAL);
	for (int i = 0; i < this->KPoint.size(); i++) {
		Point p(this->KPoint[i].y, this->KPoint[i].x);
		circle(this->SourceImg, p, 2, Scalar(0, 255, 0));
	}
	imshow(s, this->SourceImg);
	waitKey(0);
}
void sift::MatchSift(sift cont) {
	int count = 0;
	vector<int*> match;
	for (int i = 0; i < this->KPoint.size(); i++) {
		double minDist=9999;
		double temp;
		int *matchIdx=new int[2];
		matchIdx[0] = i;
		matchIdx[1] = -1;
		for (int j = 0; j < cont.KPoint.size(); j++) {
			temp = norm(this->KPoint[i].descriptor - cont.KPoint[j].descriptor);
			int flag = 0;
			if (temp < minDist) {//�ҵ���ʷ��͵����ж��Ƿ��ѱ�ѡ��
				for (int k = 0; k < match.size(); k++) {
					if (j == match[k][1])flag = 1;
				}
				if (flag == 1)continue;
				minDist = temp;
				matchIdx[1] = j;
			}
		}
		if (minDist < 0.2) {
			count++;
			match.push_back(matchIdx);
		}
	}
	Mat SHOW=Mat::zeros(max(this->SourceImg.rows, cont.SourceImg.rows), this->SourceImg.cols + cont.SourceImg.cols, CV_8UC3);
	this->SourceImg.copyTo(SHOW(Rect(0,0,this->SourceImg.cols,this->SourceImg.rows)));
	cont.SourceImg.copyTo(SHOW(Rect(this->SourceImg.cols, 0, cont.SourceImg.cols, cont.SourceImg.rows)));
	namedWindow("match", CV_WINDOW_NORMAL);
	for (int i = 0; i < match.size(); i++) {
		Point p1(this->KPoint[match[i][0]].y,this->KPoint[match[i][0]].x);
		Point p2(cont.KPoint[match[i][1]].y + this->SourceImg.cols, cont.KPoint[match[i][1]].x );
		Scalar color = Scalar(0, 0, 255);
		line(SHOW,p1,p2,color,1);
		circle(SHOW, p1, 2, Scalar(0, 255, 0));
		circle(SHOW, p2, 2, Scalar(0, 255, 0));
	}
	imshow("match", SHOW);
	cout << "ƥ��ɹ�����Ϊ" << match.size() << endl;
	waitKey(0);
}
sift::sift(Mat img) {
	//����OCT_NUM��octave
	this->SourceImg = img;
	this->amp = Mat::zeros(img.rows,img.cols,CV_32F);
	this->ori = Mat::zeros(img.rows, img.cols, CV_32F);
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	gray.convertTo(gray, 5);
	pyramidOctave mid(0, SIGMA, gray);
	this->octave.push_back(mid);
	for (int i = 1; i < OCT_NUM; i++) {
		pyramidLevel mUp = *((this->octave.end() - 1)->PL.end() - (S + 1));//ȡ�ò�octave�ĵ�����S��level��Ϊ��һ��octave�ĵײ�
		Mat mDown;
		pyrDown(mUp.LevelMat, mDown, Size(mUp.LevelMat.cols / 2, mUp.LevelMat.rows / 2));//������
		pyramidOctave mid(i, mUp.sigma, mDown);
		this->octave.push_back(mid);
	}
	this->DoGStruct();
	this->ExtractEval();
	this->ExtractDescriptor();
	//this->Display("asd");
}
