#ifndef _MYSTRUCT_H_
#define _MYSTRUCT_H_
#endif
#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include <vector>
using namespace cv;
using namespace std;
class MyPoint {
public:
	int x, y;//特征点在原图上的位置
	int fx, fy;//特征点在金字塔octave->level的位置
	int octave, level;//特征点所在octave与level
	MyPoint(int x, int y);
	
	Mat descriptor;//描述子
	int PrimeDir;//主方向
};
class pyramidLevel {//金字塔octave的每个level
public:
	int num;//当前octave的level编号
	float sigma;//当前level的k次幂delta
	Mat LevelMat;//每个level只包含一张图像
	pyramidLevel(int num, float sigma, Mat img);
	pyramidLevel();
};
class pyramidOctave {//金字塔每一个octave
public:
	int num;//octave编号
	vector<pyramidLevel> PL;
	pyramidOctave(int num, float sigma, Mat m);//建立初始底层level并拓展
	pyramidOctave();
};
class sift {
public:
	Mat SourceImg;
	Mat ori;//梯度方向
	Mat amp;//梯度幅值
	vector<pyramidOctave> octave;
	vector<pyramidLevel> DOG;//DOG金字塔
	vector<MyPoint> KPoint;
	sift(Mat img);
	void DoGStruct();//构造高斯金字塔、DOG金字塔
	void ExtractEval();//提取极值点并寻找精确解
	void ExtractDescriptor();//提取SIFT描述子并矫正
	void Display(string c);//画原图以及特征点
	void MatchSift(sift cont);//对另一个SIFT对象进行匹配
};