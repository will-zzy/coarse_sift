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
	int x, y;//��������ԭͼ�ϵ�λ��
	int fx, fy;//�������ڽ�����octave->level��λ��
	int octave, level;//����������octave��level
	MyPoint(int x, int y);
	
	Mat descriptor;//������
	int PrimeDir;//������
};
class pyramidLevel {//������octave��ÿ��level
public:
	int num;//��ǰoctave��level���
	float sigma;//��ǰlevel��k����delta
	Mat LevelMat;//ÿ��levelֻ����һ��ͼ��
	pyramidLevel(int num, float sigma, Mat img);
	pyramidLevel();
};
class pyramidOctave {//������ÿһ��octave
public:
	int num;//octave���
	vector<pyramidLevel> PL;
	pyramidOctave(int num, float sigma, Mat m);//������ʼ�ײ�level����չ
	pyramidOctave();
};
class sift {
public:
	Mat SourceImg;
	Mat ori;//�ݶȷ���
	Mat amp;//�ݶȷ�ֵ
	vector<pyramidOctave> octave;
	vector<pyramidLevel> DOG;//DOG������
	vector<MyPoint> KPoint;
	sift(Mat img);
	void DoGStruct();//�����˹��������DOG������
	void ExtractEval();//��ȡ��ֵ�㲢Ѱ�Ҿ�ȷ��
	void ExtractDescriptor();//��ȡSIFT�����Ӳ�����
	void Display(string c);//��ԭͼ�Լ�������
	void MatchSift(sift cont);//����һ��SIFT�������ƥ��
};