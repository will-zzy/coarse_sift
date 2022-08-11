#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include <vector>
#include "mySift.h"
#include <string.h>
using namespace cv;
using namespace std;
int main(int argc, char** argv) {
	if (argc == 3) {
		if (!strcmp(argv[2], "display")) {
			string a = "./";
			Mat src = imread(argv[1]);
			sift DISP(src);
			cout <<"sift���������Ϊ"<<DISP.KPoint.size() <<endl;
			DISP.Display(argv[1]);
		}
		else {
			Mat left = imread(argv[1]);
			Mat right = imread(argv[2]);
			sift lll(left);
			cout << "��һ��ͼ���������Ϊ" << lll.KPoint.size() << endl;
			sift rrr(right);
			cout << "�ڶ���ͼ���������Ϊ" << lll.KPoint.size() << endl;
			lll.MatchSift(rrr);
		}
	}
	else {
		cout << "�밴�ո�ʽ" << endl;
		cout<< "SIFT.exe image1.jpg image2.jpg" << endl;
		cout<< "���룡" << endl;
	}
	return 0;


}




