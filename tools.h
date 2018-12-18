#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

namespace tools
{
	template <class T> void Remap(const Mat& input, Mat& output, int = 0);
	int Normalize255(float value, float minVal, float maxVal);
	void Gradient(const Mat& Ic, Mat& G);
};

