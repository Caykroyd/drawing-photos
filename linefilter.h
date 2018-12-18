#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

#include "tools.h"

using namespace cv;
using namespace std;
using namespace tools;


class LineFilter {
private:
	Mat Line[8];
	Point kernel_anchor;
	int kernel_length;

	Mat C[8];

	inline Mat ApplyLineConvolution(const Mat& Gradient, const Mat& Line, const Point kernelAnchor, const int ddepth) const;
public:
	LineFilter(int line_length);

	template <class T> void Classify(const Mat& Gradient);

	void ApplyLineShaping(Mat& S);

	const Mat& getC(int i) const;
};