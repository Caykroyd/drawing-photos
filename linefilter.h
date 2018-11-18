#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;


class LineFilter {
private:
	Mat Line[8];
	Point kernel_anchor[8];
	int kernel_length;

	inline Mat& ApplyLineConvolution(const Mat& Gradient, const Mat& Line, const Point kernelAnchor) const;
public:
	LineFilter(int line_length);
	void Classify(const Mat& Gradient);
};