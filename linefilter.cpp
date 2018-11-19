#include "linefilter.h"

using namespace cv;
using namespace std;

LineFilter::LineFilter(int line_length) {

	int r = std::floor(line_length / 2.);
	this->kernel_length = 2*r+1;

	Mat L_horiz = Mat::zeros(kernel_length, kernel_length, CV_32F);
	L_horiz.row(r) = Mat::ones(1, kernel_length, CV_32F);
	Point2f pt(r, r);
	for (int k = 0; k < 8; k++) {
		Mat Rot = getRotationMatrix2D(pt, k * 22.5, 1.0);
		warpAffine(L_horiz, this->Line[k], Rot, Size(kernel_length, kernel_length));
		imshow("Line " + std::to_string(k), Line[k]);
	}
	// The kernel anchors define the center point on the kernel array. In other words, they define the direction "pointed" by the line vector
	// Note that the kernel anchors are set as (x,y) on the image, and NOT as in a matrix (i,j)
	this->kernel_anchor = Point(r, r);
}

inline Mat LineFilter::ApplyLineConvolution(const Mat& Gradient, const Mat& Line, const Point kernel_anchor, const int ddepth = -1) const{
	
	Mat G_i;

	filter2D(Gradient, G_i, ddepth, Line, kernel_anchor, BORDER_ISOLATED);
	// BORDER_ISOLATED means that the kernel gets truncated if it leaves the image bounds

	return G_i;
}

template <class T>
void LineFilter::Classify(const Mat& Gradient) {
	
	Mat G[8];
	for (int k = 0; k < 8; k++)
		G[k] = ApplyLineConvolution(Gradient, this->Line[k], this->kernel_anchor, -1);
	
	int rows = Gradient.rows, cols = Gradient.cols;
	
	for (int k = 0; k < 8; k++)
		this->C[k] = Mat::zeros(rows, cols, Gradient.type());
	
	// Ci(p) =  G(p) if argmax_i{Gi(p)} = i // IS IT ARGMIN OR ARGMAX?
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			
			int max_index = 0;
			for (int k = 0; k < 8; k++) 
				if (G[k].at<T>(i, j) > G[max_index].at<T>(i, j))
					max_index = k;
			
			C[max_index].at<T>(i, j) = Gradient.at<T>(i, j);
		}
	}
}

template void LineFilter::Classify<float>(const Mat& Gradient);
template void LineFilter::Classify<uchar>(const Mat& Gradient);

void LineFilter::ApplyLineShaping(Mat& S) {

	S = Mat::zeros(C[0].rows, C[0].cols, C[0].type());

	for (int k = 1; k < 8; k++)
		add(S, ApplyLineConvolution(this->C[k], this->Line[k], this->kernel_anchor), S);
	
	normalize(S, S, 255.0);

	S = Mat::ones(C[0].rows, C[0].cols, C[0].type()) - S;
}

const Mat& LineFilter::getC(int i) const {
	return this->C[i];
}