#include "linefilter.h"

using namespace cv;
using namespace std;

LineFilter::LineFilter(int line_length) {

	this->kernel_length = line_length;

	//#include "mat.hpp" ?
	Mat L_horiz = Mat::ones(1, kernel_length, CV_8U);
	Mat L_vert = Mat::ones(kernel_length, 1, CV_8U);
	Mat L_diag = Mat::eye(kernel_length, kernel_length, CV_8U);
	Mat L_diagT = L_diag.t();

	this->Line[0] = L_horiz; // RIGHT
	this->Line[1] = L_diag; // BOTTOM-RIGHT
	this->Line[2] = L_vert; // DOWN
	this->Line[3] = L_diagT; // BOTTOM-LEFT
	this->Line[4] = L_horiz; // LEFT
	this->Line[5] = L_diag; // TOP-LEFT
	this->Line[6] = L_vert; // UP
	this->Line[7] = L_diagT; // TOP-RIGHT

	// The kernel anchors define the center point on the kernel array. In other words, they define the direction "pointed" by the line vector
	this->kernel_anchor[0] = Point(0, 0); // RIGHT
	this->kernel_anchor[1] = Point(0, 0); // BOTTOM-RIGHT
	this->kernel_anchor[2] = Point(0, 0); // DOWN
	this->kernel_anchor[3] = Point(0, kernel_length - 1); // BOTTOM-LEFT
	this->kernel_anchor[4] = Point(0, kernel_length - 1); // LEFT
	this->kernel_anchor[5] = Point(kernel_length - 1, kernel_length - 1); // TOP-LEFT
	this->kernel_anchor[6] = Point(kernel_length - 1, 0); // UP
	this->kernel_anchor[7] = Point(kernel_length - 1, 0); // TOP-RIGHT
}

inline Mat& LineFilter::ApplyLineConvolution(const Mat& Gradient, const Mat& Line, const Point kernel_anchor) const{

	Mat G_i;
	filter2D(Gradient, G_i, -1, Line, kernel_anchor, BORDER_ISOLATED);
	// BORDER_ISOLATED means that the kernel gets truncated if it leaves the image bounds

	return G_i;
}

void LineFilter::Classify(const Mat& Gradient) {

	Mat G[8];
	for (int k = 0; k < 8; k++)
		G[k] = ApplyLineConvolution(Gradient, this->Line[k], this->kernel_anchor[k]);

	int rows = Gradient.rows, cols = Gradient.cols;

	Mat C[8];
	for (int k = 0; k < 8; k++)
		C[k] = Mat(rows, cols, Gradient.type());

	// Ci(p) =  G(p) if argmin_i{Gi(p)} = i // IS IT ARGMIN OR ARGMAX?
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			
			int max_index = 0;
			for (int k = 0; k < 8; k++) {

				if (G[k].at<uchar>(i, j) > G[max_index].at<uchar>(i, j))
					max_index = k;

				G[k].at<uchar>(i, j) = 0;
			}
			G[max_index].at<uchar>(i, j) = Gradient.at<uchar>(i, j);
		}
	}


}