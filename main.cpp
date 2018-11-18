#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include "linefilter.h"

using namespace cv;
using namespace std;

void gradient(const Mat& Ic, Mat& G)
{
	Mat I;
	cvtColor(Ic, I, CV_BGR2GRAY);

	int m = I.rows, n = I.cols;
	G = Mat(m, n, CV_32F);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			// Compute squared gradient (except on borders)
			// ...
			// G2.at<float>(i, j) = ...
			float dx, dy;

			if(i > 0 && i < m - 1)
				dy = (float(I.at<uchar>(i + 1, j)) - float(I.at<uchar>(i - 1, j))) / 2;
			else
				dy = 0;
			
			if(j > 0 && j < n - 1)
				dx = (float(I.at<uchar>(i, j + 1)) - float(I.at<uchar>(i, j - 1))) / 2;
			else
				dx = 0;

			G.at<float>(i, j) = sqrt(dx * dx + dy * dy);
		}
	}
}

int main()
{

	VideoCapture webcam(0); // open the default camera
	if (!webcam.isOpened())  // check if we succeeded
		return -1;

	namedWindow("Effects", 1);
		
	for (;;)
	{
		Mat frame;

		//webcam >> frame; // get a new frame from camera
		frame = imread("../plate.jpg");

		// Transform Color to Greyscale Image
		/*Mat grey_scale;
		cvtColor(frame, grey_scale, COLOR_BGR2GRAY);

		// Compute Gradient
		Mat Gx, Gy;
		Scharr(grey_scale, Gx, -1, 1, 0, 3);
		Scharr(grey_scale, Gy, -1, 0, 1, 3);

		// Get Gradient Norm
		Mat G;
		addWeighted(Gx.mul(Gx), 0.5, Gy.mul(Gy), 0.5, 0, G, CV_32F);
		sqrt(G, G);*/

		Mat G;
		gradient(frame, G);

		int m = G.rows, n = G.cols;
		Mat C(m, n, CV_8U);

		double minVal, maxVal;
		minMaxLoc(G, &minVal, &maxVal);
		
		int threshold = 15;
		for (int i = 0; i < m; i++){
			for (int j = 0; j < n; j++){
				//C.at<uchar>(i, j) = (G.at<float>(i, j) > threshold ? 255 : 0);
				C.at<uchar>(i, j) = char(255 * ((G.at<float>(i,j) - minVal) / (maxVal - minVal)));
			}
		}

		LineFilter sketcher(std::min(frame.cols, frame.rows) / 30);
		sketcher.Classify<float>(G);
		
		/*for(int k = 0; k < 8; k++)
			imshow("C"+std::to_string(k), sketcher.getC(k));*/

		Mat drawing;
		sketcher.ApplyLineShaping(drawing);

		Mat drawingImg(m, n, CV_8U);

		minMaxLoc(drawing, &minVal, &maxVal);
		
		for (int i = 0; i < m; i++){
			for (int j = 0; j < n; j++)
				drawingImg.at<uchar>(i, j) = char(255 * ((drawing.at<float>(i,j) - minVal) / (maxVal - minVal)));
		}

		imshow("Gradient", C);
		imshow("Effects", drawingImg);
		//if (waitKey(30) >= 0) break;
		waitKey();
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}
