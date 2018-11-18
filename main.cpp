#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include "linefilter.h"

using namespace cv;
using namespace std;

int main()
{

	VideoCapture webcam(0); // open the default camera
	if (!webcam.isOpened())  // check if we succeeded
		return -1;

	namedWindow("Effects", 1);
		
	for (;;)
	{
		Mat frame;

		webcam >> frame; // get a new frame from camera

		// Transform Color to Greyscale Image
		Mat grey_scale;
		cvtColor(frame, grey_scale, COLOR_BGR2GRAY);

		// Compute Gradient
		Mat Gx, Gy;
		Sobel(grey_scale, Gx, -1, 1, 0, 1);
		Sobel(grey_scale, Gy, -1, 0, 1, 1);

		// Get Gradient Norm
		Mat G;
		addWeighted(Gx.mul(Gx), 0.5, Gy.mul(Gy), 0.5, 0, G, CV_32F);
		sqrt(G, G);

		LineFilter sketcher(std::min(frame.cols, frame.rows) / 30);
		Mat C_0 = sketcher.Classify<float>(G);

		imshow("Effects", C_0);
		if (waitKey(30) >= 0) break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}
