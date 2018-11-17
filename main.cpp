#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	Mat edges;
	namedWindow("edges", 1);
	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera
		GaussianBlur(frame, frame, Size(7, 7), 1.5, 1.5);
		cvtColor(frame, edges, COLOR_BGR2GRAY);
		Mat blurred;
		bilateralFilter(frame, blurred, 9, 300, 300);
		Canny(edges, edges, 20, 40, 3);
		cvtColor(edges, edges, CV_GRAY2BGR);

		imshow("edges", blurred + edges);
		if (waitKey(30) >= 0) break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}
