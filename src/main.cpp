#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include "linefilter.h"
#include "tonemapper.h"
#include "tools.h"

using namespace cv;
using namespace std;
using namespace tools;

void DrawSketch(const Mat& frame, Mat& sketch, bool show_grad = false, bool show_classes = false)
{

	Mat G;
	Gradient(frame, G);
	int m = G.rows, n = G.cols;
	
	if (show_grad)
	{
		int threshold = 0; // 15
		Mat G_char = Mat(G.rows, G.cols, CV_8U);
		Remap<float>(G, G_char, threshold);
		imshow("Gradient", G_char);
	}

	LineFilter sketcher(std::min(G.cols, G.rows) / 30);
	sketcher.Classify<float>(G);

	if (show_classes) 
	{
		for (int k = 0; k < 8; k++)	
			imshow("C_" + std::to_string(k), sketcher.getC(k));
	}


	sketch = Mat(m, n, CV_8U);

	Mat sketch_float;
	sketcher.ApplyLineShaping(sketch_float);

	Remap<float> (sketch_float, sketch);
}

void combineDrawing(const Mat& sketch, const Mat& texture, Mat& drawing) {
    int m = sketch.rows, n = sketch.cols;

    Mat drawing_float(m, n, CV_32F), sketch_float(m, n, CV_32F), texture_float(m, n, CV_32F);

	// Convert sketch and texture to float for multiplication
	sketch.convertTo(sketch_float, CV_32F);
	texture.convertTo(texture_float, CV_32F);

	multiply(sketch_float, texture_float, drawing_float);
	Remap<float>(drawing_float, drawing, 0);
}

int main() {
	Mat frame;
	//webcam >> frame; // get a new frame from camera
	frame = imread("../../image/photo.jpg");
	int m = frame.rows, n = frame.cols;

	Mat sketch = Mat();
	DrawSketch(frame, sketch);
	imshow("Sketch", sketch);

	// Transform Color to Greyscale Image
	Mat grey_scale(m, n, CV_32F);
	cvtColor(frame, grey_scale, COLOR_BGR2GRAY);

	// Get YUV channels
	Mat colored;
    cvtColor(frame, colored, COLOR_BGR2YUV);
    vector<Mat> YUV;
    split(colored, YUV);
	
	ToneMapper tone_mapper = ToneMapper(42, 29, 29);

	Mat tone_image = Mat(grey_scale.rows, grey_scale.cols, CV_8U);
	tone_mapper.ComputeToneImage<uchar>(grey_scale, tone_image);
	imshow("Tone", tone_image);

	// TODO:
	//Mat& pencil_texture = imread("../../texture/pencil_texture.png");
	//Mat& beta_image;
	// tone_mapper.SolveConjugateGradient(tone_image, pencil_texture, beta_image);
	//Mat& final_texture = tone_mapper.MultipliedTextureMap(pencil_texture, beta_image);

	// Calculate the sum of the sketch + tone: R = S.T (element-wise multiplication)
	Mat drawing(m, n, CV_8U);
	combineDrawing(sketch, tone_image, drawing);

    imshow("Drawing", drawing);

    // Generate colored drawing
    YUV[0] = drawing;
    merge(YUV, colored);
    cvtColor(colored, colored, COLOR_YUV2BGR);
    imshow("Colored", colored);

	waitKey();
	//if (waitKey(30) >= 0) break;

	return 0;
}
