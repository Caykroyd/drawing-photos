#include "tools.h"

namespace tools {

	// Normalizes a matrix to the 0-255 range as the uchar format, cutting off values below optional threshold.
	template <class T>
	void Remap(const Mat& input, Mat& output, int threshold) {
		int m = input.rows, n = input.cols;

		double minVal, maxVal;
		minMaxLoc(input, &minVal, &maxVal);
		cout << minVal << " / " << maxVal << endl;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				output.at<uchar>(i, j) = (char)Normalize255(input.at<T>(i, j), minVal, maxVal);

				if (output.at<uchar>(i, j) < threshold)
					output.at<uchar>(i, j) = 0;
			}
		}
	}

	template void Remap<uchar>(const Mat& input, Mat& output, int threshold);
	template void Remap<float>(const Mat& input, Mat& output, int threshold);

	int Normalize255(float value, float minVal, float maxVal)
	{
		return (int)round(255 * (value - minVal) / (maxVal - minVal));
	}

	void Gradient(const Mat& Ic, Mat& G)
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

				if (i > 0 && i < m - 1)
					dy = (float(I.at<uchar>(i + 1, j)) - float(I.at<uchar>(i - 1, j))) / 2;
				else
					dy = 0;

				if (j > 0 && j < n - 1)
					dx = (float(I.at<uchar>(i, j + 1)) - float(I.at<uchar>(i, j - 1))) / 2;
				else
					dx = 0;

				G.at<float>(i, j) = sqrt(dx * dx + dy * dy);
			}
		}
		/*
		// Compute Gradient
		Mat Gx, Gy;
		Scharr(grey_scale, Gx, -1, 1, 0, 3);
		Scharr(grey_scale, Gy, -1, 0, 1, 3);

		// Get Gradient Norm
		Mat G;
		addWeighted(Gx.mul(Gx), 0.5, Gy.mul(Gy), 0.5, 0, G, CV_32F);
		sqrt(G, G); */
	}
}