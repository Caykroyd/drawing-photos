#include "tonemapper.h"

const double PI = 3.14159265358979323846;

// NOTE: The paper is not consistent as first it defines 'v' as being normalized between 0 and 1,
// and then it confuses formulas where 'v' should be mapped between 0-255 (P2, P3) and 0-1 (P1)
// so we have decided to use the 0-255 range.

// To Gabriel: Check if you agree with my P1, P2, P3 function definitions

// This Laplace function is not perfectly normalized, since its values are only between 0-255
inline float ToneMapper::P1(float v) const {
	return v <= 255 ? 1 / sigma_b * expf(-(255 - v) / sigma_b) : 0;
}

inline float ToneMapper::P2(float v) const {
	return u_a < v && v < u_b ? 1 / (u_b - u_a) : 0;
}

// P3: The paper appears to have incorrectly normalized by sqrt(sigma_d) instead of sigma_d
inline float ToneMapper::P3(float v) const {
	return 1 / sqrt(2 * PI) / sigma_d * exp(-pow(v - mu_d, 2) / 2 / pow(sigma_d, 2));
}

float ToneMapper::P(float v) const {
	return 1 / (w_1 + w_2 + w_3) * (w_1 * P1(v) + w_2 * P2(v) + w_3 * P3(v));
}

void ToneMapper::GenerateModelHistogram()
{
	for (int v = 0; v < 256; v++)
		this->model_histogram[v] = P(v);
}

inline int Normalize255(float value, float minVal, float maxVal)
{
	return (int) round(255 * (value - minVal) / (maxVal - minVal));
}

template<class T>
int* ToneMapper::MatchHistograms(const Mat& input, float reference_bin[256]) const
{
	float input_bin[256];
	for (int k = 0; k < 256; k++)
		input_bin[k] = 0;

	double minVal, maxVal;
	minMaxLoc(input, &minVal, &maxVal);
	int m = input.rows, n = input.cols;
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			input_bin[Normalize255(input.at<uchar>(i, j), minVal, maxVal)] += 1.0f / (m*n);
		}

	// Assure our bins are correctly summing to unity.
	float A = 0, B = 0;
	for (int k = 0; k < 256; k++) {
		A += input_bin[k];
		B += reference_bin[k];
	}
	cout << "Input bin =" << A << ", Reference bin =" << B << endl;

	int* mapping = new int[256];
	float S_in = 0, S_ref = 0;
	for (int k = 0, q = 0; k < 256; k++)
	{
		while (S_in > S_ref && q < 256) {
			S_ref += reference_bin[q];
			q++;
		}

		mapping[k] = q;
		//cout << k << ":" << q << endl;

		S_in += input_bin[k];
	}

	return mapping;
}
template int* ToneMapper::MatchHistograms<uchar>(const Mat& input, float reference_bin[256]) const;
template int* ToneMapper::MatchHistograms<float>(const Mat& input, float reference_bin[256]) const;

// In the paper, w1, w2, w3 appears to be inverted. I have inverted the values from the table.
ToneMapper::ToneMapper(float w_b, float w_m, float w_d, float sigma_b, float u_a, float u_b, float mu_d, float sigma_d) :
						w_1(w_b), w_2(w_m), w_3(w_d), sigma_b(sigma_b), u_a(u_a), u_b(u_b), mu_d(mu_d), sigma_d(sigma_d)
{
	GenerateModelHistogram();
}

template <class T>
void ToneMapper::ComputeToneImage(const Mat& input, Mat& tone_image) {
	
	int m = input.rows, n = input.cols;

	int* histogram = MatchHistograms<T>(input, model_histogram);
	
	//for (int k = 0; k < 256; k++)
	//	cout << histogram[k] << endl;

	// remember to scale values to [0,255] (min-max) to cancel illumination differences
	double minVal, maxVal;
	minMaxLoc(input, &minVal, &maxVal);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			int bin_index = Normalize255(input.at<T>(i, j), minVal, maxVal);
			tone_image.at<uchar>(i, j) = histogram[bin_index];
		}
	}
}

template void ToneMapper::ComputeToneImage<float>(const Mat& input, Mat& tone_image);
template void ToneMapper::ComputeToneImage<uchar>(const Mat& input, Mat& tone_image);

void ToneMapper::MultipliedTextureMap(const Mat& PencilTexture, const Mat& BetaImage, Mat& T) const
{
	int m = PencilTexture.rows, n = PencilTexture.cols;
	   
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			T.at<float>(i, j) = pow(PencilTexture.at<float>(i, j), BetaImage.at<float>(i, j));
			
}
