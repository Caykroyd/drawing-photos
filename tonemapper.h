#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>
#include <random>
#include <iostream>

using namespace cv;
using namespace std;

class ToneMapper {
private:
	const float w_1, w_2, w_3;
	const float sigma_b;
	const float u_a, u_b;
	const float sigma_d, mu_d;

	// functions returning the probability distributions for v in [0,255]
	inline float P1(float v) const;
	inline float P2(float v) const;
	inline float P3(float v) const;
	inline float P(float v) const;

	float model_histogram[256];

	// Generates a discrete function representing P : [0,255] -> [0,255]
	void GenerateModelHistogram();

	// Returns a mapping to the reference histogram from the histogram found by counting pixel intensities on input
	template <class T> int* MatchHistograms(const Mat& input, float reference[256]) const;

public:
	ToneMapper(float = 42, float = 29, float = 29, float = 9, float = 105, float = 225, float = 90, float = 11);

	// Calculates the Tonal Image
	template <class T> void ComputeToneImage(const Mat& input, Mat& tone_image);

	// TODO: Returns the beta parameter matrix by solving the conjugate gradient. ToneImage is type uchar.
	void SolveConjugateGradient(const Mat& ToneImage, const Mat& PencilTexture, Mat& BetaImage) const;

	// Returns the final texture map obtained by exponentiation T(i,j) = PencilTexture(i,j) ^ BetaImage(i,j) 
	void MultipliedTextureMap(const Mat& PencilTexture, const Mat& BetaImage, Mat& T) const;
};

