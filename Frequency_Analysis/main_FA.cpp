/**
	Image Processing and Computer Vision 2017 Micro project, Frequency_Analysis.cpp
	Purpose: Demonstrate the use of the Discrete Fourier Tranformation on 2D images and the correlation between spatial and frequency filters

	@author Joakim Bruslund Haurum
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>


enum filterType { LOWPASS, HIGHPASS, BANDPASS };
char *filterError[] =
{
	"Low Pass filter",
	"High Pass filter",
	"Band Pass filter"
};


using namespace cv;
int windowCounter = 0;

/**
	Discrete Fourier Transform:
	Input domain: Spatial
	Output domain: Frequency

	@param iImg - A real valued input image
	@param oReal - Output matrix for the real values components of the complex output result - Same size as inputImage
	@param oImaginary - Output matrix for the imaginary values components of the complex output result - Same size as inputImage
*/
void DFT(Mat const iImg, Mat& oReal, Mat& oImaginary) {
	double imgWidth = (double)iImg.cols;
	double imgHeight = (double)iImg.rows;
	double PI2 = 2 * M_PI;
	
	for (int l = 0; l < imgHeight; l++) {
		for (int k = 0; k < imgWidth; k++) {
			double realPart = 0.0;
			double imagPart = 0.0;
			double columnComponent = PI2 * (double)k / imgWidth;
			double rowComponent = PI2 * (double)l / imgHeight;
			for (int y = 0; y < imgHeight; y++) {
				for (int x = 0; x < imgWidth; x++) {
					double exponant = columnComponent*x + rowComponent*y;
					realPart += iImg.at<double>(y, x) * cos(exponant);
					imagPart -= iImg.at<double>(y, x) * sin(exponant);
				}
			}
			oReal.at<double>(l, k) = realPart;
			oImaginary.at<double>(l, k) = imagPart;
		}
	}
}

/**
	Inverse Discrete Fourier Transform:
	Input domain: Frequency
	Output domain: Spatial

	@param iReal - Input matrix for the real values components of the complex representation - Same size as outputImg
	@param iImaginary - Input matrix for the imaginary values components of the complex representation - Same size as outputImg
	@param oImg - A real valued output image
*/
void InvDFT(Mat const iReal, Mat const iImaginary, Mat& oImg) {
	oImg = Mat(iReal.rows, iReal.cols, CV_64F);
	double imgWidth = (double)iReal.cols;
	double imgHeight = (double)iReal.rows;
	double PI2 = 2 * M_PI;
	double normalizingFactor = 1 / (imgWidth*imgHeight);

	for (int l = 0; l < imgHeight; l++) {
		for (int k = 0; k < imgWidth; k++) {
			double realPart = 0.0;
			double imagPart = 0.0;
			double columnComponent = PI2 * (double)k / imgWidth;
			double rowComponent = PI2 * (float)l / imgHeight;
			for (int y = 0; y < imgHeight; y++) {
				for (int x = 0; x < imgWidth; x++) {
					double exponant = columnComponent*x + rowComponent*y;
					realPart += iReal.at<double>(y, x) * cos(exponant) - iImaginary.at<double>(y, x) * sin(exponant);
					imagPart += iReal.at<double>(y, x) * sin(exponant) + iImaginary.at<double>(y, x) * cos(exponant);
				}
			}
			oImg.at<double>(l, k) = normalizingFactor*(realPart + imagPart);
		}
	}
}

/**
	Shows the magnitude of the different frequencies in the image in log form
	Input/Output domain: Frequency
	
	@param realMat - Matrix with real components of the complex number
	@param imaginaryMat - Matrix with imaginary components of the complex number
*/
void showMag(Mat const realMat, Mat const imaginaryMat) {
	Mat magI;
	
	magnitude(realMat, imaginaryMat, magI);
	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);
	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a viewable image form (float between values 0 and 1).

	std::string windowName = "DFT - Magnitude" + std::to_string(windowCounter);
	imshow(windowName, magI);		// Show our image inside it.
	windowCounter++;

}

/**
	Shows the phase of the different frequencies in the image
	Input/Output domain: Frequency

	@param realMat - Matrix with real components of the complex number
	@param imaginaryMat - Matrix with imaginary components of the complex number
*/
void showPhase(Mat const realMat, Mat const imaginaryMat) {
	Mat phase = Mat::zeros(realMat.rows, realMat.cols, CV_64F);

	for (int y = 0; y < phase.rows; y++)
		for (int x = 0; x < phase.cols; x++)
			phase.at<double>(y, x) = atan2(imaginaryMat.at<double>(y,x), realMat.at<double>(y,x));

	std::string windowName = "DFT - Phase" + std::to_string(windowCounter);
	imshow(windowName, phase);		// Show our image inside it.
	windowCounter++;
}

/**
	Sort indecies so that the DC frequency is in the middle of the image
	The function is its own inverse - i.e. call it again to reverse the sorting
	Input/Output domain: Frequency

	@param matrix - The matrix which indecies has to be rearranged
*/
void sortIndecies(Mat& matrix) {
	int cx = matrix.cols / 2;
	int cy = matrix.rows / 2;

	Mat q0(matrix, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(matrix, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(matrix, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(matrix, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

/**
	Ideal filter - attenuates every frequency under/over the cut off frequency - results in ringing effects
	Assumes the image is square, if not the cut off frequency is calculated based on the longest side of the image
	Input/Output domain: Frequency

	@param realMat - Matrix with real components of the complex number
	@param imaginaryMat - Matrix with imaginary components of the complex number
	@param filter - Designates the filter type - Should be assigned as either LOWPASS, HIGHPASS or BANDPASS
	@param cutOffDec(Low/High) - Cut off frequency as a value between 0 and 1
*/
void idealFilter(Mat& realMat, Mat& imaginaryMat, filterType const filter, double const cutOffDec) {
	double multiplier = realMat.cols > realMat.rows ? realMat.cols / 2.0 : realMat.rows / 2.0;
	double cutOffFreq = cutOffDec * multiplier;

	if (filter == BANDPASS) {
		std::cerr << "FILTER TYPE IS A BAND PASS BUT ONLY 1 CUT OFF FREQUENCY HAS BEEN SUPPLIED" << std::endl;
		return;
	}

	for (int y = 0; y < realMat.rows; y++) {
		for (int x = 0; x < realMat.cols; x++) {
			double freq = sqrt(pow((x - multiplier), 2) + pow((y - multiplier), 2));
			bool cut = false;

			if (filter == LOWPASS)
				if (freq > cutOffFreq) cut = true;
			if (filter == HIGHPASS)
				if (freq < cutOffFreq) cut = true;

			if (cut) {
				realMat.at<double>(y, x) = 0.0f;
				imaginaryMat.at<double>(y, x) = 0.0f;
			}
		}
	}
}
void idealFilter(Mat& realMat, Mat& imaginaryMat, filterType const filter, double const cutOffDecLow, double const cutOffDecHigh) {
	if (filter != BANDPASS) {
		std::cerr << "FILTER TYPE IS NOT BAND PASS BUT: " << filterError[filter] << std::endl;
		return;
	}
	if (cutOffDecLow >= cutOffDecHigh) {
		std::cerr << "CUT OFF FREQUENCIES ARE SET INCORRECTLY:" << std::endl << "Low cut off: " << cutOffDecLow << std::endl << "High cut off: " << cutOffDecHigh << std::endl;
		return;
	}

	double multiplier = realMat.cols > realMat.rows ? realMat.cols / 2.0 : realMat.rows / 2.0;
	double cutOffFreqLow = cutOffDecLow * multiplier;
	double cutOffFreqHigh = cutOffDecHigh * multiplier;

	for (int y = 0; y < realMat.rows; y++) {
		for (int x = 0; x < realMat.cols; x++) {
			double freq = sqrt(pow((x - multiplier), 2) + pow((y - multiplier), 2));

			if((freq < cutOffFreqLow) || (freq > cutOffFreqHigh)){
				realMat.at<double>(y, x) = 0.0f;
				imaginaryMat.at<double>(y, x) = 0.0f;
			}
		}
	}
}

/**
	Butterworth filter - non ideal filter with fewer ringing effects - Gaussian approximation
	Assumes the image is square, if not the cut off frequency is calculated based on the longest side of the image
	Input/Output domain: Frequency

	@param realMat - Matrix with real components of the complex number
	@param imaginaryMat - Matrix with imaginary components of the complex number
	@param order - Order of the filter
	@param filter - Designates the filter type - Should be assigned as either LOWPASS, HIGHPASS or BANDPASS
	@param cutOffDec(Low/High) - Cut off frequency as a value between 0 and 1
*/
void butterworthFilter(Mat& realMat, Mat& imaginaryMat, int const order, filterType const filter, double const cutOffDec) {
	if (filter == BANDPASS) {
		std::cerr << "FILTER TYPE IS A BAND PASS BUT ONLY 1 CUT OFF FREQUENCY HAS BEEN SUPPLIED" << std::endl;
		return;
	}

	double multiplier = realMat.cols > realMat.rows ? realMat.cols / 2.0 : realMat.rows / 2.0;
	double cutOffFreq = cutOffDec * multiplier;

	for (int y = 0; y < realMat.rows; y++) {
		for (int x = 0; x < realMat.cols; x++) {
			double freq = sqrt(pow((x - multiplier), 2) + pow((y - multiplier), 2));
			double attenuation = 1 / (1 + pow((freq / cutOffFreq), 2*order));
			
			if (filter == HIGHPASS)
				attenuation = 1 - attenuation;

			realMat.at<double>(y, x) *= attenuation;
			imaginaryMat.at<double>(y, x) *= attenuation;
		}
	}
}
void butterworthFilter(Mat& realMat, Mat& imaginaryMat, int const order, filterType const filter, float const cutOffDecLow, float const cutOffDecHigh) {
	if (filter != BANDPASS) {
		std::cerr << "FILTER TYPE IS NOT BAND PASS BUT: " << filterError[filter] << std::endl;
		return;
	}
	if (cutOffDecLow >= cutOffDecHigh) {
		std::cerr << "CUT OFF FREQUENCIES ARE SET INCORRECTLY:" << std::endl << "Low cut off: " << cutOffDecLow << std::endl << "High cut off: " << cutOffDecHigh << std::endl;
		return;
	}

	double multiplier = realMat.cols > realMat.rows ? realMat.cols / 2.0 : realMat.rows / 2.0;
	double cutOffFreqLow = cutOffDecLow * multiplier;
	double cutOffFreqHigh = cutOffDecHigh * multiplier;

	for (int y = 0; y < realMat.rows; y++) {
		for (int x = 0; x < realMat.cols; x++) {
			double freq = sqrt(pow((x - multiplier), 2) + pow((y - multiplier), 2));
			double attenuationLow = 1 / (1 + pow((freq / cutOffFreqHigh), 2 * order));	//High cut off frequency used here as everything would otherwise be attenuated
			double attenuationHigh = 1 / (1 + pow((freq / cutOffFreqLow), 2 * order));	//Low cut off frequency used here as the filter will be inversed 
			double attenuation = attenuationLow * (1 - attenuationHigh);

			realMat.at<double>(y, x) *= attenuation;
			imaginaryMat.at<double>(y, x) *= attenuation;
		}
	}
}

/**
	Filter mask - Filters the input based on a supplied filter mask
	Input/Output domain: Frequency

	@param realMat - Matrix with real components of the complex number
	@param imaginaryMat - Matrix with imaginary components of the complex number
	@param const - Matrix with the filter mask that is being applied - 	The mask should have same dimensions as original image - The mask is converted to 64-bit matrix and normalized to 0 and 1
*/
void filterMask(Mat& realMat, Mat& imaginaryMat, Mat const mask) {
	if ((mask.rows != realMat.rows) || (mask.cols != realMat.cols)) {
		std::cerr << "FILTER MASK DIMENSION IS NOT IDENTICAL TO INPUT IMAGE" << std::endl;
		return;
	}

	for (int y = 0; y < realMat.rows; y++) {
		for (int x = 0; x < realMat.cols; x++) {
			realMat.at<double>(y, x) *= mask.at<double>(y, x);
			imaginaryMat.at<double>(y, x) *= mask.at<double>(y, x);
		}
	}
}

/**
	Applies a Gaussian Blur filter on the input image.
	Currently counteracts the border problem by padding with 0s i.e. ignoring kernel elements outside the image borders
	Input/Output domain: Spatial

	@param img - A 1-channel 64-bit matrix representing the grayscale image
	@param radius - An integer determining the radius of the Gaussian Filter. e.g. radius 2 = 5x5 gaussian filter
	@param sigma - The standard deviation applied in the gaussian filter. Also known as "weight"
*/
void gaussianBlur(Mat& img, int const radius = 2, double const sigma = 1.0) {
	int size = radius * 2 + 1;		//Width & Height
	Mat blurredImg = Mat::zeros(img.rows, img.cols , CV_64F);
	Mat kernel(size, size, CV_64F);

	double denom = 2.0 * sigma*sigma;	
	double a = 1 / (denom*M_PI);

	/* Calculate values for the Gaussian Blur Kernel*/
	double sum = 0.0;
	for (int hy = 0; hy < size; hy++) {
		int y = (hy - radius);
		int ysq = y*y;
		for (int hx = 0; hx < size; hx++) {
			int x = (hx - radius);
			double b = -(x*x + ysq) / denom;

			kernel.at<double>(hy, hx) = a*exp(b);
			sum += kernel.at<double>(hy, hx);
		}
	}

	/* Normalize the values in the Gaussian Blur Kernel*/
	for (int hy = 0; hy < size; hy++) {
		for (int hx = 0; hx < size; hx++) {
			kernel.at<double>(hy, hx) /= sum;
		}
	}

	/* Convolve the Gaussian blur kernel with the inout image "img" */
	for (int y = 0; y < img.rows ; y++) {
		for (int x = 0; x < img.cols; x++) {
			double tmp = 0.0;
			for (int hy = -radius; hy <= radius; hy++) {
				for (int hx = -radius; hx <= radius; hx++) {
					if(y+hy >= 0 && y+hy < img.rows && x+hx >= 0 && x+hx < img.cols)
						tmp += kernel.at<double>(hy + radius, hx+ radius) * img.at<double>(y + hy, x + hx);
				}
			}
			blurredImg.at<double>(y, x) = tmp;
		}
	}
	img = blurredImg;
}

/**
	Applies a Mean Blur filter on the input image.
	Currently counteracts the border problem by padding with 0s i.e. ignoring kernel elements outside the image borders
	Input/Output domain: Spatial

	@param img - A 1-channel 64-bit matrix representing the grayscale image
	@param radius - An integer determining the radius of the Mean Filter. e.g. radius 2 = 5x5 mean filter
*/
void meanBlur(Mat& img, int const radius = 1) {
	int size = radius * 2 + 1;
	double weight = 1.0 / (size*size);
	Mat blurredImg = Mat::zeros(img.rows, img.cols, CV_64F);

	std::cout << weight << " " << weight*(size*size)<<  std::endl;

	for (int y = 0; y < img.rows ; y++) {
		for (int x = 0; x < img.cols ; x++) {
			double tmp = 0.0;
			for (int hy = -radius; hy <= radius; hy++) {
				for (int hx = -radius; hx <= radius; hx++) {
					if(y+hy >= 0 && y+hy < img.rows && x+hx >= 0 && x+hx < img.cols)
						tmp += img.at<double>(y + hy, x + hx);
				}
			}
			blurredImg.at<double>(y, x) = weight * tmp;
		}
	}
	img = blurredImg;
}

/**
	Applies horizontal and vertical 3x3 Sobel edge operators onto the input Matrix via the horizontal and vertical Sobel operators.
	Returns a 1-channel 64-bit Matrix containing the gradient magnitude
	Input/Output domain: Spatial

	@param img - A 1-channel 64-bit matrix representing the grayscale image
*/
void sobelOperator(Mat& img) {
	Mat edgeImg(img.rows, img.cols, CV_64F);

	Mat sobelX = (Mat_<double>(3, 3) << -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0);
	Mat sobelY = (Mat_<double>(3, 3) << -1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0);
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			double tmpx = 0.0;
			double tmpy = 0.0;
			for (int hy = -1; hy <= 1; hy++) {
				for (int hx = -1; hx <= 1; hx++) {
					if (y + hy >= 0 && y + hy < img.rows && x + hx >= 0 && x + hx < img.cols) {
						tmpx += sobelX.at<double>(hy + 1, hx + 1) * img.at<double>(y + hy, x + hx);
						tmpy += sobelY.at<double>(hy + 1, hx + 1) * img.at<double>(y + hy, x + hx);
					}
				}
			}
			double magnitude_Gradient = hypot(tmpx, tmpy);
			edgeImg.at<double>(y, x) = magnitude_Gradient;
		}
	}
	img = edgeImg;
}


int main(){
	/*******************************/
	/*       INITIALIZATION        */
	/*******************************/

	std::string filepath = "Lenna.png";
	Mat realPart, imagPart, image, reconstructedImage;
	bool ownFunctions = true;

	image = imread(filepath, CV_LOAD_IMAGE_GRAYSCALE);
	if (image.empty())
		return -1;

	//image.convertTo(image, CV_64F);
	//image.copyTo(reconstructedImage);
	//gaussianBlur(reconstructedImage, 2, 3);
	//meanBlur(reconstructedImage, 2);
	//sobelOperator(reconstructedImage);

	/*******************************/
	/* SPATIAL -> FREQUENCY DOMAIN */
	/*******************************/

	if (ownFunctions) {	//	Use own DFT functions
		image.convertTo(image, CV_64F, 1/255.0);
		resize(image, image, Size(), 0.125, 0.125, CV_INTER_AREA);

		realPart = Mat::zeros(image.rows, image.cols, CV_64F);
		imagPart = Mat::zeros(image.rows, image.cols, CV_64F);

		DFT(image, realPart, imagPart);
	}
	else {	// Use built-in OpenCV DFT functions
		Mat padded;								//Expand input image to optimal size
		int m = getOptimalDFTSize(image.rows);
		int n = getOptimalDFTSize(image.cols);
		copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));	   // on the border add zero values

		Mat planesDFT[] = { Mat_<double>(padded), Mat::zeros(padded.size(), CV_64F) };
		Mat complexI;
		merge(planesDFT, 2, complexI);			// Add to the expanded another plane with zeros

		dft(complexI, complexI);				// Perform the DFT

		split(complexI, planesDFT);				// planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
		realPart = planesDFT[0];
		imagPart = planesDFT[1];
	}


	/*******************************/
	/*     FREQUENCY FILTERING     */
	/*******************************/

	sortIndecies(realPart);
	sortIndecies(imagPart);

	showMag(realPart, imagPart);
	showPhase(realPart, imagPart);

	//idealFilter(realPart, imagPart, HIGHPASS, 0.2);
	//idealFilter(realPart, imagPart, BANDPASS, 0.3, 0.8);

	butterworthFilter(realPart, imagPart, 1, HIGHPASS, 0.2);
	//butterworthFilter(realPart, imagPart, 1, BANDPASS, 0.3, 0.8);

	/*Mat mask = imread("mask.png", CV_LOAD_IMAGE_GRAYSCALE);
	mask.convertTo(mask, CV_64F, 1 / 255.0);
	filterMask(realPart, imagPart, mask);*/

	showMag(realPart, imagPart);
	showPhase(realPart, imagPart);
	
	sortIndecies(realPart);
	sortIndecies(imagPart);


	/*******************************/
	/* FREQUENCY -> SPATIAL DOMAIN */
	/*******************************/

	if (ownFunctions) {	//	Use own DFT functions
		InvDFT(realPart, imagPart, reconstructedImage);
	}
	else {	// Use built-in OpenCV DFT functions
		Mat planesIDFT[] = { Mat_<double>(realPart), Mat_<double>(imagPart) };
		Mat complexIDFT;
		merge(planesIDFT, 2, complexIDFT);
		idft(complexIDFT, reconstructedImage, DFT_SCALE | DFT_REAL_OUTPUT);
	}

	//reconstructedImage = abs(reconstructedImage);	//USE IF BINARY GRADIENT ILLUSTRATION
	normalize(reconstructedImage, reconstructedImage, 0, 255, CV_MINMAX);
	reconstructedImage.convertTo(reconstructedImage, CV_8U);
	//threshold(reconstructedImage, reconstructedImage, 10, 255, CV_THRESH_BINARY);	//USE IF BINARY GRADIENT ILLUSTRATION

	imshow("Grayscale input image", image);					// Show the input grayscale image
	imshow("DFT - Reconstructed", reconstructedImage);		// Show the reconstruted input image after DFT and IDFT

	waitKey();										        // Wait for a keystroke in the window
	return 0;
}