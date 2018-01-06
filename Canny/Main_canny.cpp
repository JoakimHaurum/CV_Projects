#define _USE_MATH_DEFINES

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
//#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <math.h>
using namespace cv;
using namespace std;


/*
!!! NOT YET IMPLEMENTED !!!

8-Connectivity Grassfire algorithm
Checks whether the pixels are connected and whether the blobs are connected to a strong edge
if yes, the pixel is set to white
if no, the pixel is set to black

!!! NOT YET IMPLEMENTED !!!
*/
/*
Mat Grassfire(Mat img){
	int dx[4] = { -1, -1, 0, 1 };
	int dy[4] = { 0, -1, -1, -1 };

	Mat labeled = Mat::zeros(img.rows, img.cols, CV_8UC1);
	int curLabel = 1; //Starting value

	for (int row = 0; row < img.rows; row++){
		for (int col = 0; col < img.cols; col++){
			
			int NeighbourLabel[4] = { 0 }; //0 = West, 1 = North West, 2 = North, 3 = North East
			int NeighbourInts[4] = { 0 };
			int counter = 0;
			
			if (img.at<uchar>(row, col) != 0 && labeled.at<uchar>(row, col) == 0){
				for (int i = 0; i < 4; i++){
					if (col + dx[i] >= 0 && row + dy[i] >= 0){
						NeighbourInts[i] = img.at<uchar>(row + dy[i], col + dx[1]);
						NeighbourLabel[i] = labeled.at<uchar>(row + dy[i], col + dx[i]);
						counter++;
					}
					else{
						NeighbourLabel[i] = 0;
						NeighbourInts[i] = 0;
					}
				}
			}

			//Compare Neighbours 
			if (counter == 0){
				labeled.at<uchar>(row, col) = curLabel;
				curLabel++;
			}
			if (counter == 1){
				if (NeighbourInts[0] == 0 && NeighbourInts[1] == 0 && NeighbourInts[2] == 0 && NeighbourInts[3] != 0)
					labeled.at<uchar>(row, col) = NeighbourLabel[3];

				else if (NeighbourInts[0] == 0 && NeighbourInts[1] == 0 && NeighbourInts[2] != 0 && NeighbourInts[3] == 0)
					labeled.at<uchar>(row, col) = NeighbourLabel[2];

				else if (NeighbourInts[0] == 0 && NeighbourInts[1] != 0 && NeighbourInts[2] == 0 && NeighbourInts[3] == 0)
					labeled.at<uchar>(row, col) = NeighbourLabel[1];

				else if (NeighbourInts[0] != 0 && NeighbourInts[1] == 0 && NeighbourInts[2] == 0 && NeighbourInts[3] == 0)
					labeled.at<uchar>(row, col) = NeighbourLabel[0];
			}





			

		}
	}


}
*/


/*
	Converts the 3-channels input matrix from colour to grayscale, and returns a 1-channel 8-bit matrix
*/
Mat ColorToGray(Mat img){
	Mat grayscale(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < grayscale.rows; y++){
		for (int x = 0; x < grayscale.cols; x++){
			int tmp = img.at<Vec3b>(y, x)[0] +img.at<Vec3b>(y, x)[1] +img.at<Vec3b>(y, x)[2];
			grayscale.at<uchar>(y, x) = tmp/3;
		}
	}
	return grayscale;
}

/*
	Applies a Gaussian Blur filter on the input filter.
	Takes the following inputs:
		Mat img : A 1-channel 8-bit matrix representing the grayscale image
		int radius : An integer determining the radius of the Gaussian Filter. e.g. radius 2 = 5x5 gaussian filter
		double sigma : The standard deviation applied in the gaussian filter. Also known as "weight"
	Currently does not take care of border problem and there for returns a Matrix containing a smaller image than the input image.
*/
Mat GaussianBlur(Mat img, int radius, double sigma){ //No taking care of border problem
	int size = radius * 2 + 1;
	int k = radius - 1;
	Mat blurredImg(img.rows - 4, img.cols - 4, CV_8UC1);
	double * kernelGaus;
	kernelGaus = new double[size*size];

	double denom = 2.0 * sigma*sigma;
	double a = 1 / (denom*M_PI);

	/* Calculate values for the Gaussian Blur Kernel*/
	double sum = 0.0;
	for (int hy = 0; hy < size; hy++){
		int y = (hy - k - 1);
		int ysq = y*y;
		for (int hx = 0; hx < size; hx++){
			int x = (hx - k - 1);
			double b = -(x*x + ysq) / denom;

			kernelGaus[hy*size + hx] = a*exp(b);
			sum += kernelGaus[hy*size + hx];
		}
	}

	/* Normalize the values in the Gaussian Blur Kernel*/
	for (int hy = 0; hy < size; hy++){
		for (int hx = 0; hx < size; hx++){
			kernelGaus[hy*size + hx] = (1 / sum)*kernelGaus[hy*size + hx];
		}
	}

	/* Convolve the Gaussian blur kernel with the inout image "img" */
	for (int y = radius; y < img.rows - radius; y++){
		for (int x = radius; x < img.cols - radius; x++){
			double tmp = 0.0;
			for (int hy = -radius; hy <= radius; hy++){
				for (int hx = -radius; hx <= radius; hx++){
					tmp += kernelGaus[(hy + radius)*size + (hx + radius)] * (double)img.at<uchar>(y + hy, x + hx);
				}
			}
			blurredImg.at<uchar>(y - radius, x - radius) = (int)tmp;
		}
	}

	delete[] kernelGaus;
	return blurredImg;

	
	
	
	/*
	Mat blurredImg(img.rows-4, img.cols-4, CV_8UC1);
	float kernelGaus[5][5] = {
		{ 2, 4, 5, 4, 2 },
		{ 4, 9, 12, 9, 4 },
		{ 5, 12, 15, 12, 5 },
		{ 4, 9, 12, 9, 4 },
		{ 2, 4, 5, 4, 2 }
	};

	for (int hy = 0; hy <= 4; hy++){
		for (int hx = 0; hx <= 4; hx++){
			kernelGaus[hy][hx] = kernelGaus[hy][hx] * 1 / 159;
		}
	}

	for (int y = 2; y < img.rows - 2; y++){
		for (int x = 2; x < img.cols - 2; x++){
			float tmp = 0.0f;
				for (int hy = -2; hy <= 2; hy++){
					for (int hx = -2; hx <= 2; hx++){
						tmp += kernelGaus[hy + 2][hx + 2] * (float) img.at<uchar>(y+hy,x+hx);
					}
				}
			blurredImg.at<uchar>(y-2, x-2) = (int) tmp;
		}
	}
	return blurredImg;
*/
}

/*
	Applies edge detection onto the input Matrix via the horizontal and vertical Sobel operators.
	Returns a 2-channel 8-bit Matrix slightly smalle than the input matrix, containing the gradient magnitude in channel 0 and gradient direction in channel 1
*/
Mat Sobel(Mat img){
	Mat edgeImg(img.rows-2, img.cols-2, CV_8UC2);

	int sobelX[3][3]{
		{ -1, 0, 1 },
		{ -2, 0, 2 },
		{ -1, 0, 1 }
	};
	int sobelY[3][3]{
		{ -1, -2, -1 },
		{ 0, 0, 0 },
		{ 1, 2, 1 }
	};


	int count0 = 0;
	int count45 = 0;
	int count90 = 0;
	int count135 = 0;
	int countNon = 0;
	for (int y = 1; y < img.rows - 1; y++){
		for (int x = 1; x < img.cols - 1; x++){
			int tmpx = 0;
			int tmpy = 0;
			int tmpx2 = 0;
			int tmpy2 = 0;
			for (int hy = -1; hy <= 1; hy++){
				for (int hx = -1; hx <= 1; hx++){
					tmpx += sobelX[hy + 1][hx + 1] * img.at<uchar>(y + hy, x + hx); //correlation
					tmpy += sobelY[hy + 1][hx + 1] * img.at<uchar>(y + hy, x + hx); //correlation
					tmpx2 += sobelX[hy + 1][hx + 1] * img.at<uchar>(y - hy, x - hx); //convolution, not correlation!!
					tmpy2 += sobelY[hy + 1][hx + 1] * img.at<uchar>(y - hy, x - hx); //convolution, not correlation!!
				}
			}
			//cout << "tmpx: " << tmpx << " tmpy: " << tmpy << "    tmpx2:" << tmpx2 << " tmpy2:" << tmpy2 << endl;
			//tmpx = tmpx/ 8; //Normalizing the sobel operator
			//tmpy = tmpy /8; //Normalizing the sobel operator
			int magnitude_Gradient = (int) hypot(tmpx, tmpy);
			if (magnitude_Gradient > 255) magnitude_Gradient = 255;
			else if (magnitude_Gradient < 0) magnitude_Gradient = 0;

			double direction_Gradient = 0.0;
			if (magnitude_Gradient == 0){ //No edge
				direction_Gradient = 255;
				countNon++;
			}
			else{
				direction_Gradient = atan2((double)tmpy2, (double)tmpx)*180/M_PI; //Calculates the direction of the gradient. Is always perpendisular to the edge itself. One of the sums has to be negated to get the correct result
				//cout << "The arc tangent for (" << tmpx << ", " << tmpy << ") is "<< direction_Gradient << endl;
				if (direction_Gradient < 0.0) direction_Gradient += 180.0;
				//cout << "The arc tangent for (" << tmpx << ", " << tmpy << ") is " << direction_Gradient << endl;

				//Sets direction to either 0, 45, 90 or 135 degrees
				if ((direction_Gradient >= 0.0 && direction_Gradient < 22.5) || (direction_Gradient >= 157.5 && direction_Gradient <= 180.0)){
					direction_Gradient = 0;
					count0++;
				}
				else if (direction_Gradient >= 22.5 && direction_Gradient < 67.5){
					direction_Gradient = 45;
					count45++;
				}
				else if (direction_Gradient >= 67.5 && direction_Gradient < 112.5){
					direction_Gradient = 90;
					count90++;
				}
				else if (direction_Gradient >= 112.5 && direction_Gradient < 157.5){
					direction_Gradient = 135;
					count135++;
				}
			}


			edgeImg.at<Vec2b>(y - 1, x - 1)[0] = magnitude_Gradient;
			edgeImg.at<Vec2b>(y - 1, x - 1)[1] = (int) direction_Gradient;
		}
	}

	cout << "0: " << count0 << " , 45: " << count45 << " , 90: " << count90 << " , 135: " << count135 << " , NON: " << countNon << endl;
	return edgeImg;
}

/*
	!!!   CURRENTLY IN DEVELOPMENT   !!!

	Applies Non-Maximum Suppresion on the input matrix. 
	 - The input matrix should be a 2-channel matrix with the gradient magnitude in channel 0 and direction in channel 1.
	Returns a 1-channel 8-bit matrix the same size as the input image, but with thinned edges

	Should it be >= or just > when comparing magnitudes?
	Is the 135 and 45 switched?

	!!!   CURRENTLY IN DEVELOPMENT   !!!
*/
Mat NonMaxSuprresion(Mat img){
	Mat thinnedImg(img.rows, img.cols, CV_8UC1);

	int count = 0;
	int count2 = 0;

	int count0 = 0;
	int count45 = 0;
	int count90 = 0;
	int count135 = 0;
	int countNon = 0;

	for (int y = 0; y < img.rows; y++){
		for (int x = 0; x < img.cols; x++){

			//If pixel has a non taken care of gradient direction
			if (img.at<Vec2b>(y, x)[1] != 255 && img.at<Vec2b>(y, x)[1] != 0 && img.at<Vec2b>(y, x)[1] != 45 && img.at<Vec2b>(y, x)[1] != 90 && img.at<Vec2b>(y, x)[1] != 135){
				cout << (int)img.at<Vec2b>(y, x)[1] << endl;
				count2++;
			}

			if (img.at<Vec2b>(y,x)[1] == 0){ //Edge goes from left to right or reverse
				count0++;
				int west = 0;
				int east = 0;

				if (x - 1 >= 0)			west = img.at<Vec2b>(y, x - 1)[0];
				if (x + 1 < img.cols)	east = img.at<Vec2b>(y, x + 1)[0];

				if ((img.at<Vec2b>(y, x)[0] > west) && (img.at<Vec2b>(y, x)[0] > east)){
					thinnedImg.at<uchar>(y, x) = img.at<Vec2b>(y, x)[0];
					count++;
				}
				else thinnedImg.at<uchar>(y, x) = 0;
			}

			else if (img.at<Vec2b>(y, x)[1] == 45){ //Edge goes from southwest to northeast or reverse (????)
				count45++;
				int northeast = 0;
				int southwest = 0;

				if ((y + 1 < img.rows) && (x - 1 >= 0))	southwest = img.at<Vec2b>(y + 1, x - 1)[0];
				if ((y - 1 >= 0) && (x + 1 < img.cols)) northeast = img.at<Vec2b>(y - 1, x + 1)[0];

				if ((img.at<Vec2b>(y, x)[0] > southwest) && (img.at<Vec2b>(y, x)[0] > northeast)){
					thinnedImg.at<uchar>(y, x) = img.at<Vec2b>(y, x)[0];
					count++;
				}
				else thinnedImg.at<uchar>(y, x) = 0;
			}

			else if (img.at<Vec2b>(y, x)[1] == 90){ //Edge goes from north tosouthor reverse
				count90++;
				int north = 0;
				int south = 0;

				if (y - 1 >= 0)			north = img.at<Vec2b>(y - 1, x)[0];
				if (y + 1  < img.rows)  south = img.at<Vec2b>(y + 1, x)[0];
				
				if ((img.at<Vec2b>(y, x)[0] > north) && (img.at<Vec2b>(y, x)[0] > south)){
					thinnedImg.at<uchar>(y, x) = img.at<Vec2b>(y, x)[0];
					count++;
				}
				else thinnedImg.at<uchar>(y, x) = 0;
			}

			else if (img.at<Vec2b>(y, x)[1] == 135){ //Edge goes from northwest to southeast or reverse (???)
				count135++;
				int northwest = 0;
				int southeast = 0;

				if ((y - 1 >= 0) && (x - 1 >= 0))				northwest = img.at<Vec2b>(y - 1, x - 1)[0];
				if ((y + 1 < img.rows) && (x + 1 < img.cols))	southeast = img.at<Vec2b>(y + 1, x + 1)[0];

				if ((img.at<Vec2b>(y, x)[0] > southeast) && (img.at<Vec2b>(y, x)[0] > northwest)){
					thinnedImg.at<uchar>(y, x) = img.at<Vec2b>(y, x)[0];
					count++;
				}
				else thinnedImg.at<uchar>(y, x) = 0;
			}
			else if (img.at <Vec2b>(y, x)[1] == 255){ //If the pixel is not at an edge
				countNon++;
				thinnedImg.at<uchar>(y, x) = img.at<Vec2b>(y, x)[0];
			}
			else{ cout << endl << "INVALID GRADIENT DIRECTION. DIRECTION: " << (int)img.at<Vec2b>(y, x)[1] << " IS NOT ACCOUNTED FOR" << endl; }
		}
	}
	cout << count << " " << count2 << endl;
	cout << "0: " << count0 << " , 45: " << count45 << " , 90: " << count90 << " , 135: " << count135 << " , NON: " << countNon << endl;
	return thinnedImg;
}

/*
	Receives a 1-channel matrix and the 2 threshold values used for detecting strong and weak edges.
	Returns a 1-channel 8-bit matrix the same size as the input matrix
*/
Mat doubleThreshold(Mat img, int max, int min){
	Mat thresholded(img.rows, img.cols, CV_8UC1);

	for (int y = 0; y < img.rows; y++){
		for (int x = 0; x < img.cols; x++){
			if (img.at<uchar>(y, x) >= max) thresholded.at<uchar>(y, x) = 255;
			else if (img.at<uchar>(y, x) >= min) thresholded.at<uchar>(y, x) = 128;
			else thresholded.at<uchar>(y, x) = 0;
		}
	}
	return thresholded;
}



void main(){
	//String filename = "Lena.png";
	//String filename = "woman.jpg";
	//String filename = "motor1.jpg";
	String filename = "motor.png";
	String filename2 = "edgetest.jpg";

	int threshold_max = 100;
	int threshold_min = 40;
	int gaussianRadius = 3;
	double gaussianSigma = 1.4;

	Mat img = imread(filename, CV_LOAD_IMAGE_COLOR);
	Mat img2 = imread(filename2, CV_LOAD_IMAGE_COLOR);

	if (img.empty())
	{
		cout << "IMAGE CALLED: " << filename << " WAS NOT FOUND"  << endl;
		system("Pause");
		return;
	}
	//namedWindow("Loaded image", CV_WINDOW_AUTOSIZE);
	//imshow("Loaded image", img);

	//Convert image from a 3-channel image to grayscale
	Mat gray = ColorToGray(img);
	cout << "Converted to grayscale" << endl;

	namedWindow("gray", CV_WINDOW_AUTOSIZE);
	imshow("gray", gray);
	imwrite("gray.jpg", gray);


	//Apply gaussian blur YxY filter with no border correction. 4 less columns and rows
	Mat blurred = GaussianBlur(gray,gaussianRadius,gaussianSigma);
	cout << "Applied a normalized Gaussian " << gaussianRadius*2+1 << "x" << gaussianRadius*2+1 << " kernel with " << gaussianSigma << " standard deviation" << endl;
	namedWindow("blurred", CV_WINDOW_AUTOSIZE);
	imshow("blurred", blurred);
	imwrite("blurred.jpg", blurred);

	//Apply 3x3 Sobel filter with no border correction. 2 less columns and rows
	Mat edge = Sobel(blurred);
	vector<Mat> channels;
	split(edge, channels);
	cout << "Edge detection with Sobel operators applied" << endl;
	namedWindow("Gradient Magnitude", CV_WINDOW_AUTOSIZE);
	imshow("Gradient Magnitude", channels[0]);
	namedWindow("Gradient Direction", CV_WINDOW_AUTOSIZE);
	imshow("Gradient Direction", channels[1]);
	imwrite("GradientMagnitude.jpg", channels[0]);

	//Apply Non-Maximum Suprresion
	Mat thinned = NonMaxSuprresion(edge);
	cout << "Non-Maximum Suppresion applied" << endl;
	namedWindow("NMS", CV_WINDOW_AUTOSIZE);
	imshow("NMS", thinned);
	imwrite("NMS.jpg", thinned);

	//Apply a double threshold for strong and weak edges
	Mat thresholded = doubleThreshold(thinned, threshold_max, threshold_min);
	cout << "Double threshold applied with thresholds: " << threshold_max << " & " << threshold_min << endl;
	namedWindow("Double Thresh", CV_WINDOW_AUTOSIZE);
	imshow("Double Thresh", thresholded);
	imwrite("thresholded.jpg", thresholded);


	//Canny(blurred, blurred, threshold_min, threshold_max, 3);
	//imshow("Canny", blurred);

	/*
	double x, y, result;
	x = 10.0;
	y = 0.0;
	result = atan2(y, x) * 180 / M_PI;
	printf("The arc tangent for (x=%f, y=%f) is %f degrees\n", x, y, result);

	x = 10.0;
	y = 10.0;
	result = atan2(y, x) * 180 / M_PI;
	printf("The arc tangent for (x=%f, y=%f) is %f degrees\n", x, y, result);

	x = 0.0;
	y = 10.0;
	result = atan2(y, x) * 180 / M_PI;
	printf("The arc tangent for (x=%f, y=%f) is %f degrees\n", x, y, result);

	x = -10.0;
	y = 10.0;
	result = atan2(y, x) * 180 / M_PI;
	printf("The arc tangent for (x=%f, y=%f) is %f degrees\n", x, y, result);
	*/

	//imwrite("ygradient.jpg", thresholded);
	//imwrite("ygradientMag.jpg", channels[0]);
	//imwrite("ygradientDir.jpg", channels[1]);



	// Wait for user
	waitKey(0);
	// End quietly
	return;
}