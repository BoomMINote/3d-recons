#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <ctime>
using namespace std;
using namespace cv;

//steger光条中心提取
class Steger
{
public:
	Steger(const Mat img, double sigma, float thresh ,Si1ze  kernel_size);
	void  GetCenterPoint(vector<Scalar> &vec);
	void  SetGaussParam(float thresh, float sigma);
private:
	void  AcquireScalarVec(vector<Scalar> &vec);
	Mat  img_src_;
	Size  kernel_size_;
	double  sigma_;
	float thresh_ToZero_;
};

Steger::Steger(const Mat img, double sigma, float thresh, Size  kernel_size) :
	sigma_(sigma),
	kernel_size_(kernel_size),
	thresh_ToZero_(thresh)
{
	img.convertTo(img_src_, CV_32FC1);
}

void Steger::GetCenterPoint(vector<Scalar> &vec) {
	AcquireScalarVec(vec);
}

void  Steger::SetGaussParam(float thresh, float sigma) {
	sigma_ = sigma;
	thresh_ToZero_ = thresh;
}

void Steger::AcquireScalarVec(vector<Scalar> &vec) {
	Mat kx_ = (Mat_<float>(1, 2) << -1, 1);
	Mat ky_ = (Mat_<float>(2, 1) << -1, 1);
	Mat kxx_ = (Mat_<float>(1, 3) << 1, -2, 1);
	Mat kyy_ = (Mat_<float>(3, 1) << 1, -2, 1);
	Mat kxy_ = (Mat_<float>(2, 2) << 1, -1, -1, 1);
	Mat  dx_, dy_, dxx_, dyy_, dxy_;
	GaussianBlur(img_src_, img_src_, kernel_size_, sigma_, sigma_);
	filter2D(img_src_, dx_, CV_32FC1, kx_);
	filter2D(img_src_, dy_, CV_32FC1, ky_);
	filter2D(img_src_, dxx_, CV_32FC1, kxx_);
	filter2D(img_src_, dyy_, CV_32FC1, kyy_);
	filter2D(img_src_, dxy_, CV_32FC1, kxy_);

	for (int i = 0; i < img_src_.cols; ++i) {
		for (int j = 0; j < img_src_.rows; ++j) {
			if (img_src_.at<float>(j, i) > thresh_ToZero_) {
				Mat hessian(2, 2, CV_32FC1);
				hessian.at<float>(0, 0) = dxx_.at<float>(j, i);
				hessian.at<float>(0, 1) = dxy_.at<float>(j, i);
				hessian.at<float>(1, 0) = dxy_.at<float>(j, i);
				hessian.at<float>(1, 1) = dyy_.at<float>(j, i);
				Mat eig_val, eig_vec;
				eigen(hessian, eig_val, eig_vec);
				float nx = 0, ny = 0, fmax_dist = 0;

				if (fabs(eig_val.at<float>(0, 0)) >= fabs(eig_val.at<float>(1, 0))) {
					nx = eig_vec.at<float>(0, 0);
					ny = eig_vec.at<float>(0, 1);
					fmax_dist = eig_val.at<float>(0, 0);
				}
				else {
					nx = eig_vec.at<float>(1, 0);
					ny = eig_vec.at<float>(1, 1);
					fmax_dist = eig_val.at<float>(1, 0);
				}
				float t = -(nx*dx_.at<float>(j, i) + ny*dy_.at<float>(j, i)) / (nx*nx*dxx_.at<float>(j, i) + 2 * nx*ny*dxy_.at<float>(j, i) + ny*ny*dyy_.at<float>(j, i));
				if (fabs(t*nx) <= 0.5 && fabs(t*ny) <= 0.5) {
					vec.push_back(Scalar(i + t*nx, j + t*ny, atan2(ny, nx), fabs(fmax_dist)));
				}
			}
		}
	}
}

#if 1
int main() {

	cv::Mat srcImg = imread("../test3.png",0);// original img
	cv::Mat dstImg = imread("../test3.png");// the img that we draw circle on
    cout<<srcImg.size()<<endl;
//	Size kernel(17, 17);
    Size kernel(17, 17);

	cv::Mat bwImg;
	double thresh = threshold(srcImg, bwImg, 0, 255, CV_THRESH_OTSU);
    cout<<thresh<<endl;
    cv::namedWindow("tt", 0);
    cv::imshow("tt", srcImg);
    cv::waitKey(0);
//	Steger s(srcImg, 3, 20, kernel);
    Steger s(srcImg, 3, 20, kernel);
	std::vector<Scalar> vec;
	s.GetCenterPoint(vec);

	fstream lightdata;
	lightdata.open("./light.txt", fstream::out);

	for (std::vector<Scalar>::iterator it = vec.begin(); it != vec.end(); ++it) {
		
		lightdata << it->val[0] << " " << it->val[1] << endl;;


		cv::circle(dstImg, cv::Point(it->val[0], it->val[1]), 1, cv::Scalar(255, 255, 0), 1, 8);
	}
    imwrite("../output.jpg", dstImg);
	cv::namedWindow("test", 0);
	cv::imshow("test", dstImg);
	cv::waitKey(0);
	return 0;
}
#endif
