#include "pch.h"
#include "framework.h"
#include "MFCApp.h"
#include "MFCAppDlg.h"

#include<iostream>
#include<algorithm>
#include<stdio.h>
#include<stdlib.h>
#include<opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include<opencv2/dnn.hpp>
#include<opencv2/cudaarithm.hpp>
#include<opencv2/cudaoptflow.hpp>
#include<opencv2/cudaimgproc.hpp>
#include<opencv2/cudafeatures2d.hpp>
#include<opencv2/cudaobjdetect.hpp>
#include<opencv2/cudawarping.hpp>
#include<opencv2/cudafilters.hpp>
#include<vector>
#include<string>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>

using namespace cv;
using namespace std;
using namespace cv::cuda;
using namespace cv::dnn;

void drawHistogram(Mat& image, string str);//绘制直方图

//灰度直方图均衡
void grayProcess()
{
	Mat h_img = imread("F:/cuda_pictures/girl.jpg", 0);
	GpuMat d_img, d_result;
	d_img.upload(h_img);
	cv::cuda::equalizeHist(d_img, d_result);
	Mat h_result;
	d_result.download(h_result);
	imshow("原始图像", h_img);
	//原始图像直方图
	drawHistogram(h_img, "原始图像直方图");
	imshow("直方图均衡", h_result);
	//图像均衡后直方图
	drawHistogram(h_result, "图像均衡后直方图");
	waitKey(0);
}

//彩色直方图均衡
int myColorProcess()
{
	cv::Mat src_host = imread("F:/cuda_pictures/tree.jpg");
	//cv::Mat src_host = imread("F:/cuda_pictures/senna.jpg");
	if (!src_host.data)
	{
		cout << "读取图片错误，请重新输入正确路径！\n";
		system("pause");
		return -1;
	}

	GpuMat src, h_result, g_result;
	src.upload(src_host);
	//HSV   h色调  s饱和度  v明度
	cuda::cvtColor(src, h_result, cv::COLOR_BGR2HSV);
	std::vector<GpuMat> vec_channels;
	cuda::split(h_result, vec_channels);
	//cuda::split(src, vec_channels);
	//cuda::equalizeHist(vec_channels[0], vec_channels[0]);
	//cuda::equalizeHist(vec_channels[1], vec_channels[1]);
	cuda::equalizeHist(vec_channels[2], vec_channels[2]);
	cuda::merge(vec_channels, h_result);
	cuda::merge(vec_channels, src);
	cuda::cvtColor(h_result, g_result, cv::COLOR_HSV2BGR);
	Mat result;
	g_result.download(result);
	//src.download(result);

	imshow("原始图像", src_host);
	//原始图像直方图
	drawHistogram(src_host, "原始图像直方图");
	imshow("直方图均衡", result);
	//图像均衡后直方图
	drawHistogram(result, "图像均衡后直方图");
	waitKey(0);
	return 0;
}

//直方图均衡+时间效率比较
string colorProcess(Mat &image)
{
	string str;
	//cv::Mat h_img1 = imread("F:/cuda_pictures/valley.jpg");
	cv::Mat h_img1 = image;
	cv::Mat h_img2, h_result;
	clock_t start, end;
	start = clock();
	cv::cvtColor(h_img1, h_img2, cv::COLOR_BGR2HSV);//BGR转为HSV
	//拆分成三通道,分别计算
	std::vector<cv::Mat> vec_channels;
	cv::split(h_img2, vec_channels);
	//色调与饱和度通道包含颜色信息，无需均衡
	//只需对值通道进行均衡
	cv::equalizeHist(vec_channels[2], vec_channels[2]);
	cv::merge(vec_channels, h_img2);
	cv::cvtColor(h_img2, h_result, cv::COLOR_HSV2BGR);

	end = clock();
	double cpu_time = double(end - start) / CLOCKS_PER_SEC;
	//cout << "time = " << cpu_time << "s" << endl;

	//初始化
	cv::Mat img_init = imread("F:/cuda_pictures/sea.jpg");
	GpuMat src_init, init_result;
	src_init.upload(img_init);
	cuda::cvtColor(src_init, init_result, cv::COLOR_BGR2HSV);

	GpuMat src, h_result_cuda, g_result;
	clock_t start_cuda, end_cuda;
	src.upload(h_img1);
	start_cuda = clock();//开始计时
	//BGR转HSV，便于进行直方图均衡化
	cuda::cvtColor(src, h_result_cuda, cv::COLOR_BGR2HSV);
	std::vector<GpuMat> vec_channels_cuda;
	cuda::split(h_result_cuda, vec_channels_cuda);
	cuda::equalizeHist(vec_channels_cuda[2], vec_channels_cuda[2]);
	cuda::merge(vec_channels_cuda, h_result_cuda);
	cuda::cvtColor(h_result_cuda, g_result, cv::COLOR_HSV2BGR);
	end_cuda = clock();
	Mat result;
	g_result.download(result);
	//并行计算耗时
	double cuda_time = double(end_cuda - start_cuda) / CLOCKS_PER_SEC;
	//cout << "cuda time = " << cuda_time << "s" << endl;

	//加速比
	double sp = cpu_time / cuda_time;
	//cout << "加速比 = " << sp << endl;

	//字符串拼接结果
	str = to_string(cpu_time);
	str.append("/");
	str.append(to_string(cuda_time));
	str.append("/");
	str.append(to_string(sp));

	//cv::imshow("原始图像", h_img1);
	drawHistogram(h_img1, "原图直方图");

	//namedWindow("直方图均衡", 0);
	//resizeWindow("直方图均衡", 240, 427);
	//cv::imshow("直方图均衡", h_result);
	//drawHistogram(h_result, "均衡后直方图");

	//namedWindow("cuda直方图均衡", 0);
	//resizeWindow("cuda直方图均衡", 240, 427);
	//cv::imshow("cuda直方图均衡", result);
	remove("F:/image_result/result.jpg");
	imwrite("F:/image_result/result.jpg", result);
	drawHistogram(result, "cuda直方图");

	waitKey(0);
	return str;
}


//自定义大小调整
void myChangeSize()
{
	Mat host_img = imread("F:/cuda_pictures/girl.jpg", 0);
	int width, height;
	cout << "请输入需要调整的宽度：" << endl;
	cin >> width;
	cout << "请输入需要调整的高度：" << endl;
	cin >> height;
	GpuMat device_img, device_result;
	device_img.upload(host_img);
	cout << "原始宽度:" << device_img.cols;
	cout << "原始高度:" << device_img.size().height << endl;
	cuda::resize(device_img, device_result, cv::Size(width, height), cv::INTER_CUBIC);//双三次
	Mat host_result;
	device_result.download(host_result);
	cv::imshow("原始图像", host_img);
	cv::imshow("调整大小", host_result);
	waitKey(0);
}

//绘制直方图
void drawHistogram(Mat& srcImage, string str)
{
	int dims = srcImage.channels();//图片通道数
	if (dims == 3)//彩色
	{
		int bins = 256;
		int histsize[] = { bins };
		float range[] = { 0, 256 };
		const float* histRange = { range };
		Mat  b_Hist, g_Hist, r_Hist;
		//图像通道的分离，3个通道B、G、R
		vector<Mat> rgb_channel;
		split(srcImage, rgb_channel);
		//计算各个通道的直方图
		//B-通道
		calcHist(&rgb_channel[0], 1, 0, Mat(), b_Hist, 1, histsize, &histRange, true, false);
		//G-通道
		calcHist(&rgb_channel[1], 1, 0, Mat(), g_Hist, 1, histsize, &histRange, true, false);
		//R-通道
		calcHist(&rgb_channel[2], 1, 0, Mat(), r_Hist, 1, histsize, &histRange, true, false);
		//设置直方图绘图参数
		int hist_h = 360;
		int hist_w = bins * 3;
		int bin_w = cvRound((double)hist_w / bins);
		//创建黑底图像
		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
		//直方图归一化到[0,histImage.rows]
		//B-通道
		cv::normalize(b_Hist, b_Hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		//G-通道
		cv::normalize(g_Hist, g_Hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		//R-通道
		cv::normalize(r_Hist, r_Hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		//绘制图像
		for (int i = 1; i < bins; i++)
		{
			//绘制B通道的直方图信息
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_Hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(b_Hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
			//绘制G通道的直方图信息
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_Hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(g_Hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
			//绘制R通道的直方图信息
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_Hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(r_Hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
		}
		imshow(str, histImage);
	}
	else//灰度图
	{
		const int channels[1] = { 0 };
		const int bins[1] = { 256 };
		float hranges[2] = { 0,255 };
		const float* ranges[1] = { hranges };
		Mat hist;
		// 计算Blue, Green, Red通道的直方图
		calcHist(&srcImage, 1, 0, Mat(), hist, 1, bins, ranges);
		//设置直方图绘图参数
		int hist_h = 360;
		int hist_w = bins[0] * 3;
		int bin_w = cvRound((double)hist_w / bins[0]);
		Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
		// 归一化直方图数据
		cv::normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		// 绘制直方图曲线
		for (int i = 1; i < bins[0]; i++) {
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		}
		imshow(str, histImage);
	}
}

//平均滤波器
//归一化块滤波
//线性滤波
void myAverageFilter(Mat &image)
{
	//Mat host_img = imread("F:/cuda_pictures/girl.jpg", 0);
	Mat host_img = image;
	Mat result;
	double temp = (double)1 / 9;
	Mat kernel = (Mat_<float>(3, 3) << temp, temp, temp, temp, temp, temp, temp, temp, temp);
	//filter2D(host_img, result, CV_8UC3, kernel);
	/*
	Mat zero_mat = Mat::zeros(Size(host_img.cols, host_img.rows), CV_8UC1);
	Mat roi(zero_mat, Rect(100, 2, 1, 280));
	roi = Scalar(255, 255, 255);
	vector<Mat> channel(3);
	split(host_img, channel);
	vector<Mat> channel4;
	channel4.push_back(channel[0]);//b
	channel4.push_back(channel[1]);//g
	channel4.push_back(channel[2]);//r
	channel4.push_back(zero_mat);//alpha
	Mat argb;
	merge(channel4, argb);
	//imshow("argb", argb);
	*/
	GpuMat d_img, d_res;
	d_img.upload(host_img);
	cuda::cvtColor(d_img, d_res, cv::COLOR_RGB2RGBA);//增加alpha通道
	cv::Ptr<cv::cuda::Filter> filter;
	GpuMat src, res;
	//src.upload(argb);
	//src.upload(host_img);
	filter = cuda::createLinearFilter(CV_8UC4, CV_8UC4, kernel);
	filter->apply(d_res, res);
	res.download(result);
	
	remove("F:/image_result/result.jpg");
	imwrite("F:/image_result/result.jpg",result);
	//imshow("原始图像", host_img);
	//imshow("3X3平均滤波", result);
	waitKey(0);
}

//高斯滤波器
//线性滤波
void myGaussFilter(Mat &image)
{
	//Mat host_img = imread("F:/cuda_pictures/agera.jpg", 0);
	Mat host_img = image;
	Mat result;
	double temp1 = (double)1 / 16;
	double temp2 = (double)2 / 16;
	double temp3 = (double)4 / 16;

	Mat kernel = (Mat_<float>(3, 3) << temp1, temp2, temp1, temp2, temp3, temp2, temp1, temp2, temp1);
	//filter2D(host_img, result, CV_8UC3, kernel);
	GpuMat d_img, d_res;
	d_img.upload(host_img);
	cuda::cvtColor(d_img, d_res, cv::COLOR_RGB2RGBA);//增加alpha通道
	cv::Ptr<cv::cuda::Filter> filter;
	GpuMat res;
	filter = cuda::createLinearFilter(CV_8UC4, CV_8UC4, kernel);
	filter->apply(d_res, res);
	res.download(result);
	//cuda::createGaussianFilter()
	//imshow("原始图像", host_img);
	//imshow("3X3高斯滤波", result);
	remove("F:/image_result/result.jpg");
	imwrite("F:/image_result/result.jpg", result);
	waitKey(0);
}

//中值过滤
//非线性滤波
int myMedianFilter()
{
	Mat host_img = imread("F:/cuda_pictures/salt_pepper.jpg", 0);
	if (!host_img.data)
	{
		cout << "读取图片错误，请重新输入正确路径！\n";
		system("pause");
		return -1;
	}
	Mat host_result;
	cv::medianBlur(host_img, host_result, 3);
	imshow("原始图像", host_img);
	imshow("中值滤波", host_result);
	waitKey(0);
	return 0;
}
//添加椒盐噪声
void salt(Mat& image, int n) {
	for (int k = 0; k < n; k++) {
		int i = rand() % image.cols;
		int j = rand() % image.rows;

		if (image.channels() == 1) {   //判断是一个通道
			image.at<uchar>(j, i) = 255;
		}
		else {
			image.at<cv::Vec3b>(j, i)[0] = 255;
			image.at<cv::Vec3b>(j, i)[1] = 255;
			image.at<cv::Vec3b>(j, i)[2] = 255;
		}
	}
	remove("F:/image_result/result_salt1.jpg");//添加噪声后图像
	imwrite("F:/image_result/result_salt1.jpg", image);
}
//中值过滤
int MedianProcess(Mat &host_src)
{
	Mat host_result;
	//host_result = host_src.clone();
	//cv::medianBlur(host_src, host_result, 3);
	
	GpuMat src_init, init_result, d_res;
	src_init.upload(host_src);
	cuda::cvtColor(src_init, init_result, cv::COLOR_BGR2HSV);//转为HSV色彩空间
	std::vector<GpuMat> vec_channels_init;
	cuda::split(init_result, vec_channels_init);
	cv::Ptr<cv::cuda::Filter> filter;
	filter = cuda::createMedianFilter(CV_8UC1, 3);//中值过滤器,只能对单通道进行
	filter->apply(vec_channels_init[2], vec_channels_init[2]);//对值通道进行中值过滤
	cuda::merge(vec_channels_init, init_result);
	cuda::cvtColor(init_result, d_res, cv::COLOR_HSV2BGR);
	d_res.download(host_result);

	remove("F:/image_result/result_salt2.jpg");//添加噪声后去噪图像
	imwrite("F:/image_result/result_salt2.jpg", host_result);
	waitKey(0);
	return 0;
}

//腐蚀
int myPicErode(Mat &image)
{
	Mat host_img = image;
	GpuMat d_img, d_result;
	//定义形态操作的结构元素
	Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	d_img.upload(host_img);
	cv::Ptr<cuda::Filter> filter;
	filter = cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, element);
	filter->apply(d_img, d_result);
	Mat h_result;
	d_result.download(h_result);

	//imshow("原始图像", host_img);
	//imshow("腐蚀图像", h_result);
	remove("F:/image_result/result.jpg");
	imwrite("F:/image_result/result.jpg", h_result);

	waitKey(0);
	return 0;
}

//膨胀
int myPicDilate(Mat& image)
{
	Mat host_img = image;
	GpuMat d_img, d_result;
	//定义形态操作的结构元素
	Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	d_img.upload(host_img);
	//d_img = gray;
	cv::Ptr<cuda::Filter> filter;
	filter = cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, element);
	filter->apply(d_img, d_result);
	Mat h_result;
	d_result.download(h_result);

	//imshow("原始图像", host_img);
	//imshow("膨胀图像", h_result);
	remove("F:/image_result/result.jpg");
	imwrite("F:/image_result/result.jpg", h_result);

	waitKey(0);
	return 0;
}

//sobel算子
int mySobel(Mat &image)
{
	Mat host_img = image;
	Mat result, result_x, result_y;

	Mat kernel_x = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	//filter2D(host_img, result_x, CV_8UC3, kernel_x);
	Mat kernel_y = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	//filter2D(host_img, result_y, CV_8UC3, kernel_y);
	//cv::add(result_x, result_y, result);

	GpuMat d_img, d_res, res;
	d_img.upload(host_img);
	cuda::cvtColor(d_img, d_res, cv::COLOR_RGB2RGBA);//增加alpha通道
	cv::Ptr<cv::cuda::Filter> filter_x, filter_y;
	GpuMat res_x, res_y;
	filter_x = cuda::createLinearFilter(CV_8UC4, CV_8UC4, kernel_x);
	filter_x->apply(d_res, res_x);
	filter_y = cuda::createLinearFilter(CV_8UC4, CV_8UC4, kernel_y);
	filter_y->apply(d_res, res_y);
	cuda::add(res_x, res_y, res);
	res.download(result);
	//cv::imshow("Sobel锐化", result);
	remove("F:/image_result/result.jpg");
	imwrite("F:/image_result/result.jpg", result);

	waitKey(0);
	return 0;
}

//拉普拉斯滤波器
int myLaplacianFilter(Mat& image)
{
	Mat host_img = image;
	//将原图转化为灰度图
	GpuMat src, gray;
	clock_t start_cuda, end_cuda;
	GpuMat d_img, d_result;
	cv::Ptr<cv::cuda::Filter> filter;
	int x = 3;//内核大小
	src.upload(host_img);
	start_cuda = clock();
	cuda::cvtColor(src, gray, COLOR_BGR2GRAY);
	d_img = gray;
	filter = cv::cuda::createLaplacianFilter(CV_8UC1, CV_8UC1, x);
	filter->apply(d_img, d_result);
	end_cuda = clock();
	Mat h_result;
	d_result.download(h_result);
	//cv::imshow("原始图像", host_img);
	//cv::imshow("拉普拉斯轮廓提取", h_result);
	remove("F:/image_result/result.jpg");
	imwrite("F:/image_result/result.jpg", h_result);

	waitKey(0);
	return 0;
}
//拉普拉斯图像锐化
int myLaplacianSharpen(Mat &image)
{
	//Mat host_img = imread("F:/cuda_pictures/tree.jpg");
	Mat host_img = image;
	/*
	//调整大小
	GpuMat d_img, d_result;
	d_img.upload(host_img);
	int width = d_img.cols;
	int height = d_img.size().height;
	cuda::resize(d_img, d_result, cv::Size(0.7 * width, 0.7 * height), cv::INTER_LINEAR);
	Mat h_result;
	d_result.download(h_result);
	*/
	//imshow("原图像", host_img);
	Mat imageEnhance;
	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
	//Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);

	GpuMat d_img, d_res, res;
	d_img.upload(host_img);
	cuda::cvtColor(d_img, d_res, cv::COLOR_RGB2RGBA);//增加alpha通道
	cv::Ptr<cv::cuda::Filter> filter;
	filter = cuda::createLinearFilter(CV_8UC4, CV_8UC4, kernel);
	filter->apply(d_res, res);
	res.download(imageEnhance);

	//filter2D(host_img, imageEnhance, CV_8UC3, kernel);
	//imshow("拉普拉斯算子图像增强效果", imageEnhance);
	remove("F:/image_result/result.jpg");
	imwrite("F:/image_result/result.jpg", imageEnhance);

	waitKey(0);
	return 0;
}

//自定义算子
int customCalculaters()
{
	Mat host_img = imread("F:/cuda_pictures/tree.jpg");
	if (!host_img.data)
	{
		cout << "读取图片错误，请重新输入正确路径！\n";
		system("pause");
		return -1;
	}
	//调整大小
	GpuMat d_img, d_result;
	d_img.upload(host_img);
	int width = d_img.cols;
	int height = d_img.size().height;
	cuda::resize(d_img, d_result, cv::Size(0.7 * width, 0.7 * height), cv::INTER_LINEAR);
	Mat h_result;
	d_result.download(h_result);

	imshow("原图像", host_img);

	Mat result;
	//Mat kernel = (Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
	//Mat kernel = (Mat_<float>(3, 3) << 1, 1, 1, 0, 0, 0, -1, -1, -1);
	//Mat kernel = (Mat_<float>(3, 3) << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5);

	//Mat kernel = (Mat_<float>(3, 3) << -1, -1 , -1, -1, 9, -1, -1, -1, -1);
	//Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
	//Mat kernel = (Mat_<float>(3, 3) << 0, 1, 0, 1, 4, 1, 0, 1, 0);
	//Mat kernel = (Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 4, 0, 0, -1, 0);

	filter2D(h_result, result, CV_8UC3, kernel);
	imshow("算子图像处理效果", result);

	waitKey(0);
	return 0;
}


//二值化处理
Mat SrcImage;
Mat GrayImage;
Mat BinaryImage;
void on_trackbar(int pos, void*)
{
	//转化为二值图
	cv::threshold(GrayImage, BinaryImage, pos, 255, CV_THRESH_BINARY);
	imshow("二值图", BinaryImage);
}
void to_GrayImage()
{
	//创建与原图同类型和同大小的矩阵
	GrayImage.create(SrcImage.size(), SrcImage.type());
	//将原图转换为灰度图像
	cv::cvtColor(SrcImage, GrayImage, CV_BGR2GRAY);
	//imshow("灰度图", GrayImage);
	//remove("F:/image_result/result.jpg");
	//imwrite("F:/image_result/result.jpg", GrayImage);

}
void creat_trackbar()
{
	int nThreshold = 0;
	createTrackbar("二值图阈值", "二值图", &nThreshold, 254, on_trackbar);
}
int myBinaryProcess(Mat& image)
{
	SrcImage = image;
	to_GrayImage();
	on_trackbar(1, 0);
	creat_trackbar();

	waitKey(0);
	return 0;
}
