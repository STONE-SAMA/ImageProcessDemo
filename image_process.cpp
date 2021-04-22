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

void drawHistogram(Mat& image, string str);//����ֱ��ͼ

//�Ҷ�ֱ��ͼ����
void grayProcess()
{
	Mat h_img = imread("F:/cuda_pictures/girl.jpg", 0);
	GpuMat d_img, d_result;
	d_img.upload(h_img);
	cv::cuda::equalizeHist(d_img, d_result);
	Mat h_result;
	d_result.download(h_result);
	imshow("ԭʼͼ��", h_img);
	//ԭʼͼ��ֱ��ͼ
	drawHistogram(h_img, "ԭʼͼ��ֱ��ͼ");
	imshow("ֱ��ͼ����", h_result);
	//ͼ������ֱ��ͼ
	drawHistogram(h_result, "ͼ������ֱ��ͼ");
	waitKey(0);
}

//��ɫֱ��ͼ����
int myColorProcess()
{
	cv::Mat src_host = imread("F:/cuda_pictures/tree.jpg");
	//cv::Mat src_host = imread("F:/cuda_pictures/senna.jpg");
	if (!src_host.data)
	{
		cout << "��ȡͼƬ����������������ȷ·����\n";
		system("pause");
		return -1;
	}

	GpuMat src, h_result, g_result;
	src.upload(src_host);
	//HSV   hɫ��  s���Ͷ�  v����
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

	imshow("ԭʼͼ��", src_host);
	//ԭʼͼ��ֱ��ͼ
	drawHistogram(src_host, "ԭʼͼ��ֱ��ͼ");
	imshow("ֱ��ͼ����", result);
	//ͼ������ֱ��ͼ
	drawHistogram(result, "ͼ������ֱ��ͼ");
	waitKey(0);
	return 0;
}

//ֱ��ͼ����+ʱ��Ч�ʱȽ�
void colorProcess()
{

	//cv::Mat h_img1 = imread("F:/cuda_pictures/sea.jpg");
	cv::Mat h_img1 = imread("F:/cuda_pictures/valley.jpg");

	cv::Mat h_img2, h_result;
	clock_t start, end;
	start = clock();
	cv::cvtColor(h_img1, h_img2, cv::COLOR_BGR2HSV);//BGRתΪHSV
	//��ֳ���ͨ��,�ֱ����
	std::vector<cv::Mat> vec_channels;
	cv::split(h_img2, vec_channels);
	//ɫ���뱥�Ͷ�ͨ��������ɫ��Ϣ���������
	//ֻ���ֵͨ�����о���
	cv::equalizeHist(vec_channels[2], vec_channels[2]);
	cv::merge(vec_channels, h_img2);
	cv::cvtColor(h_img2, h_result, cv::COLOR_HSV2BGR);

	end = clock();
	double cpu_time = double(end - start) / CLOCKS_PER_SEC;
	cout << "time = " << cpu_time << "s" << endl;

	//��ʼ��
	cv::Mat img_init = imread("F:/cuda_pictures/sea.jpg");
	GpuMat src_init, init_result;
	src_init.upload(img_init);
	cuda::cvtColor(src_init, init_result, cv::COLOR_BGR2HSV);

	GpuMat src, h_result_cuda, g_result;
	clock_t start_cuda, end_cuda;
	src.upload(h_img1);
	start_cuda = clock();
	cuda::cvtColor(src, h_result_cuda, cv::COLOR_BGR2HSV);
	std::vector<GpuMat> vec_channels_cuda;
	cuda::split(h_result_cuda, vec_channels_cuda);
	cuda::equalizeHist(vec_channels_cuda[2], vec_channels_cuda[2]);
	cuda::merge(vec_channels_cuda, h_result_cuda);
	cuda::cvtColor(h_result_cuda, g_result, cv::COLOR_HSV2BGR);
	end_cuda = clock();
	Mat result;
	g_result.download(result);
	//���м����ʱ
	double cuda_time = double(end_cuda - start_cuda) / CLOCKS_PER_SEC;
	cout << "cuda time = " << cuda_time << "s" << endl;

	//���ٱ�
	double sp = cpu_time / cuda_time;
	cout << "���ٱ� = " << sp << endl;

	namedWindow("ԭʼͼ��", 0);
	resizeWindow("ԭʼͼ��", 240, 427);
	cv::imshow("ԭʼͼ��", h_img1);
	drawHistogram(h_img1, "ԭͼֱ��ͼ");

	//namedWindow("ֱ��ͼ����", 0);
	//resizeWindow("ֱ��ͼ����", 240, 427);
	//cv::imshow("ֱ��ͼ����", h_result);
	//drawHistogram(h_result, "�����ֱ��ͼ");

	namedWindow("cudaֱ��ͼ����", 0);
	resizeWindow("cudaֱ��ͼ����", 240, 427);
	cv::imshow("cudaֱ��ͼ����", result);
	drawHistogram(result, "cudaֱ��ͼ");

	waitKey(0);
}


//�Զ����С����
void myChangeSize()
{
	Mat host_img = imread("F:/cuda_pictures/girl.jpg", 0);
	int width, height;
	cout << "��������Ҫ�����Ŀ�ȣ�" << endl;
	cin >> width;
	cout << "��������Ҫ�����ĸ߶ȣ�" << endl;
	cin >> height;
	GpuMat device_img, device_result;
	device_img.upload(host_img);
	cout << "ԭʼ���:" << device_img.cols;
	cout << "ԭʼ�߶�:" << device_img.size().height << endl;
	cuda::resize(device_img, device_result, cv::Size(width, height), cv::INTER_CUBIC);//˫����
	Mat host_result;
	device_result.download(host_result);
	cv::imshow("ԭʼͼ��", host_img);
	cv::imshow("������С", host_result);
	waitKey(0);
}

//����ֱ��ͼ
void drawHistogram(Mat& srcImage, string str)
{
	int dims = srcImage.channels();//ͼƬͨ����
	if (dims == 3)//��ɫ
	{
		int bins = 256;
		int histsize[] = { bins };
		float range[] = { 0, 256 };
		const float* histRange = { range };
		Mat  b_Hist, g_Hist, r_Hist;
		//ͼ��ͨ���ķ��룬3��ͨ��B��G��R
		vector<Mat> rgb_channel;
		split(srcImage, rgb_channel);
		//�������ͨ����ֱ��ͼ
		//B-ͨ��
		calcHist(&rgb_channel[0], 1, 0, Mat(), b_Hist, 1, histsize, &histRange, true, false);
		//G-ͨ��
		calcHist(&rgb_channel[1], 1, 0, Mat(), g_Hist, 1, histsize, &histRange, true, false);
		//R-ͨ��
		calcHist(&rgb_channel[2], 1, 0, Mat(), r_Hist, 1, histsize, &histRange, true, false);
		//����ֱ��ͼ��ͼ����
		int hist_h = 360;
		int hist_w = bins * 3;
		int bin_w = cvRound((double)hist_w / bins);
		//�����ڵ�ͼ��
		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
		//ֱ��ͼ��һ����[0,histImage.rows]
		//B-ͨ��
		cv::normalize(b_Hist, b_Hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		//G-ͨ��
		cv::normalize(g_Hist, g_Hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		//R-ͨ��
		cv::normalize(r_Hist, r_Hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		//����ͼ��
		for (int i = 1; i < bins; i++)
		{
			//����Bͨ����ֱ��ͼ��Ϣ
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_Hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(b_Hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
			//����Gͨ����ֱ��ͼ��Ϣ
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_Hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(g_Hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
			//����Rͨ����ֱ��ͼ��Ϣ
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_Hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(r_Hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
		}
		imshow(str, histImage);
	}
	else//�Ҷ�ͼ
	{
		const int channels[1] = { 0 };
		const int bins[1] = { 256 };
		float hranges[2] = { 0,255 };
		const float* ranges[1] = { hranges };
		Mat hist;
		// ����Blue, Green, Redͨ����ֱ��ͼ
		calcHist(&srcImage, 1, 0, Mat(), hist, 1, bins, ranges);
		//����ֱ��ͼ��ͼ����
		int hist_h = 360;
		int hist_w = bins[0] * 3;
		int bin_w = cvRound((double)hist_w / bins[0]);
		Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
		// ��һ��ֱ��ͼ����
		cv::normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		// ����ֱ��ͼ����
		for (int i = 1; i < bins[0]; i++) {
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		}
		imshow(str, histImage);
	}
}

//ƽ���˲���
//��һ�����˲�
//�����˲�
void myAverageFilter(Mat &image)
{
	//Mat host_img = imread("F:/cuda_pictures/girl.jpg", 0);
	Mat host_img = image;
	Mat result;
	double temp = (double)1 / 9;
	Mat kernel = (Mat_<float>(3, 3) << temp, temp, temp, temp, temp, temp, temp, temp, temp);
	filter2D(host_img, result, CV_8UC3, kernel);
	remove("F:/image_result/result.jpg");
	imwrite("F:/image_result/result.jpg",result);
	//imshow("ԭʼͼ��", host_img);
	//imshow("3X3ƽ���˲�", result);
	waitKey(0);
}

//��˹�˲���
//�����˲�
void myGaussFilter(Mat &image)
{
	//Mat host_img = imread("F:/cuda_pictures/agera.jpg", 0);
	Mat host_img = image;
	Mat result;
	double temp1 = (double)1 / 16;
	double temp2 = (double)2 / 16;
	double temp3 = (double)4 / 16;

	Mat kernel = (Mat_<float>(3, 3) << temp1, temp2, temp1, temp2, temp3, temp2, temp1, temp2, temp1);
	filter2D(host_img, result, CV_8UC3, kernel);

	//imshow("ԭʼͼ��", host_img);
	//imshow("3X3��˹�˲�", result);
	remove("F:/image_result/result.jpg");
	imwrite("F:/image_result/result.jpg", result);

	waitKey(0);
}

//��ֵ����
//�������˲�
int myMedianFilter()
{
	Mat host_img = imread("F:/cuda_pictures/salt_pepper.jpg", 0);
	if (!host_img.data)
	{
		cout << "��ȡͼƬ����������������ȷ·����\n";
		system("pause");
		return -1;
	}
	Mat host_result;
	cv::medianBlur(host_img, host_result, 3);
	imshow("ԭʼͼ��", host_img);
	imshow("��ֵ�˲�", host_result);
	waitKey(0);
	return 0;
}
//��ӽ�������
void salt(Mat& image, int n) {
	for (int k = 0; k < n; k++) {
		int i = rand() % image.cols;
		int j = rand() % image.rows;

		if (image.channels() == 1) {   //�ж���һ��ͨ��
			image.at<uchar>(j, i) = 255;
		}
		else {
			image.at<cv::Vec3b>(j, i)[0] = 255;
			image.at<cv::Vec3b>(j, i)[1] = 255;
			image.at<cv::Vec3b>(j, i)[2] = 255;
		}
	}
	remove("F:/image_result/result_salt1.jpg");//���������ͼ��
	imwrite("F:/image_result/result_salt1.jpg", image);
}
//��ֵ����
int MedianProcess(Mat &host_src)
{
	//Mat host_src = imread("F:/cuda_pictures/agera.jpg",IMREAD_COLOR);
	//Mat host_src = imread("F:/cuda_pictures/girl.jpg", IMREAD_COLOR);
	if (!host_src.data)
	{
		cout << "��ȡͼƬ����������������ȷ·����\n";
		system("pause");
		return -1;
	}
	//salt(host_src, 10000);
	Mat host_result;
	host_result = host_src.clone();
	cv::medianBlur(host_src, host_result, 3);
	//imshow("ԭʼͼ��", host_src);
	//imshow("��ֵ�˲�", host_result);
	remove("F:/image_result/result_salt2.jpg");//���������ͼ��
	imwrite("F:/image_result/result_salt2.jpg", host_result);
	waitKey(0);
	return 0;
}

//��ʴ
int myPicErode()
{
	Mat host_img = imread("F:/cuda_pictures/blobs.png", 0);
	if (!host_img.data)
	{
		cout << "��ȡͼƬ����������������ȷ·����\n";
		system("pause");
		return -1;
	}
	GpuMat d_img, d_result;
	//������̬�����ĽṹԪ��
	Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	d_img.upload(host_img);
	cv::Ptr<cuda::Filter> filter;
	filter = cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, element);
	filter->apply(d_img, d_result);
	Mat h_result;
	d_result.download(h_result);

	imshow("ԭʼͼ��", host_img);
	imshow("��ʴͼ��", h_result);

	waitKey(0);
	return 0;
}

//����
int myPicDilate()
{
	Mat host_img = imread("F:/cuda_pictures/blobs.png", 0);
	//Mat host_img = imread("F:/cuda_pictures/senna.jpg");
	if (!host_img.data)
	{
		cout << "��ȡͼƬ����������������ȷ·����\n";
		system("pause");
		return -1;
	}
	//��ԭͼת��Ϊ�Ҷ�ͼ
	//GpuMat src, gray;
	//src.upload(host_img);
	//cuda::cvtColor(src, gray, COLOR_BGR2GRAY);
	GpuMat d_img, d_result;
	//������̬�����ĽṹԪ��
	Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	d_img.upload(host_img);
	//d_img = gray;
	cv::Ptr<cuda::Filter> filter;
	filter = cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, element);
	filter->apply(d_img, d_result);
	Mat h_result;
	d_result.download(h_result);

	imshow("ԭʼͼ��", host_img);
	imshow("����ͼ��", h_result);

	waitKey(0);
	return 0;
}

//sobel����
int mySobel()
{
	Mat host_img = imread("F:/cuda_pictures/senna.jpg");
	if (!host_img.data)
	{
		cout << "��ȡͼƬ����������������ȷ·����\n";
		system("pause");
		return -1;
	}
	imshow("ԭͼ��", host_img);
	Mat result, result_x, result_y;

	Mat kernel_x = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	filter2D(host_img, result_x, CV_8UC3, kernel_x);

	Mat kernel_y = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	filter2D(host_img, result_y, CV_8UC3, kernel_y);

	cv::add(result_x, result_y, result);
	cv::imshow("Sobel��", result);

	waitKey(0);
	return 0;
}

//������˹�˲���
int myLaplacianFilter()
{
	//Mat host_img = imread("F:/cuda_pictures/blobs.png", 0);
	Mat host_img = imread("F:/cuda_pictures/senna.jpg");
	if (!host_img.data)
	{
		cout << "��ȡͼƬ����������������ȷ·����\n";
		system("pause");
		return -1;
	}
	//��ԭͼת��Ϊ�Ҷ�ͼ
	GpuMat src, gray;

	clock_t start_cuda, end_cuda;
	GpuMat d_img, d_result;
	cv::Ptr<cv::cuda::Filter> filter;
	int x = 3;//�ں˴�С
	src.upload(host_img);

	start_cuda = clock();

	cuda::cvtColor(src, gray, COLOR_BGR2GRAY);
	//d_img.upload(host_img);
	d_img = gray;
	filter = cv::cuda::createLaplacianFilter(CV_8UC1, CV_8UC1, x);
	filter->apply(d_img, d_result);

	end_cuda = clock();

	Mat h_result;
	d_result.download(h_result);
	cv::imshow("ԭʼͼ��", host_img);
	cv::imshow("������˹��", h_result);

	cout << "cuda time = " << double(end_cuda - start_cuda) / CLOCKS_PER_SEC << "s" << endl;

	waitKey(0);
	return 0;
}
//������˹ͼ����
int myLaplacianSharpen()
{
	Mat host_img = imread("F:/cuda_pictures/tree.jpg");
	if (!host_img.data)
	{
		cout << "��ȡͼƬ����������������ȷ·����\n";
		system("pause");
		return -1;
	}
	//������С
	GpuMat d_img, d_result;
	d_img.upload(host_img);
	int width = d_img.cols;
	int height = d_img.size().height;
	cuda::resize(d_img, d_result, cv::Size(0.7 * width, 0.7 * height), cv::INTER_LINEAR);
	Mat h_result;
	d_result.download(h_result);

	imshow("ԭͼ��", host_img);
	Mat imageEnhance;
	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
	//Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 4, 0, 0, -1, 0);
	filter2D(h_result, imageEnhance, CV_8UC3, kernel);
	imshow("������˹����ͼ����ǿЧ��", imageEnhance);

	waitKey(0);
	return 0;
}

//�Զ�������
int customCalculaters()
{
	Mat host_img = imread("F:/cuda_pictures/tree.jpg");
	if (!host_img.data)
	{
		cout << "��ȡͼƬ����������������ȷ·����\n";
		system("pause");
		return -1;
	}
	//������С
	GpuMat d_img, d_result;
	d_img.upload(host_img);
	int width = d_img.cols;
	int height = d_img.size().height;
	cuda::resize(d_img, d_result, cv::Size(0.7 * width, 0.7 * height), cv::INTER_LINEAR);
	Mat h_result;
	d_result.download(h_result);

	imshow("ԭͼ��", host_img);

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
	imshow("����ͼ����Ч��", result);

	waitKey(0);
	return 0;
}


//��ֵ������
Mat SrcImage;
Mat GrayImage;
Mat BinaryImage;
void on_trackbar(int pos, void*)
{
	//ת��Ϊ��ֵͼ
	cv::threshold(GrayImage, BinaryImage, pos, 255, CV_THRESH_BINARY);
	imshow("��ֵͼ", BinaryImage);
}
void to_GrayImage()
{
	//������ԭͼͬ���ͺ�ͬ��С�ľ���
	GrayImage.create(SrcImage.size(), SrcImage.type());
	//��ԭͼת��Ϊ�Ҷ�ͼ��
	cv::cvtColor(SrcImage, GrayImage, CV_BGR2GRAY);
	//imshow("�Ҷ�ͼ", GrayImage);
	//remove("F:/image_result/result.jpg");
	//imwrite("F:/image_result/result.jpg", GrayImage);

}
void creat_trackbar()
{
	int nThreshold = 0;
	createTrackbar("��ֵͼ��ֵ", "��ֵͼ", &nThreshold, 254, on_trackbar);
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
