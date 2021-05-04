
// MFCAppDlg.cpp: 实现文件
//

#include "pch.h"
#include "framework.h"
#include "MFCApp.h"
#include "MFCAppDlg.h"
#include "afxdialogex.h"

#include"CMydialog.h"

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

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace std;

// CMFCAppDlg 对话框

CMFCAppDlg::CMFCAppDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_MFCAPP_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CMFCAppDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, edit_zaodian, text_num);
	DDX_Control(pDX, pic_src, pic_src_control);
	DDX_Control(pDX, edit_filepath, text_filepath);
	DDX_Control(pDX, pic_res, pic_res_control);
	DDX_Control(pDX, edit_rate, text_rate);
	DDX_Control(pDX, edit_CPU_time, text_cpu_time);
	DDX_Control(pDX, edit_GPU_time, text_cuda_time);
}

BEGIN_MESSAGE_MAP(CMFCAppDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()

	ON_BN_CLICKED(Ibtn_laplacian_filter, &CMFCAppDlg::OnBnClickedlaplacianfilter)
	ON_BN_CLICKED(btn_average_filter, &CMFCAppDlg::OnBnClickedaveragefilter)
	ON_BN_CLICKED(btn_gauss_filter, &CMFCAppDlg::OnBnClickedgaussfilter)
	ON_BN_CLICKED(btn_median_filter, &CMFCAppDlg::OnBnClickedmedianfilter)
	ON_BN_CLICKED(btn_choose, &CMFCAppDlg::OnBnClickedchoose)
	ON_BN_CLICKED(btn_binary, &CMFCAppDlg::OnBnClickedbinary)
	ON_BN_CLICKED(btn_erode, &CMFCAppDlg::OnBnClickederode)
	ON_BN_CLICKED(btn_dilate, &CMFCAppDlg::OnBnClickeddilate)
	ON_BN_CLICKED(btn_laplacian, &CMFCAppDlg::OnBnClickedlaplacian)
	ON_BN_CLICKED(btn_sobel, &CMFCAppDlg::OnBnClickedsobel)
	ON_BN_CLICKED(btn_invisible, &CMFCAppDlg::OnBnClickedinvisible)
	ON_BN_CLICKED(btn_load_result, &CMFCAppDlg::OnBnClickedloadresult)
	ON_BN_CLICKED(btn_gray_histogram, &CMFCAppDlg::OnBnClickedgrayhistogram)
	ON_BN_CLICKED(btn_rgb_histogram, &CMFCAppDlg::OnBnClickedrgbhistogram)
END_MESSAGE_MAP()


// CMFCAppDlg 消息处理程序

BOOL CMFCAppDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 初始化代码
	SetDlgItemText(edit_zaodian, "10000");

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CMFCAppDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CMFCAppDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


/*
void CMFCAppDlg::OnBnClickedButton1()
{
	// TODO: 在此添加控件通知处理程序代码
	CMydialog mydlg;
	mydlg.DoModal();
}
*/

//图片文件路径
CString strFilePath;
//图片文件名
CString strFileName;

//选取图片
void CMFCAppDlg::OnBnClickedchoose()
{
	//选取图片
	CFileDialog fileDlg(TRUE, _T("jpg"), NULL, 0, _T("image Files(*.bmp; *.jpg;*.png)|*.JPG;*.PNG;*.BMP|All Files (*.*) |*.*|"), this);
	fileDlg.DoModal();
	
	//UpdateWindow();
	//MessageBox(strFilePath);
	//CString EntName = fileDlg.GetFileExt();//文件扩展名

	if (fileDlg.GetPathName().IsEmpty())
	{
		MessageBox("选取图片不能为空！", "error");
	}
	else
	{
		strFilePath = fileDlg.GetPathName();//文件路径
		strFileName = fileDlg.GetFileName();//文件名

		text_filepath.SetWindowTextA(strFilePath);

		CRect rect;
		CImage image;
		image.Load(strFilePath);
		int cx = image.GetWidth();
		int cy = image.GetHeight();

		CWnd* pWnd = NULL;
		pWnd = GetDlgItem(pic_src);//获取控件句柄
		//获取Picture Control控件的区域的大小  
		pWnd->GetClientRect(&rect);
		CDC* pDc = NULL;
		pDc = pWnd->GetDC();//获取picture control的DC  
		//设置指定设备环境中的位图拉伸模式
		int ModeOld = SetStretchBltMode(pDc->m_hDC, STRETCH_HALFTONE);
		//从源矩形中复制一个位图到目标矩形，按目标设备设置的模式进行图像的拉伸或压缩
		image.StretchBlt(pDc->m_hDC, rect, SRCCOPY);
		SetStretchBltMode(pDc->m_hDC, ModeOld);
		//释放资源
		ReleaseDC(pDc);
	}

	//重启显示结果的图像控件，实现清除图像
	GetDlgItem(pic_res)->ShowWindow(FALSE);
	GetDlgItem(pic_res)->ShowWindow(TRUE);
}


//锐化
void CMFCAppDlg::OnBnClickedlaplacianfilter()
{
	OnBnClickedinvisible();
	//拉普拉斯锐化
	int myLaplacianSharpen(Mat & image);
	string str_src = strFilePath.GetBuffer(0);
	Mat host_src = imread(str_src, IMREAD_COLOR);
	if (!host_src.data)
	{
		MessageBox("读取图片错误，请重新输入正确路径！", "error");
		return;
	}

	myLaplacianSharpen(host_src);

	OnBnClickedloadresult();

}

//平滑
void CMFCAppDlg::OnBnClickedaveragefilter()
{
	OnBnClickedinvisible();
	//平均滤波
	void myAverageFilter(Mat & image);
	string str_src = strFilePath.GetBuffer(0);
	Mat host_src = imread(str_src, IMREAD_COLOR);
	if (!host_src.data)
	{
		MessageBox("读取图片错误，请重新输入正确路径！", "error");
		return;
	}
	myAverageFilter(host_src);

	OnBnClickedloadresult();
}

//高斯滤波
void CMFCAppDlg::OnBnClickedgaussfilter()
{
	OnBnClickedinvisible();

	void myGaussFilter(Mat & image);
	string str_src = strFilePath.GetBuffer(0);
	Mat host_src = imread(str_src, IMREAD_COLOR);
	if (!host_src.data)
	{
		MessageBox("读取图片错误，请重新输入正确路径！", "error");
		return;
	}
	myGaussFilter(host_src);
	OnBnClickedloadresult();

}

//中值滤波，并去除椒盐噪声
void CMFCAppDlg::OnBnClickedmedianfilter()
{
	CString str_text;
	text_num.GetWindowTextA(str_text);
	int num = _ttoi(str_text);//CString转int
	string str_src = strFilePath.GetBuffer(0);//CString转string
	Mat host_src = imread(str_src, IMREAD_COLOR);
	if (!host_src.data)
	{
		MessageBox("读取图片错误，请重新输入正确路径！", "error");
		return;
	}
	void salt(Mat & image, int n);
	salt(host_src, num);

	CString result_path1 = "F:/image_result/result_salt1.jpg";
	CRect rect1;
	CImage image1;
	image1.Load(result_path1);
	int cx1 = image1.GetWidth();
	int cy1 = image1.GetHeight();
	CWnd* pWnd1 = GetDlgItem(pic_src);//获取控件句柄
	//获取Picture Control控件的区域的大小  
	pWnd1->GetClientRect(&rect1);
	CDC* pDc1 = pWnd1->GetDC();//获取picture control的DC  
	//设置指定设备环境中的位图拉伸模式
	int ModeOld = SetStretchBltMode(pDc1->m_hDC, STRETCH_HALFTONE);
	//从源矩形中复制一个位图到目标矩形，按目标设备设置的模式进行图像的拉伸或压缩
	image1.StretchBlt(pDc1->m_hDC, rect1, SRCCOPY);
	SetStretchBltMode(pDc1->m_hDC, ModeOld);
	//释放资源
	ReleaseDC(pDc1);

	int  MedianProcess(Mat & host_src);
	MedianProcess(host_src);

	CString result_path2 = "F:/image_result/result_salt2.jpg";
	CRect rect2;
	CImage image2;
	image2.Load(result_path2);
	int cx2 = image2.GetWidth();
	int cy2 = image2.GetHeight();
	CWnd* pWnd2 = GetDlgItem(pic_res);//获取控件句柄
	//获取Picture Control控件的区域的大小  
	pWnd2->GetClientRect(&rect2);
	CDC* pDc2 = pWnd2->GetDC();//获取picture control的DC  
	//设置指定设备环境中的位图拉伸模式
	ModeOld = SetStretchBltMode(pDc2->m_hDC, STRETCH_HALFTONE);
	//从源矩形中复制一个位图到目标矩形，按目标设备设置的模式进行图像的拉伸或压缩
	image2.StretchBlt(pDc2->m_hDC, rect2, SRCCOPY);
	SetStretchBltMode(pDc2->m_hDC, ModeOld);
	//释放资源
	ReleaseDC(pDc2);
}

// 二值化
void CMFCAppDlg::OnBnClickedbinary()
{
	OnBnClickedinvisible();

	int myBinaryProcess(Mat & image);
	string str_src = strFilePath.GetBuffer(0);//CString转string
	Mat host_src = imread(str_src, IMREAD_COLOR);
	if (!host_src.data)
	{
		MessageBox("读取图片错误，请重新输入正确路径！", "error");
		return;
	}

	remove("F:/image_result/result.jpg");
	GpuMat src, gray;
	src.upload(host_src);
	cuda::cvtColor(src, gray, COLOR_BGR2GRAY);
	Mat gray_host;
	gray.download(gray_host);
	imwrite("F:/image_result/result.jpg", gray_host);
	CString result_path = "F:/image_result/result.jpg";
	CRect rect;
	CImage image;
	image.Load(result_path);
	int cx = image.GetWidth();
	int cy = image.GetHeight();
	CWnd* pWnd = GetDlgItem(pic_res);//获取控件句柄
	//获取Picture Control控件的区域的大小  
	pWnd->GetClientRect(&rect);
	CDC* pDc = pWnd->GetDC();//获取picture control的DC  
	//设置指定设备环境中的位图拉伸模式
	int ModeOld = SetStretchBltMode(pDc->m_hDC, STRETCH_HALFTONE);
	//从源矩形中复制一个位图到目标矩形，按目标设备设置的模式进行图像的拉伸或压缩
	image.StretchBlt(pDc->m_hDC, rect, SRCCOPY);
	SetStretchBltMode(pDc->m_hDC, ModeOld);
	//释放资源
	ReleaseDC(pDc);

	myBinaryProcess(host_src);
}

//腐蚀
void CMFCAppDlg::OnBnClickederode()
{
	OnBnClickedinvisible();
	int myPicErode(Mat & image);
	string str_src = strFilePath.GetBuffer(0);//CString转string
	Mat host_src = imread(str_src, IMREAD_GRAYSCALE);
	if (!host_src.data)
	{
		MessageBox("读取图片错误，请重新输入正确路径！", "error");
		return;
	}
	myPicErode(host_src);
	OnBnClickedloadresult();
}

//膨胀
void CMFCAppDlg::OnBnClickeddilate()
{
	OnBnClickedinvisible();
	int myPicDilate(Mat & image);
	string str_src = strFilePath.GetBuffer(0);//CString转string
	Mat host_src = imread(str_src, IMREAD_GRAYSCALE);
	if (!host_src.data)
	{
		MessageBox("读取图片错误，请重新输入正确路径！", "error");
		return;
	}
	myPicDilate(host_src);
	
	OnBnClickedloadresult();
}

//laplacian滤波
void CMFCAppDlg::OnBnClickedlaplacian()
{
	OnBnClickedinvisible();
	int myLaplacianFilter(Mat & image);

	string str_src = strFilePath.GetBuffer(0);//CString转string
	Mat host_src = imread(str_src, IMREAD_COLOR);
	if (!host_src.data)
	{
		MessageBox("读取图片错误，请重新输入正确路径！", "error");
		return;
	}
	myLaplacianFilter(host_src);
	
	OnBnClickedloadresult();
}

//sobel滤波
void CMFCAppDlg::OnBnClickedsobel()
{
	OnBnClickedinvisible();
	int mySobel(Mat & image);

	string str_src = strFilePath.GetBuffer(0);//CString转string
	Mat host_src = imread(str_src, IMREAD_COLOR);
	if (!host_src.data)
	{
		MessageBox("读取图片错误，请重新输入正确路径！", "error");
		return;
	}
	mySobel(host_src);

	OnBnClickedloadresult();

}

//重新加载原图，避免原图被修改
void CMFCAppDlg::OnBnClickedinvisible()
{
	string str_src = strFilePath.GetBuffer(0);//CString转string
	Mat host_src = imread(str_src, IMREAD_COLOR);
	if (!host_src.data)
	{
		//MessageBox("读取图片错误，请重新输入正确路径！", "error");
		return;
	}

	CRect rect;
	CImage image;
	image.Load(strFilePath);
	int cx = image.GetWidth();
	int cy = image.GetHeight();

	CWnd* pWnd = NULL;
	pWnd = GetDlgItem(pic_src);//获取控件句柄
	//获取Picture Control控件的区域的大小  
	pWnd->GetClientRect(&rect);
	CDC* pDc = NULL;
	pDc = pWnd->GetDC();//获取picture control的DC  
	//设置指定设备环境中的位图拉伸模式
	int ModeOld = SetStretchBltMode(pDc->m_hDC, STRETCH_HALFTONE);
	//从源矩形中复制一个位图到目标矩形，按目标设备设置的模式进行图像的拉伸或压缩
	image.StretchBlt(pDc->m_hDC, rect, SRCCOPY);
	SetStretchBltMode(pDc->m_hDC, ModeOld);
	//释放资源
	ReleaseDC(pDc);

	//重启显示结果的图像控件，实现清除图像
	GetDlgItem(pic_res)->ShowWindow(FALSE);
	GetDlgItem(pic_res)->ShowWindow(TRUE);
}

//加载结果
void CMFCAppDlg::OnBnClickedloadresult()
{
	CString result_path = "F:/image_result/result.jpg";
	CRect rect;
	CImage image;
	image.Load(result_path);
	int cx = image.GetWidth();
	int cy = image.GetHeight();
	CWnd* pWnd = GetDlgItem(pic_res);//获取控件句柄
	//获取Picture Control控件的区域的大小  
	pWnd->GetClientRect(&rect);
	CDC* pDc = pWnd->GetDC();//获取picture control的DC  
	//设置指定设备环境中的位图拉伸模式
	int ModeOld = SetStretchBltMode(pDc->m_hDC, STRETCH_HALFTONE);
	//从源矩形中复制一个位图到目标矩形，按目标设备设置的模式进行图像的拉伸或压缩
	image.StretchBlt(pDc->m_hDC, rect, SRCCOPY);
	SetStretchBltMode(pDc->m_hDC, ModeOld);
	//释放资源
	ReleaseDC(pDc);
}

//灰度直方图均衡
void CMFCAppDlg::OnBnClickedgrayhistogram()
{
	void drawHistogram(Mat & srcImage, string str);
	
	string str_src = strFilePath.GetBuffer(0);//CString转string
	Mat host_src = imread(str_src, IMREAD_GRAYSCALE);
	if (!host_src.data)
	{
		MessageBox("读取图片错误，请重新输入正确路径！", "error");
		return;
	}
	remove("F:/image_result/gray.jpg");
	imwrite("F:/image_result/gray.jpg",host_src);

	CRect rect;
	CImage image;
	image.Load("F:/image_result/gray.jpg");
	int cx = image.GetWidth();
	int cy = image.GetHeight();

	CWnd* pWnd = NULL;
	pWnd = GetDlgItem(pic_src);//获取控件句柄
	//获取Picture Control控件的区域的大小  
	pWnd->GetClientRect(&rect);
	CDC* pDc = NULL;
	pDc = pWnd->GetDC();//获取picture control的DC  
	//设置指定设备环境中的位图拉伸模式
	int ModeOld = SetStretchBltMode(pDc->m_hDC, STRETCH_HALFTONE);
	//从源矩形中复制一个位图到目标矩形，按目标设备设置的模式进行图像的拉伸或压缩
	image.StretchBlt(pDc->m_hDC, rect, SRCCOPY);
	SetStretchBltMode(pDc->m_hDC, ModeOld);
	//释放资源
	ReleaseDC(pDc);

	//CPU实现直方图均衡
	Mat h_img = host_src;
	clock_t start, end;
	start = clock();
	Mat cpu_result;
	cv::equalizeHist(h_img, cpu_result);
	end = clock();
	double cpu_time = double(end - start) / CLOCKS_PER_SEC;

	//CUDA环境初始化
	cv::Mat img_init = imread("F:/cuda_pictures/sea.jpg");
	GpuMat src_init, init_result;
	src_init.upload(img_init);
	cuda::cvtColor(src_init, init_result, cv::COLOR_BGR2HSV);
	//CUDA实现直方图均衡
	clock_t start_cuda, end_cuda;
	GpuMat d_img, d_result;
	//开始计时
	start_cuda = clock();
	d_img.upload(h_img);
	
	cv::cuda::equalizeHist(d_img, d_result);
	//结束计时
	end_cuda = clock();
	Mat h_result;
	d_result.download(h_result);
	//并行运算耗时
	double cuda_time = double(end_cuda - start_cuda) / CLOCKS_PER_SEC;
	//原始图像直方图
	drawHistogram(h_img, "原始图像直方图");
	//图像均衡后直方图
	drawHistogram(h_result, "图像均衡后直方图");
	remove("F:/image_result/result.jpg");
	imwrite("F:/image_result/result.jpg", h_result);
	
	//加速比
	double sp = cpu_time / cuda_time;

	//加载结果
	OnBnClickedloadresult();

	//显示耗时
	CString cstr;
	cstr.Format(_T("%.3lf"), cpu_time);
	text_cpu_time.SetWindowTextA(cstr);
	cstr.Format(_T("%.3lf"), cuda_time);
	text_cuda_time.SetWindowTextA(cstr);
	cstr.Format(_T("%.3lf"), sp);
	text_rate.SetWindowTextA(cstr);

	waitKey(0);
}

//RGB直方图均衡
void CMFCAppDlg::OnBnClickedrgbhistogram()
{
	OnBnClickedinvisible();
	void drawHistogram(Mat & srcImage, string str);
	
	string str_src = strFilePath.GetBuffer(0);//CString转string
	Mat host_src = imread(str_src, IMREAD_COLOR);
	if (!host_src.data)
	{
		MessageBox("读取图片错误，请重新输入正确路径！", "error");
		return;
	}
	
	//CPU实现
	cv::Mat h_img1 = host_src;
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

	//初始化
	cv::Mat img_init = imread("F:/cuda_pictures/sea.jpg");
	GpuMat src_init, init_result, res_init;
	src_init.upload(img_init);
	cuda::cvtColor(src_init, init_result, cv::COLOR_BGR2HSV);
	
	std::vector<GpuMat> vec_channels_init;
	cuda::split(init_result, vec_channels_init);
	cuda::equalizeHist(vec_channels_init[2], vec_channels_init[2]);
	cuda::merge(vec_channels_init, init_result);
	cuda::cvtColor(init_result, res_init, cv::COLOR_HSV2BGR);
	
	//CUDA实现直方图均衡
	GpuMat src, h_result_cuda, g_result;
	clock_t start_cuda, end_cuda;
	start_cuda = clock();//开始计时
	src.upload(h_img1);
	
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

	//加速比
	double sp = cpu_time / cuda_time;
	
	//绘制直方图
	drawHistogram(h_img1, "原图直方图");
	remove("F:/image_result/result.jpg");
	imwrite("F:/image_result/result.jpg", result);
	//绘制直方图
	drawHistogram(result, "cuda直方图");

	//加载结果
	OnBnClickedloadresult();

	//显示耗时
	CString cstr;
	cstr.Format(_T("%.3lf"), cpu_time);
	text_cpu_time.SetWindowTextA(cstr);
	cstr.Format(_T("%.3lf"), cuda_time);
	text_cuda_time.SetWindowTextA(cstr);
	cstr.Format(_T("%.3lf"), sp);
	text_rate.SetWindowTextA(cstr);

	waitKey(0);

}
