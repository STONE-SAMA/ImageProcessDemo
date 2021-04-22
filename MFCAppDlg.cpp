
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

	strFilePath = fileDlg.GetPathName();//文件路径
	strFileName = fileDlg.GetFileName();//文件名
	
	text_filepath.SetWindowTextA(strFilePath);

	//MessageBox(strFilePath);
	//CString EntName = fileDlg.GetFileExt();//文件扩展名

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

//锐化
void CMFCAppDlg::OnBnClickedlaplacianfilter()
{
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

//平滑
void CMFCAppDlg::OnBnClickedaveragefilter()
{
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

//高斯滤波
void CMFCAppDlg::OnBnClickedgaussfilter()
{
	void myGaussFilter(Mat & image);
	string str_src = strFilePath.GetBuffer(0);
	Mat host_src = imread(str_src, IMREAD_COLOR);
	if (!host_src.data)
	{
		MessageBox("读取图片错误，请重新输入正确路径！", "error");
		return;
	}
	myGaussFilter(host_src);
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
	int myPicErode(Mat & image);
	string str_src = strFilePath.GetBuffer(0);//CString转string
	Mat host_src = imread(str_src, IMREAD_GRAYSCALE);
	if (!host_src.data)
	{
		MessageBox("读取图片错误，请重新输入正确路径！", "error");
		return;
	}
	myPicErode(host_src);
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

//膨胀
void CMFCAppDlg::OnBnClickeddilate()
{
	int myPicDilate(Mat & image);
	string str_src = strFilePath.GetBuffer(0);//CString转string
	Mat host_src = imread(str_src, IMREAD_GRAYSCALE);
	if (!host_src.data)
	{
		MessageBox("读取图片错误，请重新输入正确路径！", "error");
		return;
	}
	myPicDilate(host_src);
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

//laplacian滤波
void CMFCAppDlg::OnBnClickedlaplacian()
{
	int myLaplacianFilter(Mat & image);

	string str_src = strFilePath.GetBuffer(0);//CString转string
	Mat host_src = imread(str_src, IMREAD_COLOR);
	if (!host_src.data)
	{
		MessageBox("读取图片错误，请重新输入正确路径！", "error");
		return;
	}
	myLaplacianFilter(host_src);
	
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

//sobel滤波
void CMFCAppDlg::OnBnClickedsobel()
{
	int mySobel(Mat & image);
	string str_src = strFilePath.GetBuffer(0);//CString转string
	Mat host_src = imread(str_src, IMREAD_COLOR);
	if (!host_src.data)
	{
		MessageBox("读取图片错误，请重新输入正确路径！", "error");
		return;
	}
	mySobel(host_src);

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
