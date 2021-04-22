
// MFCAppDlg.h: 头文件
//

#pragma once


// CMFCAppDlg 对话框
class CMFCAppDlg : public CDialogEx
{
// 构造
public:
	CMFCAppDlg(CWnd* pParent = nullptr);	// 标准构造函数

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_MFCAPP_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButton1();
	CEdit edit_input;
	afx_msg void OnBnClickedButton2();
	afx_msg void OnBnClickedButton3();
	afx_msg void OnBnClickedButton4();
	afx_msg void OnBnClickedlaplacianfilter();
	afx_msg void OnBnClickedaveragefilter();
	afx_msg void OnBnClickedgaussfilter();
	afx_msg void OnBnClickedmedianfilter();
	// 噪点个数
	CEdit text_num;
	afx_msg void OnBnClickedchoose();
	// 图片控制器，用于显示原图
	CStatic pic_src_control;
	CEdit text_filepath;
	afx_msg void OnBnClickedbinary();
	afx_msg void OnEnChangefilepath();
	// 图像处理结果显示
	CStatic pic_res_control;
};
