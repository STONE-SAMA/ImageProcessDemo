// CMydialog.cpp: 实现文件
//

#include "pch.h"
#include "MFCApp.h"
#include "CMydialog.h"
#include "afxdialogex.h"


// CMydialog 对话框

IMPLEMENT_DYNAMIC(CMydialog, CDialogEx)

CMydialog::CMydialog(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIALOG1, pParent)
{

}

CMydialog::~CMydialog()
{
}

void CMydialog::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CMydialog, CDialogEx)
END_MESSAGE_MAP()


// CMydialog 消息处理程序
