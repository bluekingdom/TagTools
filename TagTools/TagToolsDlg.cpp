
// TagToolsDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "TagTools.h"
#include "TagToolsDlg.h"
#include "afxdialogex.h"

#include <sstream>
#include <fstream>
#include <boost/filesystem.hpp>
#include "../utils/utils.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
	ON_WM_RBUTTONUP()
END_MESSAGE_MAP()


// CTagToolsDlg 对话框



CTagToolsDlg::CTagToolsDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CTagToolsDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CTagToolsDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CTagToolsDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CTagToolsDlg::OnBnClickedButton1)
	ON_LBN_SELCHANGE(IDC_LIST_FILELIST, &CTagToolsDlg::OnLbnSelchangeListFilelist)
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONUP()
	ON_WM_RBUTTONUP()
	ON_BN_CLICKED(IDOK, &CTagToolsDlg::OnBnClickedOk)
//	ON_BN_CLICKED(IDDEL, &CTagToolsDlg::OnBnClickedDel)
	ON_BN_CLICKED(IDC_BUTTON_DEL, &CTagToolsDlg::OnBnClickedButtonDel)
	ON_WM_CLOSE()
	ON_WM_KEYUP()
END_MESSAGE_MAP()


// CTagToolsDlg 消息处理程序

BOOL CTagToolsDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO:  在此添加额外的初始化代码

	GetInitImageRect();

	auto listbox = (CListBox*)GetDlgItem(IDC_LIST_FILELIST);
	listbox->SetHorizontalExtent(1000);

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CTagToolsDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CTagToolsDlg::OnPaint()
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
HCURSOR CTagToolsDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CTagToolsDlg::OnBnClickedButton1()
{
	// TODO:  在此添加控件通知处理程序代码
	setlocale(LC_ALL, "Chinese-simplified");

	CString FilePathName;
	CEdit* pBoxOne;
	pBoxOne = (CEdit*)GetDlgItem(IDC_EDIT1);
	pBoxOne->GetWindowText(FilePathName);

	if (LoadMatFromRoot(FilePathName.GetString()))
	{
		RefreshListBox();
		((CListBox*)GetDlgItem(IDC_LIST_FILELIST))->SetCurSel(0);
		OnLbnSelchangeListFilelist();
	}
	else {
		return;
	}

}

bool CTagToolsDlg::LoadMatFromRoot(const std::string& root)
{
	m_sDicomRoot = root;

	std::vector<std::string> vFiles;

	scanFilesUseRecursive(m_sDicomRoot, vFiles);

	int nFilesLen = vFiles.size();
	if (nFilesLen == 0)
	{
		MessageBox("该目录没有文件！");
		return false;
	}

	//m_vMats.clear();
	for (const auto& file : vFiles)
	{
		int idx = file.rfind('.');
		if (idx == -1)
			continue;

		int len = file.size();
		if (len - idx != 4)
			continue;

		std::string ext = file.substr(idx + 1);
		
		if (ext != "jpg")
		{
			continue;
		}

		//cv::Mat img = cv::imread(file);
		//if (img.empty()) continue;

		m_vFiles.push_back(file);
		//m_vMats.push_back(img);
	}

	m_vTagged.resize(m_vFiles.size(), false);

	return true;
}

void CTagToolsDlg::RefreshListBox()
{
	CListBox* listbox_filepath = (CListBox*)GetDlgItem(IDC_LIST_FILELIST); //取得显示文件路径LISTBOX句柄

	listbox_filepath->ResetContent();
	int nFilesLen = m_vFiles.size();
	for (int i = 0; i < nFilesLen; i++)
	{
		CString file = CString(m_vFiles[i].c_str());

		CString line;
		if (m_vTagged[i])
			line.Format("t(%d) %s", i + 1, file);
		else
			line.Format(" (%d) %s", i + 1, file);

		listbox_filepath->InsertString(i, line);
	}
}



void CTagToolsDlg::OnLbnSelchangeListFilelist()
{
	// TODO:  在此添加控件通知处理程序代码
	if (true == SaveRect2Txt())
		m_vTagged[m_nCurFileIdx] = true;

	RefreshCurListString();

	CListBox* listbox = (CListBox*)GetDlgItem(IDC_LIST_FILELIST);
	m_nCurFileIdx = listbox->GetCurSel();

	ResetRectInfo();

	ShowCurSelImg();

	if (LoadExistTxt())
	{
		m_vTagged[m_nCurFileIdx] = true;
		RefreshCurListString();
	}

	listbox->SetCurSel(m_nCurFileIdx);

	RefreshRectList();

	Redraw();
}

bool CTagToolsDlg::ShowCurSelImg()
{
	//CListBox* listbox_filepath = (CListBox*)GetDlgItem(IDC_LIST_FILELIST);
	//int idx = listbox_filepath->GetCurSel();
	int idx = m_nCurFileIdx;

	CWnd *pWnd = GetDlgItem(IDC_IMAGE); //获得pictrue控件窗口的句柄   
	CRect rect;
	pWnd->GetClientRect(&rect);//获得pictrue控件所在的矩形区域   

	if (idx < 0)
	{
		return false;
	}

	std::string file = m_vFiles[idx];

	cv::Mat img = cv::imread(file);

	if (img.empty())
	{
		CString line;
		line.Format("不能读取图片： %s", file.c_str());
		MessageBox(line);
		return false;
	}

	cv::Mat resize_img;
	AdjustInputMatSizeAndRect(img, resize_img, rect);
	m_mResizeImg = resize_img;
	m_mOriImg = img.clone();

	//DrawClickPoints(resize_img);

	ShowImg(resize_img);

	return true;

}

bool CTagToolsDlg::ShowImg(const cv::Mat& src)
{
	CWnd *pWnd = GetDlgItem(IDC_IMAGE); //获得pictrue控件窗口的句柄   
	pWnd->UpdateWindow();

	pWnd->SetWindowPos(NULL, m_rImageInit.left, m_rImageInit.top, m_rImageInit.Width(), m_rImageInit.Height(), 
		SWP_NOZORDER | SWP_NOMOVE);

	CRect rect(0, 0, src.cols, src.rows);
	//pWnd->GetClientRect(&rect);//获得pictrue控件所在的矩形区域   

	CImage c_mat;
	MatToCImage(src, c_mat);

	CDC *pDC = pWnd->GetDC();//获得pictrue控件的DC   

	c_mat.Draw(pDC->m_hDC, rect); //将图片画到Picture控件表示的矩形区域   

	ReleaseDC(pDC);

	return true;
}

bool CTagToolsDlg::AdjustInputMatSizeAndRect(const cv::Mat& src, cv::Mat&dst, CRect& rect)
{
	double m_rate = double(src.cols) / double(src.rows);
	double cr_rate = double(rect.Width()) / double(rect.Height());
	int width = 0, height = 0;
	if (cr_rate >= m_rate)
	{
		height = rect.Height();
		width = height*m_rate;
	}
	else
	{
		width = rect.Width();
		height = width / m_rate;
	}

	rect.bottom = rect.top + height;
	rect.right = rect.left + width;

	resize(src, dst, cv::Size(width, height), cv::INTER_LANCZOS4);

	return true;
}

void CTagToolsDlg::MatToCImage(const cv::Mat& mat, CImage& cImage)
{
	int width = mat.cols;
	int height = mat.rows;
	int channels = mat.channels();

	cImage.Destroy();//这一步是防止重复利用造成内存问题  
	cImage.Create(width, height, 8 * channels);

	const uchar* ps;
	uchar* pimg = (uchar*)cImage.GetBits(); //获取CImage的像素存贮区的指针  
	int step = cImage.GetPitch();//每行的字节数,注意这个返回值有正有负  

	// 如果是1个通道的图像(灰度图像) DIB格式才需要对调色板设置    
	// CImage中内置了调色板，我们要对他进行赋值：  
	if (1 == channels)
	{
		RGBQUAD* ColorTable;
		int MaxColors = 256;
		//这里可以通过CI.GetMaxColorTableEntries()得到大小(如果你是CI.Load读入图像的话)    
		ColorTable = new RGBQUAD[MaxColors];
		cImage.GetColorTable(0, MaxColors, ColorTable);//这里是取得指针    
		for (int i = 0; i<MaxColors; i++)
		{
			ColorTable[i].rgbBlue = (BYTE)i;
			//BYTE和uchar一回事，但MFC中都用它    
			ColorTable[i].rgbGreen = (BYTE)i;
			ColorTable[i].rgbRed = (BYTE)i;
		}
		cImage.SetColorTable(0, MaxColors, ColorTable);
		delete[]ColorTable;
	}


	for (int i = 0; i < height; i++)
	{
		ps = mat.ptr<uchar>(i);
		for (int j = 0; j < width; j++)
		{
			if (1 == channels)
			{
				*(pimg + i*step + j) = ps[j];
				//*(pimg + i*step + j) = 105;  
			}
			else if (3 == channels)
			{
				*(pimg + i*step + j * 3) = ps[j * 3];
				*(pimg + i*step + j * 3 + 1) = ps[j * 3 + 1];
				*(pimg + i*step + j * 3 + 2) = ps[j * 3 + 2];
			}
		}
	}
	//string str = CString2StdString(_T("C:\\sample1020.bmp"));  
	//imwrite(str,mat);  
	//这句话就是用来测试cimage有没有被赋值  
	//cImage.Save(_T("C:\\sample1024.bmp"));  
	//return;
}

void CTagToolsDlg::GetInitImageRect()
{
	CWnd *pWnd = GetDlgItem(IDC_IMAGE); //获得pictrue控件窗口的句柄   
	CRect rect, rect_windows;
	pWnd->GetClientRect(&rect);//获得pictrue控件所在的矩形区域 
	rect_windows = rect;
	ScreenToClient(&rect_windows);
	m_rImageInit = rect_windows;
	m_mBgMat = cv::Mat(rect.Height(), rect.Width(), CV_8UC3, cv::Scalar(255, 255, 255));
}

void CTagToolsDlg::OnLButtonDown(UINT nFlags, CPoint point)
{
	// TODO:  在此添加消息处理程序代码和/或调用默认值

	CDialogEx::OnLButtonDown(nFlags, point);
}

void CTagToolsDlg::OnLButtonUp(UINT nFlags, CPoint point)
{
	// TODO:  在此添加消息处理程序代码和/或调用默认值

	if (m_mResizeImg.empty())
	{
		MessageBox("请先点击图片列表!");
		return;
	}

	AddClickPoint(point);

	cv::Mat drawImg = m_mResizeImg.clone();
	DrawClickPoints(drawImg);

	ShowImg(drawImg);

	CDialogEx::OnLButtonUp(nFlags, point);
}

void CTagToolsDlg::DrawClickPoints(cv::Mat& srcImg)
{
	for (const auto& point : m_vClickPoints)
	{
		cv::circle(srcImg, point, 5, cv::Scalar(0, 0, 255), 5, -1);
	}

	const auto GetRect = [](const std::vector<cv::Point>& vPts) -> cv::Rect {
		int min_x = INT_MAX, min_y = INT_MAX, max_x = INT_MIN, max_y = INT_MIN;

		for (auto pt : vPts)
		{
			if (pt.x < min_x) min_x = pt.x;
			if (pt.x > max_x) max_x = pt.x;
			if (pt.y < min_y) min_y = pt.y;
			if (pt.y > max_y) max_y = pt.y;
		}

		return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
	};

	std::stringstream ss;

	int nPtVecLen = m_vPointVecs.size();

	for (int i = 0; i < nPtVecLen; i++)
	{
		ss.str("");
		ss << i;
		const auto& vPts = m_vPointVecs[i];
		auto rect = GetRect(vPts);
		cv::putText(srcImg, ss.str(), cv::Point(rect.x, rect.y - 2), 1, 1, cv::Scalar(255, 255, 255));
		cv::rectangle(srcImg, rect, cv::Scalar(255, 0, 0), 2);
	}
}

void CTagToolsDlg::AddClickPoint(const CPoint& point)
{
	CRect Prect1;          //定义图片的矩形
	CRect Prect;         //图片矩形框

	GetDlgItem(IDC_IMAGE)->GetWindowRect(&Prect1);    //得到图片的矩//形大小
	ScreenToClient(&Prect1);   //将图片框的绝对矩形大小

	//判断是否在图片框内，不处理不在图片框内的点击
	if (point.x<Prect1.left || point.x>Prect1.right || point.y<Prect1.top || point.y>Prect1.bottom)
		return;

	m_vClickPoints.push_back(cv::Point(point.x - Prect1.left, point.y - Prect1.top));
}


void CAboutDlg::OnRButtonUp(UINT nFlags, CPoint point)
{
	// TODO:  在此添加消息处理程序代码和/或调用默认值

	CDialogEx::OnRButtonUp(nFlags, point);
}


void CTagToolsDlg::OnRButtonUp(UINT nFlags, CPoint point)
{
	// TODO:  在此添加消息处理程序代码和/或调用默认值
	m_vClickPoints.clear();

	ShowImg(m_mResizeImg.clone());

	CDialogEx::OnRButtonUp(nFlags, point);
}


void CTagToolsDlg::OnBnClickedOk()
{
	// TODO:  在此添加控件通知处理程序代码

	if (m_vClickPoints.size() <= 2)
	{
		MessageBox("请先点好区域!");
		return;
	}

	m_vPointVecs.push_back(m_vClickPoints);
	m_vClickPoints.clear();

	RefreshRectList();

	Redraw();
}


//void CTagToolsDlg::OnBnClickedDel()
//{
//	// TODO:  在此添加控件通知处理程序代码
//
//	EndDialog(-1);
//}

void CTagToolsDlg::RefreshRectList()
{
	CListBox* listbox = (CListBox*)GetDlgItem(IDC_LIST_RECTLIST);

	listbox->ResetContent();

	int nPtVecLen = m_vPointVecs.size();

	for (int i = 0; i < nPtVecLen; i++)
	{
		CString line;
		line.Format("区域: %d", i);
		listbox->InsertString(i, line);
	}
}

void CTagToolsDlg::Redraw()
{
	ShowImg(m_mBgMat);
	if (m_mResizeImg.empty())
		return;

	cv::Mat img = m_mResizeImg.clone();
	DrawClickPoints(img);
	ShowImg(img);
}

bool CTagToolsDlg::SaveRect2Txt()
{
	const std::string txtRoot = "RectTxt";
	if (m_vPointVecs.size() == 0)
	{
		return false;
	}

	//CListBox* listbox = (CListBox*)GetDlgItem(IDC_LIST_FILELIST);
	//int idx = listbox->GetCurSel();
	int idx = m_nCurFileIdx;

	if (idx < 0)
		return false;

	auto file = m_vFiles[idx];

	std::string relativePath = file.substr(m_sDicomRoot.size() + 1);

	idx = relativePath.rfind('\\');

	std::string parentPath = "";
	std::string filename = "";

	CString path;

	if (idx != -1)
	{
		parentPath = relativePath.substr(0, idx);
		filename = relativePath.substr(idx + 1, relativePath.find_last_of('.') - idx - 1);
		path.Format("%s\\%s", txtRoot.c_str(), parentPath.c_str());
	}
	else {
		filename = relativePath.substr(0, relativePath.find_last_of('.') - idx - 1);
		path.Format("%s", txtRoot.c_str());
	}

	if (false == PathIsDirectory(path))
	{
		boost::filesystem::create_directories(path.GetString());
	}

	std::string txtPath = path.GetString();
	txtPath += "\\" + filename + ".txt";

	std::ofstream out(txtPath);

	out << relativePath << std::endl;
	out << m_vPointVecs.size() << std::endl;

	const auto SetRect = [&](const std::vector<cv::Point>& vPts) {
		// todo 坐标转换

		auto oriImgSize = m_mOriImg.size();
		auto resizeImgSize = m_mResizeImg.size();
		float radio_x = (float)oriImgSize.width / (float)resizeImgSize.width;
		float radio_y = (float)oriImgSize.height / (float)resizeImgSize.height;
		
		for (const auto& pt : vPts)
		{
			out << int(pt.x * radio_x + 0.5f) << "," << int(pt.y * radio_y + 0.5f) << " ";
		}
	};

	for (const auto& vPts : m_vPointVecs)
	{
		SetRect(vPts);
		out << std::endl;
	}

	return true;
}

void CTagToolsDlg::ResetRectInfo()
{
	m_vPointVecs.clear();
	m_vClickPoints.clear();
}


void CTagToolsDlg::OnBnClickedButtonDel()
{
	// TODO:  在此添加控件通知处理程序代码
	CListBox* listbox = (CListBox*)GetDlgItem(IDC_LIST_RECTLIST);

	int idx = listbox->GetCurSel();

	if (idx < 0)
	{
		MessageBox("请先在列表中选择要删除的区域！");
		return;
	}

	auto temp = m_vPointVecs;

	int nPtVecLen = m_vPointVecs.size();

	int cnt = 0;
	for (auto iter = temp.begin(); iter != temp.end(); iter++)
	{
		if (cnt++ != idx)
		{ 
			continue;
		}

		temp.erase(iter);
		break;
	}

	m_vPointVecs = temp;

	RefreshRectList();
	Redraw();
}


void CTagToolsDlg::OnClose()
{
	// TODO:  在此添加消息处理程序代码和/或调用默认值

	CDialogEx::OnClose();
}

bool CTagToolsDlg::LoadExistTxt()
{
	int idx = m_nCurFileIdx;

	if (idx < 0)
		return false;

	auto file = m_vFiles[idx];
	auto relativePath = file.substr(m_sDicomRoot.size() + 1);

	idx = relativePath.rfind('\\');
	auto filename = relativePath.substr(idx + 1, relativePath.size() - idx - 5);
	relativePath = relativePath.substr(0, idx);


	auto rectTxtFile = c_sRectTxtPath + "\\" + relativePath + "\\" + filename + ".txt";

	if (false == boost::filesystem::exists(rectTxtFile))
		return false;

	std::vector<std::vector<cv::Point>> vRects;
	std::string relativePathFromInfo, errorMsg;

	if (false == ParseTxtInfo(rectTxtFile, vRects, relativePathFromInfo, errorMsg))
	{
		MessageBox(errorMsg.c_str());
		return false;
	}

	auto oriImgSize = m_mOriImg.size();
	auto resizeImgSize = m_mResizeImg.size();
	float radio_x = (float)oriImgSize.width / (float)resizeImgSize.width;
	float radio_y = (float)oriImgSize.height / (float)resizeImgSize.height;

	m_vPointVecs.clear();
	for (auto vPts : vRects)
	{
		std::vector<cv::Point> pts;
		for (auto pt : vPts)
		{
			int x = int(pt.x / radio_x);
			int y = int(pt.y / radio_y);

			pts.push_back(cv::Point(x, y));
		}
		m_vPointVecs.push_back(pts);
	}

	return true;
}


void CTagToolsDlg::OnKeyUp(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	// TODO:  在此添加消息处理程序代码和/或调用默认值

	//if (nChar == VK_RIGHT)
	//{
	//	MessageBox("");
	//}

	CDialogEx::OnKeyUp(nChar, nRepCnt, nFlags);
}

void CTagToolsDlg::RefreshCurListString()
{
	if (m_nCurFileIdx < 0)
		return;

	CListBox* listbox = (CListBox*)GetDlgItem(IDC_LIST_FILELIST);

	listbox->DeleteString(m_nCurFileIdx);
	CString file = CString(m_vFiles[m_nCurFileIdx].c_str());

	CString line;
	if (m_vTagged[m_nCurFileIdx])
		line.Format("t(%d) %s", m_nCurFileIdx + 1, file);
	else
		line.Format(" (%d) %s", m_nCurFileIdx + 1, file);

	listbox->InsertString(m_nCurFileIdx, line);
}

