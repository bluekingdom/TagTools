
// TagToolsDlg.cpp : ʵ���ļ�
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


// ����Ӧ�ó��򡰹��ڡ��˵���� CAboutDlg �Ի���

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// �Ի�������
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��

// ʵ��
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

// CTagToolsDlg �Ի���

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
	ON_WM_CHAR()
	ON_WM_MOUSEMOVE()
END_MESSAGE_MAP()


// CTagToolsDlg ��Ϣ�������

BOOL CTagToolsDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// ��������...���˵�����ӵ�ϵͳ�˵��С�

	// IDM_ABOUTBOX ������ϵͳ���Χ�ڡ�
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

	// ���ô˶Ի����ͼ�ꡣ  ��Ӧ�ó��������ڲ��ǶԻ���ʱ����ܽ��Զ�
	//  ִ�д˲���
	SetIcon(m_hIcon, TRUE);			// ���ô�ͼ��
	SetIcon(m_hIcon, FALSE);		// ����Сͼ��

	// TODO:  �ڴ���Ӷ���ĳ�ʼ������

	GetInitImageRect();

	auto listbox = (CListBox*)GetDlgItem(IDC_LIST_FILELIST);
	listbox->SetHorizontalExtent(1000);

	m_bEditMode = false;
	m_bAddMode = false;

	ShowMode();

	return TRUE;  // ���ǽ��������õ��ؼ������򷵻� TRUE
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

// �����Ի��������С����ť������Ҫ����Ĵ���
//  �����Ƹ�ͼ�ꡣ  ����ʹ���ĵ�/��ͼģ�͵� MFC Ӧ�ó���
//  �⽫�ɿ���Զ���ɡ�

void CTagToolsDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // ���ڻ��Ƶ��豸������

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// ʹͼ���ڹ����������о���
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// ����ͼ��
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//���û��϶���С������ʱϵͳ���ô˺���ȡ�ù��
//��ʾ��
HCURSOR CTagToolsDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CTagToolsDlg::OnBnClickedButton1()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	setlocale(LC_ALL, "Chinese-simplified");

	CString FilePathName;
	CEdit* pBoxOne;
	pBoxOne = (CEdit*)GetDlgItem(IDC_EDIT1);
	pBoxOne->GetWindowText(FilePathName);

	if (LoadMatFromRoot(FilePathName.GetString()))
	{
		RefreshListBox();
		for (int i = m_vTagged.size() - 1; i >= 0; i--)
		{
			if (true == m_vTagged[i])
			{
				m_nCurFileIdx = std::min(int(m_vTagged.size()) - 1, i + 1);
				break;
			}
		}
		PostChangeSel();
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
		MessageBox("��Ŀ¼û���ļ���");
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

	RefreshTagged();

	if (m_vFiles.size() == 0)
	{
		MessageBox("��Ŀ¼û���ļ���");
		return false;
	}

	return true;
}

void CTagToolsDlg::RefreshListBox()
{
	CListBox* listbox_filepath = (CListBox*)GetDlgItem(IDC_LIST_FILELIST); //ȡ����ʾ�ļ�·��LISTBOX���

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
	// TODO:  �ڴ���ӿؼ�֪ͨ����������

	if (m_vClickPointRects.size() != 0)
	{
		MessageBox("��������ǵð��س�����");
		return;
	}

	PreChangeSel();

	CListBox* listbox = (CListBox*)GetDlgItem(IDC_LIST_FILELIST);
	m_nCurFileIdx = listbox->GetCurSel();

	PostChangeSel();
}

bool CTagToolsDlg::ShowCurSelImg()
{
	//CListBox* listbox_filepath = (CListBox*)GetDlgItem(IDC_LIST_FILELIST);
	//int idx = listbox_filepath->GetCurSel();
	int idx = m_nCurFileIdx;

	CWnd *pWnd = GetDlgItem(IDC_IMAGE); //���pictrue�ؼ����ڵľ��   
	CRect rect;
	pWnd->GetClientRect(&rect);//���pictrue�ؼ����ڵľ�������   

	if (idx < 0)
	{
		return false;
	}

	std::string file = m_vFiles[idx];

	cv::Mat img = cv::imread(file);

	if (img.empty())
	{
		CString line;
		line.Format("���ܶ�ȡͼƬ�� %s", file.c_str());
		MessageBox(line);
		return false;
	}

	cv::Mat resize_img;

	if (m_vValidRects.size() == 1)
	{
		img = img(m_vValidRects[0]);
	}

	AdjustInputMatSizeAndRect(img, resize_img, rect);
	m_mResizeImg = resize_img;
	m_mOriImg = img.clone();

	//DrawClickPoints(resize_img);

	ShowImg(resize_img);

	return true;

}

bool CTagToolsDlg::ShowImg(const cv::Mat& src)
{
	CWnd *pWnd = GetDlgItem(IDC_IMAGE); //���pictrue�ؼ����ڵľ��   
	pWnd->UpdateWindow();

	pWnd->SetWindowPos(NULL, m_rImageInit.left, m_rImageInit.top, m_rImageInit.Width(), m_rImageInit.Height(), 
		SWP_NOZORDER | SWP_NOMOVE);

	CRect rect(0, 0, src.cols, src.rows);
	//pWnd->GetClientRect(&rect);//���pictrue�ؼ����ڵľ�������   

	CImage c_mat;
	MatToCImage(src, c_mat);

	CDC *pDC = pWnd->GetDC();//���pictrue�ؼ���DC   

	c_mat.Draw(pDC->m_hDC, rect); //��ͼƬ����Picture�ؼ���ʾ�ľ�������   

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

	cImage.Destroy();//��һ���Ƿ�ֹ�ظ���������ڴ�����  
	cImage.Create(width, height, 8 * channels);

	const uchar* ps;
	uchar* pimg = (uchar*)cImage.GetBits(); //��ȡCImage�����ش�������ָ��  
	int step = cImage.GetPitch();//ÿ�е��ֽ���,ע���������ֵ�����и�  

	// �����1��ͨ����ͼ��(�Ҷ�ͼ��) DIB��ʽ����Ҫ�Ե�ɫ������    
	// CImage�������˵�ɫ�壬����Ҫ�������и�ֵ��  
	if (1 == channels)
	{
		RGBQUAD* ColorTable;
		int MaxColors = 256;
		//�������ͨ��CI.GetMaxColorTableEntries()�õ���С(�������CI.Load����ͼ��Ļ�)    
		ColorTable = new RGBQUAD[MaxColors];
		cImage.GetColorTable(0, MaxColors, ColorTable);//������ȡ��ָ��    
		for (int i = 0; i<MaxColors; i++)
		{
			ColorTable[i].rgbBlue = (BYTE)i;
			//BYTE��ucharһ���£���MFC�ж�����    
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
	//��仰������������cimage��û�б���ֵ  
	//cImage.Save(_T("C:\\sample1024.bmp"));  
	//return;
}

void CTagToolsDlg::GetInitImageRect()
{
	CWnd *pWnd = GetDlgItem(IDC_IMAGE); //���pictrue�ؼ����ڵľ��   
	CRect rect, rect_windows;
	pWnd->GetClientRect(&rect);//���pictrue�ؼ����ڵľ������� 
	rect_windows = rect;
	ScreenToClient(&rect_windows);
	m_rImageInit = rect_windows;
	m_mBgMat = cv::Mat(rect.Height(), rect.Width(), CV_8UC3, cv::Scalar(255, 255, 255));
}

void CTagToolsDlg::OnLButtonDown(UINT nFlags, CPoint point)
{
	// TODO:  �ڴ������Ϣ�����������/�����Ĭ��ֵ

	m_pBegPt = point;
	m_bIsLBPushing = true;
	m_bIsMouseMoving = false;

	CDialogEx::OnLButtonDown(nFlags, point);
}

void CTagToolsDlg::OnLButtonUp(UINT nFlags, CPoint point)
{
	// TODO:  �ڴ������Ϣ�����������/�����Ĭ��ֵ

	if (m_mResizeImg.empty())
	{
		MessageBox("���ȵ��ͼƬ�б�!");
		m_bIsLBPushing = false;
		m_bIsMouseMoving = false;
		return;
	}

	if (true == m_bIsMouseMoving)
	{
		if (false == m_bEditMode)
		{
			AddRect(m_pBegPt, m_pCurPt);
		}
		else 
		{
			ReplaceRect(m_pBegPt, m_pCurPt);
		}
	}

	m_bIsLBPushing = false;
	m_bIsMouseMoving = false;

	RefreshRectList();
	Redraw();

	//cv::Mat drawImg = m_mResizeImg.clone();
	//DrawClickPoints(drawImg);

	//ShowImg(drawImg);

	CDialogEx::OnLButtonUp(nFlags, point);
}

void CTagToolsDlg::OnMouseMove(UINT nFlags, CPoint point)
{
	// TODO:  �ڴ������Ϣ�����������/�����Ĭ��ֵ

	m_pCurPt = point;
	if (m_bIsLBPushing)
	{
		cv::Rect rect(m_pBegPt.x, m_pBegPt.y, m_pCurPt.x - m_pBegPt.x, m_pCurPt.y - m_pBegPt.y);
		if (rect.area() > 5)
			m_bIsMouseMoving = true;
		Redraw();
	}

	CDialogEx::OnMouseMove(nFlags, point);
}

void CTagToolsDlg::DrawClickPointRects(cv::Mat& srcImg, int id_offset)
{
	for (auto rect: m_vClickPointRects)
	{
		ImgRect2SrceenRect(rect);
		cv::rectangle(srcImg, rect, cv::Scalar(0, 0, 255), 1);
	}

	const auto GetBBoxRect = [](const std::vector<cv::Rect2f>& vRects) -> cv::Rect {
		int min_x = INT_MAX, min_y = INT_MAX, max_x = INT_MIN, max_y = INT_MIN;

		const float margin = 1;
		for (auto rect: vRects)
		{
			auto pt = rect.tl();
			if (pt.x < min_x) min_x = pt.x - margin;
			if (pt.y < min_y) min_y = pt.y - margin;

			pt = rect.br();
			if (pt.x > max_x) max_x = pt.x + margin;
			if (pt.y > max_y) max_y = pt.y + margin;
		}

		return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
	};

	std::stringstream ss;

	int nPtVecLen = m_vPointRectVecs.size();

	for (int i = 0; i < nPtVecLen; i++)
	{
		for (auto rect : m_vPointRectVecs[i])
		{
			ImgRect2SrceenRect(rect);
			cv::rectangle(srcImg, rect, cv::Scalar(255, 0, 0), 1);
		}

		ss.str("");
		ss << i + id_offset;
		const auto& vRects = m_vPointRectVecs[i];
		auto rect = GetBBoxRect(vRects);
		ImgRect2SrceenRect(rect);
		cv::putText(srcImg, ss.str(), cv::Point(rect.x, rect.y - 2), 1, 1, cv::Scalar(255, 255, 255));
		cv::rectangle(srcImg, rect, cv::Scalar(0, 255, 255), 1);

	}
}

void CTagToolsDlg::AddClickPoint(const CPoint& point)
{
	//CRect Prect1;          //����ͼƬ�ľ���
	//CRect Prect;         //ͼƬ���ο�

	//GetDlgItem(IDC_IMAGE)->GetWindowRect(&Prect1);    //�õ�ͼƬ�ľ�//�δ�С
	//ScreenToClient(&Prect1);   //��ͼƬ��ľ��Ծ��δ�С

	////�ж��Ƿ���ͼƬ���ڣ���������ͼƬ���ڵĵ��
	//if (point.x<Prect1.left || point.x>Prect1.right || point.y<Prect1.top || point.y>Prect1.bottom)
	//	return;

	//m_vClickPointRects.push_back(cv::Point(point.x - Prect1.left, point.y - Prect1.top));
}


void CAboutDlg::OnRButtonUp(UINT nFlags, CPoint point)
{
	// TODO:  �ڴ������Ϣ�����������/�����Ĭ��ֵ

	CDialogEx::OnRButtonUp(nFlags, point);
}

void CTagToolsDlg::OnRButtonUp(UINT nFlags, CPoint point)
{
	// TODO:  �ڴ������Ϣ�����������/�����Ĭ��ֵ
	if (m_bAddMode == true)
	{
		WinPos2ScreenPos(point);
		cv::Point pt(point.x, point.y);

		auto temp = m_vAdditionRects;
		for (auto iter = temp.begin(); iter != temp.end(); iter++)
		{
			auto rect = *iter;
			ImgRect2SrceenRect(rect);
			if (rect.contains(pt))
			{
				temp.erase(iter);
				break;
			}
		}
		m_vAdditionRects = temp;
	}
	else 
	{
		if (m_vClickPointRects.size() > 0)
		{
			m_vClickPointRects.erase(m_vClickPointRects.end() - 1);
		}
	}

	Redraw();
	RefreshRectList();

	CDialogEx::OnRButtonUp(nFlags, point);
}


void CTagToolsDlg::OnBnClickedOk()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������

	if (m_vValidRects.size() == 0 && m_vClickPointRects.size() > 4)
	{
		MessageBox("��ô��󣬼ǵ��Ȱ��س���!");
		return;
	}

	if (m_vValidRects.size() == 0 && m_vClickPointRects.size() == 1)
	{
		m_vValidRects.push_back(m_vClickPointRects[0]);
		if (m_vValidRects.size() == 1)
		{
			for (auto& vRect : m_vPointRectVecs)
			{
				for (auto& rect : vRect)
				{
					rect.x -= m_vValidRects[0].x;
					rect.y -= m_vValidRects[0].y;
				}
			}

			for (auto& rect : m_vAdditionRects)
			{
				rect.x -= m_vValidRects[0].x;
				rect.y -= m_vValidRects[0].y;
			}		

		}

		RefreshTxtInfo();
		return;
	}
	else 
	{
		if (m_vClickPointRects.size() <= 2)
		{
			MessageBox("���ȵ������!");
			return;
		}

		m_vPointRectVecs.push_back(m_vClickPointRects);
	}

	m_vClickPointRects.clear();

	RefreshRectList();

	Redraw();
}


//void CTagToolsDlg::OnBnClickedDel()
//{
//	// TODO:  �ڴ���ӿؼ�֪ͨ����������
//
//	EndDialog(-1);
//}

void CTagToolsDlg::RefreshRectList()
{
	CListBox* listbox = (CListBox*)GetDlgItem(IDC_LIST_RECTLIST);

	listbox->ResetContent();

	int totalRectLen = m_vValidRects.size() + m_vPointRectVecs.size() + m_vAdditionRects.size();

	for (int i = 0; i < totalRectLen; i++)
	{
		CString line;
		line.Format("����: %d", i);
		listbox->InsertString(i, line);
	}

}

void CTagToolsDlg::Redraw()
{
	if (m_mResizeImg.empty())
		return;

	cv::Mat img = m_mResizeImg.clone();

	DrawDragRect(img);

	if (m_vValidRects.size() != 1)
		DrawRects(img, 0);

	int offset = m_vValidRects.size();
	DrawClickPointRects(img, offset);

	offset += m_vPointRectVecs.size();
	DrawAdditionRects(img, offset);

	ShowImg(img);
}

bool CTagToolsDlg::SaveRect2Txt()
{
	const std::string txtRoot = "RectTxt";
	if (m_vPointRectVecs.size() + m_vValidRects.size() == 0)
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

	auto vValidRects = m_vValidRects;
	auto vPointRectVecs = m_vPointRectVecs;
	auto vAdditionRects = m_vAdditionRects;

	if (vValidRects.size() == 1)
	{
		for (auto& vRect : vPointRectVecs)
		{
			for (auto& rect : vRect)
			{
				rect.x += vValidRects[0].x;
				rect.y += vValidRects[0].y;
			}
		}
		for (auto& rect : vAdditionRects)
		{
			rect.x += vValidRects[0].x;
			rect.y += vValidRects[0].y;
		}
	}

	std::string txtFilePath, errorMsg;
	if (false == SaveInfo2Txt(vPointRectVecs, vValidRects, vAdditionRects, relativePath, txtRoot, txtFilePath, errorMsg))
	{
		MessageBox(errorMsg.c_str());
		return false;
	}

	return true;
}

void CTagToolsDlg::ResetRectInfo()
{
	m_vPointRectVecs.clear();
	m_vClickPointRects.clear();
	m_vValidRects.clear();
	m_vAdditionRects.clear();
}

void CTagToolsDlg::OnBnClickedButtonDel()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	CListBox* listbox = (CListBox*)GetDlgItem(IDC_LIST_RECTLIST);

	int idx = listbox->GetCurSel();

	if (idx < 0)
	{
		MessageBox("�������б���ѡ��Ҫɾ��������");
		return;
	}

	int nRectLen = m_vValidRects.size();
	int nPtsRectLen = m_vPointRectVecs.size();

	if (idx < nRectLen) {
		if (nRectLen == 1)
		{
			for (auto& vRect : m_vPointRectVecs)
			{
				for (auto& rect : vRect)
				{
					rect.x += m_vValidRects[0].x;
					rect.y += m_vValidRects[0].y;
				}
			}

			for (auto& rect : m_vAdditionRects)
			{
				rect.x += m_vValidRects[0].x;
				rect.y += m_vValidRects[0].y;
			}

		}

		auto temp = m_vValidRects;

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

		m_vValidRects = temp;

		RefreshTxtInfo();
	}
	else if (idx >= nRectLen && idx < nRectLen + nPtsRectLen)
	{
		auto temp = m_vPointRectVecs;

		int nPtVecLen = m_vPointRectVecs.size();

		int cnt = 0;
		for (auto iter = temp.begin(); iter != temp.end(); iter++)
		{
			if (cnt++ != (idx - nRectLen))
			{
				continue;
			}

			temp.erase(iter);
			break;
		}

		m_vPointRectVecs = temp;
	}
	else
	{
		auto temp = m_vAdditionRects;

		int cnt = 0;
		for (auto iter = temp.begin(); iter != temp.end(); iter++)
		{
			if (cnt++ != (idx - nRectLen - nPtsRectLen))
			{
				continue;
			}

			temp.erase(iter);
			break;
		}

		m_vAdditionRects = temp;
	}

	RefreshRectList();
	Redraw();
}


void CTagToolsDlg::OnClose()
{
	// TODO:  �ڴ������Ϣ�����������/�����Ĭ��ֵ

	CDialogEx::OnClose();
}

bool CTagToolsDlg::LoadExistTxt(int curIdx)
{
	//int idx = m_nCurFileIdx;
	int idx = curIdx;

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

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects, vAddRects;
	std::string relativePathFromInfo, errorMsg;

	if (false == ParseTxtInfo(rectTxtFile, vPtsRects, vImgRects, vAddRects, relativePathFromInfo, errorMsg))
	{
		MessageBox(errorMsg.c_str());
		return false;
	}

	if (vImgRects.size() == 1)
	{
		auto imgRect = vImgRects[0];
		for (auto iter = vAddRects.begin(); iter != vAddRects.end();)
		{
			auto& rect = *iter;
			int reduce = 2;
			while (reduce-- > 0 &&
				(false == imgRect.contains(rect.tl()) || false == imgRect.contains(rect.br())))
			{
				rect.x -= imgRect.x;
				rect.y -= imgRect.y;
			}

			if (false == imgRect.contains(rect.tl()) || false == imgRect.contains(rect.br()))
			{
				iter = vAddRects.erase(iter);
			}
			else
			{
				iter++;
			}
		}
	}

	if (vImgRects.size() == 1)
	{
		for (auto& vRect : vPtsRects)
		{
			for (auto& rect : vRect)
			{
				rect.x -= vImgRects[0].x;
				rect.y -= vImgRects[0].y;
			}
		}

		for (auto& rect : vAddRects)
		{
			rect.x -= vImgRects[0].x;
			rect.y -= vImgRects[0].y;
		}
	}

	m_vPointRectVecs = vPtsRects;
	m_vValidRects = vImgRects;
	m_vAdditionRects = vAddRects;

	return true;
}

void CTagToolsDlg::OnKeyUp(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	// TODO:  �ڴ������Ϣ�����������/�����Ĭ��ֵ

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

void CTagToolsDlg::OnChar(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	// TODO:  �ڴ������Ϣ�����������/�����Ĭ��ֵ

	CDialogEx::OnChar(nChar, nRepCnt, nFlags);
}

//BOOL CTagToolsDlg::PreTranslateMessage(MSG* pMsg)
//{
//	// TODO:  �ڴ����ר�ô����/����û���
//
//	//CString keyCode;
//	//if (pMsg->message == WM_KEYUP && false == m_mResizeImg.empty())
//	//{
//	//	UINT nChar = (UINT)pMsg->wParam;
//	//	keyCode.Format("%c", nChar);
//	//	keyCode.MakeLower();
//	//	MessageBox(keyCode, "KeyPressed", MB_OK);
//	//}
//
//	return CDialogEx::PreTranslateMessage(pMsg);
//}


BOOL CTagToolsDlg::PreTranslateMessage(MSG* pMsg)
{
	// TODO:  �ڴ����ר�ô����/����û���

	CString keyCode;
	if (pMsg->message == WM_KEYDOWN && false == m_mResizeImg.empty())
	{
		UINT nChar = (UINT)pMsg->wParam;

		if (nChar == VK_RETURN)
		{
			OnBnClickedOk();
		}
		else if (nChar == VK_DOWN)
		{
			if (m_vClickPointRects.size() != 0)
			{
				MessageBox("��������ǵð��س�����");
				return TRUE;
			}
			PreChangeSel();

			m_nCurFileIdx = std::min(m_nCurFileIdx + 1, (int)m_vFiles.size() - 1);

			PostChangeSel();
		}
		else if (nChar == VK_UP)
		{
			if (m_vClickPointRects.size() != 0)
			{
				MessageBox("��������ǵð��س�����");
				return TRUE;
			}
			PreChangeSel();

			m_nCurFileIdx = std::max(0, m_nCurFileIdx - 1);

			PostChangeSel();
		}
		else if (nChar == 'p' || nChar == 'P')
		{
			m_bEditMode = !m_bEditMode;
			ShowMode();
		}
		else if (nChar == 'a' || nChar == 'A' || nChar == 'O' || nChar == 'o')
		{
			m_bAddMode = !m_bAddMode;
			ShowMode();
		}

		//keyCode.Format("%c", nChar);
		//keyCode.MakeLower();
		//MessageBox(keyCode, "KeyPressed", MB_OK);

		return TRUE;
	}
	else if (pMsg->message == WM_KEYUP)
	{
		UINT nChar = (UINT)pMsg->wParam;
		if (nChar == 'p' || nChar == 'P')
		{
			//m_bEditMode = false;
		}
	}
	else if (pMsg->message == WM_MOUSEWHEEL)
	{
		int zDelta = GET_WHEEL_DELTA_WPARAM(pMsg->wParam);

		if (zDelta < 0)
		{
			PreChangeSel();

			m_nCurFileIdx = std::min(m_nCurFileIdx + 1, (int)m_vFiles.size() - 1);

			PostChangeSel();
		}
		else if (zDelta > 0)
		{
			PreChangeSel();

			m_nCurFileIdx = std::max(0, m_nCurFileIdx - 1);

			PostChangeSel();
		}

	}

	return CDialogEx::PreTranslateMessage(pMsg);
}

void CTagToolsDlg::PreChangeSel()
{
	if (true == SaveRect2Txt())
		m_vTagged[m_nCurFileIdx] = true;

	RefreshCurListString();
}

void CTagToolsDlg::PostChangeSel()
{
	ResetRectInfo();

	if (LoadExistTxt(m_nCurFileIdx))
	{
		m_vTagged[m_nCurFileIdx] = true;
		RefreshCurListString();
	}

	ShowCurSelImg();

	CListBox* listbox = (CListBox*)GetDlgItem(IDC_LIST_FILELIST);
	listbox->SetCurSel(m_nCurFileIdx);
	listbox->UpdateData();

	RefreshRectList();

	ShowImg(m_mBgMat);
	Redraw();
}

void CTagToolsDlg::RefreshTagged()
{
	//std::vector<std::string> files;
	m_vTagged.resize(m_vFiles.size());

	for (int n = 0; n < m_vFiles.size(); n++)
	{
		if (LoadExistTxt(n))
		{
			m_vTagged[n] = true;
			//files.push_back(m_vFiles[n]);
		}
	}

	//m_vFiles = files;
	//m_vTagged.resize(m_vFiles.size());

	//for (int n = 0; n < m_vFiles.size(); n++)
	//{
	//	if (LoadExistTxt(n))
	//	{
	//		m_vTagged[n] = true;
	//	}
	//}
}

void CTagToolsDlg::DrawDragRect(cv::Mat& drawing)
{
	if (false == m_bIsLBPushing)
		return;

	auto begPt = m_pBegPt;
	auto endPt = m_pCurPt;

	WinPos2ScreenPos(begPt);
	WinPos2ScreenPos(endPt);

	cv::Rect rect(begPt.x, begPt.y, endPt.x - begPt.x, endPt.y - begPt.y);

	if (rect.area() > m_nDragMinArea)
	{
		cv::rectangle(drawing, rect, cv::Scalar(255, 0, 255), 2);
	}
}

void CTagToolsDlg::DrawRects(cv::Mat& srcImg, int id_offset)
{
	std::stringstream ss;
	for (int i = 0; i < m_vValidRects.size(); i++)
	{
		auto rect = m_vValidRects[i];
		ImgRect2SrceenRect(rect);
		ss.str("");
		ss << i + id_offset;
		cv::putText(srcImg, ss.str(), cv::Point(rect.x, rect.y - 2), 1, 1, cv::Scalar(255, 255, 255));
		cv::rectangle(srcImg, rect, cv::Scalar(0, 255, 0), 1);
	}
}

void CTagToolsDlg::AddRect(CPoint p1, CPoint p2)
{
	auto begPt = m_pBegPt;
	auto endPt = m_pCurPt;

	WinPos2ScreenPos(begPt);
	WinPos2ScreenPos(endPt);

	cv::Rect2f rect(begPt.x, begPt.y, endPt.x - begPt.x, endPt.y - begPt.y);

	if (rect.width < 0)
	{
		rect.x += rect.width;
		rect.width = -rect.width;
	}

	if (rect.height < 0)
	{
		rect.y += rect.height;
		rect.height = -rect.height;
	}

	ScreenRect2ImgRect(rect);

	if (rect.area() > m_nDragMinArea)
	{
		if (m_bAddMode)
			m_vAdditionRects.push_back(rect);
		else
			m_vClickPointRects.push_back(rect);
	}
}

void CTagToolsDlg::ImgRects2ScreenRects(std::vector<std::vector<cv::Rect>>& vPtsRects, std::vector<cv::Rect>& vImgRects)
{
	auto oriImgSize = m_mOriImg.size();
	auto resizeImgSize = m_mResizeImg.size();
	float radio_x = (float)oriImgSize.width / (float)resizeImgSize.width;
	float radio_y = (float)oriImgSize.height / (float)resizeImgSize.height;

	for (auto vRect: vPtsRects)
	{
		std::vector<cv::Rect2f> rects;
		for (auto rect: vRect)
		{
			float x = (rect.x / radio_x);
			float y = (rect.y / radio_y);
			float w = (rect.width / radio_x);
			float h = (rect.height / radio_y);

			rects.push_back(cv::Rect2f(x, y, w, h));
		}

		m_vPointRectVecs.push_back(rects);
	}

	for (auto& rect : vImgRects)
	{
		rect.x = (rect.x / radio_x );
		rect.y = (rect.y / radio_y );
		rect.width = (rect.width / radio_x );
		rect.height = (rect.height / radio_y );
	}
}

void CTagToolsDlg::ScreenRects2ImgRects(std::vector<std::vector<cv::Rect>>& vPtsRects, std::vector<cv::Rect>& vImgRects)
{
	auto oriImgSize = m_mOriImg.size();
	auto resizeImgSize = m_mResizeImg.size();
	float radio_x = (float)oriImgSize.width / (float)resizeImgSize.width;
	float radio_y = (float)oriImgSize.height / (float)resizeImgSize.height;

	for (auto& vRects : vPtsRects)
	{
		for (auto& rect : vRects)
		{
			rect.x = (rect.x * radio_x );
			rect.y = (rect.y * radio_y );
			rect.width = (rect.width * radio_x );
			rect.height = (rect.height * radio_y );
		}
	}

	for (auto& rect : vImgRects)
	{
		rect.x = (rect.x * radio_x );
		rect.y = (rect.y * radio_y );
		rect.width = (rect.width * radio_x );
		rect.height = (rect.height * radio_y );
	}
}

template<typename T>
void CTagToolsDlg::ImgRect2SrceenRect(T& rect)
{
	auto oriImgSize = m_mOriImg.size();
	auto resizeImgSize = m_mResizeImg.size();
	float radio_x = (float)oriImgSize.width / (float)resizeImgSize.width;
	float radio_y = (float)oriImgSize.height / (float)resizeImgSize.height;

	rect.x = (rect.x / radio_x);
	rect.y = (rect.y / radio_y);
	rect.width = (rect.width / radio_x);
	rect.height = (rect.height / radio_y);
}

template<typename T>
void CTagToolsDlg::ScreenRect2ImgRect(T& rect)
{
	auto oriImgSize = m_mOriImg.size();
	auto resizeImgSize = m_mResizeImg.size();
	float radio_x = (float)oriImgSize.width / (float)resizeImgSize.width;
	float radio_y = (float)oriImgSize.height / (float)resizeImgSize.height;

	rect.x = (rect.x * radio_x);
	rect.y = (rect.y * radio_y);
	rect.width = (rect.width * radio_x);
	rect.height = (rect.height * radio_y);
}

void CTagToolsDlg::WinPos2ScreenPos(CPoint& pt)
{
	CRect Prect1;          //����ͼƬ�ľ���

	GetDlgItem(IDC_IMAGE)->GetWindowRect(&Prect1);    //�õ�ͼƬ�ľ�//�δ�С
	ScreenToClient(&Prect1);   //��ͼƬ��ľ��Ծ��δ�С

	const auto SetPt = [&](CPoint& pt) {

		pt.x = std::min(std::max(pt.x, Prect1.left), Prect1.right);
		pt.y = std::min(std::max(pt.y, Prect1.top), Prect1.bottom);

		pt.x -= Prect1.left;
		pt.y -= Prect1.top;
	};

	SetPt(pt);
}

void CTagToolsDlg::ReplaceRect(CPoint p1, CPoint p2)
{
	auto begPt = m_pBegPt;
	auto endPt = m_pCurPt;

	WinPos2ScreenPos(begPt);
	WinPos2ScreenPos(endPt);

	cv::Rect2f rect(begPt.x, begPt.y, endPt.x - begPt.x, endPt.y - begPt.y);

	if (rect.width < 0)
	{
		rect.x += rect.width;
		rect.width = -rect.width;
	}

	if (rect.height < 0)
	{
		rect.y += rect.height;
		rect.height = -rect.height;
	}

	ScreenRect2ImgRect(rect);

	if (rect.area() <= m_nDragMinArea)
	{
		return;
	}

	cv::Rect2f* pRect = nullptr;
	float minDist = FLT_MAX;

	auto ct1 = (rect.tl() + rect.br()) / 2;

	if (false == m_bAddMode)
	{
		for (int ind1 = 0; ind1 < m_vPointRectVecs.size(); ind1++)
		{
			for (int ind2 = 0; ind2 < m_vPointRectVecs[ind1].size(); ind2++)
			{
				auto r = m_vPointRectVecs[ind1][ind2];

				auto ct2 = (r.tl() + r.br()) / 2;
				auto dd = (ct1 - ct2);
				float d = dd.dot(dd);

				if (d < minDist)
				{
					pRect = &m_vPointRectVecs[ind1][ind2];
					minDist = d;
				}
			}
		}
	}
	else 
	{
		for (int ind1 = 0; ind1 < m_vAdditionRects.size(); ind1++)
		{
			auto r = m_vAdditionRects[ind1];

			auto ct2 = (r.tl() + r.br()) / 2;
			auto dd = (ct1 - ct2);
			float d = dd.dot(dd);

			if (d < minDist)
			{
				pRect = &m_vAdditionRects[ind1];
				minDist = d;
			}
		}

	}

	if (!pRect)
	{
		MessageBox("�޷��ҵ�ƥ��ľ��ο��밴p���л�Ϊ '����' ģʽ��");
		return;
	}

	pRect->x = rect.x;
	pRect->y = rect.y;
	pRect->width = rect.width;
	pRect->height = rect.height;

}

void CTagToolsDlg::RefreshTxtInfo()
{
	if (true == SaveRect2Txt())
		m_vTagged[m_nCurFileIdx] = true;

	PostChangeSel();
}

void CTagToolsDlg::ShowMode()
{
	std::string text = "";
	if (m_bAddMode == true)
	{
		text = "���ӿ� ";
	}
	
	if (m_bEditMode == true)
	{
		text += "�༭";
	}
	else 
	{
		text += "����";
	}

	GetDlgItem(IDC_LABEL_MODE)->SetWindowText(text.c_str());
}

void CTagToolsDlg::DrawAdditionRects(cv::Mat& srcImg, int id_offset /*= 0*/)
{
	std::stringstream ss;

	int nPtVecLen = m_vAdditionRects.size();

	for (int i = 0; i < nPtVecLen; i++)
	{
		auto rect = m_vAdditionRects[i];
		ss.str("");
		ss << i + id_offset;
		ImgRect2SrceenRect(rect);
		cv::putText(srcImg, ss.str(), cv::Point(rect.x, rect.y - 2), 1, 1, cv::Scalar(255, 255, 255));
		cv::rectangle(srcImg, rect, cv::Scalar(255, 255, 0), 1);
	}
}

