
// TagToolsDlg.h : ͷ�ļ�
//

#pragma once

#include <string>

#include <opencv2/opencv.hpp>

// CTagToolsDlg �Ի���
class CTagToolsDlg : public CDialogEx
{
// ����
public:
	CTagToolsDlg(CWnd* pParent = NULL);	// ��׼���캯��

// �Ի�������
	enum { IDD = IDD_TAGTOOLS_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV ֧��


// ʵ��
protected:
	HICON m_hIcon;

	// ���ɵ���Ϣӳ�亯��
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButton1();

protected:
	bool LoadMatFromRoot(const std::string& root);
	void RefreshListBox();
	bool ShowCurSelImg();
	bool ShowImg(const cv::Mat& img);
	void GetInitImageRect();
	bool AdjustInputMatSizeAndRect(const cv::Mat& srcImg, cv::Mat& dstImg, CRect& rect);
	void MatToCImage(const cv::Mat& srcImg, CImage& cImage);
	void DrawClickPointRects(cv::Mat& srcImg, int id_offset = 0);
	void AddClickPoint(const CPoint& point);
	void RefreshRectList();
	void Redraw();
	bool SaveRect2Txt();
	void ResetRectInfo();
	bool LoadExistTxt(int curIdx);
	void RefreshCurListString();
	void PreChangeSel();
	void PostChangeSel();
	void RefreshTagged();
	void DrawDragRect(cv::Mat& drawing);
	void DrawRects(cv::Mat& srcImg, int id_offset = 0);
	void AddRect(CPoint p1, CPoint p2);


private:
	std::string m_sDicomRoot;
	std::vector<std::string> m_vFiles;
	CRect m_rImageInit;
	cv::Mat m_mBgMat;
	cv::Mat m_mOriImg;
	cv::Mat m_mResizeImg;
	//std::vector<cv::Mat> m_vMats;
	std::vector<cv::Rect2f> m_vClickPointRects;
	std::vector<std::vector<cv::Rect2f>> m_vPointRectVecs;
	int m_nCurFileIdx;
	std::vector<bool> m_vTagged;
	bool m_bIsLBPushing; // �������Ƿ����ڰ���
	bool m_bIsMouseMoving; // ����ƶ���
	CPoint m_pBegPt; // ����������λ��
	CPoint m_pCurPt; // ����϶�λ��
	std::vector<cv::Rect2f> m_vValidRects; // 

	const int m_nDragMinArea = 10;

public:
	const std::string c_sRectTxtPath = "RectTxt"; // ��ע�ļ�Ŀ¼

public:
	afx_msg void OnLbnSelchangeListFilelist();
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnLButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnBnClickedOk();
//	afx_msg void OnBnClickedDel();
	afx_msg void OnBnClickedButtonDel();
	afx_msg void OnClose();
	afx_msg void OnKeyUp(UINT nChar, UINT nRepCnt, UINT nFlags);
	afx_msg void OnChar(UINT nChar, UINT nRepCnt, UINT nFlags);
//	virtual BOOL PreTranslateMessage(MSG* pMsg);
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
};
