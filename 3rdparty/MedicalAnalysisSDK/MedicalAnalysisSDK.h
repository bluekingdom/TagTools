/*!
 * \file MedicalAnalysisSDK.h
 * \date 2017/05/09 16:20
 *
 * \author blue
 * Contact: yang.wang@shangyiyun.com
 *
 * \brief 
 *
 * description: sdk header
 *
 * \note
*/

#pragma once

#include "ErrorCode.h"

#ifdef MEDICAL_ANALYSIS_SDK_EXPORTS
#define MEDICAL_ANALYSIS_SDK_API __declspec(dllexport) 
#else
#define MEDICAL_ANALYSIS_SDK_API __declspec(dllimport) 
#endif

#ifndef IN
#define IN
#endif

#ifndef OUT
#define OUT
#endif

#ifndef INOUT
#define INOUT
#endif

namespace SYY { 

	struct Rect { 
		int x, y, w, h; 
	};

	struct Image
	{
		Image() : pData(nullptr), nWidth(0), nHeight(0), nChannels(0) {}
		Image(char* p, int w, int h, int c) : pData(p), nWidth(w), nHeight(h), nChannels(c) {}

		char* pData;
		int nWidth;
		int nHeight;
		int nChannels;
	};

	typedef unsigned long long HANDLE;

	MEDICAL_ANALYSIS_SDK_API ErrorCode InitSDK();

	MEDICAL_ANALYSIS_SDK_API ErrorCode ReleaseSDK();

namespace Inpainting{
	enum InpaintMode
	{
		PatchMatch = 0x01,
		Criminisi_P1 = 0x10,
		Criminisi_P3 = 0x20,
		Criminisi_P5 = 0x40,
		Criminisi_P7 = 0x80,
	};

	MEDICAL_ANALYSIS_SDK_API ErrorCode InitInpaint(
		OUT HANDLE& hHandle,
		IN unsigned long nMode
		);
	MEDICAL_ANALYSIS_SDK_API ErrorCode ReleaseInpaint(
		INOUT HANDLE& hHandle
		);
	MEDICAL_ANALYSIS_SDK_API ErrorCode ExecuteInpaint(
		IN HANDLE hHandle,
		IN Image srcImg,
		IN Image maskImg,
		OUT Image& inpaintImg
		);
}

namespace MedicalAnalysis {

	struct BUAnalysisResult {
		BUAnalysisResult() : pLessionRects(nullptr), nLessionsCount(0) {}

		Rect rCropRect;				// 有效图片区域

		int nLessionsCount;			// 病灶数量
		Rect* pLessionRects;		// 病灶区域
	};

	enum BUAnalysisMode{
		None = 0x1,
		Crop_V1 = 0x2,
		Crop_V2 = 0x4,
	};


	MEDICAL_ANALYSIS_SDK_API ErrorCode InitBUAnalysis(
		OUT HANDLE& hHandle,
		IN unsigned long nMode
		);

	MEDICAL_ANALYSIS_SDK_API ErrorCode ReleaseBUAnalysis(
		INOUT HANDLE& hHandle
		);

	MEDICAL_ANALYSIS_SDK_API ErrorCode ExecuteBUAnalysis(
		IN HANDLE hHandle, 
		IN char* pImg, 
		IN int nImgWidth, 
		IN int nImgHeight,
		OUT BUAnalysisResult* pResult
		);
}
}
