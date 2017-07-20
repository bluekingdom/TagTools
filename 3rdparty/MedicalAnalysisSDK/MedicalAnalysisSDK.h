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
#define MEDICAL_ANALYSIS_SDK_API extern "C" __declspec(dllexport) 
#else
#define MEDICAL_ANALYSIS_SDK_API extern "C" __declspec(dllimport) 
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

	enum LessionGrading
	{
		LG1a = 0,
		LG_OTHER = 1,
		//LG4a = 2,
		//LG4b = 3,
		//LG4c = 4,
		//LG5 = 5,
		//LG6 = 6,
	};
	enum LessionType
	{
		NO_LESSION = 0,
		LESSION = 1,
	};

	struct BUAnalysisResult {
		BUAnalysisResult() : nLessionsCount(0) {}

		const static int MAX_LEN = 128;

		Rect rCropRect;					// 有效图片区域

		LessionGrading nGrading;		// 病况分级

		int nLessionsCount;				// 病灶数量
		Rect pLessionRects[MAX_LEN];		// 病灶区域
		float pLessionConfidence[MAX_LEN];	// 病灶置信值
		LessionType pLessionTypes[MAX_LEN];	// 病灶类型
	};

	enum BUAnalysisMode{
		None = 0x1,
		Crop_V1 = 0x2,
		Crop_V2 = 0x4,
		Crop_V3 = 0x8,
		DetectMore = 0x10,
		DetectAccurate = 0x20,
	};

	MEDICAL_ANALYSIS_SDK_API ErrorCode InitBUAnalysisWithMode(
		OUT HANDLE& hHandle,
		IN unsigned long nMode
		);

	MEDICAL_ANALYSIS_SDK_API ErrorCode InitBUAnalysis(
		OUT HANDLE& hHandle
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

	MEDICAL_ANALYSIS_SDK_API ErrorCode ExecuteBUAnalysisFromFile(
		IN HANDLE hHandle, 
		IN Image* pImage, 
		OUT BUAnalysisResult* pResult
		);

	MEDICAL_ANALYSIS_SDK_API ErrorCode DrawResult2Image(
		INOUT Image* pImage,
		IN BUAnalysisResult* pResult
		);

}
}
