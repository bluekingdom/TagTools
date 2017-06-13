#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

const std::vector<std::string>& scanFilesUseRecursive(
	const std::string& rootPath, 
	std::vector<std::string>& container
	);

bool ParseTxtInfo(
	const std::string& txtFile,
	std::vector<std::vector<cv::Point>>& vPtss,
	std::vector<cv::Rect>& vRects,
	std::string& relativePath,
	std::string& errorMsg = std::string("")
	);

bool SaveInfo2Txt(
	const std::vector<std::vector<cv::Point>>& vPtss,
	const std::vector<cv::Rect>& vRects,
	const std::string& relativePath,
	const std::string& txtRoot,
	std::string& txtFile,
	std::string& errorMsg = std::string("")
	);

bool ParseTxtInfo(
	const std::string& txtFile,
	std::vector<std::vector<cv::Rect2f>>& vPtsRects,
	std::vector<cv::Rect2f>& vImgRects,
	std::string& relativePath,
	std::string& errorMsg = std::string("")
	);

bool SaveInfo2Txt(
	const std::vector<std::vector<cv::Rect2f>>& vPtsRects,
	const std::vector<cv::Rect2f>& vImgRects,
	const std::string& relativePath,
	const std::string& txtRoot,
	std::string& txtFile,
	std::string& errorMsg = std::string("")
	);

bool ParseTxtInfo(
	const std::string& txtFile,
	std::vector<std::vector<cv::Rect2f>>& vPtsRects,
	std::vector<cv::Rect2f>& vImgRects,
	std::vector<cv::Rect2f>& vAddRects,
	std::string& relativePath,
	std::string& errorMsg = std::string("")
	);

bool SaveInfo2Txt(
	const std::vector<std::vector<cv::Rect2f>>& vPtsRects,
	const std::vector<cv::Rect2f>& vImgRects,
	std::vector<cv::Rect2f>& vAddRects,
	const std::string& relativePath,
	const std::string& txtRoot,
	std::string& txtFile,
	std::string& errorMsg = std::string("")
	);

