#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

const std::vector<std::string>& scanFilesUseRecursive(
	const std::string& rootPath, 
	std::vector<std::string>& container
	);


bool ParseTxtInfo(
	const std::string& txtFile,
	std::vector<std::vector<cv::Point>>& vRects,
	std::string& relativePath,
	std::string& errorMsg = std::string("")
	);