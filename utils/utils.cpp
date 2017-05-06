#include "stdafx.h"
#include "../utils/utils.h"
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

const std::vector<std::string>& scanFilesUseRecursive(
	const std::string& rootPath, std::vector<std::string>& container){
	namespace fs = boost::filesystem;
	fs::path fullpath(rootPath, fs::native);
	std::vector<std::string> &ret = container;

	if (!fs::exists(fullpath)){ return ret; }
	fs::recursive_directory_iterator end_iter;
	for (fs::recursive_directory_iterator iter(fullpath); iter != end_iter; iter++){
		try{
			if (fs::is_directory(*iter)){
				//ret.push_back(iter->path().string());
			}
			else{
				ret.push_back(iter->path().string());
			}
		}
		catch (const std::exception & ex){
			//std::cerr << ex.what() << std::endl;
			continue;
		}
	}
	return ret;
}

bool ParseTxtInfo(const std::string& txtFile, 
	std::vector<std::vector<cv::Point>>& vRects, 
	std::string& relativePath, 
	std::string& errorMsg)
{
	std::ifstream in(txtFile);
	if (false == in.is_open())
	{
		errorMsg = "can not open txt: " + txtFile;
		return false;
	}

	std::string line;
	std::vector<std::string> lines;

	while (getline(in, line))
		lines.push_back(line);

	if (lines.size() < 2)
	{
		errorMsg = "error when parse txt: " + txtFile;
		return false;
	}

	relativePath = lines[0];
	int nRectCount = atoi(lines[1].c_str());

	//std::vector<std::vector<cv::Point>> vRects;
	vRects.clear();

	for (int i = 0; i < nRectCount; i++)
	{
		std::string line = lines[i + 2];
		std::stringstream ss(line);
		std::vector<cv::Point> vPts;

		std::string str;
		while (ss >> str)
		{
			int idx = str.find(',');
			int x = atoi(str.substr(0, idx).c_str());
			int y = atoi(str.substr(idx + 1).c_str());
			vPts.push_back(cv::Point(x, y));
		}

		vRects.push_back(vPts);
	}

	return true;
}
