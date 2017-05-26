#include "stdafx.h"
#include "../utils/utils.h"
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>

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
	std::vector<std::vector<cv::Point>>& vPtss,
	std::vector<cv::Rect>& vRects,
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

	//std::vector<std::vector<cv::Point>> vPtss;
	vPtss.clear();

	for (int i = 0; i < nRectCount; i++)
	{
		std::string line = lines[i + 2];
		std::stringstream ss(line);
		std::vector<cv::Point> vPts;

		std::string str;
		std::vector<std::string> words;

		while (ss >> str)
			words.push_back(str);

		if (words[0] == "r") {
			int x = atoi(words[1].c_str());
			int y = atoi(words[2].c_str());
			int w = atoi(words[3].c_str());
			int h = atoi(words[4].c_str());

			vRects.push_back(cv::Rect(x, y, w, h));
		}
		else {
			for (auto word : words) {
				int idx = word.find(',');
				int x = atoi(word.substr(0, idx).c_str());
				int y = atoi(word.substr(idx + 1).c_str());
				vPts.push_back(cv::Point(x, y));
			}
			vPtss.push_back(vPts);
		}

	}

	return true;
}


bool ParseTxtInfo(const std::string& txtFile, 
	std::vector<std::vector<cv::Rect2f>>& vPtsRects, 
	std::vector<cv::Rect2f>& vImgRects, 
	std::string& relativePath, std::string& errorMsg /*= std::string("") */)
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

	//std::vector<std::vector<cv::Point>> vPtss;
	vPtsRects.clear();

	for (int i = 0; i < nRectCount; i++)
	{
		std::string line = lines[i + 2];
		std::stringstream ss(line);
		std::vector<cv::Rect2f> rects;

		std::string str;
		std::vector<std::string> words;

		while (ss >> str)
			words.push_back(str);

		if (words[0] == "r") {
			float x = atof(words[1].c_str());
			float y = atof(words[2].c_str());
			float w = atof(words[3].c_str());
			float h = atof(words[4].c_str());

			vImgRects.push_back(cv::Rect2f(x, y, w, h));
		}
		else {
			for (auto word : words) {
				int idx1 = word.find(',');
				int idx2 = word.find(',', idx1 + 1);
				int idx3 = word.find(',', idx2 + 1);
				float x = atof(word.substr(0, idx1).c_str());
				float y = atof(word.substr(idx1 + 1, idx2 - idx1).c_str());
				float w = atof(word.substr(idx2 + 1, idx3 - idx2).c_str());
				float h = atof(word.substr(idx3 + 1).c_str());
				rects.push_back(cv::Rect(x, y, w, h));
			}
			vPtsRects.push_back(rects);
		}
	}

	return true;
}



bool SaveInfo2Txt(
	const std::vector<std::vector<cv::Point>>& vPtss, 
	const std::vector<cv::Rect>& vRects, 
	const std::string& relativePath, 
	const std::string& txtRoot, 
	std::string& txtFile, std::string& errorMsg /*= std::string("") */)
{
	int idx = relativePath.rfind('\\');

	std::string parentPath = "";
	std::string filename = "";

	std::stringstream ss;

	if (idx != -1)
	{
		parentPath = relativePath.substr(0, idx);
		filename = relativePath.substr(idx + 1, relativePath.find_last_of('.') - idx - 1);
		ss << txtRoot << "\\" << parentPath;
	}
	else {
		filename = relativePath.substr(0, relativePath.find_last_of('.') - idx - 1);
		ss << txtRoot;
	}

	std::string txtPath = ss.str();

	if (false == boost::filesystem::exists(txtPath))
	{
		boost::filesystem::create_directories(txtPath);
	}

	txtPath += "\\" + filename + ".txt";

	std::ofstream out(txtPath);

	out << relativePath << '\n';
	out << vPtss.size() + vRects.size() << '\n';

	for (const auto& rect : vRects)
	{
		out << "r " 
			<< rect.x << " " << rect.y << " "
			<< rect.width << " " << rect.height << "\n";
	}

	for (const auto& vPts : vPtss)
	{
		for (const auto& pt : vPts)
		{
			out << pt.x << "," << pt.y << " ";
		}
		out << '\n';
	}

	return true;
}



bool SaveInfo2Txt(const std::vector<std::vector<cv::Rect2f>>& vPtsRects, 
	const std::vector<cv::Rect2f>& vImgRects, 
	const std::string& relativePath, 
	const std::string& txtRoot, 
	std::string& txtFile, std::string& errorMsg /*= std::string("") */)
{
	int idx = relativePath.rfind('\\');

	std::string parentPath = "";
	std::string filename = "";

	std::stringstream ss;

	if (idx != -1)
	{
		parentPath = relativePath.substr(0, idx);
		filename = relativePath.substr(idx + 1, relativePath.find_last_of('.') - idx - 1);
		ss << txtRoot << "\\" << parentPath;
	}
	else {
		filename = relativePath.substr(0, relativePath.find_last_of('.') - idx - 1);
		ss << txtRoot;
	}

	std::string txtPath = ss.str();

	if (false == boost::filesystem::exists(txtPath))
	{
		boost::filesystem::create_directories(txtPath);
	}

	txtPath += "\\" + filename + ".txt";

	std::ofstream out(txtPath);

	out << relativePath << '\n';
	out << vPtsRects.size() + vImgRects.size() << '\n';

	for (const auto& rect : vImgRects)
	{
		out << "r " 
			<< rect.x << " " << rect.y << " "
			<< rect.width << " " << rect.height << "\n";
	}

	for (const auto& vRects : vPtsRects)
	{
		for (const auto& rect: vRects)
		{
			out << rect.x << "," << rect.y << "," << rect.width << "," << rect.height << " ";
		}
		out << '\n';
	}

	return true;
}
