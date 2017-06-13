#include <fstream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include "MedicalAnalysisSDK.h"

#include "../utils/utils.h"
#include <omp.h>

typedef void(*Func)(const std::string& param);

const double PI = 3.1415926535f;

void ProcessAllPath(int argc, char** argv, Func func) {

	std::vector<std::string> paths;
	for (int i = 1; i < argc; i++)
	{
		paths.push_back(argv[i]);
	}

	for (auto path : paths)
	{
		func(path);
	}
}

void ProcessAllFile(int argc, char** argv, Func func, int stop_no = 0)
{
	std::vector<std::string> paths;
	for (int i = 1; i < argc; i++)
	{
		paths.push_back(argv[i]);
	}

	static int process_no = 0;

	for (auto path : paths)
	{
		std::vector<std::string> files;
		scanFilesUseRecursive(path, files);

		int numOfFiles = files.size();
		for (int i = 0; i < numOfFiles; i++)
		{
			auto file = files[i];
			std::cout << "(" << i << "," << numOfFiles << ") " << file << std::endl;
			func(file);
			if (stop_no > 0 && process_no++ >= stop_no){
				std::cout << "stop processing for reaching the stop count: " << stop_no << std::endl;
				return;
			}
		}
	}
}

void ProcessAllFileMP(int argc, char** argv, Func func)
{
	std::vector<std::string> paths;
	for (int i = 1; i < argc; i++)
	{
		paths.push_back(argv[i]);
	}

	for (auto path : paths)
	{
		std::vector<std::string> files;
		scanFilesUseRecursive(path, files);

		int numOfFiles = files.size();

		int num_of_threads = omp_get_max_threads();
		printf("max number of threads: %d\n", num_of_threads);
		omp_set_num_threads(num_of_threads / 2);
		printf("number of threads: %d\n", omp_get_num_threads());

#pragma omp parallel for
		for (int i = 0; i < numOfFiles; i++)
		{
			auto file = files[i];
			std::cout << "(" << i << "," << numOfFiles << ") " << file << std::endl;
			func(file);
		}
	}
}


const auto GetRect = [](const std::vector<cv::Rect2f>& vRects) -> cv::Rect {

	int min_x = INT_MAX, min_y = INT_MAX, max_x = INT_MIN, max_y = INT_MIN;

	for (auto rect : vRects)
	{
		auto pt = (rect.tl() + rect.br()) / 2;

		if (pt.x < min_x) min_x = pt.x;
		if (pt.x > max_x) max_x = pt.x;
		if (pt.y < min_y) min_y = pt.y;
		if (pt.y > max_y) max_y = pt.y;
	}

	if (vRects.size() == 0)
		return cv::Rect(0, 0, 0, 0);

	return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
};

std::string Phase = "test";
//const std::string root = "D:\\blue\\data\\训练文件\\检测图\\";
const std::string root = "D:\\blue\\data\\训练文件\\检测病灶\\";
const std::string trainImgsRoot = root + "imgs\\";
const std::string trainAnnotsFile = root + "\\" + Phase + ".txt";
std::ofstream annotFile(trainAnnotsFile);

void ParseTxt(const std::string& txt)
{
	//const std::string ImgRoot = "D:\\blue\\data\\乳腺癌图片\\";
	//const std::string ImgRoot = "D:\\迅雷下载\\inpaint_imgs\\";
	const std::string ImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs\\";

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return;
	}

	std::string img_path = ImgRoot + relativePath;
	cv::Mat img = cv::imread(img_path);
	if (img.empty())
	{
		std::cout << "error when open img: " << img_path << std::endl;
		return;
	}

	if (vImgRects.size() == 0 || vPtsRects.size() == 0)
	{
		//boost::filesystem::remove(txt);
		return;
	}

	static int img_no = 0;

	cv::Rect r = vImgRects[0];
	auto part = img(r);

	//auto imgFileName = relativePath.substr(0, relativePath.find_last_of('\\') + 1);
	//cv::imwrite(trainImgsRoot + imgFileName, part);

	auto fullPath = trainImgsRoot + relativePath;
	auto fullRelativePath = fullPath.substr(0, fullPath.find_last_of('\\'));

	if (false == boost::filesystem::exists(fullRelativePath))
	{
		boost::filesystem::create_directories(fullRelativePath);
	}

	cv::imwrite(fullPath, part);

	const auto GetRect = [](const std::vector<cv::Rect2f>& vRects) -> cv::Rect {
		int min_x = INT_MAX, min_y = INT_MAX, max_x = INT_MIN, max_y = INT_MIN;

		for (auto rect: vRects)
		{
			auto pt = (rect.tl() + rect.br()) / 2;

			if (pt.x < min_x) min_x = pt.x;
			if (pt.x > max_x) max_x = pt.x;
			if (pt.y < min_y) min_y = pt.y;
			if (pt.y > max_y) max_y = pt.y;
		}

		return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
	};

	for (const auto& vRects : vPtsRects)
	{
		auto bbox = GetRect(vRects);
		bbox.x -= r.x;
		bbox.y -= r.y;

		if (bbox.area() < 1000)
		{
			std::cout << "	small area!\n";
			//cv::rectangle(part, bbox, cv::Scalar(255, 255, 255));
			//cv::imshow("small", part);
			//cv::waitKey();
			//return;
		}
	}

	annotFile << "# " << img_no++ << std::endl;
	annotFile << relativePath << std::endl;
	annotFile << vPtsRects.size() << std::endl;

	for (const auto& vRects : vPtsRects)
	{
		auto bbox = GetRect(vRects);
		bbox.x -= r.x;
		bbox.y -= r.y;

		float radio = 0.05;

		bbox.x = int(bbox.x - bbox.width * radio + 0.5f);
		bbox.y = int(bbox.y - bbox.height * radio + 0.5f);
		bbox.width = int(bbox.width * (1 + 2 * radio) + 0.5f);
		bbox.height = int(bbox.height * (1 + 2 * radio) + 0.5f);

		cv::rectangle(part, bbox, cv::Scalar(255, 255, 255), 2);

		auto tl = bbox.tl();
		auto br = bbox.br();
		tl.x = std::max(0, std::min(tl.x, r.width));
		tl.y = std::max(0, std::min(tl.y, r.height));
		br.x = std::max(tl.x, std::min(br.x, r.width));
		br.y = std::max(tl.y, std::min(br.y, r.height));

		annotFile << "1 " << tl.x << " " << tl.y << " " << br.x << " " << br.y << " 0" << std::endl;

		/*for (const auto& pt : vPts)
		{
			cv::circle(img, pt, 5, cv::Scalar(0, 0, 255), 5, -1);
		}*/
	}

	annotFile.flush();

	cv::imshow("part", part);
	cv::waitKey(1);

	//for (const auto& rect : vRects)
	//{
	//	cv::rectangle(img, rect, cv::Scalar(255, 0, 255), 2);
	//}

	//if (img.size().width > 1024)
	//	cv::resize(img, img, img.size() / 2);

	//cv::imshow("img", img);
	//cv::waitKey(1000);

}

void ParseTxtForImgBBox(const std::string& txt)
{
	const std::string ImgRoot = "D:\\迅雷下载\\inpaint_imgs\\";

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return;
	}

	std::string img_path = ImgRoot + relativePath;
	cv::Mat img = cv::imread(img_path);
	if (img.empty())
	{
		std::cout << "error when open img: " << img_path << std::endl;
		return;
	}

	if (vImgRects.size() == 0)
	{
		//boost::filesystem::remove(txt);
		return;
	}

	static int img_no = 0;

	//auto imgFileName = relativePath.substr(0, relativePath.find_last_of('\\') + 1);
	//cv::imwrite(trainImgsRoot + imgFileName, part);

	auto fullPath = trainImgsRoot + relativePath;
	auto fullRelativePath = fullPath.substr(0, fullPath.find_last_of('\\'));

	if (false == boost::filesystem::exists(fullRelativePath))
	{
		boost::filesystem::create_directories(fullRelativePath);
	}

	cv::imwrite(fullPath, img);

	annotFile << "# " << img_no++ << std::endl;
	annotFile << relativePath << std::endl;
	annotFile << vImgRects.size() << std::endl;

	cv::Size2f r = img.size();

	for (const auto& rect : vImgRects)
	{
		auto bbox = rect;

		float radio = 0.0f;

		bbox.x = int(bbox.x - bbox.width * radio + 0.5f);
		bbox.y = int(bbox.y - bbox.height * radio + 0.5f);
		bbox.width = int(bbox.width * (1 + 2 * radio) + 0.5f);
		bbox.height = int(bbox.height * (1 + 2 * radio) + 0.5f);

		cv::rectangle(img, bbox, cv::Scalar(255, 255, 255), 2);

		auto tl = bbox.tl();
		auto br = bbox.br();
		tl.x = (int)std::max(0.f, std::min(tl.x, r.width));
		tl.y = (int)std::max(0.f, std::min(tl.y, r.height));
		br.x = (int)std::max(tl.x, std::min(br.x, r.width));
		br.y = (int)std::max(tl.y, std::min(br.y, r.height));

		annotFile << "1 " << tl.x << " " << tl.y << " " << br.x << " " << br.y << " 0" << std::endl;

		/*for (const auto& pt : vPts)
		{
			cv::circle(img, pt, 5, cv::Scalar(0, 0, 255), 5, -1);
		}*/
	}

	annotFile.flush();

	cv::imshow("img", img);
	cv::waitKey(1);

	//for (const auto& rect : vRects)
	//{
	//	cv::rectangle(img, rect, cv::Scalar(255, 0, 255), 2);
	//}

	//if (img.size().width > 1024)
	//	cv::resize(img, img, img.size() / 2);

	//cv::imshow("img", img);
	//cv::waitKey(1000);
}

void ChangeRelativePath(const std::string& txt)
{
	const std::string ImgRoot = "D:\\blue\\data\\乳腺癌图片\\";
	const std::string txtSaveRoot = "D:\\blue\\codes\\TagTools\\TagTools\\RectTxtProcess\\replace\\";
	const std::string parentPath = "4a类02\\";


	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return;
	}

	relativePath = parentPath + relativePath;

	std::string img_path = ImgRoot + relativePath;
	cv::Mat img = cv::imread(img_path);
	if (img.empty())
	{
		std::cout << "error when open img: " << img_path << std::endl;
		return;
	}

	std::string relativeFilename = relativePath.substr(0, relativePath.find_last_of('.'));
	std::string relativeTxtFile = relativeFilename + ".txt";

	std::string txtFile;
	if (false == SaveInfo2Txt(vPtsRects, vImgRects, relativePath, txtSaveRoot, txtFile, errorMsg))
	{
		std::cout << "SaveInfo2Txt error: " << errorMsg << "\n";
		return;
	}
}

void DrawGT(const std::string& txt)
{
	//const std::string ImgRoot = "D:\\blue\\data\\乳腺癌图片\\";
	const std::string ImgRoot = "D:\\迅雷下载\\results\\";
	const std::string ResultImgRoot = "D:\\迅雷下载\\results1\\";

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return;
	}

	std::string img_path = ImgRoot + relativePath;
	cv::Mat img = cv::imread(img_path);
	if (img.empty())
	{
		std::cout << "error when open img: " << img_path << std::endl;
		return;
	}

	if (vImgRects.size() == 0 || vPtsRects.size() == 0)
	{
		cv::imwrite(ResultImgRoot + relativePath, img);
		return;
	}


	const auto GetRect = [](const std::vector<cv::Rect2f>& vRects) -> cv::Rect {
		int min_x = INT_MAX, min_y = INT_MAX, max_x = INT_MIN, max_y = INT_MIN;

		for (auto rect : vRects)
		{
			auto pt = (rect.tl() + rect.br()) / 2;

			if (pt.x < min_x) min_x = pt.x;
			if (pt.x > max_x) max_x = pt.x;
			if (pt.y < min_y) min_y = pt.y;
			if (pt.y > max_y) max_y = pt.y;
		}

		return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
	};

	for (auto vRects : vPtsRects)
	{
		auto rect = GetRect(vRects);
		rect.x -= vRects[0].x;
		rect.y -= vRects[0].y;

		cv::rectangle(img, rect, cv::Scalar(255, 0, 255), 2);
	}

	cv::imshow("img", img);
	cv::waitKey(1);

	auto fullPath = ResultImgRoot + relativePath;
	auto fullRelativePath = fullPath.substr(0, fullPath.find_last_of('\\'));
	if (false == boost::filesystem::exists(fullRelativePath))
		boost::filesystem::create_directories(fullRelativePath);
	cv::imwrite(fullPath, img);
}

void RemoveImage(const std::string& txt)
{
	const std::string ImgRoot = "D:\\blue\\data\\乳腺癌图片\\test\\";

	std::vector<std::vector<cv::Rect2f>> vPtss;
	std::vector<cv::Rect2f> vRects;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vPtss, vRects, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return;
	}

	std::string img_path = ImgRoot + relativePath;
	cv::Mat img = cv::imread(img_path);
	if (img.empty())
	{
		std::cout << "error when open img: " << img_path << std::endl;
		return;
	}

	static int count = 0;
	boost::filesystem::remove(img_path);
	std::cout << "remove count: " << count++;
}

void CalHVProj(const cv::Mat& srcImg, cv::Mat& hProj, cv::Mat& vProj, int col = 1)
{
	cv::Mat img = srcImg.clone();
	if (img.channels() != 1)
		cv::cvtColor(img, img, CV_RGB2GRAY);

	auto imgSize = img.size();
	hProj = cv::Mat::zeros(imgSize.width, col, CV_32F);
	vProj = cv::Mat::zeros(imgSize.height, col, CV_32F);

	for (int r = 0; r < imgSize.height; r++)
	{
		int s = 0;
		for (int c = 0; c < imgSize.width; c++)
		{
			s += img.at<uchar>(r, c);
		}
		for (int i = 0; i < col; i++)
			vProj.at<float>(r, i) = s;
	}

	for (int c = 0; c < imgSize.width; c++)
	{
		int s = 0;
		for (int r = 0; r < imgSize.height; r++)
		{
			s += img.at<uchar>(r, c);
		}
		for (int i = 0; i < col; i++)
			hProj.at<float>(c, i) = s;
	}

	//cv::reduce(srcImg, hProj, 0, CV_REDUCE_SUM, CV_32SC1);
	//cv::reduce(srcImg, vProj, 1, CV_REDUCE_SUM, CV_32SC1);
}

cv::Mat GetHistImg(const cv::MatND& hist)
{
	double maxVal = 0;
	double minVal = 0;

	//找到直方图中的最大值和最小值
	minMaxLoc(hist, &minVal, &maxVal, 0, 0);
	int histSize = hist.rows;
	cv::Mat histImg(histSize, histSize, CV_8U, cv::Scalar(255));
	// 设置最大峰值为图像高度的90%
	int hpt = static_cast<int>(0.9*histSize);

	//int meanVal = cv::mean(hist)[0];

	for (int h = 0; h < histSize; h++)
	{
		int binVal = int(hist.at<float>(h, 0));
		int intensity = 0;

		//if (binVal >= meanVal)
			intensity = static_cast<int>((float)binVal*hpt / (float)maxVal);

		line(histImg, cv::Point(h, histSize), cv::Point(h, histSize - intensity), cv::Scalar::all(0));
	}

	cv::resize(histImg, histImg, histImg.size() / 2);

	return histImg;
}

// 合并矩形框
void ClusterRegions(const cv::Mat& srcImg, std::vector<cv::Rect>& regions)
{
	cv::Point pt1, pt2;
	int c = 1;
	float radio;

	while (1)
	{
		bool isCluster = false;
		std::vector<int> Ids(regions.size());
		for (int i = 0; i < regions.size(); i++)
		{
			if (Ids[i] != 0) continue;
			Ids[i] = c++;

			for (int j = i + 1; j < regions.size(); j++)
			{
				if ((regions[i] & regions[j]).area() > 0)
				{
					Ids[j] = Ids[i];
					isCluster = true;
				}
			}
		}

		if (false == isCluster)
			break;

		std::map<int, cv::Rect> id2rect;
		for (int i = 0; i < regions.size(); i++)
		{
			auto id = Ids[i];
			if (id2rect.find(id) == id2rect.end())
				id2rect[id] = regions[i];
			else
				id2rect[id] |= regions[i];
		}

		regions.clear();
		for (const auto& p : id2rect)
		{
			regions.push_back(p.second);
		}
	}

	//std::stringstream ss;
	//cv::Mat drawing1 = srcImg.clone();
	//for (int i = 0; i < regions.size(); i++)
	//{
	//	//auto id = Ids[i];
	//	auto rect = regions[i];
	//	ss.str("");
	//	cv::rectangle(drawing1, rect, cv::Scalar(255, 255, 255));
	//	//ss << id;
	//	//cv::putText(drawing1, ss.str(), rect.tl(), 1, 1, cv::Scalar(255, 255, 255));
	//}

	//cv::Mat drawing2 = srcImg.clone();

	//for (auto p : id2rect)
	//{
	//	auto id = p.first;
	//	auto rect = p.second;

	//	ss.str("");
	//	cv::rectangle(drawing2, rect, cv::Scalar(255, 255, 255));
	//	ss << id;
	//	cv::putText(drawing2, ss.str(), rect.tl(), 1, 1, cv::Scalar(255, 255, 255));
	//}

	//cv::resize(drawing1, drawing1, drawing1.size() / 2);
	//cv::resize(drawing2, drawing2, drawing1.size() / 2);
	//cv::imshow("cluster1", drawing1);
	//cv::imshow("cluster2", drawing2);
	//cv::waitKey();

	//return true;
}

void GetContoursBBox(const cv::Mat& srcImg, std::vector<cv::Rect>& bbox)
{
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(srcImg.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	int maxArea = 0;

	//bbox.clear();
	for (int i = 0; i < contours.size(); i++)
	{
		auto h = hierarchy[i];
		if (h[3] != -1)
			continue;

		auto contour = contours[i];
		auto rect = cv::boundingRect(contour);

		bbox.push_back(rect);
	}
}

// 调整矩形框
void AdjustRegion(const cv::Mat& srcImg, std::vector<cv::Rect>& regions)
{
	cv::Mat gray, h, v, res1, res2, res3;
	cv::cvtColor(srcImg, gray, CV_RGB2GRAY);

	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

	cv::morphologyEx(gray, res1, cv::MORPH_OPEN, element);

	cv::adaptiveThreshold(res1, res2, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 5, 3);
	element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	cv::morphologyEx(res2, res3, cv::MORPH_CLOSE, element);

	CalHVProj(res3, h, v);
	int ksize = 5;
	cv::medianBlur(h, h, ksize);
	cv::medianBlur(v, v, ksize);

	auto hShow = GetHistImg(h);
	auto vShow = GetHistImg(v);
	cv::imshow("h", hShow);
	cv::imshow("v", vShow);

	//cv::waitKey();

	for (auto& rect : regions)
	{
		cv::rectangle(gray, rect, cv::Scalar(255, 0, 0), 1);

		auto r = cv::Rect(0, rect.x, 1, rect.width);
		int hMean = int(cv::mean(h(r))[0] * 0.3f);

		int cx = rect.x + rect.width / 2;
		
		for (int c = cx; c >= 0; c--)
		{
			//std::cout << "h: " << h.at<float>(c, 0) << " mean: " << hMean << std::endl;
			if (h.at<float>(c, 0) >= hMean)
				continue;
			rect.x = c;
			break;
		}

		for (int c = cx; c < h.rows; c++)
		{
			if (h.at<float>(c, 0) >= hMean)
				continue;
			rect.width = c - rect.x;
			break;
		}

		r = cv::Rect(0, rect.y, 1, rect.height);
		int vMean = int(cv::mean(v(r))[0] * 0.3f);
		int cy = rect.y + rect.height / 2;

		for (int c = cy; c >= 0; c--)
		{
			if (v.at<float>(c, 0) >= vMean)
				continue;
			rect.y = c;
			break;
		}

		for (int c = cy; c < h.rows; c++)
		{
			if (v.at<float>(c, 0) >= vMean)
				continue;
			rect.height = c - rect.y;
			break;
		}

		cv::rectangle(gray, rect, cv::Scalar(255, 0, 0), 2);
		cv::imshow("gray", gray);
		cv::waitKey();
	}

}

void AdjustRegion2(const cv::Mat& srcImg, std::vector<cv::Rect>& regions)
{
	cv::Mat gray, h, v, res1, res2, res3;
	gray = srcImg.clone();

	if (gray.channels() != 1)
		cv::cvtColor(gray, gray, CV_RGB2GRAY);

	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
	for (auto rect : regions)
	{
		res1 = gray(rect).clone();

		cv::morphologyEx(res1, res1, cv::MORPH_OPEN, element);
		//cv::threshold(res1, res2, 0, 255, cv::THRESH_OTSU);
		cv::adaptiveThreshold(res1, res2, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 3);
		std::vector<cv::Rect> bboxes;
		GetContoursBBox(res2, bboxes);

		for (auto bbox : bboxes)
		{
			cv::rectangle(res1, bbox, cv::Scalar(255, 255, 255));
		}

		cv::imshow("res1", res1);
		cv::imshow("res2", res2);
		//cv::waitKey();
	}

}

// 获取最优矩形
void GetFinalRegion(const cv::Mat& srcImg, std::vector<cv::Rect>& regions)
{
	auto imgSize = srcImg.size();
	if (regions.size() == 1)
	{
		auto rect = regions[0];
	}
}

void Method1(cv::Mat img) {
	int morph_size = 1;
	cv::Mat res1, res2;
	while (morph_size < 5)
	{
		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1));

		cv::morphologyEx(img, res1, cv::MORPH_OPEN, element);
		//cv::imshow("res1", res1);

		//cv::morphologyEx(res1, res2, cv::MORPH_GRADIENT, element);
		cv::cvtColor(res1, res2, CV_RGB2GRAY);

		//cv::adaptiveThreshold(res2, res2, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 5, 3);
		cv::adaptiveThreshold(res2, res2, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 5, 3);
		element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
		cv::morphologyEx(res2, res2, cv::MORPH_CLOSE, element);
		cv::imshow("res2", res2);

		cv::Mat h, v, hShow, vShow;
		int ksize = 5;
		CalHVProj(res2, h, v);
		cv::medianBlur(h, h, ksize);
		cv::medianBlur(v, v, ksize);
		hShow = GetHistImg(h);
		vShow = GetHistImg(v);
		cv::imshow("h", hShow);
		cv::imshow("v", vShow);

		//cv::waitKey();

		//cv::morphologyEx(res1, res2, cv::MORPH_CLOSE, element);
		//cv::imshow("res2", res2);

		//cv::imshow("img", img);
		//cv::waitKey();

		cv::Mat edge;
		cv::Canny(res1, edge, 0, 255);
		cv::Mat grey;
		cv::cvtColor(img, grey, CV_RGB2GRAY);
		//cv::threshold(grey, edge, 10, 255, cv::THRESH_BINARY);
		//edge = grey;

		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::RNG rng(12345);
		cv::findContours(res2.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		/// Draw contours
		//cv::Mat drawing = cv::Mat::zeros(edge.size(), CV_8UC3);
		int maxArea = 0;
		cv::Mat drawing = grey.clone();
		std::vector<cv::Rect> rects;
		for (int i = 0; i < contours.size(); i++)
		{
			auto h = hierarchy[i];
			if (h[3] != -1)
				continue;

			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			//cv::drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
			auto contour = contours[i];
			auto rect = cv::boundingRect(contour);
			rects.push_back(rect);
			//if (rect.area() < maxArea) continue;

			//maxArea = rect.area();
			//cv::rectangle(drawing, rect, color, 3);
		}

		std::sort(rects.begin(), rects.end(), [](cv::Rect a, cv::Rect b)->bool { return a.area() > b.area(); } );

		std::vector<cv::Rect> temp;
		int count = int(rects.size() * 0.1f);
		for (int i = 0; i < count; i++)
		{
			temp.push_back(rects[i]);
		}
		rects = temp;

		ClusterRegions(img, rects);

		cv::rectangle(drawing, rects[0], cv::Scalar(255, 255, 255));

		cv::imshow("drawing", drawing);
		//cv::imshow("edge", edge);
		cv::imshow("img", img);

		cv::waitKey();

		morph_size += 1;
	}

	cv::destroyAllWindows();
}

void Method2(cv::Mat img)
{
	//int morph_size = 2;
	cv::Mat res1, res2, res3;
	//std::vector<cv::Rect> all_rects;
	std::vector<std::pair<cv::Rect, int>> all_rects;
	//while (morph_size < 5)
	for (int morph_size = 1; morph_size < 3; morph_size++)
	{
		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1) );

		cv::morphologyEx(img, res1, cv::MORPH_OPEN, element);
		cv::cvtColor(res1, res2, CV_RGB2GRAY);
		cv::Canny(res2, res3, 0, 255);

		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;

		cv::findContours(res3.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		int maxArea = 0;
		cv::Mat drawing = res3.clone();
		std::vector<std::pair<cv::Rect, int>> rects;
		for (int i = 0; i < contours.size(); i++)
		{
			auto h = hierarchy[i];
			if (h[3] != -1)
				continue;

			auto contour = contours[i];
			auto rect = cv::boundingRect(contour);
			int val = cv::sum(res2(rect)).dot(cv::Scalar::ones());
			rects.push_back(std::pair<cv::Rect, int>(rect, val));
		}

		std::sort(rects.begin(), rects.end(), 
			[](std::pair<cv::Rect, int> a, std::pair<cv::Rect, int> b)->bool 
		{ 
			return a.first.area() > b.first.area(); 
		} );

		int count = std::min(2, int(rects.size()));

		for (int i = 0; i < count; i++)
		{
			all_rects.push_back(rects[i]);
		}
	}

	std::sort(all_rects.begin(), all_rects.end(),
		[](std::pair<cv::Rect, int> a, std::pair<cv::Rect, int> b)->bool
	{
		return a.second > b.second;
	});

	std::vector<cv::Rect> result_rects;
	int count = std::min(5, int(all_rects.size()));

	for (int i = 0; i < count; i++)
	{
		result_rects.push_back(all_rects[i].first);
	}

	ClusterRegions(img, result_rects);
	AdjustRegion(img, result_rects);

	for (auto rect : result_rects)
	{
		cv::rectangle(img, rect, cv::Scalar(255, 255, 255), 2);
	}

	if (img.size().height > 600)
		cv::resize(img, img, img.size() / 2);

	cv::imshow("img", img);
	cv::waitKey();
}

void DrawLines(const cv::Mat& srcImg, const std::vector<cv::Vec2f>& lines)
{
	cv::Mat drawing = srcImg.clone();
	auto it = lines.begin();
	while (it != lines.end())
	{
		float rho = (*it)[0];
		float theta = (*it)[1];

		if (theta < PI / 4 || theta > 3 * PI / 4)
		{
			cv::Point pt1(rho / cos(theta), 0);
			cv::Point pt2((rho - srcImg.rows * sin(theta)) / cos(theta), srcImg.rows);

			cv::line(drawing, pt1, pt2, cv::Scalar(255), 1);
		}
		else {
			cv::Point pt1(0, rho / sin(theta));
			cv::Point pt2(srcImg.cols, (rho - srcImg.cols * cos(theta)) / sin(theta));

			cv::line(drawing, pt1, pt2, cv::Scalar(255), 1);
		}

		++it;
	}
	cv::imshow("draw line", drawing);
	//cv::waitKey();
}

void Method3(cv::Mat img)
{
	cv::Mat gray, h, v, res1, res2, res3;
	auto imgSize = img.size();

	if (img.size().width > 1024)
		cv::resize(img, img, cv::Size(1024, int(img.size().height * 1024.f / img.size().width)));

	cv::cvtColor(img, gray, CV_RGB2GRAY);

	std::vector<cv::Rect> bboxes;

	cv::Mat close_element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
	for (int ksize = 1; ksize <= 21; ksize += 10)
	{
		cv::Mat open_element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksize, ksize));

		cv::morphologyEx(gray, res1, cv::MORPH_OPEN, open_element);
		cv::morphologyEx(res1, res1, cv::MORPH_CLOSE, open_element);

		cv::threshold(res1, res2, 0, 255, cv::THRESH_OTSU);

		GetContoursBBox(res2, bboxes);
	}

	cv::Mat drawing = gray.clone();
	std::vector<cv::Rect> temp;
	const float disable_region_radio = 0.05f;
	cv::Vec4i disable_regions = {
		int(imgSize.width * disable_region_radio), 
		int(imgSize.width * (1 - disable_region_radio)), 
		int(imgSize.height * disable_region_radio), 
		int(imgSize.height * (1 - disable_region_radio)), 
	};
	for (auto rect : bboxes)
	{
		float r1 = (float)rect.area() / (float)gray.size().area();
		if ( r1 < 0.005f || r1 > 0.9f)
			continue;

		if (rect.x < disable_regions[0] ||
			rect.x + rect.width > disable_regions[1] ||
			rect.y < disable_regions[2] ||
			rect.y + rect.height > disable_regions[3]
			)
			continue;

		cv::rectangle(drawing, rect, cv::Scalar(255, 255, 255), 1);

		temp.push_back(rect);
	}

	//cv::imshow("drawing", drawing);
	//cv::waitKey();

	bboxes = temp;

	std::cout << "bbox count: " << bboxes.size() << std::endl;

	ClusterRegions(gray, bboxes);

	std::vector<std::pair<cv::Rect, int>> rects;
	for (auto rect: bboxes)
	{
		float r1 = (float)rect.area() / (float)gray.size().area();
		if ( r1 < 0.025f || r1 > 0.9f)
			continue;


		float r2 = (float)rect.width / (float)rect.height;
		if (r2 < 1.0f || r2 > 5.0f)
			continue;

		int val = cv::sum(res2(rect)).dot(cv::Scalar::ones());
		rects.push_back(std::pair<cv::Rect, int>(rect, val));
	}

	int count = std::min(2, (int)rects.size());
	if (rects.size() >= 2)
	{
		std::partial_sort(rects.begin(), rects.begin() + count, rects.end(),
			[](std::pair<cv::Rect, int> a, std::pair<cv::Rect, int> b)->bool
		{
			float r1 = (float)a.second / (float)a.first.area();
			float r2 = (float)b.second / (float)b.first.area();
			return r1 > r2;
		});
	}

	int i = 0;
	std::vector<cv::Rect> all_rects;
	for (auto p : rects)
	{
		if (i++ >= count)
			break;

		auto rect = p.first;

		float r = (float)rect.height / (float)rect.width;
		float imgR = (float)imgSize.height / (float)imgSize.width;

		if (r < imgR)
		{
			rect.height = int(rect.width * imgR);
			if (rect.y + rect.height >= imgSize.height)
				continue;
		}

		all_rects.push_back(rect);

	}

	if (all_rects.size() == 0)
		return;

	cv::Rect r = all_rects[0];
	for (int i = 1; i < all_rects.size(); i++)
	{
		//cv::rectangle(drawing, rect, cv::Scalar(255, 255, 255), 3);
		r |= all_rects[i];
	}
	//cv::rectangle(drawing, r, cv::Scalar(255, 255, 255), 3);
	cv::rectangle(img, r, cv::Scalar(255, 255, 255), 3);

	cv::imshow("img", img);

	//cv::imshow("res1", res1);
	//cv::imshow("res2", res2);
	//cv::imshow("res3", res3);
	//cv::imshow("drawing", drawing);
	cv::waitKey(1);

}

void Preprocess(const std::string& imgFile)
{
	auto img = cv::imread(imgFile);
	if (img.empty())
		return;
	
	//Method1(img);
	//Method2(img);
	Method3(img);
}

void Inpainting(const std::string& txt)
{
	SYY::HANDLE hHandle, hHandle2;
	if (SYY::SYY_NO_ERROR != SYY::Inpainting::InitInpaint(hHandle, SYY::Inpainting::PatchMatch))
	{
		std::cerr << "InitPaint error!\n" << std::endl;
		return;
	}
	if (SYY::SYY_NO_ERROR != SYY::Inpainting::InitInpaint(hHandle2, SYY::Inpainting::Criminisi_P5))
	{
		std::cerr << "InitPaint error!\n" << std::endl;
		return;
	}

	const std::string ImgRoot = "D:\\blue\\data\\乳腺癌图片\\";
	//const std::string InpaintImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs\\";
	const std::string InpaintImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs_crimisi\\";

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects, vAddRects;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, vAddRects, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return;
	}

	std::string inpaint_img_path = InpaintImgRoot + relativePath;
	std::string inpaintRelativePath = inpaint_img_path.substr(0, inpaint_img_path.find_last_of('\\'));

	if (false == boost::filesystem::exists(inpaintRelativePath))
		boost::filesystem::create_directories(inpaintRelativePath);

	if (true == boost::filesystem::exists(inpaint_img_path))
		return;

	std::string img_path = ImgRoot + relativePath;
	cv::Mat img = cv::imread(img_path);
	if (img.empty())
	{
		std::cout << "error when open img: " << img_path << std::endl;
		return;
	}

	cv::Mat inpaintImg;
	if (vPtsRects.size() == 0)
	{
		inpaintImg = img;
	}
	else 
	{
		auto srcImg = img.clone();
		cv::Mat element5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
		cv::Mat element3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::Mat maskImg = cv::Mat::zeros(img.size(), CV_8UC1);

		for (auto vRects : vPtsRects)
		{
			for (auto rect : vRects)
			{
				const int margin = 1.0f;
				rect.x = (int)std::max(0.f, rect.x - margin);
				rect.y = (int)std::max(0.f, rect.y - margin);
				rect.width = (int)std::min((float)srcImg.cols, rect.width + 2 * margin + 0.5f);
				rect.height = (int)std::min((float)srcImg.rows, rect.height + 2 * margin + 0.5f);

				auto region = img(rect);
				cv::cvtColor(region, region, CV_RGB2GRAY);

				cv::Mat m = maskImg(rect);
				cv::threshold(region, m, 0, 255, CV_THRESH_OTSU);
				cv::morphologyEx(m, m, cv::MORPH_DILATE, element5x5);

			}
		}

		static SYY::Image inpaint;

		for (int i = 0; i < 1; i++)
		{
			SYY::Image 
				src((char*)srcImg.data, srcImg.cols, srcImg.rows, srcImg.channels()),
				mask((char*)maskImg.data, maskImg.cols, maskImg.rows, maskImg.channels());

			//if (SYY::SYY_NO_ERROR != SYY::Inpainting::ExecuteInpaint(hHandle, src, mask, inpaint))
				//return;

			//inpaintImg = cv::Mat(inpaint.nHeight, inpaint.nWidth, CV_8UC3, inpaint.pData).clone();
			cv::inpaint(srcImg, maskImg, inpaintImg, 5, cv::INPAINT_NS);

			//cv::imshow("inpaingImg1", inpaintImg);

			src = SYY::Image((char*)inpaintImg.data, inpaintImg.cols, inpaintImg.rows, inpaintImg.channels());

			if (SYY::SYY_NO_ERROR != SYY::Inpainting::ExecuteInpaint(hHandle2, src, mask, inpaint))
				return;

			inpaintImg = cv::Mat(inpaint.nHeight, inpaint.nWidth, CV_8UC3, inpaint.pData).clone();

			//srcImg = inpaintImg.clone();

			//cv::imshow("inpaingImg2", inpaintImg);
			//cv::waitKey();
			//cv::dilate(maskImg, maskImg, element3x3);
			//cv::imshow("mask", maskImg);
			//cv::imshow("inpaingImg", inpaintImg);
			//cv::waitKey();
		}

		//cv::imshow("img", img);
		//cv::imshow("mask", maskImg);
		//cv::imshow("inpaingImg", inpaintImg);
		//cv::waitKey(1);
		//cv::destroyAllWindows();
	}



	cv::imwrite(inpaint_img_path, inpaintImg);

	SYY::Inpainting::ReleaseInpaint(hHandle);
	SYY::Inpainting::ReleaseInpaint(hHandle2);
}

cv::Mat getHistImg(const cv::MatND& hist)
{
	double maxVal = 0;
	double minVal = 0;

	//找到直方图中的最大值和最小值
	cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);
	int histSize = hist.rows;
	cv::Mat histImg(histSize, histSize, CV_8U, cv::Scalar(255));
	// 设置最大峰值为图像高度的90%
	int hpt = static_cast<int>(0.9*histSize);

	for (int h = 0; h < histSize; h++)
	{
		float binVal = hist.at<float>(h);
		int intensity = static_cast<int>(binVal*hpt / maxVal);
		line(histImg, cv::Point(h, histSize), cv::Point(h, histSize - intensity), cv::Scalar::all(0));
	}

	return histImg;
}

void ShowHistImg(const cv::Mat& img)
{
	const int channels[1] = { 0 };
	const int histSize[1] = { 256 };
	float hranges[2] = { 0, 255 };
	const float* ranges[1] = { hranges };
	cv::MatND hist;
	cv::calcHist(&img, 1, channels, cv::Mat(), hist, 1, histSize, ranges);

	cv::imshow("hist", getHistImg(hist));
}

void Inpainting_old(const std::string& txt)
{
	SYY::HANDLE hHandle, hHandle2;
	if (SYY::SYY_NO_ERROR != SYY::Inpainting::InitInpaint(hHandle, SYY::Inpainting::PatchMatch))
	{
		std::cerr << "InitPaint error!\n" << std::endl;
		return;
	}
	if (SYY::SYY_NO_ERROR != SYY::Inpainting::InitInpaint(hHandle2, SYY::Inpainting::Criminisi_P5))
	{
		std::cerr << "InitPaint error!\n" << std::endl;
		return;
	}

	const std::string ImgRoot = "D:\\blue\\data\\乳腺癌图片\\";
	const std::string InpaintImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs\\";

	std::vector<std::vector<cv::Point>> vPtss;
	std::vector<cv::Rect> vRects;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vPtss, vRects, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return;
	}

	std::string img_path = ImgRoot + relativePath;
	cv::Mat img = cv::imread(img_path);
	if (img.empty())
	{
		std::cout << "error when open img: " << img_path << std::endl;
		return;
	}

	cv::Mat inpaintImg;
	if (vPtss.size() == 0)
	{
		inpaintImg = img;
	}
	else
	{
		auto srcImg = img.clone();
		cv::Mat element5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
		cv::Mat element3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::Mat maskImg = cv::Mat::zeros(img.size(), CV_8UC1);
		int radius = 20;
		//if (srcImg.cols > 1000) radius = 15;

		for (auto vPts : vPtss)
		{
			for (auto pt : vPts)
			{
				auto rect = cv::Rect(pt.x + 1 - radius / 2, pt.y + 1 - radius / 2, radius, radius);

				//auto rect = cv::Rect(pt.x, pt.y, radius, radius);

				maskImg(rect) = 255;
				srcImg(rect) = cv::Scalar(255, 255, 255);
				//cv::rectangle(srcImg, rect, cv::Scalar(255, 255, 255));

				//cv::circle(maskImg, pt, radius, cv::Scalar(255), -1);

				auto region = img(rect);
				//cv::cvtColor(region, region, CV_RGB2GRAY);
				//cv::morphologyEx(region, region, cv::MORPH_DILATE, element3x3);

				cv::Mat m = maskImg(rect);
				//cv::Canny(region, m, 0, 255);
				//cv::threshold(region, m, 0, 255, CV_THRESH_OTSU);
				//cv::morphologyEx(m, m, cv::MORPH_DILATE, element5x5);
				//cv::morphologyEx(m, m, cv::MORPH_CLOSE, element);
				//cv::dilate(m, m, element3x3);
				//cv::erode(m, m, element3x3);

				//ShowHistImg(region);

				//cv::imshow("region", region);
				//cv::imshow("m", m);
				//cv::waitKey();

				//std::vector<cv::Rect> bboxes;
				//GetContoursBBox(m, bboxes);

				//if (bboxes.size() > 1)
				//{
				//	std::sort(bboxes.begin(), bboxes.end(), [](const cv::Rect& a, const cv::Rect& b){
				//		return a.area() > b.area();
				//	});
				//}

				//for (auto bbox : bboxes) cv::rectangle(m, bbox, cv::Scalar(128));

				//int min_len = std::min(1, int(bboxes.size()));
				//for (int i = 0; i < min_len; i++)
				//{
				//	img(rect)(bboxes[i]) = cv::Scalar(255, 255, 255);
				//	m(bboxes[i]) = 255;
				//}

				//cv::dilate(m, m, element3x3);

				//cv::imshow("region", region);
				//cv::imshow("m", m);
				//cv::imshow("img", img);
				//cv::imshow("mask", maskImg);
				//cv::waitKey();
				//cv::circle(maskImg, pt, radius, cv::Scalar(255), -1);
				//cv::circle(img, pt, radius, cv::Scalar(255), -1);
			}
		}

		//cv::imshow("srcImg", srcImg);
		//cv::imshow("mask", maskImg);
		//cv::waitKey();

	//	static SYY::Image inpaint;

	//	for (int i = 0; i < 1; i++)
	//	{
	//		SYY::Image
	//			src((char*)srcImg.data, srcImg.cols, srcImg.rows, srcImg.channels()),
	//			mask((char*)maskImg.data, maskImg.cols, maskImg.rows, maskImg.channels());

	//		//if (SYY::SYY_NO_ERROR != SYY::Inpainting::ExecuteInpaint(hHandle, src, mask, inpaint))
	//		//	return;

	//		//inpaintImg = cv::Mat(inpaint.nHeight, inpaint.nWidth, CV_8UC3, inpaint.pData).clone();
	//		cv::inpaint(srcImg, maskImg, inpaintImg, 5, cv::INPAINT_NS);

	//		//cv::imshow("inpaingImg1", inpaintImg);

	//		src = SYY::Image((char*)inpaintImg.data, inpaintImg.cols, inpaintImg.rows, inpaintImg.channels());

	//		if (SYY::SYY_NO_ERROR != SYY::Inpainting::ExecuteInpaint(hHandle2, src, mask, inpaint))
	//			return;

	//		inpaintImg = cv::Mat(inpaint.nHeight, inpaint.nWidth, CV_8UC3, inpaint.pData);

	//		srcImg = inpaintImg.clone();

	//		//cv::imshow("inpaingImg2", inpaintImg);
	//		//cv::waitKey();
	//		//cv::dilate(maskImg, maskImg, element3x3);
	//		//cv::imshow("mask", maskImg);
	//		//cv::imshow("inpaingImg", inpaintImg);
	//		//cv::waitKey();
	//	}

		cv::imshow("img", srcImg);
		cv::imshow("mask", maskImg);
	//	//cv::imshow("inpaingImg", inpaintImg);
		cv::waitKey();
	//	//cv::destroyAllWindows();
	}

	//std::string inpaint_img_path = InpaintImgRoot + relativePath;
	//std::string inpaintRelativePath = inpaint_img_path.substr(0, inpaint_img_path.find_last_of('\\'));

	//if (false == boost::filesystem::exists(inpaintRelativePath))
	//	boost::filesystem::create_directories(inpaintRelativePath);

	//cv::imwrite(inpaint_img_path, inpaintImg);

	SYY::Inpainting::ReleaseInpaint(hHandle);
	SYY::Inpainting::ReleaseInpaint(hHandle2);
}

void ChangeTxtInfo(const std::string& txt)
{
	const std::string ImgRoot = "D:\\blue\\data\\乳腺癌图片\\";
	const std::string txtSaveRoot = "D:\\blue\\codes\\TagTools\\TagTools\\RectTxtProcess\\new_txtRect\\";

	std::vector<std::vector<cv::Rect2f>> vPtss;
	std::vector<cv::Rect2f> vRects, vAdds;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vPtss, vRects, vAdds, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return;
	}

	std::string img_path = ImgRoot + relativePath;
	cv::Mat img = cv::imread(img_path);
	if (img.empty())
	{
		std::cout << "error when open img: " << img_path << std::endl;
		return;
	}

	std::string relativeFilename = relativePath.substr(0, relativePath.find_last_of('.'));
	std::string relativeTxtFile = relativeFilename + ".txt";

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects;

	const cv::Mat element3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	const cv::Mat element5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	const int margin = 1.0f;

	for (const auto& vPts : vPtss)
	{
		std::vector<cv::Rect2f> rects;
		for (auto rect : vPts)
		{
			rect.x = (int)std::max(0.f, rect.x - margin);
			rect.y = (int)std::max(0.f, rect.y - margin);
			rect.width = (int)std::min((float)img.cols, rect.width + 2 * margin + 0.5f);
			rect.height = (int)std::min((float)img.rows, rect.height + 2 * margin + 0.5f);

			auto region = img(rect);

			cv::cvtColor(region, region, CV_RGB2GRAY);
			cv::threshold(region, region, 0, 255, cv::THRESH_OTSU);
			cv::morphologyEx(region, region, cv::MORPH_DILATE, element5x5);
			std::vector<cv::Rect> bboxes;
			GetContoursBBox(region, bboxes);

			cv::Rect bbox;

			if (bboxes.size() == 0)
			{
				bbox = rect;
			}
			else if (bboxes.size() >= 1)
			{
				if (bboxes.size() >= 2)
				{
					std::sort(bboxes.begin(), bboxes.end(), [](const cv::Rect& a, const cv::Rect& b){
						return a.area() > b.area();
					});
				}
				bbox = bboxes[0];
			}
			bbox.x += rect.x;
			bbox.y += rect.y;

			//cv::rectangle(img, bbox, cv::Scalar(255, 255, 255));
			//cv::imshow("img", img);
			//cv::waitKey();

			rects.push_back(bbox);
		}

		vPtsRects.push_back(rects);
	}

	for (auto rect : vRects)
	{
		cv::Rect r = rect;
		vImgRects.push_back(r);
	}

	std::string txtFile;
	if (false == SaveInfo2Txt(vPtsRects, vImgRects, relativePath, txtSaveRoot, txtFile, errorMsg))
	{
		std::cout << "SaveInfo2Txt error: " << errorMsg << "\n";
		return;
	}
}

void CreatePosNegImgs4Adaboost(const std::string& txt)
{
	const std::string ImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs\\";
	const std::string posImgsRoot = "D:\\blue\\data\\训练文件\\adaboost\\b-scan\\pos\\";
	const std::string negImgsRoot = "D:\\blue\\data\\训练文件\\adaboost\\b-scan\\neg\\";

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return;
	}

	std::string img_path = ImgRoot + relativePath;
	cv::Mat img = cv::imread(img_path);
	if (img.empty())
	{
		std::cout << "error when open img: " << img_path << std::endl;
		return;
	}

	std::string relativeFilename = relativePath.substr(0, relativePath.find_last_of('.'));
	std::string PosRelativePath = posImgsRoot + relativePath.substr(0, relativePath.find_last_of('\\'));

	if (false == boost::filesystem::exists(PosRelativePath))
		boost::filesystem::create_directories(PosRelativePath);

	std::string NegRelativePath = negImgsRoot + relativePath.substr(0, relativePath.find_last_of('\\'));

	if (false == boost::filesystem::exists(NegRelativePath))
		boost::filesystem::create_directories(NegRelativePath);

	std::vector<cv::Rect> gt_bboxes;

	const auto GetRect = [](const std::vector<cv::Rect2f>& vRects) -> cv::Rect {

		int min_x = INT_MAX, min_y = INT_MAX, max_x = INT_MIN, max_y = INT_MIN;

		for (auto rect : vRects)
		{
			auto pt = (rect.tl() + rect.br()) / 2;

			if (pt.x < min_x) min_x = pt.x;
			if (pt.x > max_x) max_x = pt.x;
			if (pt.y < min_y) min_y = pt.y;
			if (pt.y > max_y) max_y = pt.y;
		}

		if (vRects.size() == 0)
			return cv::Rect(0, 0, 0, 0);

		return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
	};

	const float radio = 0.1f;

	auto negImg = img.clone();

	int rect_no = 0;
	for (const auto& vRects : vPtsRects)
	{
		auto bbox = GetRect(vRects);

		bbox.x = int(bbox.x - bbox.width * radio);
		bbox.y = int(bbox.y - bbox.height * radio);
		bbox.width = int(bbox.width * (1 + 2 * radio) + 0.5f);
		bbox.height = int(bbox.height * (1 + 2 * radio) + 0.5f);

		gt_bboxes.push_back(bbox);

		auto region = img(bbox);

		negImg(bbox) = cv::Scalar(0, 0, 0);

		std::stringstream ss;
		ss << posImgsRoot << relativeFilename << "_" << rect_no << ".jpg";

		cv::imwrite(ss.str(), region);
	}

	if (vImgRects.size() != 1)
		return;

	cv::Rect img_bbox = vImgRects[0];

	static int neg_no = 0;

	std::stringstream ss;
	ss << negImgsRoot << relativeFilename << "_" << neg_no++ << ".jpg";

	cv::imwrite(ss.str(), negImg(img_bbox));
}

void CreatePosNegCrossImgs4Adaboost(const std::string& txt)
{
	const std::string ImgRoot = "D:\\blue\\data\\乳腺癌图片\\";
	const std::string txtSaveRoot = "D:\\blue\\codes\\TagTools\\TagTools\\RectTxt\\";
	const std::string posImgsRoot = "D:\\blue\\data\\训练文件\\adaboost\\cross\\pos\\";
	const std::string negImgsRoot = "D:\\blue\\data\\训练文件\\adaboost\\cross\\neg\\";

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return;
	}

	if (vPtsRects.size() == 0)
	{
		return;
	}

	std::string img_path = ImgRoot + relativePath;
	cv::Mat img = cv::imread(img_path);
	if (img.empty())
	{
		std::cout << "error when open img: " << img_path << std::endl;
		return;
	}

	// process image
	cv::Mat gray, edge;
	cv::cvtColor(img, gray, CV_RGB2GRAY);
	cv::Canny(gray, edge, 50, 300);
	//cv::adaptiveThreshold(gray, gray, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 3, 0);

	//cv::imshow("img", img);
	//cv::imshow("gray", gray);
	//cv::imshow("edge", edge);
	//cv::waitKey();

	//

	std::string relativeFilename = relativePath.substr(0, relativePath.find_last_of('.'));
	std::string PosRelativePath = posImgsRoot + relativePath.substr(0, relativePath.find_last_of('\\'));

	if (false == boost::filesystem::exists(PosRelativePath))
		boost::filesystem::create_directories(PosRelativePath);

	std::string NegRelativePath = negImgsRoot + relativePath.substr(0, relativePath.find_last_of('\\'));

	if (false == boost::filesystem::exists(NegRelativePath))
		boost::filesystem::create_directories(NegRelativePath);

	std::vector<cv::Rect> gt_bboxes;

	const auto GetRect = [](const std::vector<cv::Rect2f>& vRects) -> cv::Rect {

		int min_x = INT_MAX, min_y = INT_MAX, max_x = INT_MIN, max_y = INT_MIN;

		for (auto rect : vRects)
		{
			auto pt = (rect.tl() + rect.br()) / 2;

			if (pt.x < min_x) min_x = pt.x;
			if (pt.x > max_x) max_x = pt.x;
			if (pt.y < min_y) min_y = pt.y;
			if (pt.y > max_y) max_y = pt.y;
		}

		if (vRects.size() == 0)
			return cv::Rect(0, 0, 0, 0);

		return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
	};

	const float margin = 0.0f;

	static int pos_no = 0;
	static int neg_no = 0;

	auto negImg = edge.clone();

	for (const auto& vRects : vPtsRects)
	{
		for (auto rect : vRects)
		{
			rect.x -= margin;
			rect.y -= margin;
			rect.width += 2 * margin;
			rect.height += 2 * margin;

			auto region = edge(rect);

			std::stringstream ss;
			ss << posImgsRoot << relativeFilename << "_" << pos_no++ << ".jpg";


			cv::imwrite(ss.str(), region);
			negImg(rect) = cv::Scalar(0, 0, 0);

		}
	}

	if (vImgRects.size() != 1)
		return;

	std::stringstream ss;
	ss << negImgsRoot << relativeFilename << "_" << neg_no++ << ".jpg";

	cv::imwrite(ss.str(), negImg);
}

void DetectCross(const std::string& imgFile)
{
	static cv::CascadeClassifier g_crossDetector("D:\\blue\\data\\训练文件\\adaboost\\cross\\cascade.xml");
	auto srcImg = cv::imread(imgFile);
	if (srcImg.empty())
	{
		std::cout << "can not open image file: " << imgFile << std::endl;
		return;
	}

	std::vector<cv::Rect> detections;
	const float scaleFactor = 1.01f;
	cv::Mat img_processed;
	cv::cvtColor(srcImg, img_processed, CV_RGB2GRAY);
	cv::Canny(img_processed, img_processed, 50, 300);
	g_crossDetector.detectMultiScale(img_processed, detections, scaleFactor, 2, 0, cv::Size(10, 10), cv::Size(30, 30));

	for (auto rect : detections)
	{
		cv::rectangle(srcImg, rect, cv::Scalar(255, 255, 255));
	}

	cv::imshow("srcImg", srcImg);
	cv::imshow("img_processed", img_processed);
	cv::waitKey();
}

void DetectNode(const std::string& imgFile)
{
	static cv::CascadeClassifier g_crossDetector("D:\\blue\\data\\训练文件\\adaboost\\b-scan\\cascade.xml");
	auto srcImg = cv::imread(imgFile);
	if (srcImg.empty())
	{
		std::cout << "can not open image file: " << imgFile << std::endl;
		return;
	}

	std::vector<cv::Rect> detections;
	const float scaleFactor = 1.1f;
	g_crossDetector.detectMultiScale(srcImg, detections, scaleFactor, 5, 0);

	for (auto rect : detections)
	{
		cv::rectangle(srcImg, rect, cv::Scalar(255, 255, 255));
	}

	cv::imshow("srcImg", srcImg);
	cv::waitKey();
}

void Create_VOC_Annots(const std::string& txt) {
	using namespace boost::property_tree;
	ptree pt;

	const std::string ImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs_crimisi\\";
	const std::string AnnotSaveRoot = "D:\\blue\\data\\训练文件\\ssd\\检测病灶\\annots\\";
	const std::string ImgSaveRoot = "D:\\blue\\data\\训练文件\\ssd\\检测病灶\\imgs\\";

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return;
	}

	if (vPtsRects.size() == 0)
	{
		return;
	}

	std::string img_path = ImgRoot + relativePath;
	cv::Mat img = cv::imread(img_path);
	if (img.empty())
	{
		std::cout << "error when open img: " << img_path << std::endl;
		return;
	}

	if (vImgRects.size() != 1)
		return;

	static int img_no = 0;

	cv::Rect imgBox = vImgRects[0];

	const auto& set_node = [&](const std::string& part_name, const cv::Rect& rect){
		ptree& node = pt.add("annotation.object", "");
		node.put("name", part_name);
		node.put("difficult", 0);

		int xmin = std::max(0, rect.x);
		int ymin = std::max(0, rect.y);
		int xmax = std::min(xmin + rect.width, img.cols - 1);
		int ymax = std::min(ymin + rect.height, img.rows - 1);

		//cv::Rect roi(xmin, ymin, xmax - xmin, ymax - ymin);
		//cv::Mat image = img(imgBox)(roi);
		//cv::imshow("1", image);
		//cv::waitKey(1000);

		ptree& bndbox = node.add("bndbox", "");
		bndbox.add("xmin", xmin);
		bndbox.add("ymin", ymin);
		bndbox.add("xmax", xmax);
		bndbox.add("ymax", ymax);
	};

	const auto GetRect = [](const std::vector<cv::Rect2f>& vRects) -> cv::Rect {

		int min_x = INT_MAX, min_y = INT_MAX, max_x = INT_MIN, max_y = INT_MIN;

		for (auto rect : vRects)
		{
			auto pt = (rect.tl() + rect.br()) / 2;

			if (pt.x < min_x) min_x = pt.x;
			if (pt.x > max_x) max_x = pt.x;
			if (pt.y < min_y) min_y = pt.y;
			if (pt.y > max_y) max_y = pt.y;
		}

		if (vRects.size() == 0)
			return cv::Rect(0, 0, 0, 0);

		return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
	};

	pt.add("annotation.size.height", imgBox.height);
	pt.add("annotation.size.width", imgBox.width);

	for (auto vRects : vPtsRects)
	{
		auto rect = GetRect(vRects);
		rect.x -= imgBox.x;
		rect.y -= imgBox.y;

		set_node("node", rect);
	}

	std::stringstream ss;
	ss << AnnotSaveRoot << img_no << ".xml";
	std::string annot_file = ss.str();
	std::ofstream xml_out(annot_file);
	write_xml(xml_out, pt);

	ss.str("");
	ss << ImgSaveRoot << img_no << ".jpg";
	std::string img_file = ss.str();
	cv::imwrite(img_file, img(imgBox));

	img_no += 1;
}

void Create_VOC_Annots_With_Filter(const std::string& txt) {
	using namespace boost::property_tree;
	ptree pt;

	const std::string ImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs_crimisi\\";
	//const std::string ImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs\\";

	const std::string AnnotSaveRoot = "D:\\blue\\data\\训练文件\\ssd\\检测病灶\\annots_v2\\";
	const std::string ImgSaveRoot = "D:\\blue\\data\\训练文件\\ssd\\检测病灶\\imgs_v2\\";

	const std::string TrainAnnotSaveRoot = AnnotSaveRoot + "train\\";
	const std::string TrainImgSaveRoot = ImgSaveRoot + "train\\";

	const std::string TestAnnotSaveRoot = AnnotSaveRoot + "test\\";
	const std::string TestImgSaveRoot = ImgSaveRoot + "test\\";

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects, vAddRects;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, vAddRects, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return;
	}

	if (vPtsRects.size() == 0 && vAddRects.size() == 0) 
		return;

	if (vImgRects.size() != 1) 
		return;

	std::string img_path = ImgRoot + relativePath;
	cv::Mat img = cv::imread(img_path);
	if (img.empty())
	{
		std::cout << "error when open img: " << img_path << std::endl;
		return;
	}

	static int img_no = 0;

	cv::Rect imgBox = vImgRects[0];

	const auto& set_node = [&](const std::string& part_name, const cv::Rect& rect){
		ptree& node = pt.add("annotation.object", "");
		node.put("name", part_name);
		node.put("difficult", 0);

		int xmin = std::max(0, rect.x);
		int ymin = std::max(0, rect.y);
		int xmax = std::min(xmin + rect.width, img.cols - 1);
		int ymax = std::min(ymin + rect.height, img.rows - 1);

		//cv::Rect roi(xmin, ymin, xmax - xmin, ymax - ymin);
		//cv::Mat image = img(imgBox)(roi);
		//cv::imshow("1", image);
		//cv::waitKey(1000);

		ptree& bndbox = node.add("bndbox", "");
		bndbox.add("xmin", xmin);
		bndbox.add("ymin", ymin);
		bndbox.add("xmax", xmax);
		bndbox.add("ymax", ymax);
	};

	pt.add("annotation.size.height", imgBox.height);
	pt.add("annotation.size.width", imgBox.width);

	for (auto vRects : vPtsRects)
	{
		auto rect = GetRect(vRects);
		rect.x -= imgBox.x;
		rect.y -= imgBox.y;

		set_node("node", rect);
	}

	for (auto rect : vAddRects)
	{
		rect.x -= imgBox.x;
		rect.y -= imgBox.y;

		set_node("node", rect);
	}

	std::stringstream ss;
	std::string annot_file, img_file;

	if (vAddRects.size() > 0 && vPtsRects.size() == 0)
	{
		ss << TestAnnotSaveRoot << img_no << ".xml";
		annot_file = ss.str();

		ss.str("");
		ss << TestImgSaveRoot << img_no << ".jpg";
		img_file = ss.str();
	}
	else {
		ss << TrainAnnotSaveRoot << img_no << ".xml";
		annot_file = ss.str();

		ss.str("");
		ss << TrainImgSaveRoot << img_no << ".jpg";
		img_file = ss.str();
	}

	auto img_file_root = img_file.substr(0, img_file.find_last_of('\\'));
	if (false == boost::filesystem::exists(img_file_root))
		boost::filesystem::create_directories(img_file_root);

	cv::imwrite(img_file, img(imgBox));

	auto annot_file_root = annot_file.substr(0, annot_file.find_last_of('\\'));
	if (false == boost::filesystem::exists(annot_file_root))
		boost::filesystem::create_directories(annot_file_root);

	std::ofstream xml_out(annot_file);
	write_xml(xml_out, pt);

	img_no += 1;
}

float CalcIOU(cv::Rect a, cv::Rect b)
{
	float i = (a & b).area();
	float u = std::abs(a.area() + b.area() - i);

	return i / u;
}

SYY::HANDLE hHandleBUAnalysis;
void GenerateAdditionRect(const std::string& txt)
{
	const std::string ImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs_crimisi\\";
	const std::string txtSaveRoot = "D:\\blue\\codes\\TagTools\\x64\\标注工具\\RectTxtAddtion\\";

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return;
	}

	auto txtSaveFile = txtSaveRoot + relativePath.substr(0, relativePath.find_last_of('.')) + ".txt";
	if (true == boost::filesystem::exists(txtSaveFile))
	{
		std::cout << "txt exist!\n";
		return;
	}

	std::string img_path = ImgRoot + relativePath;
	cv::Mat img = cv::imread(img_path);
	if (img.empty())
	{
		std::cout << "error when open img: " << img_path << std::endl;
		return;
	}

	static SYY::MedicalAnalysis::BUAnalysisResult result;
	if (SYY::SYY_NO_ERROR != SYY::MedicalAnalysis::ExecuteBUAnalysis(hHandleBUAnalysis, (char*)img.data, img.cols, img.rows, &result))
	{
		std::cout << "execute BU analysis error!\n";
		return;
	}

	const float max_overlay = 0.5f;

	std::vector<cv::Rect2f> vAddRects;
	for (int i = 0; i < result.nLessionsCount; i++)
	{
		SYY::Rect detRect = result.pLessionRects[i];
		cv::Rect2f r(detRect.x, detRect.y, detRect.w, detRect.h);

		bool isMatch = false;
		for (auto vRect : vPtsRects)
		{
			auto rect = GetRect(vRect);
			if (max_overlay < CalcIOU(rect, r))
			{
				isMatch = true;
				break;
			}
		}

		if (isMatch) continue;

		cv::rectangle(img, r, cv::Scalar(255, 255, 255));

		vAddRects.push_back(r);
	}

	std::string txtFile;
	if (false == SaveInfo2Txt(vPtsRects, vImgRects, vAddRects, relativePath, txtSaveRoot, txtFile, errorMsg))
	{
		std::cout << "save info error: " << errorMsg << "\n";
		return;	
	}

	cv::imshow("img", img);
	cv::waitKey(1);
}

void main(int argc, char** argv)
{
	if (SYY::SYY_NO_ERROR != SYY::InitSDK())
	{
		std::cerr << "InitSDK error!" << std::endl;
		return;
	}

	if (SYY::SYY_NO_ERROR != SYY::MedicalAnalysis::InitBUAnalysis(hHandleBUAnalysis, SYY::MedicalAnalysis::Crop_V1))
	{
		std::cerr << "InitBUAnalysis error!\n" << std::endl;
		return;
	}

	//argc = 2;
	//argv[1] = "D:\\blue\\data\\训练文件\\test\\";
	//ProcessAllFile(argc, argv, ParseTxt);

	//argc = 2;
	//argv[1] = (char*)std::string("D:\\blue\\data\\训练文件\\" + Phase + "\\").c_str();
	//ProcessAllFile(argc, argv, ParseTxtForImgBBox);

	//argc = 2;
	//argv[1] = "D:\\blue\\data\\乳腺癌图片";
	//argv[1] = "D:\\blue\\codes\\TagTools\\testImgs";
	//ProcessAllFile(argc, argv, Preprocess);

	//argc = 2;
	//argv[1] = "D:\\blue\\codes\\TagTools\\TagTools\\RectTxtProcess\\unchange\\";
	//ProcessAllFile(argc, argv, ChangeRelativePath);

	//argc = 2;
	//argv[1] = "D:\\blue\\codes\\TagTools\\TagTools\\RectTxt\\4a类01\\";
	//ProcessAllFile(argc, argv, DrawGT);

	//argc = 2;
	//argv[1] = "D:\\blue\\codes\\TagTools\\TagTools\\RectTxt\\";
	//ProcessAllFile(argc, argv, RemoveImage);

	//argc = 2;
	//argv[1] = "D:\\blue\\codes\\TagTools\\x64\\标注工具\\RectTxt\\";
	//ProcessAllFile(argc, argv, Inpainting);

	//argc = 2;
	//argv[1] = "D:\\blue\\codes\\TagTools\\x64\\标注工具\\RectTxt\\";
	//ProcessAllFile(argc, argv, CreatePosNegImgs4Adaboost);

	//argc = 2;
	//argv[1] = "D:\\blue\\codes\\TagTools\\x64\\标注工具\\RectTxt\\";
	//ProcessAllFile(argc, argv, CreatePosNegCrossImgs4Adaboost, 0);

	//argc = 2;
	//argv[1] = "D:\\blue\\codes\\TagTools\\TagTools\\RectTxt1\\";
	//ProcessAllFile(argc, argv, Inpainting_old);

	//argc = 2;
	//argv[1] = "D:\\blue\\codes\\TagTools\\x64\\标注工具\\RectTxt\\";
	//ProcessAllFile(argc, argv, ChangeTxtInfo);

	//argc = 2;
	//argv[1] = "D:\\blue\\data\\乳腺癌图片\\";
	//ProcessAllFile(argc, argv, DetectCross);

	//argc = 2;
	//argv[1] = "D:\\blue\\data\\训练文件\\检测病灶\\imgs\\";
	//ProcessAllFile(argc, argv, DetectNode);

	argc = 2;
	argv[1] = "D:\\blue\\codes\\TagTools\\x64\\标注工具\\RectTxt\\";
	ProcessAllFile(argc, argv, Create_VOC_Annots_With_Filter);

	//argc = 2;
	//argv[1] = "D:\\blue\\codes\\TagTools\\x64\\标注工具\\RectTxt\\";
	//ProcessAllFile(argc, argv, GenerateAdditionRect);

	//cv::Mat m = cv::Mat::zeros(3, 3, CV_8UC1);
	//cv::Mat h, v;
	//m.at<uchar>(0, 1) = 1;
	//m.at<uchar>(2, 1) = 1;
	//CalHVProj(m, h, v);

	//std::cout << m << std::endl;
	//std::cout << h << std::endl;
	//std::cout << v << std::endl;
	SYY::MedicalAnalysis::ReleaseBUAnalysis(hHandleBUAnalysis);
	SYY::ReleaseSDK();

	system("pause");
}
