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

void ProcessAllFile(int argc, char** argv, Func func, int stop_no = 0, bool is_reserve = false)
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

		if (is_reserve)
		{
			std::vector<std::string> temp;
			for (int i = files.size() - 1; i >= 0; i--)
				temp.push_back(files[i]);
			files = temp;
		}

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

const auto Check_Create_Path = [](std::string path){
	if (false == boost::filesystem::exists(path))
		boost::filesystem::create_directories(path);
};

void ParseTxt(const std::string& txt)
{
	//const std::string ImgRoot = "D:\\blue\\data\\乳腺癌图片\\";
	//const std::string ImgRoot = "D:\\迅雷下载\\inpaint_imgs\\";
	const std::string ImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs\\";

	const std::string root = "D:\\blue\\data\\训练文件\\检测病灶\\";
	const std::string trainingImgsRoot = root + "imgs\\";
	const std::string trainvalAnnotsFilePath = root + "\\" + "train.txt";
	const std::string testAnnotsFilePath = root + "\\" + "test.txt";

	static std::ofstream trainAnnotFile(trainvalAnnotsFilePath);
	static std::ofstream testAnnotFile(testAnnotsFilePath);

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects, vAddRects;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, vAddRects, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return;
	}

	if (vImgRects.size() != 1)
	{
		return;
	}

	if (vPtsRects.size() == 0 && vAddRects.size() == 0)
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

	static int img_no = 0;
	static int trainval_img_no = 0;
	static int test_img_no = 0;

	cv::Rect r = vImgRects[0];
	auto part = img(r);

	auto fullPath = trainingImgsRoot + relativePath;
	auto fullRelativePath = fullPath.substr(0, fullPath.find_last_of('\\'));

	if (false == boost::filesystem::exists(fullRelativePath))
	{
		boost::filesystem::create_directories(fullRelativePath);
	}

	cv::imwrite(fullPath, part);

	std::vector<cv::Rect> all_bboxes;
	for (const auto& vRects : vPtsRects)
	{
		auto bbox = GetRect(vRects);
		bbox.x -= r.x;
		bbox.y -= r.y;

		all_bboxes.push_back(bbox);
	}

	for (auto bbox : vAddRects)
	{
		bbox.x -= r.x;
		bbox.y -= r.y;

		all_bboxes.push_back(bbox);
	}

	std::vector<cv::Vec4i> all_coords;
	for (auto bbox : all_bboxes)
	{
		float radio = 0.01f;

		bbox.x = int(bbox.x - bbox.width * radio + 0.5f);
		bbox.y = int(bbox.y - bbox.height * radio + 0.5f);
		bbox.width = int(bbox.width * (1 + 2 * radio) + 0.5f);
		bbox.height = int(bbox.height * (1 + 2 * radio) + 0.5f);

		if (bbox.x > part.cols || bbox.y > part.rows)
		{
			std::cout << "bbox out of img size!\n";
			continue;
		}

		cv::rectangle(part, bbox, cv::Scalar(255, 255, 255), 2);

		auto tl = bbox.tl();
		auto br = bbox.br();
		tl.x = std::max(0, std::min(tl.x, r.width));
		tl.y = std::max(0, std::min(tl.y, r.height));
		br.x = std::max(tl.x, std::min(br.x, r.width));
		br.y = std::max(tl.y, std::min(br.y, r.height));

		all_coords.push_back(cv::Vec4i(tl.x, tl.y, br.x, br.y));
	}

	if (vPtsRects.size() != 0)
	{
		trainAnnotFile << "# " << trainval_img_no++ << std::endl;
		trainAnnotFile << relativePath << std::endl;
		trainAnnotFile << vPtsRects.size() + vAddRects.size() << std::endl;

		for (auto coords : all_coords)
		{
			trainAnnotFile << "1 " << coords[0] << " " << coords[1] << " " 
				<< coords[2] << " " << coords[3] << " 0" << std::endl;
		}
	}
	else if (vPtsRects.size() == 0 && vAddRects.size() != 0)
	{
		testAnnotFile << "# " << test_img_no++ << std::endl;
		testAnnotFile << relativePath << std::endl;
		testAnnotFile << vAddRects.size() << std::endl;

		for (auto coords : all_coords)
		{
			testAnnotFile << "1 " << coords[0] << " " << coords[1] << " " 
				<< coords[2] << " " << coords[3] << " 0" << std::endl;
		}
	}

	cv::imshow("part", part);
	cv::waitKey(1);
}

void ParseTxtForImgBBox(const std::string& txt)
{
	//const std::string ImgRoot = "D:\\迅雷下载\\inpaint_imgs\\"; 
	//const std::string root = "D:\\blue\\data\\训练文件\\检测病灶\\"; 
	const std::string ImgRoot = "E:\\data\\inpaint_imgs_crimisi\\"; 
	const std::string root = "E:\\data\\train_files\\detect_lessions\\"; 


	const std::string trainingImgsRoot = root + "imgs\\";
	const std::string trainvalAnnotsFilePath = root + "\\" + "train.txt";
	const std::string testAnnotsFilePath = root + "\\" + "test.txt";

	Check_Create_Path(root);

	static std::ofstream trainAnnotFile(trainvalAnnotsFilePath);
	static std::ofstream testAnnotFile(testAnnotsFilePath);

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects, vAddRects;
	std::string relativePath, errorMsg;

	std::map<std::string, std::string> mAttrs;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, vAddRects, mAttrs, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return;
	}

	if (vImgRects.size() != 1)
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

	static int img_no = 0;
	static int trainval_img_no = 0;
	static int test_img_no = 0;

	cv::Rect bbox = vImgRects[0];

	auto fullPath = trainingImgsRoot + relativePath;
	auto fullRelativePath = fullPath.substr(0, fullPath.find_last_of('\\'));

	if (false == boost::filesystem::exists(fullRelativePath))
	{
		boost::filesystem::create_directories(fullRelativePath);
	}

	cv::imwrite(fullPath, img);


	std::vector<cv::Vec4i> all_coords;

	float radio = 0.005f;

	bbox.x = int(bbox.x - bbox.width * radio + 0.5f);
	bbox.y = int(bbox.y - bbox.height * radio + 0.5f);
	bbox.width = int(bbox.width * (1 + 2 * radio) + 0.5f);
	bbox.height = int(bbox.height * (1 + 2 * radio) + 0.5f);

	auto tl = bbox.tl();
	auto br = bbox.br();
	tl.x = std::max(0, std::min(tl.x, img.cols));
	tl.y = std::max(0, std::min(tl.y, img.rows));
	br.x = std::max(tl.x, std::min(br.x, img.cols));
	br.y = std::max(tl.y, std::min(br.y, img.rows));

	bbox = cv::Rect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);

	cv::rectangle(img, bbox, cv::Scalar(255, 255, 255));

	auto coords = cv::Vec4i(tl.x, tl.y, br.x, br.y);

	int randNum = std::rand() % 10;

	if (randNum < 8)
	{
		trainAnnotFile << "# " << trainval_img_no++ << std::endl;
		trainAnnotFile << relativePath << std::endl;
		trainAnnotFile << 1 << std::endl;

		trainAnnotFile << "1 " << coords[0] << " " << coords[1] << " "
			<< coords[2] << " " << coords[3] << " 0" << std::endl;
	}
	else if (vPtsRects.size() == 0 && vAddRects.size() != 0)
	{
		testAnnotFile << "# " << test_img_no++ << std::endl;
		testAnnotFile << relativePath << std::endl;
		testAnnotFile << 1 << std::endl;

		testAnnotFile << "1 " << coords[0] << " " << coords[1] << " "
			<< coords[2] << " " << coords[3] << " 0" << std::endl;
	}

	cv::imshow("img", img);
	cv::waitKey(1);
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
	std::vector<cv::Rect2f> vImgRects, vAddRects;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, vAddRects, relativePath, errorMsg))
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
	std::map<std::string, std::string> mAttrs;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, vAddRects, mAttrs, relativePath, errorMsg))
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
//
//void Inpainting_old(const std::string& txt)
//{
//	SYY::HANDLE hHandle, hHandle2;
//	if (SYY::SYY_NO_ERROR != SYY::Inpainting::InitInpaint(hHandle, SYY::Inpainting::PatchMatch))
//	{
//		std::cerr << "InitPaint error!\n" << std::endl;
//		return;
//	}
//	if (SYY::SYY_NO_ERROR != SYY::Inpainting::InitInpaint(hHandle2, SYY::Inpainting::Criminisi_P5))
//	{
//		std::cerr << "InitPaint error!\n" << std::endl;
//		return;
//	}
//
//	const std::string ImgRoot = "D:\\blue\\data\\乳腺癌图片\\";
//	const std::string InpaintImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs\\";
//
//	std::vector<std::vector<cv::Point>> vPtss;
//	std::vector<cv::Rect> vRects;
//	std::string relativePath, errorMsg;
//
//	if (false == ParseTxtInfo(txt, vPtss, vRects, relativePath, errorMsg))
//	{
//		std::cout << errorMsg << std::endl;
//		return;
//	}
//
//	std::string img_path = ImgRoot + relativePath;
//	cv::Mat img = cv::imread(img_path);
//	if (img.empty())
//	{
//		std::cout << "error when open img: " << img_path << std::endl;
//		return;
//	}
//
//	cv::Mat inpaintImg;
//	if (vPtss.size() == 0)
//	{
//		inpaintImg = img;
//	}
//	else
//	{
//		auto srcImg = img.clone();
//		cv::Mat element5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
//		cv::Mat element3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
//		cv::Mat maskImg = cv::Mat::zeros(img.size(), CV_8UC1);
//		int radius = 20;
//		//if (srcImg.cols > 1000) radius = 15;
//
//		for (auto vPts : vPtss)
//		{
//			for (auto pt : vPts)
//			{
//				auto rect = cv::Rect(pt.x + 1 - radius / 2, pt.y + 1 - radius / 2, radius, radius);
//
//				//auto rect = cv::Rect(pt.x, pt.y, radius, radius);
//
//				maskImg(rect) = 255;
//				srcImg(rect) = cv::Scalar(255, 255, 255);
//				//cv::rectangle(srcImg, rect, cv::Scalar(255, 255, 255));
//
//				//cv::circle(maskImg, pt, radius, cv::Scalar(255), -1);
//
//				auto region = img(rect);
//				//cv::cvtColor(region, region, CV_RGB2GRAY);
//				//cv::morphologyEx(region, region, cv::MORPH_DILATE, element3x3);
//
//				cv::Mat m = maskImg(rect);
//				//cv::Canny(region, m, 0, 255);
//				//cv::threshold(region, m, 0, 255, CV_THRESH_OTSU);
//				//cv::morphologyEx(m, m, cv::MORPH_DILATE, element5x5);
//				//cv::morphologyEx(m, m, cv::MORPH_CLOSE, element);
//				//cv::dilate(m, m, element3x3);
//				//cv::erode(m, m, element3x3);
//
//				//ShowHistImg(region);
//
//				//cv::imshow("region", region);
//				//cv::imshow("m", m);
//				//cv::waitKey();
//
//				//std::vector<cv::Rect> bboxes;
//				//GetContoursBBox(m, bboxes);
//
//				//if (bboxes.size() > 1)
//				//{
//				//	std::sort(bboxes.begin(), bboxes.end(), [](const cv::Rect& a, const cv::Rect& b){
//				//		return a.area() > b.area();
//				//	});
//				//}
//
//				//for (auto bbox : bboxes) cv::rectangle(m, bbox, cv::Scalar(128));
//
//				//int min_len = std::min(1, int(bboxes.size()));
//				//for (int i = 0; i < min_len; i++)
//				//{
//				//	img(rect)(bboxes[i]) = cv::Scalar(255, 255, 255);
//				//	m(bboxes[i]) = 255;
//				//}
//
//				//cv::dilate(m, m, element3x3);
//
//				//cv::imshow("region", region);
//				//cv::imshow("m", m);
//				//cv::imshow("img", img);
//				//cv::imshow("mask", maskImg);
//				//cv::waitKey();
//				//cv::circle(maskImg, pt, radius, cv::Scalar(255), -1);
//				//cv::circle(img, pt, radius, cv::Scalar(255), -1);
//			}
//		}
//
//		//cv::imshow("srcImg", srcImg);
//		//cv::imshow("mask", maskImg);
//		//cv::waitKey();
//
//	//	static SYY::Image inpaint;
//
//	//	for (int i = 0; i < 1; i++)
//	//	{
//	//		SYY::Image
//	//			src((char*)srcImg.data, srcImg.cols, srcImg.rows, srcImg.channels()),
//	//			mask((char*)maskImg.data, maskImg.cols, maskImg.rows, maskImg.channels());
//
//	//		//if (SYY::SYY_NO_ERROR != SYY::Inpainting::ExecuteInpaint(hHandle, src, mask, inpaint))
//	//		//	return;
//
//	//		//inpaintImg = cv::Mat(inpaint.nHeight, inpaint.nWidth, CV_8UC3, inpaint.pData).clone();
//	//		cv::inpaint(srcImg, maskImg, inpaintImg, 5, cv::INPAINT_NS);
//
//	//		//cv::imshow("inpaingImg1", inpaintImg);
//
//	//		src = SYY::Image((char*)inpaintImg.data, inpaintImg.cols, inpaintImg.rows, inpaintImg.channels());
//
//	//		if (SYY::SYY_NO_ERROR != SYY::Inpainting::ExecuteInpaint(hHandle2, src, mask, inpaint))
//	//			return;
//
//	//		inpaintImg = cv::Mat(inpaint.nHeight, inpaint.nWidth, CV_8UC3, inpaint.pData);
//
//	//		srcImg = inpaintImg.clone();
//
//	//		//cv::imshow("inpaingImg2", inpaintImg);
//	//		//cv::waitKey();
//	//		//cv::dilate(maskImg, maskImg, element3x3);
//	//		//cv::imshow("mask", maskImg);
//	//		//cv::imshow("inpaingImg", inpaintImg);
//	//		//cv::waitKey();
//	//	}
//
//		cv::imshow("img", srcImg);
//		cv::imshow("mask", maskImg);
//	//	//cv::imshow("inpaingImg", inpaintImg);
//		cv::waitKey();
//	//	//cv::destroyAllWindows();
//	}
//
//	//std::string inpaint_img_path = InpaintImgRoot + relativePath;
//	//std::string inpaintRelativePath = inpaint_img_path.substr(0, inpaint_img_path.find_last_of('\\'));
//
//	//if (false == boost::filesystem::exists(inpaintRelativePath))
//	//	boost::filesystem::create_directories(inpaintRelativePath);
//
//	//cv::imwrite(inpaint_img_path, inpaintImg);
//
//	SYY::Inpainting::ReleaseInpaint(hHandle);
//	SYY::Inpainting::ReleaseInpaint(hHandle2);
//}
//
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

void DetectCrossWithAdaboost(const std::string& imgFile)
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

void DetectCross(const std::string imgFile)
{
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
	const float scaleFactor = 1.05f;
	g_crossDetector.detectMultiScale(srcImg, detections, scaleFactor, 3, 0);

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

	//const std::string ImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs_crimisi\\";
	//const std::string AnnotSaveRoot = "D:\\blue\\data\\训练文件\\ssd\\检测病灶\\annots_v2\\";
	//const std::string ImgSaveRoot = "D:\\blue\\data\\训练文件\\ssd\\检测病灶\\imgs_v2\\";
	const std::string ImgRoot = "E:\\data\\inpaint_imgs_crimisi\\";
	const std::string AnnotSaveRoot = "E:\\data\\train_files\\ssd\\detect_lession\\annots_v2\\";
	const std::string ImgSaveRoot = "E:\\data\\train_files\\ssd\\detect_lession\\imgs_v2\\";

	const std::string TrainAnnotSaveRoot = AnnotSaveRoot + "train\\";
	const std::string TrainImgSaveRoot = ImgSaveRoot + "train\\";

	const std::string TestAnnotSaveRoot = AnnotSaveRoot + "test\\";
	const std::string TestImgSaveRoot = ImgSaveRoot + "test\\";

	Check_Create_Path(TrainImgSaveRoot);
	Check_Create_Path(TrainAnnotSaveRoot);
	Check_Create_Path(TestImgSaveRoot);
	Check_Create_Path(TestAnnotSaveRoot);

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects, vAddRects;
	std::string relativePath, errorMsg;
	std::map<std::string, std::string> mAttrs;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, vAddRects, mAttrs, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return;
	}

	//if (vPtsRects.size() == 0 && vAddRects.size() == 0) return;

	if (vImgRects.size() != 1) 
		return;

	if (mAttrs.find("LessionType") == mAttrs.end())
		return;

	auto type = mAttrs["LessionType"];

	if (type == "0")
		return;

	auto typeName = "lession";

	//if (type == "1")
	//{
	//	typeName = "lession";
	//}
	//else if (type == "2")
	//{
	//	typeName = "lymphaden";
	//}

	auto cls = relativePath.substr(0, relativePath.find_first_of('\\'));

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

	std::vector<cv::Rect> all_rects;

	for (auto vRects : vPtsRects)
	{
		auto rect = GetRect(vRects);
		rect.x -= imgBox.x;
		rect.y -= imgBox.y;

		all_rects.push_back(rect);
	}

	//for (auto rect : vAddRects)
	//{
	//	rect.x -= imgBox.x;
	//	rect.y -= imgBox.y;

	//	all_rects.push_back(rect);
	//}

	std::stringstream ss;
	std::string annot_file, img_file;

	// train data
	if (cls == "1a类" || (vPtsRects.size() == 0 && vAddRects.size() == 0))
	{
		pt.clear();

		pt.add("annotation.size.height", imgBox.height);
		pt.add("annotation.size.width", imgBox.width);

		ss.str("");
		ss << TrainAnnotSaveRoot << img_no << ".xml";
		annot_file = ss.str();

		ss.str("");
		ss << TrainImgSaveRoot << img_no << ".jpg";
		img_file = ss.str();

		cv::imwrite(img_file, img(imgBox));

		std::ofstream xml_out(annot_file);
		write_xml(xml_out, pt);

		img_no += 1;
		return;
	}

	// test data
	if (vAddRects.size() > 0 && vPtsRects.size() == 0)
	{
		for (auto rect : vAddRects)
		{
			rect.x -= imgBox.x;
			rect.y -= imgBox.y;

			all_rects.push_back(rect);
		}

		pt.clear();

		pt.add("annotation.size.height", imgBox.height);
		pt.add("annotation.size.width", imgBox.width);

		for (auto rect : all_rects)
		{
			set_node(typeName, rect);
		}

		ss.str("");
		ss << TestAnnotSaveRoot << img_no << ".xml";
		annot_file = ss.str();

		ss.str("");
		ss << TestImgSaveRoot << img_no << ".jpg";
		img_file = ss.str();

		cv::imwrite(img_file, img(imgBox));

		std::ofstream xml_out(annot_file);
		write_xml(xml_out, pt);

		img_no += 1;

		return;
	}

	/// train data
	if (all_rects.size() != 0)
	{
		int enhanceCount = 10;

		while (enhanceCount-- > 0)
		{
			// image
			ss.str("");
			ss << TrainImgSaveRoot << img_no << ".jpg";
			img_file = ss.str();

			auto rect = imgBox;
			const float resize_radio = 1.0f + float((std::rand() % 1001) - 500) / 1000.f; // 0.5 ~ 1.5
			cv::Mat part, origin = img(imgBox);

			const int new_width = origin.cols * resize_radio;
			const int new_height = origin.rows * resize_radio;

			cv::resize(origin, part, cv::Size(new_width, new_height), 0, 0, resize_radio <= 1.0f ? cv::INTER_AREA : cv::INTER_CUBIC);

			const float color_transfer_radio = 1.0f + float((std::rand() % 1001) - 500) / 2000.f; // 0.75 ~ 1.25

			part.convertTo(part, CV_32FC3);
			part *= color_transfer_radio;
			part.convertTo(part, CV_8UC3);

			cv::imwrite(img_file, part);

			// end img

			// xml

			ss.str("");
			ss << TrainAnnotSaveRoot << img_no << ".xml";
			annot_file = ss.str();

			pt.clear();
			pt.add("annotation.size.height", new_height);
			pt.add("annotation.size.width", new_width);

			for (auto rect : all_rects)
			{
				rect.x *= resize_radio;
				rect.y *= resize_radio;
				rect.width *= resize_radio;
				rect.height *= resize_radio;

				set_node(typeName, rect);

				cv::rectangle(part, rect, cv::Scalar(255, 255, 255));
			}

			std::ofstream xml_out(annot_file);
			write_xml(xml_out, pt);

			img_no += 1;

			//cv::imshow("part", part);
			//cv::waitKey();
		}

		return;
	}

	return;
}

void Create_VOC_Annots_ImgBox(const std::string& txt) {
	using namespace boost::property_tree;
	ptree pt;

	const std::string ImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs_crimisi\\";
	//const std::string ImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs\\";

	const std::string AnnotSaveRoot = "D:\\blue\\data\\训练文件\\ssd\\检测图\\annots_v2\\";
	const std::string ImgSaveRoot = "D:\\blue\\data\\训练文件\\ssd\\检测图\\imgs_v2\\";

	const std::string TrainAnnotSaveRoot = AnnotSaveRoot + "train\\";
	const std::string TrainImgSaveRoot = ImgSaveRoot + "train\\";

	const std::string TestAnnotSaveRoot = AnnotSaveRoot + "test\\";
	const std::string TestImgSaveRoot = ImgSaveRoot + "test\\";

	const auto Check_Create_Path = [](std::string path){
		if (false == boost::filesystem::exists(path))
			boost::filesystem::create_directories(path);
	};

	Check_Create_Path(TrainImgSaveRoot);
	Check_Create_Path(TrainAnnotSaveRoot);
	Check_Create_Path(TestImgSaveRoot);
	Check_Create_Path(TestAnnotSaveRoot);

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

	std::vector<cv::Rect> all_rects;

	for (auto vRects : vPtsRects)
	{
		auto rect = GetRect(vRects);
		rect.x -= imgBox.x;
		rect.y -= imgBox.y;

		all_rects.push_back(rect);
	}

	std::stringstream ss;
	std::string annot_file, img_file;

	int randomNum = std::rand() % 10;

	// test data
	if (randomNum < 2)
	{
		pt.clear();

		pt.add("annotation.size.height", img.rows);
		pt.add("annotation.size.width", img.cols);

		set_node("imgBox", imgBox);

		ss.str("");
		ss << TestAnnotSaveRoot << img_no << ".xml";
		annot_file = ss.str();

		ss.str("");
		ss << TestImgSaveRoot << img_no << ".jpg";
		img_file = ss.str();

		cv::imwrite(img_file, img);

		std::ofstream xml_out(annot_file);
		write_xml(xml_out, pt);

		img_no += 1;

		cv::rectangle(img, imgBox, cv::Scalar(255, 255, 255));
		cv::imshow("img", img);
		cv::waitKey(1);

		return;
	}
	else /// train data
	{
		int enhanceCount = 10;

		while (enhanceCount-- > 0)
		{
			// image
			ss.str("");
			ss << TrainImgSaveRoot << img_no << ".jpg";
			img_file = ss.str();

			auto rect = imgBox;
			const float resize_radio = 1.0f + float((std::rand() % 1001) - 500) / 1000.f; // 0.5 ~ 1.5
			cv::Mat part, origin = img;

			const int new_width = origin.cols * resize_radio;
			const int new_height = origin.rows * resize_radio;

			cv::resize(origin, part, cv::Size(new_width, new_height), 0, 0, resize_radio <= 1.0f ? cv::INTER_AREA : cv::INTER_CUBIC);

			const float color_transfer_radio = 1.0f + float((std::rand() % 1001) - 500) / 2000.f; // 0.75 ~ 1.25

			part.convertTo(part, CV_32FC3);
			part *= color_transfer_radio;
			part.convertTo(part, CV_8UC3);

			cv::imwrite(img_file, part);

			// end img

			// xml

			ss.str("");
			ss << TrainAnnotSaveRoot << img_no << ".xml";
			annot_file = ss.str();

			pt.clear();
			pt.add("annotation.size.height", new_height);
			pt.add("annotation.size.width", new_width);

			rect.x *= resize_radio;
			rect.y *= resize_radio;
			rect.width *= resize_radio;
			rect.height *= resize_radio;

			set_node("imgBox", rect);

			cv::rectangle(part, rect, cv::Scalar(255, 255, 255));

			std::ofstream xml_out(annot_file);
			write_xml(xml_out, pt);

			img_no += 1;

			cv::imshow("part", part);
			cv::waitKey(1);
		}

		return;
	}

	return;
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

void GetTestImg(const std::string& txt)
{
	const std::string ImgRoot = "D:\\blue\\data\\乳腺癌图片\\";
	const std::string ImgSaveRoot = "D:\\blue\\data\\测试图片\\";

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects, vAddRects;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, vAddRects, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return;
	}

	if (vPtsRects.size() != 0 || vAddRects.size() == 0)
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

	std::string img_save_path = ImgSaveRoot + relativePath;
	std::string img_file_relative = img_save_path.substr(0, img_save_path.find_last_of('\\'));
	if (false == boost::filesystem::exists(img_file_relative))
		boost::filesystem::create_directories(img_file_relative);

	cv::imwrite(img_save_path, img);

}

//
//SYY::HANDLE hHandleBUAnalysis2;
//void DrawSSDResAndGT(const std::string& txt)
//{
//	const std::string ImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs_crimisi\\";
//	const std::string txtSaveRoot = "D:\\blue\\codes\\TagTools\\x64\\标注工具\\RectTxtAddtion\\";
//
//	std::vector<std::vector<cv::Rect2f>> vPtsRects;
//	std::vector<cv::Rect2f> vImgRects, vAddRects;
//	std::string relativePath, errorMsg;
//
//	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, relativePath, errorMsg))
//	{
//		std::cout << errorMsg << std::endl;
//		return;
//	}
//
//	auto txtSaveFile = txtSaveRoot + relativePath.substr(0, relativePath.find_last_of('.')) + ".txt";
//	if (true == boost::filesystem::exists(txtSaveFile))
//	{
//		std::cout << "txt exist!\n";
//		return;
//	}
//
//	std::string img_path = ImgRoot + relativePath;
//	cv::Mat img = cv::imread(img_path);
//	if (img.empty())
//	{
//		std::cout << "error when open img: " << img_path << std::endl;
//		return;
//	}
//
//	static SYY::MedicalAnalysis::BUAnalysisResult result;
//	if (SYY::SYY_NO_ERROR != SYY::MedicalAnalysis::ExecuteBUAnalysis(hHandleBUAnalysis, (char*)img.data, img.cols, img.rows, &result))
//	{
//		std::cout << "execute BU analysis error!\n";
//		return;
//	}
//
//	const float max_overlay = 0.5f;
//
//	std::vector<cv::Rect2f> vAddRects;
//	for (int i = 0; i < result.nLessionsCount; i++)
//	{
//		SYY::Rect detRect = result.pLessionRects[i];
//		cv::Rect2f r(detRect.x, detRect.y, detRect.w, detRect.h);
//
//		bool isMatch = false;
//		for (auto vRect : vPtsRects)
//		{
//			auto rect = GetRect(vRect);
//			if (max_overlay < CalcIOU(rect, r))
//			{
//				isMatch = true;
//				break;
//			}
//		}
//
//		if (isMatch) continue;
//
//		cv::rectangle(img, r, cv::Scalar(255, 255, 255));
//
//		vAddRects.push_back(r);
//	}
//
//	std::string txtFile;
//	if (false == SaveInfo2Txt(vPtsRects, vImgRects, vAddRects, relativePath, txtSaveRoot, txtFile, errorMsg))
//	{
//		std::cout << "save info error: " << errorMsg << "\n";
//		return;	
//	}
//
//	cv::imshow("img", img);
//	cv::waitKey(1);
//}
//

cv::Mat CreateRandomRect(cv::Mat img, cv::Rect2f sample_rect, cv::Rect2f imgRect)
{
	auto rect = sample_rect;

	float posRandomRadio_x = float(std::rand() % 1001 - 500) / 5000.0f; // -0.1 ~ 0.1
	float posRandomRadio_y = float(std::rand() % 1001 - 500) / 5000.0f; // -0.1 ~ 0.1
	float lenRandomRadio = float(std::rand() % 1001 - 200) / 2000.0f; // -0.1 ~ 0.4
	float colorRandomRadio = float(std::rand() % 1001 - 500) / 1500.0f; // -0.33 ~ 0.33

	cv::Point cpox = (rect.tl() + rect.br()) / 2;
	cpox.x = cpox.x + rect.width * posRandomRadio_x;
	cpox.y = cpox.y + rect.height * posRandomRadio_y;
	rect.width *= (1 + lenRandomRadio);
	rect.height *= (1 + lenRandomRadio);
	rect.x = std::max(imgRect.x, std::min((cpox.x - rect.width / 2.f), imgRect.br().x - rect.width));
	rect.y = std::max(imgRect.y, std::min((cpox.y - rect.height / 2.f), imgRect.br().y - rect.height));
	rect.width = std::min(rect.width, (imgRect.br().x - rect.x));
	rect.height = std::min(rect.height, (imgRect.br().y - rect.y));

	auto part = img(rect);

	part.convertTo(part, CV_32FC3);
	part *= (1 + colorRandomRadio);
	part.convertTo(part, CV_8UC3);

	return part;
}

void GradingClassify4Lession(const std::string& txt)
{
	const std::string ImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs_crimisi\\";
	const std::string ImgSaveRoot = "D:\\blue\\data\\训练文件\\病灶分级\\";

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects, vAddRects;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, vAddRects, relativePath, errorMsg))
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

	if (vPtsRects.size() == 0 && vImgRects.size() != 1)
	{
		return;
	}

	std::string img_save_relative = ImgSaveRoot + relativePath;
	img_save_relative = img_save_relative.substr(0, img_save_relative.find_last_of('\\'));
	if (false == boost::filesystem::exists(img_save_relative))
		boost::filesystem::create_directories(img_save_relative);
	std::string img_save_prefix = ImgSaveRoot + relativePath;
	img_save_prefix = img_save_prefix.substr(0, img_save_prefix.find_last_of('.'));
	std::stringstream ss;

	auto imgRect = vImgRects[0];

	for (auto vRects : vPtsRects)
	{
		int sampleCount = 10;

		while (sampleCount-- > 0)
		{
			auto rect = GetRect(vRects);

			auto part = CreateRandomRect(img, rect, imgRect);

			cv::imshow("part", part);
			cv::waitKey(1);

			ss.str("");
			ss << img_save_prefix << "_" << sampleCount << ".jpg";
			std::string img_save_path = ss.str();
			cv::imwrite(img_save_path, part);
		}

	}
}

void GradingClassify4FullImage(const std::string& txt)
{
	//const std::string ImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs_crimisi\\";
	//const std::string ImgSaveRoot = "D:\\blue\\data\\训练文件\\病灶分级_全图\\";
	//const std::string TrainImgSaveRoot = "D:\\blue\\data\\训练文件\\病灶分级_全图\\train\\";
	//const std::string TestImgSaveRoot = "D:\\blue\\data\\训练文件\\病灶分级_全图\\test\\";

	const std::string ImgRoot = "E:\\data\\inpaint_imgs_crimisi\\";
	const std::string ImgSaveRoot = "E:\\data\\train_files\\gc_fullimage\\";
	const std::string TrainImgSaveRoot = "E:\\data\\train_files\\gc_fullimage\\train\\";
	const std::string TestImgSaveRoot = "E:\\data\\train_files\\gc_fullimage\\val\\";

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects, vAddRects;
	std::string relativePath, errorMsg;
	std::map<std::string, std::string> mAttrs;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, vAddRects, mAttrs, relativePath, errorMsg))
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

	//if (vPtsRects.size() == 0 && vAddRects.size() == 0) return;

	if (vImgRects.size() != 1)
	{
		return;
	}

	//auto type = mAttrs["LessionType"];
	//std::string attr_name = "";
	//if (type == "0")
	//{
	//	attr_name = "None";
	//}
	//else if (type == "1")
	//{
	//	attr_name = "Lession";
	//}
	//else if (type == "2")
	//{
	//	attr_name = "Lymphaden";
	//}
	//else 
	//{
	//	std::cout << "error Lession Type: " << type << std::endl;
	//	return;
	//}

	Check_Create_Path(TrainImgSaveRoot);
	Check_Create_Path(TestImgSaveRoot);

	auto cls = relativePath.substr(0, relativePath.find_first_of('\\'));
	if (cls == "1a类")
	{
		cls = "no";
	}
	else
	{
		cls = "yes";
	}

	std::stringstream ss;

	auto imgRect = vImgRects[0];

	int randomNum = std::rand() % 10;

	static int img_no = 0;

	// 80% used in train set
	if (randomNum < 8)
	{
		auto img_save_prefix = TrainImgSaveRoot + cls + "\\";
		Check_Create_Path(img_save_prefix);

		int sampleCount = 2;

		while (sampleCount-- > 0)
		{
			auto rect = imgRect;
			float posRandomRadio_x = float(std::rand() % 1001 - 500) / 50000.0f; // -0.01 ~ 0.01
			float posRandomRadio_y = float(std::rand() % 1001 - 500) / 50000.0f; // -0.01 ~ 0.01
			float colorRandomRadio = float(std::rand() % 1001 - 500) / 1500.0f; // -0.33 ~ 0.33

			cv::Point cpox = (rect.tl() + rect.br()) / 2;
			cpox.x = cpox.x + rect.width * posRandomRadio_x;
			cpox.y = cpox.y + rect.height * posRandomRadio_y;

			rect.x = std::max(0.f, cpox.x - rect.width / 2.f);
			rect.y = std::max(0.f, cpox.y - rect.height / 2.f);

			rect.width = std::min(rect.width, img.cols - rect.x);
			rect.height = std::min(rect.height, img.rows - rect.y);

			auto part = img(rect);

			part.convertTo(part, CV_32FC3);
			part *= (1 + colorRandomRadio);
			part.convertTo(part, CV_8UC3);

			cv::imshow("part", part);
			cv::waitKey(1);

			ss.str("");
			ss << img_save_prefix << img_no++ << ".jpg";

			std::string img_save_path = ss.str();
			cv::imwrite(img_save_path, part);
		}
	}
	else 
	{
		auto img_save_prefix = TestImgSaveRoot + cls + "\\";
		Check_Create_Path(img_save_prefix);

		auto part = img(imgRect);

		ss.str("");
		ss << img_save_prefix << img_no++ << ".jpg";

		std::string img_save_path = ss.str();
		cv::imwrite(img_save_path, part);
	}
}

void AddLessionAttrInfo(const std::string& txt)
{
	const std::string ImgRoot = "D:\\blue\\data\\乳腺癌图片\\";
	const std::string txtSaveRoot = "D:\\blue\\codes\\TagTools\\TagTools\\RectTxtProcess\\new_txtRect\\";

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects, vAddRects;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, vAddRects, relativePath, errorMsg))
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

	std::map<std::string, std::string> mAttr;
	mAttr["LessionType"] = "0";

	std::string txtFile;
	if (false == SaveInfo2Txt(vPtsRects, vImgRects, vAddRects, mAttr, relativePath, txtSaveRoot, txtFile, errorMsg))
	{
		std::cout << "save info error: " << errorMsg << "\n";
		return;	
	}
}

void GetTrueFalseRegion(const std::string& txt)
{
	//const std::string ImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs_crimisi\\";

	//const std::string TrainImgSaveRoot = "D:\\blue\\data\\训练文件\\lession_nonlession\\train\\";
	//const std::string TestImgSaveRoot = "D:\\blue\\data\\训练文件\\lession_nonlession\\val\\";

	const std::string ImgRoot = "E:\\data\\inpaint_imgs_crimisi\\";

	const std::string TrainImgSaveRoot = "E:\\data\\train_files\\lession_nonlession\\train\\";
	const std::string TestImgSaveRoot = "E:\\data\\train_files\\lession_nonlession\\val\\";

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vImgRects, vAddRects;
	std::string relativePath, errorMsg;
	std::map<std::string, std::string> mAttrs;

	Check_Create_Path(TrainImgSaveRoot);
	Check_Create_Path(TestImgSaveRoot);

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, vAddRects, mAttrs, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return;
	}

	if (vImgRects.size() != 1)
		return;

	if (vPtsRects.size() == 0 && vAddRects.size() > 0)
		return;

	auto imgRect = vImgRects[0];

	auto cls = relativePath.substr(0, relativePath.find_first_of('\\'));

	//if (cls != "1a类" && vPtsRects.size() != 0) return;

	std::string img_path = ImgRoot + relativePath;
	cv::Mat img = cv::imread(img_path);

	if (img.empty())
	{
		std::cout << "error when open img: " << img_path << std::endl;
		return;
	}

	static SYY::MedicalAnalysis::BUAnalysisResult result;

	if (SYY::SYY_NO_ERROR != SYY::MedicalAnalysis::ExecuteBUAnalysis(
		hHandleBUAnalysis, (char*)img.data, img.cols, img.rows, &result))
	{
		std::cout << "execute BU analysis error!\n";
		return;
	}

	const float max_overlay = 0.5f;

	std::vector<cv::Rect> nonLessionsRects;
	std::vector<cv::Rect> lessionsRects;

	auto draw = img.clone();

	for (auto vRect : vPtsRects)
	{
		auto rect = GetRect(vRect);
		lessionsRects.push_back(rect);

		cv::rectangle(draw, rect, cv::Scalar(0, 0, 255), 2);
	}

	for (int i = 0; i < result.nLessionsCount; i++)
	{
		SYY::Rect detRect = result.pLessionRects[i];
		cv::Rect2f r(detRect.x, detRect.y, detRect.w, detRect.h);

		cv::rectangle(draw, r, cv::Scalar(255, 255, 255), 1);

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

		if (false == isMatch) 
		{
			// create non lession train region
			nonLessionsRects.push_back(r);
		}
	}

	cv::imshow("draw", draw);
	cv::waitKey(1);

	int randomNum = std::rand() % 10;
	static int img_no = 0;

	const auto create_random_train_imgs = [&](
		const std::vector<cv::Rect>& rects, 
		const std::string& folder_name) 
	{
		for (auto rect : rects)
		{
			int sampleNum = 3;
			while (sampleNum-- > 0)
			{
				auto part = CreateRandomRect(img, rect, imgRect);
				std::stringstream ss;
				ss << folder_name << img_no++ << ".jpg";
				auto file_path = ss.str();
				cv::imwrite(file_path, part);
			}
		}
	};

	const auto create_random_test_imgs = [&](
		const std::vector<cv::Rect>& rects, 
		const std::string& folder_name) 
	{
		for (auto rect : rects)
		{
			auto part = img(rect);
			std::stringstream ss;
			ss << folder_name << img_no++ << ".jpg";
			auto file_path = ss.str();
			cv::imwrite(file_path, part);
		}
	};

	auto lession_train_path = TrainImgSaveRoot + "1_lession\\";
	auto nonlession_train_path = TrainImgSaveRoot + "0_nonlession\\";
	auto lession_test_path = TestImgSaveRoot + "1_lession\\";
	auto nonlession_test_path = TestImgSaveRoot + "0_nonlession\\";

	Check_Create_Path(lession_train_path);
	Check_Create_Path(nonlession_train_path);
	Check_Create_Path(lession_test_path);
	Check_Create_Path(nonlession_test_path);

	if (randomNum < 7)
	{
		// train set
		create_random_train_imgs(lessionsRects, lession_train_path);
		create_random_train_imgs(nonLessionsRects, nonlession_train_path);
	}
	else 
	{
		//test set
		create_random_test_imgs(lessionsRects, lession_test_path);
		create_random_test_imgs(nonLessionsRects, nonlession_test_path);
	}
}

enum JustifyType { F2T, F2F, T2F, T2T, Other, Err};

JustifyType Justify(const std::string& txt, float thr, int& t_count, int& f_count, 
	cv::Mat& origin_img, SYY::MedicalAnalysis::BUAnalysisResult& result,
	std::vector<std::vector<cv::Rect2f>>& vPtsRects, std::vector<cv::Rect2f>& vAddRects,
	std::string& relativePath
	)
{
	//const std::string ImgRoot = "D:\\blue\\data\\乳腺癌图片\\";
	//const std::string InpaintImgRoot = "D:\\blue\\data\\训练文件\\inpaint_imgs_crimisi\\";

	const std::string ImgRoot = "E:\\data\\乳腺癌图片\\";
	const std::string InpaintImgRoot = "E:\\data\\乳腺癌图片\\";
	//const std::string InpaintImgRoot = "E:\\data\\inpaint_imgs_crimisi\\";

	//std::vector<std::vector<cv::Rect2f>> vPtsRects;
	//std::vector<cv::Rect2f> vImgRects, vAddRects;
	std::vector<cv::Rect2f> vImgRects;
	//std::string relativePath, errorMsg;
	std::string errorMsg;

	std::map<std::string, std::string> mAttrs;

	if (false == ParseTxtInfo(txt, vPtsRects, vImgRects, vAddRects, mAttrs, relativePath, errorMsg))
	{
		std::cout << errorMsg << std::endl;
		return Err;
	}

	if (vImgRects.size() != 1)
		return Err;

	auto cls = relativePath.substr(0, relativePath.find_first_of('\\'));

	//if (cls != "1a类" && (vPtsRects.size() != 0 || vAddRects.size() != 0)) return Err;
	//if (cls != "1a类" && vPtsRects.size() != 0 ) return Err;

	std::vector<int>::size_type idx = 1;

	std::string img_path = ImgRoot + relativePath;
	std::string inpaint_img_path = InpaintImgRoot + relativePath;
	origin_img = cv::imread(img_path);
	if (origin_img.empty())
	{
		std::cout << "error when open img: " << img_path << std::endl;
		return Err;
	}
	cv::Mat inpaint_img = cv::imread(inpaint_img_path);
	if (inpaint_img.empty())
	{
		std::cout << "error when open img: " << inpaint_img_path << std::endl;
		return Err;
	}

	//static SYY::MedicalAnalysis::BUAnalysisResult result;

	auto img = inpaint_img;
	if (SYY::SYY_NO_ERROR != SYY::MedicalAnalysis::ExecuteBUAnalysis(
		hHandleBUAnalysis, (char*)img.data, img.cols, img.rows, &result))
	{
		std::cout << "execute BU analysis error!\n";
		return Err;
	}

	if ((vPtsRects.size() == 0 && vAddRects.size() == 0) || cls == "1a类")
		f_count += 1;
	else
		t_count += 1;

	if (result.nLessionsCount == 0 && result.nGrading == SYY::MedicalAnalysis::LG1a)
	{
		// to false 
		// false
		if ((vPtsRects.size() == 0 && vAddRects.size() == 0) || cls == "1a类")
		{
			return JustifyType::F2F;
		}

		// true image
		if ((vPtsRects.size() > 0 || vAddRects.size() > 0) && cls != "1a类")
		{
			return JustifyType::T2F;
		}
	}
	else if (result.nLessionsCount != 0 && result.nGrading == SYY::MedicalAnalysis::LG1a)
	{
		bool is_false = true;

		// to true
		for (int i = 0; i < result.nLessionsCount; i++)
		{
			if (result.pLessionConfidence[i] > thr || result.pLessionTypes[i] == SYY::MedicalAnalysis::LESSION)
			//if (result.pLessionConfidence[i] > thr)
			{
				is_false = false;

				// false image
				if ((vPtsRects.size() == 0 && vAddRects.size() == 0) || cls == "1a类")
				{
					return JustifyType::F2T;
				}

				break;
			}
		}

		// to false
		if (is_false)
		{
			// false image
			if ((vPtsRects.size() == 0 && vAddRects.size() == 0) || cls == "1a类")
			{
				return F2F;
			}

			// true image
			if ((vPtsRects.size() > 0 || vAddRects.size() > 0) && cls != "1a类")
			{
				return T2F;
			}
		}
	}
	else if (result.nLessionsCount != 0 && result.nGrading == SYY::MedicalAnalysis::LG_OTHER)
	{
		// to true
		if ((vPtsRects.size() == 0 && vAddRects.size() == 0) || cls == "1a类")
		{
			return F2T;
		}
	}
	else if (result.nLessionsCount == 0 && result.nGrading == SYY::MedicalAnalysis::LG_OTHER)
	{
		// to true

		if ((vPtsRects.size() == 0 && vAddRects.size() == 0) || cls == "1a类")
		{
			//f_t_count += 1;
			//write_img("f2t");
			return F2T;
		}

		//// true image
		//if ((vPtsRects.size() > 0 || vAddRects.size() > 0) && cls != "1a类")
		//{
		//	return T2F;
		//}
	}
	else {
		return Other;
		std::cout << "some thing else!!!\n";
	}

	return Other;
}

void CalTestCount(const std::string& txt, float thr, int& f_f_count, int& f_t_count, int& t_f_count, int& t_count, int& f_count) {
	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vAddRects;
	std::string relativePath;
	cv::Mat origin_img;
	static SYY::MedicalAnalysis::BUAnalysisResult result;

	auto type = Justify(txt, thr, t_count, f_count, origin_img, result, vPtsRects, vAddRects, relativePath);

	switch (type)
	{
	case F2T:
		f_t_count += 1;
		break;
	case F2F:
		f_f_count += 1;
		break;
	case T2F:
		boost::filesystem::remove(txt);
		t_f_count += 1;
		break;
	case T2T:
		break;
	case Other:
	case Err:
		break;
	}
}

void TestAcc(const std::string& txt)
{
	static int f_f_count = 0;
	static int f_t_count = 0;
	static int t_f_count = 0;
	static int t_count = 0;
	static int f_count = 0;

	std::vector<std::vector<cv::Rect2f>> vPtsRects;
	std::vector<cv::Rect2f> vAddRects;
	std::string relativePath;
	cv::Mat origin_img;
	static SYY::MedicalAnalysis::BUAnalysisResult result;

	auto type = Justify(txt, 0.1f, t_count, f_count, origin_img, result, vPtsRects, vAddRects, relativePath);

	const auto write_img = [&](const std::string& type) 
	{
		auto draw = origin_img.clone();
		cv::Rect r(result.rCropRect.x, result.rCropRect.y, result.rCropRect.w, result.rCropRect.h);
		cv::putText(draw, result.nGrading == SYY::MedicalAnalysis::LG1a ? "1a" : "other", r.tl(), 1, 1, cv::Scalar(255, 255, 255));
		cv::rectangle(draw, r, cv::Scalar(255, 255, 255), 1);

		std::stringstream ss;
		for (int i = 0; i < result.nLessionsCount; i++)
		{
			auto lr = result.pLessionRects[i];
			cv::Rect r(lr.x, lr.y, lr.w, lr.h);
			cv::rectangle(draw, r, cv::Scalar(255, 255, 255), 1);

			ss.str("");
			ss.clear();

			ss.width(3);
			ss << result.pLessionConfidence[i];

			cv::putText(draw, ss.str(), (r.tl() + r.br()) / 2, 1, 1, cv::Scalar(255, 255, 255));

			ss.str("");
			ss.clear();
			ss << ((result.pLessionTypes[i] == SYY::MedicalAnalysis::NO_LESSION) ? "non-l" : "lession");

			cv::putText(draw, ss.str(), r.tl(), 1, 1, cv::Scalar(255, 255, 255));
		}

		for (auto vRects : vPtsRects)
		{
			auto rect = GetRect(vRects);
			cv::rectangle(draw, rect, cv::Scalar(0, 0, 255), 2);
		}
		for (auto rect : vAddRects)
		{
			cv::rectangle(draw, rect, cv::Scalar(0, 0, 255), 2);
		}

		const std::string path = type + "\\" + relativePath;
		auto folder = path.substr(0, path.find_last_of('\\'));
		Check_Create_Path(folder);
		cv::imwrite(path, draw);
	};

	switch (type)
	{
	case F2T:
		f_t_count += 1;
		write_img("f2t");
		break;
	case F2F:
		f_f_count += 1;
		break;
	case T2F:
		t_f_count += 1;
		write_img("t2f");
		break;
	case T2T:
		break;
	case Other:
		write_img("other");
		break;
	}

	std::cout << "false2false: " << (float)f_f_count / (float)f_count<< " (" << f_f_count << ", " << f_count << ")\n";
	std::cout << "true2false: " << (float)t_f_count / (float)t_count << " (" << t_f_count << ", " << t_count << ")\n";
	std::cout << "false2true: " << (float)f_t_count / (float)f_count << " (" << f_t_count << ", " << f_count << ")\n";

	write_img("res");
}

struct Scores {
	float f2t_ratio = 1.0f;
	float f2f_ratio = 0.0f;
	float t2f_ratio = 1.0f;
};

typedef std::pair<float, Scores> thr2scores;

void AutoGetThreshold() {
	const std::string txtRectFolder = "E:\\data\\RectTxt\\";

	std::vector<std::string> files;
	scanFilesUseRecursive(txtRectFolder, files);

	std::random_shuffle(files.begin(), files.end());

	const float thr_step = 0.05f;

	std::vector<thr2scores> vThr2scores;

	for (float thr = 0.2f; thr < 0.51f; thr += thr_step)
	{
		thr2scores t2s;
		t2s.first = thr;
		int ff_c = 0, ft_c = 0, tf_c = 0, t_c = 0, f_c = 0;

		std::cout << "thr: " << thr << std::endl;
		int file_count = files.size();
		for (int i = 0; i < files.size(); i++)
		{
			auto file = files[i];
			CalTestCount(file, thr, ff_c, ft_c, tf_c, t_c, f_c);

			t2s.second.f2t_ratio = (float)ft_c / (float)f_c;
			t2s.second.f2f_ratio = (float)ff_c / (float)f_c;
			t2s.second.t2f_ratio = (float)tf_c / (float)t_c;

			// std::cout << "\rprocessing: (" << i << ", " << file_count << ")" << "\tf2t: " << t2s.second.f2t_ratio << "\tf2f: " << t2s.second.f2f_ratio << "\tt2f: " << t2s.second.t2f_ratio;
			std::cout << "\rprocessing: (" << f_c + t_c << ")" << "\tf2t: " << t2s.second.f2t_ratio << "\tf2f: " << t2s.second.f2f_ratio << "\tt2f: " << t2s.second.t2f_ratio;

			std::cout.flush();
		}
		std::cout << std::endl;

		vThr2scores.push_back(t2s);
	}

	std::sort(vThr2scores.begin(), vThr2scores.end(), [](thr2scores a, thr2scores b){
		auto as = a.second;
		auto bs = b.second;

		if (as.t2f_ratio < bs.t2f_ratio)
			return true;
		else if (as.t2f_ratio > bs.t2f_ratio)
			return false;
		else if (as.f2f_ratio > bs.f2f_ratio)
			return true;
		else
			return false;
	});

	std::ofstream out("result.txt");

	for (auto t2s : vThr2scores)
	{
		out << "thr: " << t2s.first << "\n";
		out << "t2f: " << t2s.second.t2f_ratio << "\n";
		out << "f2t: " << t2s.second.f2t_ratio << "\n";
		out << "f2f: " << t2s.second.f2f_ratio << "\n";
		out << "\n";
	}
}

void main(int argc, char** argv)
{
	if (SYY::SYY_NO_ERROR != SYY::InitSDK())
	{
		std::cerr << "InitSDK error!" << std::endl;
		return;
	}

	if (SYY::SYY_NO_ERROR != SYY::MedicalAnalysis::InitBUAnalysisWithMode(hHandleBUAnalysis, SYY::MedicalAnalysis::Crop_V2 | SYY::MedicalAnalysis::DetectAccurate))
	{
		std::cerr << "InitBUAnalysis error!\n" << std::endl;
		return;
	}

	//if (SYY::SYY_NO_ERROR != SYY::MedicalAnalysis::InitBUAnalysis(hHandleBUAnalysis2, SYY::MedicalAnalysis::Crop_V2))
	//{
	//	std::cerr << "InitBUAnalysis error!\n" << std::endl;
	//	return;
	//}

	//argc = 2;
	//argv[1] = "D:\\blue\\codes\\TagTools\\x64\\标注工具\\RectTxt\\";
	//argv[1] = "E:\\data\\RectTxt\\";
	//ProcessAllFile(argc, argv, ParseTxt);

	//argc = 2;
	////argv[1] = "D:\\blue\\codes\\TagTools\\x64\\标注工具\\RectTxt\\";
	//argv[1] = "E:\\data\\RectTxt\\";
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

	//argc = 2;
	//argv[1] = "D:\\blue\\codes\\TagTools\\x64\\标注工具\\RectTxt\\";
	//argv[1] = "E:\\data\\RectTxt\\";
	//ProcessAllFile(argc, argv, Create_VOC_Annots_With_Filter);

	//argc = 2;
	//argv[1] = "D:\\blue\\codes\\TagTools\\x64\\标注工具\\RectTxt\\";
	//ProcessAllFile(argc, argv, Create_VOC_Annots_ImgBox);

	//argc = 2;
	//argv[1] = "D:\\blue\\codes\\TagTools\\x64\\标注工具\\RectTxt\\";
	//ProcessAllFile(argc, argv, GenerateAdditionRect);

	//argc = 2;
	//argv[1] = "D:\\blue\\codes\\TagTools\\x64\\标注工具\\RectTxt\\";
	//ProcessAllFile(argc, argv, GetTestImg);

	//argc = 2;
	//argv[1] = "D:\\blue\\codes\\TagTools\\x64\\标注工具\\RectTxt\\";
	//argv[1] = "E:\\data\\RectTxt\\";
	//ProcessAllFile(argc, argv, GradingClassify4FullImage);


	//argc = 2;
	////argv[1] = "D:\\blue\\codes\\TagTools\\x64\\标注工具\\RectTxt\\";
	//argv[1] = "D:\\blue\\codes\\TagTools\\x64\\旧标注工具\\RectTxt\\";
	//ProcessAllFile(argc, argv, AddLessionAttrInfo);

	//argc = 2;
	//argv[1] = "E:\\data\\RectTxt\\";
	//ProcessAllFile(argc, argv, TestAcc, 0, true);

	argc = 2;
	//argv[1] = "D:\\blue\\codes\\TagTools\\x64\\标注工具\\RectTxt\\";
	argv[1] = "E:\\data\\RectTxt\\";
	ProcessAllFile(argc, argv, GetTrueFalseRegion);

	//AutoGetThreshold();

	//cv::Mat m = cv::Mat::zeros(3, 3, CV_8UC1);
	//cv::Mat h, v;
	//m.at<uchar>(0, 1) = 1;
	//m.at<uchar>(2, 1) = 1;
	//CalHVProj(m, h, v);

	//std::cout << m << std::endl;
	//std::cout << h << std::endl;
	//std::cout << v << std::endl;

	//SYY::MedicalAnalysis::ReleaseBUAnalysis(hHandleBUAnalysis2);
	SYY::MedicalAnalysis::ReleaseBUAnalysis(hHandleBUAnalysis);
	SYY::ReleaseSDK();

	system("pause");
}
