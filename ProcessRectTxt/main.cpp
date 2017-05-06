#include <fstream>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "../utils/utils.h"

typedef void(*Func)(const std::string& param);

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

void ProcessAllFile(int argc, char** argv, Func func)
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

		for (auto file : files)
		{
			std::cout << file << std::endl;
			func(file);
		}
	}
}

void ParseTxt(const std::string& txt)
{
	const std::string ImgRoot = "D:\\blue\\data\\乳腺癌图片\\";

	std::vector<std::vector<cv::Point>> vRects;
	std::string relativePath, errorMsg;

	if (false == ParseTxtInfo(txt, vRects, relativePath, errorMsg))
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

	for (const auto& vPts : vRects)
	{
		for (const auto& pt : vPts)
		{
			cv::circle(img, pt, 5, cv::Scalar(0, 0, 255), 5, -1);
		}
	}

	cv::imshow("img", img);
	cv::waitKey();

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

	//auto hShow = GetHistImg(h);
	//auto vShow = GetHistImg(v);
	//cv::imshow("h", hShow);
	//cv::imshow("v", vShow);

	//cv::waitKey();

	for (auto& rect : regions)
	{
		//cv::rectangle(gray, rect, cv::Scalar(255, 0, 0), 1);

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

		//cv::rectangle(gray, rect, cv::Scalar(255, 0, 0), 2);
		//cv::imshow("gray", gray);
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
	for (int morph_size = 0; morph_size < 7; morph_size++)
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

void Preprocess(const std::string& imgFile)
{
	auto img = cv::imread(imgFile);
	if (img.empty())
		return;
	
	//Method1(img);
	Method2(img);
}

void main(int argc, char** argv)
{
	//argc = 2;
	//argv[1] = "D:\\blue\\codes\\TagTools\\TagTools\\RectTxt\\";
	//ProcessAllFile(argc, argv, ParseTxt);

	argc = 2;
	argv[1] = "D:\\blue\\data\\乳腺癌图片";
	//argv[1] = "D:\\blue\\codes\\TagTools\\testImgs";
	ProcessAllFile(argc, argv, Preprocess);

	//cv::Mat m = cv::Mat::zeros(3, 3, CV_8UC1);
	//cv::Mat h, v;
	//m.at<uchar>(0, 1) = 1;
	//m.at<uchar>(2, 1) = 1;
	//CalHVProj(m, h, v);

	//std::cout << m << std::endl;
	//std::cout << h << std::endl;
	//std::cout << v << std::endl;

	system("pause");
}