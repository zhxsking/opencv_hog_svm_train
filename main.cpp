#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include "time.h"

using namespace std;
using namespace cv;

cv::Size winSize(64, 64);
cv::Size blockSize(16, 16);
cv::Size blockStride(8, 8);
cv::Size cellSize(8, 8);
int hogSize = 1764;//64*64为1764，64*128为3780，64*224为6804

void train_hog_svm()
{
	string positivePath = "D:\\pic\\muti\\hog_svm\\lotus\\pos\\";
	string negativePath = "D:\\pic\\muti\\hog_svm\\lotus\\neg\\";
	//string positivePath = "D:\\pic\\muti\\hog_svm\\wheat\\pos\\";
	//string negativePath = "D:\\pic\\muti\\hog_svm\\wheat\\neg\\";
	//string positivePath = "D:\\pic\\muti\\hog_svm\\wheat\\64-224\\pos\\";
	//string negativePath = "D:\\pic\\muti\\hog_svm\\wheat\\64-224\\neg\\";
	string suffix = ".jpg";// 图片后缀 

	int positiveSampleCount = 500;
	int negativeSampleCount = 100;
	int totalSampleCount = positiveSampleCount + negativeSampleCount;

	std::cout << "/******************************/" << std::endl;
	cout << "总样本数: " << totalSampleCount << endl;
	cout << "正样本数: " << positiveSampleCount << endl;
	cout << "负样本数: " << negativeSampleCount << endl;

	cv::Mat sampleFeaturesMat = cv::Mat::zeros(totalSampleCount, hogSize, CV_32FC1);
	cv::Mat sampleLabelMat = cv::Mat::zeros(totalSampleCount, 1, CV_32SC1);

	// 计算用时
	clock_t start, finish;
	double duration;
	start = clock();

	for (int i = 0; i < positiveSampleCount; i++) {
		stringstream path;
		path << positivePath << i+1 << suffix;
		cv::Mat img = cv::imread(path.str());
		if (img.data == NULL) {
			cout << "positive image sample load error: " << i << " " << path.str() << endl;
			system("pause");
			continue;
		}

		cv::HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, 9);
		vector<float> featureVec;

		hog.compute(img, featureVec, cv::Size(8, 8));
		int featureVecSize = (int)featureVec.size();

		for (int j = 0; j < featureVecSize; j++) {
			sampleFeaturesMat.at<float>(i, j) = featureVec[j];
		}
		sampleLabelMat.at<int>(i) = 1;
	}

	for (int i = 0; i < negativeSampleCount; i++) {
		stringstream path;
		path << negativePath << i + 1 << suffix;
		cv::Mat img = cv::imread(path.str());
		if (img.data == NULL) {
			cout << "negative image sample load error: " << path.str() << endl;
			continue;
		}

		cv::HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, 9);
		vector<float> featureVec;

		hog.compute(img, featureVec, cv::Size(8, 8));//计算HOG特征  
		int featureVecSize = (int)featureVec.size();

		for (int j = 0; j < featureVecSize; j++) {
			sampleFeaturesMat.at<float>(i + positiveSampleCount, j) = featureVec[j];
		}
		sampleLabelMat.at<int>(i + positiveSampleCount) = 0;
	}

	std::cout << "/******************************/" << std::endl;
	cout << "训练SVM..." << endl;

	// initial SVM
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::Types::C_SVC);
	svm->setKernel(cv::ml::SVM::KernelTypes::LINEAR);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, FLT_EPSILON));
	svm->setDegree(0.2);
	svm->setGamma(0.5);
	svm->setC(32);

	// train operation
	svm->train(sampleFeaturesMat, cv::ml::SampleTypes::ROW_SAMPLE, sampleLabelMat);
	svm->save("hog-svm-model.xml");
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	std::cout << "训练完毕，用时：" << duration << "s" << std::endl;
	
	// accuracy
	float cnt = 0;
	int rowsize = sampleLabelMat.rows;
	for (int i = 0; i < rowsize; ++i) {
		cv::Mat samp = sampleFeaturesMat.row(i);
		float res = svm->predict(samp);
		cnt += std::abs(res - sampleLabelMat.at<int>(i)) <= FLT_EPSILON ? 1 : 0;
	}
	std::cout << "准确率：" << cnt / rowsize * 100 << "%" << std::endl;
	cv::Mat SVmat = svm->getSupportVectors();
	std::cout << "支持向量个数：" << SVmat.rows << std::endl;
	


	// 从XML文件读取训练好的SVM模型  
	//cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("hog-svm-model.xml");
	//if (svm->empty()) {
	//	std::cout << "load svm detector failed!!!" << std::endl;
	//	return;
	//}

	//HOG描述子的维数
	int DescriptorDim;
	DescriptorDim = svm->getVarCount();//特征向量的维数，即HOG描述子的维数
	cv::Mat svecsmat = svm->getSupportVectors();//获取svecsmat，元素类型为float

	int svdim = svm->getVarCount();
	int numofsv = svecsmat.rows;

	//alphamat和svindex必须初始化，否则getDecisionFunction()函数会报错  
	cv::Mat alphamat = cv::Mat::zeros(numofsv, svdim, CV_32F);
	cv::Mat svindex = cv::Mat::zeros(1, numofsv, CV_64F);
	double rho = svm->getDecisionFunction(0, alphamat, svindex);
	alphamat.convertTo(alphamat, CV_32F);//将alphamat元素的数据类型重新转成CV_32F

	cv::Mat Result;
	Result = -1 * alphamat * svecsmat;

	std::vector<float> vec;
	for (int i = 0; i < svdim; ++i) {
		vec.push_back(Result.at<float>(0, i));
	}
	vec.push_back((float)rho);

	// 保存hog分类参数
	std::ofstream fout("hog-detector.txt");
	for (int i = 0; i < vec.size(); ++i) {
		fout << vec[i] << std::endl;
	}
	std::cout << "保存hog参数完毕！" << std::endl;
	std::cout << "/******************************/" << std::endl;
	
}

void detect_hog_svm()
{
	// 读取hog分类参数
	ifstream fin("hog-detector.txt");
	vector<float> vec;
	float tmp = 0;
	while (fin >> tmp) {
		vec.push_back(tmp);
	}
	fin.close();

	cv::HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, 9);
	hog.setSVMDetector(vec);
	std::cout << "hog描述子加载完毕！" << std::endl;
	std::cout << "识别中..." << std::endl;

	Mat src;
	//src = imread("D:\\pic\\wheat_pic\\wheat_stand_mid.jpg");
	src = imread("D:\\pic\\muti\\test_lotus\\lotus-614421__340.jpg");
	//src = imread("D:\\pic\\muti\\wheat\\wheat_132.jpg");
	//src = imread("D:\\pic\\wheat_pic\\wheat1_ps_low.jpg");
	//src = imread("C:\\Users\\Administrator\\Desktop\\timg.jpg");

	vector<Rect> found, found_filtered;//矩形框数组
	hog.detectMultiScale(src, found, 0, Size(8, 8), Size(32, 32), 1.5, 2);//对图像进行多尺度检测
	std::cout << "识别完毕！" << std::endl;

	cout << "矩形个数：" << found.size() << endl;
	//找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
	for (int i = 0; i < found.size(); i++) {
		Rect r = found[i];
		int j = 0;
		for (; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);
	}
	cout << "矩形的个数：" << found_filtered.size() << endl;

	//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
	for (int i = 0; i < found_filtered.size(); i++) {
		Rect r = found_filtered[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		rectangle(src, r.tl(), r.br(), Scalar(0, 255, 255), 2);
	}
	namedWindow("检测", WINDOW_NORMAL);
	imshow("检测", src);
	//imwrite("ImgProcessed.jpg",src);
	waitKey();//注意：imshow之后一定要加waitKey，否则无法显示图像
}
int main()
{
	train_hog_svm();
	detect_hog_svm();

	// test
	//cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("hog-svm-model.xml");

	//string pic = "D:\\pic\\muti\\hog_svm\\pedestrain\\pos\\5.png";
	//cv::Mat testimg = cv::imread(pic);

	//cv::HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, 9);
	//vector<float> featureVec;

	//hog.compute(testimg, featureVec, cv::Size(8, 8));//计算HOG特征  
	//int featureVecSize = (int)featureVec.size();
	//cv::Mat testmat = cv::Mat::zeros(1, hogSize, CV_32FC1);
	//for (int j = 0; j < featureVecSize; j++) {
	//	testmat.at<float>(0, j) = featureVec[j];
	//}
	//float predict = svm->predict(testmat);
	//std::cout << "预测结果：" << predict << std::endl;

	return 0;
}