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
int hogSize = 1764;//64*64Ϊ1764��64*128Ϊ3780��64*224Ϊ6804

void train_hog_svm()
{
	string positivePath = "D:\\pic\\muti\\hog_svm\\lotus\\pos\\";
	string negativePath = "D:\\pic\\muti\\hog_svm\\lotus\\neg\\";
	//string positivePath = "D:\\pic\\muti\\hog_svm\\wheat\\pos\\";
	//string negativePath = "D:\\pic\\muti\\hog_svm\\wheat\\neg\\";
	//string positivePath = "D:\\pic\\muti\\hog_svm\\wheat\\64-224\\pos\\";
	//string negativePath = "D:\\pic\\muti\\hog_svm\\wheat\\64-224\\neg\\";
	string suffix = ".jpg";// ͼƬ��׺ 

	int positiveSampleCount = 500;
	int negativeSampleCount = 100;
	int totalSampleCount = positiveSampleCount + negativeSampleCount;

	std::cout << "/******************************/" << std::endl;
	cout << "��������: " << totalSampleCount << endl;
	cout << "��������: " << positiveSampleCount << endl;
	cout << "��������: " << negativeSampleCount << endl;

	cv::Mat sampleFeaturesMat = cv::Mat::zeros(totalSampleCount, hogSize, CV_32FC1);
	cv::Mat sampleLabelMat = cv::Mat::zeros(totalSampleCount, 1, CV_32SC1);

	// ������ʱ
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

		hog.compute(img, featureVec, cv::Size(8, 8));//����HOG����  
		int featureVecSize = (int)featureVec.size();

		for (int j = 0; j < featureVecSize; j++) {
			sampleFeaturesMat.at<float>(i + positiveSampleCount, j) = featureVec[j];
		}
		sampleLabelMat.at<int>(i + positiveSampleCount) = 0;
	}

	std::cout << "/******************************/" << std::endl;
	cout << "ѵ��SVM..." << endl;

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
	std::cout << "ѵ����ϣ���ʱ��" << duration << "s" << std::endl;
	
	// accuracy
	float cnt = 0;
	int rowsize = sampleLabelMat.rows;
	for (int i = 0; i < rowsize; ++i) {
		cv::Mat samp = sampleFeaturesMat.row(i);
		float res = svm->predict(samp);
		cnt += std::abs(res - sampleLabelMat.at<int>(i)) <= FLT_EPSILON ? 1 : 0;
	}
	std::cout << "׼ȷ�ʣ�" << cnt / rowsize * 100 << "%" << std::endl;
	cv::Mat SVmat = svm->getSupportVectors();
	std::cout << "֧������������" << SVmat.rows << std::endl;
	


	// ��XML�ļ���ȡѵ���õ�SVMģ��  
	//cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("hog-svm-model.xml");
	//if (svm->empty()) {
	//	std::cout << "load svm detector failed!!!" << std::endl;
	//	return;
	//}

	//HOG�����ӵ�ά��
	int DescriptorDim;
	DescriptorDim = svm->getVarCount();//����������ά������HOG�����ӵ�ά��
	cv::Mat svecsmat = svm->getSupportVectors();//��ȡsvecsmat��Ԫ������Ϊfloat

	int svdim = svm->getVarCount();
	int numofsv = svecsmat.rows;

	//alphamat��svindex�����ʼ��������getDecisionFunction()�����ᱨ��  
	cv::Mat alphamat = cv::Mat::zeros(numofsv, svdim, CV_32F);
	cv::Mat svindex = cv::Mat::zeros(1, numofsv, CV_64F);
	double rho = svm->getDecisionFunction(0, alphamat, svindex);
	alphamat.convertTo(alphamat, CV_32F);//��alphamatԪ�ص�������������ת��CV_32F

	cv::Mat Result;
	Result = -1 * alphamat * svecsmat;

	std::vector<float> vec;
	for (int i = 0; i < svdim; ++i) {
		vec.push_back(Result.at<float>(0, i));
	}
	vec.push_back((float)rho);

	// ����hog�������
	std::ofstream fout("hog-detector.txt");
	for (int i = 0; i < vec.size(); ++i) {
		fout << vec[i] << std::endl;
	}
	std::cout << "����hog������ϣ�" << std::endl;
	std::cout << "/******************************/" << std::endl;
	
}

void detect_hog_svm()
{
	// ��ȡhog�������
	ifstream fin("hog-detector.txt");
	vector<float> vec;
	float tmp = 0;
	while (fin >> tmp) {
		vec.push_back(tmp);
	}
	fin.close();

	cv::HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, 9);
	hog.setSVMDetector(vec);
	std::cout << "hog�����Ӽ�����ϣ�" << std::endl;
	std::cout << "ʶ����..." << std::endl;

	Mat src;
	//src = imread("D:\\pic\\wheat_pic\\wheat_stand_mid.jpg");
	src = imread("D:\\pic\\muti\\test_lotus\\lotus-614421__340.jpg");
	//src = imread("D:\\pic\\muti\\wheat\\wheat_132.jpg");
	//src = imread("D:\\pic\\wheat_pic\\wheat1_ps_low.jpg");
	//src = imread("C:\\Users\\Administrator\\Desktop\\timg.jpg");

	vector<Rect> found, found_filtered;//���ο�����
	hog.detectMultiScale(src, found, 0, Size(8, 8), Size(32, 32), 1.5, 2);//��ͼ����ж�߶ȼ��
	std::cout << "ʶ����ϣ�" << std::endl;

	cout << "���θ�����" << found.size() << endl;
	//�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��
	for (int i = 0; i < found.size(); i++) {
		Rect r = found[i];
		int j = 0;
		for (; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);
	}
	cout << "���εĸ�����" << found_filtered.size() << endl;

	//�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����
	for (int i = 0; i < found_filtered.size(); i++) {
		Rect r = found_filtered[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		rectangle(src, r.tl(), r.br(), Scalar(0, 255, 255), 2);
	}
	namedWindow("���", WINDOW_NORMAL);
	imshow("���", src);
	//imwrite("ImgProcessed.jpg",src);
	waitKey();//ע�⣺imshow֮��һ��Ҫ��waitKey�������޷���ʾͼ��
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

	//hog.compute(testimg, featureVec, cv::Size(8, 8));//����HOG����  
	//int featureVecSize = (int)featureVec.size();
	//cv::Mat testmat = cv::Mat::zeros(1, hogSize, CV_32FC1);
	//for (int j = 0; j < featureVecSize; j++) {
	//	testmat.at<float>(0, j) = featureVec[j];
	//}
	//float predict = svm->predict(testmat);
	//std::cout << "Ԥ������" << predict << std::endl;

	return 0;
}