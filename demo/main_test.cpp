#include <iostream>
#include<opencv2/opencv.hpp>
#include "CameraArguments.h"
#include"CoreAlgorithm.h"
using namespace cv;

int minX = 0, maxX = 0, minY = 0, maxY = 0;
vector<Mat> coordinate;
Mat rgbChannel;
vector<float> color;

// ȷ��ÿ������ʵ�ʵĶ�Ӧ��
struct matchpts {
	double pts_x;
	double pts_y;
	double position;
};



// �����ֵ�ָ�
Mat OtsuAlgThreshold(Mat& src)
{
	if (src.channels() != 1)
	{
		cout << "Please input Gray-src!" << endl;
	}

	auto T = 0;
	double varValue = 0;
	double w0 = 0;
	double w1 = 0;
	double u0 = 0;
	double u1 = 0;
	double Histogram[256] = { 0 };
	uchar* data = src.data;

	double totalNum = src.rows * src.cols;

	for (auto i = 0; i < src.rows; i++)
	{
		for (auto j = 0; j < src.cols; j++)
		{
			if (src.at<float>(i, j) != 0) Histogram[data[i * src.step + j]]++;
		}
	}

	auto minpos = 0, maxpos = 0;
	for (auto i = 0; i < 255; i++)
	{
		if (Histogram[i] != 0)
		{
			minpos = i;
			break;
		}
	}

	for (auto i = 255; i > 0; i--)
	{
		if (Histogram[i] != 0)
		{
			maxpos = i;
			break;
		}
	}

	for (auto i = minpos; i <= maxpos; i++)
	{
		w1 = 0;
		u1 = 0;
		w0 = 0;
		u0 = 0;
		for (auto j = 0; j <= i; j++)
		{
			w1 += Histogram[j];
			u1 += j * Histogram[j];
		}
		if (w1 == 0)
		{
			break;
		}
		u1 = u1 / w1;
		w1 = w1 / totalNum;
		for (auto k = i + 1; k < 255; k++)
		{
			w0 += Histogram[k];
			u0 += k * Histogram[k];
		}
		if (w0 == 0)
		{
			break;
		}
		u0 = u0 / w0;
		w0 = w0 / totalNum;

		auto varValueI = w0 * w1 * (u1 - u0) * (u1 - u0);
		if (varValue < varValueI)
		{
			varValue = varValueI;
			T = i;
		}
	}
	//    cout << T << endl;
	Mat dst = src.clone();
	for (auto i = 0; i < src.rows; i++)
		for (auto j = 0; j < src.cols; j++)
			dst.at<float>(i, j) = src.at<float>(i, j) > T ? 255 : 0;
	return dst;
}

// ������������
//����De Bruijn����ʱ��ʹ����һ��ѭ�����еļ��ɣ��������ɵ������ٴθ���һ�飬���ڽ�β����n-1���ַ����γ�һ���µ����У�ʹ�������е�ÿ���Ӵ����ܹ�ǡ�ó���һ�Ρ�
vector<int> DeBruijn(int k, int n)
{
	std::vector<byte> a(k * n, 0);
	std::vector<byte> seq;

	std::function<void(int, int)> db;
	db = [&](int t, int p)
	{
		if (t > n)
		{
			if (n % p == 0)
			{
				for (int i = 1; i < p + 1; i++)
				{
					seq.push_back(a[i]);
				}
			}
		}
		else
		{
			a[t] = a[t - p];
			db(t + 1, p);
			auto j = a[t - p] + 1;
			while (j < k)
			{
				a[t] = j & 0xFF;
				db(t + 1, t);
				j++;
			}
		}
	};

	db(1, 1);
	std::string buf;
	for (auto i : seq)
	{
		buf.push_back('0' + i);
	}

	std::vector<int> res;
	std::string tmp = buf + buf.substr(0, n - 1);
	for (char i : tmp)
	{
		res.push_back(i - '0');
	}
	return res;
}

void Reconstruction(vector<vector<float>> maximas, vector<vector<float>> minimas,
	vector<vector<float>> colorLabel, /*vector<vector<float>> phases, */const Mat& Hc1,
	Mat Hp2, const double* map)
{
	for (auto i = 0; i < maximas.size(); i++)
	{
		if (maximas[i].empty())continue;
		if (maximas[i].size() < 4)continue;
		auto mark = 0;
		//        double pc = 0;
		for (auto j = 0; j < maximas[i].size(); j++)
		{
			// ͼ���ֵλ������Ӧ��Ƶ��ֵ,��ͼ����б���
			double position;
			if (j < maximas[i].size() - 3)
			{
				position = map[int(pow(3, 3) * colorLabel[i].at(j) + pow(3, 2) * colorLabel[i].at(j + 1) +
					3 * colorLabel[i].at(j + 2) + colorLabel[i].at(j + 3))];
			}
			else
			{
				auto fix = maximas[i].size() - 4;
				auto index = j - maximas[i].size() + 4;
				position = map[int(pow(3, 3) * colorLabel[i].at(fix) + pow(3, 2) * colorLabel[i].at(fix + 1) +
					3 * colorLabel[i].at(fix + 2) + colorLabel[i].at(fix + 3))] + 14.0 * index;
			}

			cout << position << endl;
			Mat matrix = Mat::zeros(cv::Size(3, 3), CV_32FC1);

			matrix.row(0) = Hc1(Rect(0, 2, 3, 1)) * maximas[i][j] - Hc1(Rect(0, 0, 3, 1));
			matrix.row(1) = Hc1(Rect(0, 2, 3, 1)) * float(i + minX) - Hc1(Rect(0, 1, 3, 1));
			matrix.row(2) = Hp2(Rect(0, 2, 3, 1)) * position - Hp2(Rect(0, 0, 3, 1));
			//cout << Hc1 << endl;
			//cout << Hc1(Rect(0, 2, 3, 1)) << endl; // 

			Mat tang = Mat::zeros(cv::Size(3, 1), CV_32FC1);
			Mat b = Mat::zeros(cv::Size(1, 3), CV_32FC1);
			b.at<float>(0, 0) = Hc1.at<float>(0, 3) - Hc1.at<float>(2, 3) * maximas[i][j];
			b.at<float>(1, 0) = Hc1.at<float>(1, 3) - Hc1.at<float>(2, 3) * float(i + minX);
			b.at<float>(2, 0) = Hp2.at<float>(0, 3) - Hp2.at<float>(2, 3) * position;


			// Match coordinates
			cout << "u1 = " << maximas[i][j] << "   " << "v1 = " << float(i + minX) << endl;
			cout << "u2 = " << position << endl;



			solve(matrix, b, tang);
			
			// ������������ϵ�µ���ά�ȵ�
			cout << maximas[i][j] << "    "<< float(i + minX) << endl;
			cout << b << endl;
			// ��ά������Ϊ maximas[i][j]�� float(i + minX)

			// ƥ��������Ϊb��������ͼ���bһ����ʱ��˵����Ӧ�Ķ�ά����һ����



			// ���������Ϣ���й���
			if (tang.at<float>(2, 0) > 750 && tang.at<float>(2, 0) < 1500)
			{
				coordinate.push_back(tang.t());

				int r = (int)rgbChannel.at<Vec3b>(i + minX, maximas[i][j])[2],
					g = rgbChannel.at<Vec3b>(i + minX, maximas[i][j])[1],
					b = rgbChannel.at<Vec3b>(i + minX, maximas[i][j])[0];
				int rgb = ((int)r << 16 | (int)g << 8 | (int)b);
				float frgb = *reinterpret_cast<float*>(&rgb);
				color.push_back(frgb);
			}
			//            if (i == 200)cout << maximas[i][j] << "," << 0 << "," << position << endl;
		/*	if (phases[i].empty())continue;
			auto pi = false;
			auto start = minimas[i][0];
			if (start > maximas[i][j]) continue;
			if (j == 0)
			{
				for (auto k = mark; k + start < maximas[i][j]; k++)
				{
					if ((start + k) < maximas[i][j] && phases[i][k] < 0)continue;
					if ((start + k) < maximas[i][j] && phases[i][k] > 0)
					{
						if (maximas[i][j] - (start + k) < 1)
						{
							continue;
						}
						mark = k + 1;
					}
					else if ((start + k) > maximas[i][j]) break;
				}
			}

			for (auto k = mark; k < phases[i].size() - 1; k++)
			{
				mark++;
				double newPosition;
				if ((start + k) < maximas[i][j] && phases[i][k] < 0) newPosition = position + phases[i][k];
				else if ((maximas[i][j] - (start + k)) > 1 && phases[i][k] > 0)
					newPosition = position + phases[i][k] - 7;
				else if ((start + k) > maximas[i][j] && phases[i][k] > 0)newPosition = position + phases[i][k];
				else if (((start + k) - maximas[i][j]) > 1 && phases[i][k] < 0)
					newPosition = position + phases[i][k] + 7;
				else continue;

				matrix.row(0) = Hc1(Rect(0, 2, 3, 1)) * (start + k) - Hc1(Rect(0, 0, 3, 1));
				matrix.row(2) = Hp2(Rect(0, 2, 3, 1)) * newPosition - Hp2(Rect(0, 0, 3, 1));
				b.row(0) = Hc1.at<float>(0, 3) - Hc1.at<float>(2, 3) * (start + k);
				b.row(2) = Hp2.at<float>(0, 3) - Hp2.at<float>(2, 3) * newPosition;
				solve(matrix, b, tang);
				if (tang.at<float>(2, 0) > 750 && tang.at<float>(2, 0) < 1500)
				{
					coordinate.push_back(tang.t());
					int r = (int)rgbChannel.at<Vec3b>(i + minX, (start + k))[2],
						g = rgbChannel.at<Vec3b>(i + minX, (start + k))[1],
						b = rgbChannel.at<Vec3b>(i + minX, (start + k))[0];
					int rgb = ((int)r << 16 | (int)g << 8 | (int)b);
					float frgb = *reinterpret_cast<float*>(&rgb);
					color.push_back(frgb);
				}

				if ((start + k) > maximas[i][j] && !pi && phases[i][k] > 0) pi = true;

				if ((start + k) > maximas[i][j] && phases[i][k] < 0 && phases[i][k + 1] > 0 && pi)break;

			}*/

		}
	}
}

void saveCoordinate()
{
	ofstream destFile("./Data/my_result/result.pcd", ios::out); //���ı�ģʽ��out.txt��д
	destFile << "# .PCD v0.7 - Point Cloud Data file format" << endl;
	destFile << "VERSION 0.7" << endl;
	destFile << "FIELDS x y z rgb" << endl;
	destFile << "SIZE 4 4 4 4" << endl;
	destFile << "TYPE F F F F" << endl;
	destFile << "COUNT 1 1 1 1" << endl;
	destFile << "WIDTH " << coordinate.size() << endl;
	destFile << "HEIGHT 1" << endl;

	destFile << "VIEWPOINT 0 0 0 1 0 0 0" << endl;
	destFile << "POINTS " << coordinate.size() << endl;
	destFile << "DATA ascii" << endl;
	for (auto i = 0; i < coordinate.size(); i++)
	{
		//        cout << i << endl;
		if (i == coordinate.size() - 1)
		{
			destFile << coordinate[i].at<float>(0, 0) << " " << coordinate[i].at<float>(0, 1) << " "
				<< coordinate[i].at<float>(0, 2) << " " << color[i];
		}
		else
		{
			destFile << coordinate[i].at<float>(0, 0) << " " << coordinate[i].at<float>(0, 1) << " "
				<< coordinate[i].at<float>(0, 2) << " " << color[i] << endl; //��������cout������ofstream����
		}
	}
	destFile.close();
}

void savePly()
{
	std::ofstream plyFile("./Data/my_result/result.ply");

	if (!plyFile)
	{
		std::cerr << "Failed to open output file." << std::endl;
		return;
	}

	plyFile << "ply" << std::endl;
	plyFile << "format ascii 1.0" << std::endl;
	plyFile << "element vertex " << coordinate.size() << std::endl;
	plyFile << "property float x" << std::endl;
	plyFile << "property float y" << std::endl;
	plyFile << "property float z" << std::endl;
	plyFile << "end_header" << std::endl;

	for (const auto& point : coordinate)
	{
		plyFile << point.at<float>(0, 0) << " " << point.at<float>(0, 1) << " " << point.at<float>(0, 2) << std::endl;
	}

	plyFile.close();

	std::cout << "PLY file saved successfully." << std::endl;
}


int main()
{
	// ��ת����
	Mat r(3, 3, CV_32F);
	double m0[3][3] = {
		{9.7004457782050868e-001, 1.3447278830863673e-002, 2.4255450466457243e-001},
		{-8.7082927494022376e-003, 9.9974988338843274e-001, -2.0599424802792338e-002},
		{-2.4277084396282392e-001, 1.7870124701864658e-002, 9.6991905639837694e-001}
	};
	for (auto i = 0; i < r.rows; i++)
		for (auto j = 0; j < r.cols; j++)
			r.at<float>(i, j) = m0[i][j];
	// ƽ�ƾ���
	Mat t(1, 3, CV_32F);
	double m1[1][3] = {
		{-1.9511179496234658e+002, 1.2627509817628756e+001, -5.9345885017522171e+001}
	};

	for (auto i = 0; i < t.rows; i++)
		for (auto j = 0; j < t.cols; j++)
			t.at<float>(i, j) = m1[i][j];

	// ������ڲ�
	Mat kc(3, 3, CV_32F);
	double m2[3][3] = {
		{2.1536653255083029e+003, 0., 6.1886776197116581e+002},
		{0., 2.1484363899666910e+003, 5.0694898820460787e+002},
		{0., 0., 1.}
	};
	for (auto i = 0; i < kc.rows; i++)
		for (auto j = 0; j < kc.cols; j++)
			kc.at<float>(i, j) = m2[i][j];

	// ������ڲ�
	Mat kp(3, 3, CV_32F);
	double m3[3][3] = {
		{1.7235093158297350e+003, 0., 4.4128195628736904e+002},
		{0., 3.4533404000869359e+003, 5.7316457428558715e+002},
		{0., 0., 1.}
	};
	for (auto i = 0; i < kp.rows; i++)
		for (auto j = 0; j < kp.cols; j++)
			kp.at<float>(i, j) = m3[i][j];

	cv::Mat tmp_matrix;
	hconcat(cv::Mat::eye(3, 3, CV_32FC1),
		cv::Mat::zeros(cv::Size(1, 3), CV_32FC1), tmp_matrix);

	// ��������ڲ�����εĳ˻�
	// 3 * 3 * 3 * 4
	Mat hc1 = kc * tmp_matrix; 
	/*cout << hc1 << endl; */
	hconcat(r, t.t(), tmp_matrix);

	// �����ת�������������ε�ƴ�Ӿ���
	Mat hp2 = kp * tmp_matrix;


	rgbChannel = imread("./Data/image/reconstruction/test.png", cv::IMREAD_UNCHANGED);
	int cols = rgbChannel.cols;
	int rows = rgbChannel.rows;

	Mat lab, hsv; // ��ʼ��lab�ռ��ͼƬ��hsv�ռ��ͼƬ
	cvtColor(rgbChannel, hsv, COLOR_BGR2HSV, 3);

	//imshow("hsv", hsv);
	//waitKey(0);

	// �洢��ÿ��ͨ���µ�����ֵ
	vector<Mat>hsvChannel;
	split(hsv, hsvChannel);

	// ת��ΪLab�ռ�
	cvtColor(rgbChannel, lab, COLOR_BGR2Lab);
	//imshow("lab", lab);
	//waitKey(0);

	// �ָ�����е�����
	Mat mask = Mat::zeros(Size(cols, rows), CV_32FC1);
	for (auto i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			mask.at<float>(i, j) = (int)rgbChannel.at<Vec3b>(i, j)[0] > (int)rgbChannel.at<Vec3b>(i, j)[1]
				? (
					(int)rgbChannel.at<Vec3b>(i, j)[0] > (int)rgbChannel.at<Vec3b>(i, j)[2]
					? (int)rgbChannel.at<Vec3b>(i, j)[0]
					: (int)rgbChannel.at<Vec3b>(i, j)[2])
				: (
					(int)rgbChannel.at<Vec3b>(i, j)[1] > (int)rgbChannel.at<Vec3b>(i, j)[2]
					? (int)rgbChannel.at<Vec3b>(i, j)[1]
					: (int)rgbChannel.at<Vec3b>(i, j)[2]);
		}
	}

	Mat tmp = OtsuAlgThreshold(mask);
	//imshow("tmp", tmp);
	//waitKey(0);

	 //����ͼ�����е���̬ѧ�����������ͺ͸�ʴ
	auto kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

	 // ��������ָ�Ƚ��и�ʴ�������ٽ������Ͳ���������ȥ��ͼ���е�С����ϸС����ͨ����
	morphologyEx(tmp, tmp, MORPH_OPEN, kernel);
	//imshow("tmp", tmp);
	//waitKey(0);


	// ���ó������Ƶ�����
	auto min = false;
	for (auto i = 0; i < rows; i++)
	{
		for (auto j = 0; j < cols; j++)
		{
			if (tmp.at<float>(i, j) == 255)
			{
				if (!min)
				{
					minX = i;
					minY = j;
					min = true;
				}

				if (j < minY) minY = j;
				if (i > maxX) maxX = i;
				if (j > maxY) maxY = j;
			}
		}
	}
	

	// ������ֵ��ͼ��ָ���Ϊ���ҵ����εı߽��������
	minX -= 50;
	minY -= 50;
	maxX += 50;
	maxY += 50;
	//cout << minX << "     " << minY << "    "<< maxX << "     "<< maxY;


	// �Բü�������ͼ��ת��Ϊ�Ҷ�ͼ��,�����Ƿŵ���32F�Ŀռ���������ʾΪȫ�׺�����Ҫת��Ϊ8λ����������
	Mat img = Mat::zeros(Size(cols, rows), CV_32FC1);

	for (auto i = minX; i < maxX; i++)
	{
		for (auto j = minY; j < maxY; j++)
		{
			/*cout << 0.2989 * (int)rgbChannel.at<Vec3b>(i, j)[2] +
				0.5907 * (int)rgbChannel.at<Vec3b>(i, j)[1] +
				0.1140 * (int)rgbChannel.at<Vec3b>(i, j)[0] << endl;*/
			img.at<float>(i, j) = 0.2989 * (int)rgbChannel.at<Vec3b>(i, j)[2] +
				0.5907 * (int)rgbChannel.at<Vec3b>(i, j)[1] +
				0.1140 * (int)rgbChannel.at<Vec3b>(i, j)[0];
		
		}
	}
	//  �Բü�������ͼ����б����㣬�������ն�
	kernel = getStructuringElement(MORPH_RECT, cv::Size(3, 3));
	morphologyEx(img, img, MORPH_CLOSE, kernel);

	GaussianBlur(img, img, Size(5, 5), 0, 0);
	Mat B = Mat::zeros(cols, rows, CV_8UC1);

	//normalize(img, img, 1.0, 0.0, NORM_MINMAX);//��һ��0~1֮��
	//img.convertTo(B, CV_8UC1, 255, 0); //ת��Ϊ0~255֮�������
	//imshow("B", B);//��ʾ 
	//waitKey(0);

	Mat derivative1 = Mat::zeros(Size(cols, rows), CV_32FC1);
	Mat derivative2 = Mat::zeros(Size(cols, rows), CV_32FC1);

	for (auto i = 0; i < rows; i++)
	{
		for (auto j = 1; j < cols - 1; j++)
		{
			derivative1.at<float>(i, j) = img.at<float>(i, j + 1) - img.at<float>(i, j);
			derivative2.at<float>(i, j) = img.at<float>(i, j + 1) + img.at<float>(i, j - 1) - 2 * img.at<float>(i, j);
		}
	}

	// �洢�����������
	// �Լ���õ���һ�׺Ͷ��׵������з����� ÿһ�м�ֵ�㱣����maximas������ֵ�㣩�� minimax(��Сֵ�㣩, colorlable -> ��ɫ���
	vector<vector<float>> maximas(0, vector<float>(0, 0));
	vector<vector<float>> minimas(0, vector<float>(0, 0));
	vector<vector<float>> colorLabel(0, vector<float>(0, 0));
	for (auto i = minX; i < maxX; i++)
	{
		maximas.resize(i - minX + 1);
		minimas.resize(i - minX + 1);
		colorLabel.resize(i - minX + 1);
		vector<double> tmpMin;
		for (auto j = minY; j < maxY; j++)
		{
			// cout << i << endl;
			if (derivative1.at<float>(i, j) > 0 && derivative1.at<float>(i, j + 1) < 0)
			{
				double k = derivative1.at<float>(i, j + 1) - derivative1.at<float>(i, j);
				double b = derivative1.at<float>(i, j) - k * j;
				double zero = -b / k;
				double k2 = derivative2.at<float>(i, j + 1) - derivative2.at<float>(i, j);
				double b2 = derivative2.at<float>(i, j) - k2 * j;
				double value = k2 * zero + b2;
				if (value < 0 && lab.at<Vec3b>(i, zero)[0] > 5)
				{
					maximas[i - minX].push_back(zero);
					if (lab.at<Vec3b>(i, zero)[2] < 126)
					{
						colorLabel[i - minX].push_back(2); //blue
					}
					else
					{
						if (lab.at<Vec3b>(i, zero)[1] >= 128)
						{
							colorLabel[i - minX].push_back(0); //red
						}
						else
						{
							colorLabel[i - minX].push_back(1); //green
						}
					}
				}
			}

			if (derivative1.at<float>(i, j) < 0 && derivative1.at<float>(i, j + 1) > 0)
			{
				double k = derivative1.at<float>(i, j + 1) - derivative1.at<float>(i, j);
				double b = derivative1.at<float>(i, j) - k * j;
				double zero = -b / k;
				double k2 = derivative2.at<float>(i, j + 1) - derivative2.at<float>(i, j);
				double b2 = derivative2.at<float>(i, j) - k2 * j;
				double value = k2 * zero + b2;
				if (value > 0)
				{
					tmpMin.push_back(zero);
				}
			}
		}
		if (!tmpMin.empty() && !maximas[i - minX].empty())
		{
			auto pos = 0;
			for (auto j = 0; j < tmpMin.size() - 1; j++)
			{

				if (tmpMin[j + 1] < maximas[i - minX][pos])
				{
					continue;
				}
				minimas[i - minX].push_back(tmpMin[j]);
				pos++;
				if (pos >= maximas[i - minX].size())break;
			}

		}
	}
	// ��ɫ��ǩ���ڽ���ֵ���Ϊ��ͬ�����, ��Сֵ�����ɫ��ǩ����Ӧ�ļ���ֵ�����ɫ��ǩ��ͬ��
	//cout << minimas.size() << endl;
	//cout << colorLabel.size() << endl;
	
	//cout << maximas.size() << endl;

	// ���ӳ��ܶ�,��ʱ�ɲ���Ҫ
	// To do:С���任

	//
	auto db = DeBruijn(3, 4);
	/*cout << db.size() << endl;*/
	double map[76]{ 0 };

	// 
	for (auto i = 0; i < 61; i++)
	{
		// ��ǰ64�����ƽ��б���Ƶ��ֵ
		int index = int(pow(3, 3) * db.at(i) + pow(3, 2) * db.at(i + 1) + 3 * db.at(i + 2) + db.at(i + 3));

		// De Bruijn������ÿ��3-mer�������г��ֵĴ����������ʽ����Դ��һƪ��Ϊ"Visualization and analysis of DNA microarrays using R and bioconductor"
		map[index] = 7.5 + 14 * i; // ��Ӧ��Ƶ��ֵ,�Ѿ���ʼ������
	}
	Reconstruction(maximas, minimas, colorLabel, hc1, hp2, map);

	cout << coordinate.size() << endl;

	ofstream destFile("./Data/my_result/result.txt", ios::out); //���ı�ģʽ��out.txt��д
	for (auto i = 0; i < coordinate.size(); i++)
	{
		if (i == coordinate.size() - 1)
		{
			destFile << coordinate[i].at<float>(0, 0) << " " << coordinate[i].at<float>(0, 1) << " "
				<< coordinate[i].at<float>(0, 2);
		}
		else
		{
			destFile << coordinate[i].at<float>(0, 0) << " " << coordinate[i].at<float>(0, 1) << " "
				<< coordinate[i].at<float>(0, 2) << endl; //��������cout������ofstream����
		}
	}

	destFile.close();
	saveCoordinate();
	savePly();
}