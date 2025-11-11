#include "CameraArguments.h"
#include "CoreAlgorithm.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct CropBounds {
	int minRow{ 0 };
	int maxRow{ 0 };
	int minCol{ 0 };
	int maxCol{ 0 };
};

struct ReconstructionResult {
	std::vector<cv::Vec3f> points;
	std::vector<float> colors;
};

cv::Mat1b otsuThreshold(const cv::Mat1b &src) {
	CV_Assert(src.channels() == 1);

	std::array<double, 256> histogram{ 0.0 };
	for (int r = 0; r < src.rows; ++r) {
		const auto *row = src.ptr<uchar>(r);
		for (int c = 0; c < src.cols; ++c) {
			const auto value = row[c];
			if (value != 0) {
				++histogram[value];
			}
		}
	}

	const double totalPixels   = static_cast<double>(src.rows) * src.cols;
	const double nonZeroPixels = std::accumulate(histogram.begin(), histogram.end(), 0.0);
	if (nonZeroPixels == 0.0) {
		return cv::Mat1b::zeros(src.size());
	}

	const double weightedSum = [&histogram]() {
		double sum = 0.0;
		for (int i = 0; i < static_cast<int>(histogram.size()); ++i) {
			sum += static_cast<double>(i) * histogram[i];
		}
		return sum;
	}();

	double bestVariance = -1.0;
	int bestThreshold   = 0;
	double cumulativeWeight{ 0.0 };
	double cumulativeMean{ 0.0 };

	for (int threshold = 0; threshold < 256; ++threshold) {
		cumulativeWeight += histogram[threshold];
		if (cumulativeWeight == 0.0) {
			continue;
		}

		const double remainingWeight = nonZeroPixels - cumulativeWeight;
		if (remainingWeight == 0.0) {
			break;
		}

		cumulativeMean += static_cast<double>(threshold) * histogram[threshold];
		const double foregroundMean = cumulativeMean / cumulativeWeight;
		const double backgroundMean = (weightedSum - cumulativeMean) / remainingWeight;

		const double foregroundWeight = cumulativeWeight / totalPixels;
		const double backgroundWeight = remainingWeight / totalPixels;
		const double variance         = foregroundWeight * backgroundWeight * std::pow(foregroundMean - backgroundMean, 2.0);

		if (variance > bestVariance) {
			bestVariance  = variance;
			bestThreshold = threshold;
		}
	}

	cv::Mat1b dst;
	cv::threshold(src, dst, bestThreshold, 255, cv::THRESH_BINARY);
	return dst;
}

CropBounds findStripeBounds(const cv::Mat1b &mask, int padding) {
	CropBounds bounds{ mask.rows, 0, mask.cols, 0 };
	bool found = false;

	for (int r = 0; r < mask.rows; ++r) {
		const auto *row = mask.ptr<uchar>(r);
		for (int c = 0; c < mask.cols; ++c) {
			if (row[c] != 255) {
				continue;
			}

			if (!found) {
				bounds.minRow = bounds.maxRow = r;
				bounds.minCol = bounds.maxCol = c;
				found                         = true;
				continue;
			}

			bounds.minRow = std::min(bounds.minRow, r);
			bounds.maxRow = std::max(bounds.maxRow, r);
			bounds.minCol = std::min(bounds.minCol, c);
			bounds.maxCol = std::max(bounds.maxCol, c);
		}
	}

	if (!found) {
		return { 0, mask.rows, 0, mask.cols };
	}

	bounds.minRow = std::max(0, bounds.minRow - padding);
	bounds.minCol = std::max(0, bounds.minCol - padding);
	bounds.maxRow = std::min(mask.rows, bounds.maxRow + padding + 1);
	bounds.maxCol = std::min(mask.cols, bounds.maxCol + padding + 1);
	return bounds;
}

std::vector<int> deBruijnSequence(int alphabet, int subseqLength) {
	std::vector<int> a(alphabet * subseqLength, 0);
	std::vector<int> sequence;
	sequence.reserve(static_cast<std::size_t>(std::pow(alphabet, subseqLength)));

	const auto db = [&](auto &&self, int t, int p) -> void {
		if (t > subseqLength) {
			if (subseqLength % p == 0) {
				for (int i = 1; i <= p; ++i) {
					sequence.push_back(a[i]);
				}
			}
			return;
		}

		a[t] = a[t - p];
		self(self, t + 1, p);
		for (int j = a[t - p] + 1; j < alphabet; ++j) {
			a[t] = j;
			self(self, t + 1, t);
		}
	};

	db(db, 1, 1);

	std::vector<int> result(sequence.begin(), sequence.end());
	const int extra = std::max(0, subseqLength - 1);
	for (int i = 0; i < extra && i < static_cast<int>(result.size()); ++i) {
		result.push_back(result[i]);
	}
	return result;
}

int encodePattern(const std::vector<int> &colors, std::size_t start) {
	CV_Assert(start + 3 < colors.size());
	constexpr int base  = 3;
	constexpr int base2 = base * base;
	constexpr int base3 = base2 * base;
	const int c0        = colors[start];
	const int c1        = colors[start + 1];
	const int c2        = colors[start + 2];
	const int c3        = colors[start + 3];
	return base3 * c0 + base2 * c1 + base * c2 + c3;
}

double lookupFrequency(const std::vector<int> &colors, std::size_t idx, std::span<const double> frequencyLut) {
	if (colors.size() < 4) {
		return 0.0;
	}

	if (idx + 3 < colors.size()) {
		const int code = encodePattern(colors, idx);
		return code < static_cast<int>(frequencyLut.size()) ? frequencyLut[code] : 0.0;
	}

	const std::size_t anchor = colors.size() - 4;
	const int code           = encodePattern(colors, anchor);
	const double baseFreq    = code < static_cast<int>(frequencyLut.size()) ? frequencyLut[code] : 0.0;
	const double offset      = static_cast<double>(idx - colors.size() + 4);
	return baseFreq + 14.0 * offset;
}

ReconstructionResult reconstructPoints(const std::vector<std::vector<float>> &maxima,
                                       const std::vector<std::vector<int>> &colorLabels,
                                       const cv::Mat &leftProjection,
                                       const cv::Mat &rightProjection,
                                       std::span<const double> frequencyLut,
                                       const CropBounds &bounds,
                                       const cv::Mat &rgbSource) {
	ReconstructionResult result;

	const auto candidateCount =
	    std::accumulate(maxima.begin(), maxima.end(), std::size_t{ 0 }, [](std::size_t acc, const auto &row) { return acc + row.size(); });
	result.points.reserve(candidateCount);
	result.colors.reserve(candidateCount);

	cv::Mat1f matrix(3, 3);
	cv::Mat1f rhs(3, 1);
	cv::Mat1f solution(3, 1);

	for (std::size_t rowIdx = 0; rowIdx < maxima.size(); ++rowIdx) {
		const auto &rowMaxima = maxima[rowIdx];
		if (rowMaxima.size() < 4 || rowIdx >= colorLabels.size()) {
			continue;
		}

		const auto &rowColors = colorLabels[rowIdx];
		if (rowColors.size() != rowMaxima.size()) {
			continue;
		}

		const int imageRow = static_cast<int>(rowIdx) + bounds.minRow;

		for (std::size_t colIdx = 0; colIdx < rowMaxima.size(); ++colIdx) {
			const double frequency = lookupFrequency(rowColors, colIdx, frequencyLut);
			const float u          = rowMaxima[colIdx];
			const float v          = static_cast<float>(imageRow);

			matrix.row(0) = leftProjection(cv::Rect(0, 2, 3, 1)) * u - leftProjection(cv::Rect(0, 0, 3, 1));
			matrix.row(1) = leftProjection(cv::Rect(0, 2, 3, 1)) * v - leftProjection(cv::Rect(0, 1, 3, 1));
			matrix.row(2) = rightProjection(cv::Rect(0, 2, 3, 1)) * static_cast<float>(frequency) - rightProjection(cv::Rect(0, 0, 3, 1));

			rhs.at<float>(0, 0) = leftProjection.at<float>(0, 3) - leftProjection.at<float>(2, 3) * u;
			rhs.at<float>(1, 0) = leftProjection.at<float>(1, 3) - leftProjection.at<float>(2, 3) * v;
			rhs.at<float>(2, 0) = rightProjection.at<float>(0, 3) - rightProjection.at<float>(2, 3) * static_cast<float>(frequency);

			if (!cv::solve(matrix, rhs, solution, cv::DECOMP_LU)) {
				continue;
			}

			const float depth = solution.at<float>(2, 0);
			if (depth < 750.0f || depth > 1500.0f) {
				continue;
			}

			result.points.emplace_back(solution.at<float>(0, 0), solution.at<float>(1, 0), depth);

			const int col    = std::clamp(static_cast<int>(std::lround(u)), 0, rgbSource.cols - 1);
			const auto pixel = rgbSource.at<cv::Vec3b>(imageRow, col);
			const int packed = (static_cast<int>(pixel[2]) << 16) | (static_cast<int>(pixel[1]) << 8) | static_cast<int>(pixel[0]);
			result.colors.push_back(std::bit_cast<float>(packed));
		}
	}

	return result;
}

void saveTxt(std::string_view path, const ReconstructionResult &result) {
	std::ofstream file{ std::string(path) };
	if (!file) {
		std::cerr << "Failed to open " << path << '\n';
		return;
	}

	for (std::size_t i = 0; i < result.points.size(); ++i) {
		const auto &p = result.points[i];
		file << p[0] << ' ' << p[1] << ' ' << p[2];
		if (i + 1 < result.points.size()) {
			file << '\n';
		}
	}
}

void savePcd(std::string_view path, const ReconstructionResult &result) {
	std::ofstream file{ std::string(path) };
	if (!file) {
		std::cerr << "Failed to open " << path << '\n';
		return;
	}

	file << "# .PCD v0.7 - Point Cloud Data file format\n";
	file << "VERSION 0.7\n";
	file << "FIELDS x y z rgb\n";
	file << "SIZE 4 4 4 4\n";
	file << "TYPE F F F F\n";
	file << "COUNT 1 1 1 1\n";
	file << "WIDTH " << result.points.size() << '\n';
	file << "HEIGHT 1\n";
	file << "VIEWPOINT 0 0 0 1 0 0 0\n";
	file << "POINTS " << result.points.size() << '\n';
	file << "DATA ascii\n";

	for (std::size_t i = 0; i < result.points.size(); ++i) {
		const auto &p   = result.points[i];
		const float rgb = i < result.colors.size() ? result.colors[i] : 0.0f;
		file << p[0] << ' ' << p[1] << ' ' << p[2] << ' ' << rgb << '\n';
	}
}

void savePly(std::string_view path, const ReconstructionResult &result) {
	std::ofstream file{ std::string(path) };
	if (!file) {
		std::cerr << "Failed to open " << path << '\n';
		return;
	}

	file << "ply\n";
	file << "format ascii 1.0\n";
	file << "element vertex " << result.points.size() << '\n';
	file << "property float x\n";
	file << "property float y\n";
	file << "property float z\n";
	file << "end_header\n";

	for (const auto &point : result.points) {
		file << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
	}
}

} // namespace

int main() {
	const cv::Mat rotation        = (cv::Mat_<float>(3, 3) << 9.7004458e-001f,
                              1.3447279e-002f,
                              2.4255451e-001f,
                              -8.7082927e-003f,
                              9.9974989e-001f,
                              -2.0599425e-002f,
                              -2.4277084e-001f,
                              1.7870125e-002f,
                              9.6991906e-001f);
	const cv::Mat translation     = (cv::Mat_<float>(3, 1) << -1.95111795e+002f, 1.26275098e+001f, -5.9345885e+001f);
	const cv::Mat leftIntrinsics  = (cv::Mat_<float>(3, 3) << 2.1536653e+003f, 0.f, 6.1886776e+002f, 0.f, 2.1484364e+003f, 5.0694899e+002f, 0.f, 0.f, 1.f);
	const cv::Mat rightIntrinsics = (cv::Mat_<float>(3, 3) << 1.7235093e+003f, 0.f, 4.4128196e+002f, 0.f, 3.4533404e+003f, 5.7316458e+002f, 0.f, 0.f, 1.f);

	cv::Mat projection;
	cv::hconcat(cv::Mat::eye(3, 3, CV_32F), cv::Mat::zeros(cv::Size(1, 3), CV_32F), projection);
	const cv::Mat leftProjection = leftIntrinsics * projection;

	cv::hconcat(rotation, translation, projection);
	const cv::Mat rightProjection = rightIntrinsics * projection;

	cv::Mat rgb = cv::imread("../Data/image/reconstruction/test.png", cv::IMREAD_COLOR);
	if (rgb.empty()) {
		std::cerr << "Failed to load input image.\n";
		return EXIT_FAILURE;
	}

	cv::Mat lab;
	cv::cvtColor(rgb, lab, cv::COLOR_BGR2Lab);

	cv::Mat1b mask(rgb.rows, rgb.cols);
	for (int r = 0; r < rgb.rows; ++r) {
		const auto *row = rgb.ptr<cv::Vec3b>(r);
		auto *maskRow   = mask.ptr<uchar>(r);
		for (int c = 0; c < rgb.cols; ++c) {
			const auto &pixel = row[c];
			maskRow[c]        = static_cast<uchar>(std::max({ pixel[0], pixel[1], pixel[2] }));
		}
	}

	cv::Mat1b binaryMask = otsuThreshold(mask);
	auto openKernel      = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::morphologyEx(binaryMask, binaryMask, cv::MORPH_OPEN, openKernel);

	const auto bounds = findStripeBounds(binaryMask, 50);

	cv::Mat gray8;
	cv::cvtColor(rgb, gray8, cv::COLOR_BGR2GRAY);
	cv::Mat1f gray;
	gray8.convertTo(gray, CV_32F);

	auto closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::morphologyEx(gray, gray, cv::MORPH_CLOSE, closeKernel);
	cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0.0);

	cv::Mat1f derivative1 = cv::Mat1f::zeros(gray.size());
	cv::Mat1f derivative2 = cv::Mat1f::zeros(gray.size());

	const int minDerivativeCol = std::max(bounds.minCol, 1);
	const int maxDerivativeCol = std::min(bounds.maxCol, gray.cols - 1);
	for (int r = bounds.minRow; r < bounds.maxRow; ++r) {
		const auto *grayRow = gray.ptr<float>(r);
		auto *d1Row         = derivative1.ptr<float>(r);
		auto *d2Row         = derivative2.ptr<float>(r);
		for (int c = minDerivativeCol; c < maxDerivativeCol; ++c) {
			d1Row[c] = grayRow[c + 1] - grayRow[c];
			d2Row[c] = grayRow[c + 1] + grayRow[c - 1] - 2.0f * grayRow[c];
		}
	}

	const int rowSpan = std::max(0, bounds.maxRow - bounds.minRow);
	std::vector<std::vector<float>> maxima(rowSpan);
	std::vector<std::vector<float>> minima(rowSpan);
	std::vector<std::vector<int>> colorLabels(rowSpan);

	for (int r = bounds.minRow; r < bounds.maxRow; ++r) {
		auto &rowMaxima = maxima[r - bounds.minRow];
		auto &rowMinima = minima[r - bounds.minRow];
		auto &rowColors = colorLabels[r - bounds.minRow];

		std::vector<float> tmpMinima;
		for (int c = bounds.minCol; c + 1 < bounds.maxCol; ++c) {
			const float d     = derivative1.at<float>(r, c);
			const float dNext = derivative1.at<float>(r, c + 1);

			const float slope = derivative1.at<float>(r, c + 1) - derivative1.at<float>(r, c);
			if (d > 0.0f && dNext < 0.0f && std::abs(slope) > std::numeric_limits<float>::epsilon()) {
				const float intercept  = d - slope * static_cast<float>(c);
				const double zero      = -static_cast<double>(intercept) / static_cast<double>(slope);
				const double k2        = derivative2.at<float>(r, c + 1) - derivative2.at<float>(r, c);
				const double b2        = derivative2.at<float>(r, c) - k2 * c;
				const double curvature = k2 * zero + b2;

				if (curvature < 0.0) {
					const int sampleCol = std::clamp(static_cast<int>(std::lround(zero)), 0, lab.cols - 1);
					const auto labPixel = lab.at<cv::Vec3b>(r, sampleCol);
					if (labPixel[0] > 5) {
						rowMaxima.push_back(static_cast<float>(zero));
						if (labPixel[2] < 126) {
							rowColors.push_back(2);
						} else if (labPixel[1] >= 128) {
							rowColors.push_back(0);
						} else {
							rowColors.push_back(1);
						}
					}
				}
			}

			if (d < 0.0f && dNext > 0.0f && std::abs(slope) > std::numeric_limits<float>::epsilon()) {
				const float intercept  = d - slope * static_cast<float>(c);
				const double zero      = -static_cast<double>(intercept) / static_cast<double>(slope);
				const double k2        = derivative2.at<float>(r, c + 1) - derivative2.at<float>(r, c);
				const double b2        = derivative2.at<float>(r, c) - k2 * c;
				const double curvature = k2 * zero + b2;
				if (curvature > 0.0) {
					tmpMinima.push_back(static_cast<float>(zero));
				}
			}
		}

		if (!tmpMinima.empty() && !rowMaxima.empty()) {
			std::size_t pos = 0;
			for (std::size_t idx = 0; idx + 1 < tmpMinima.size() && pos < rowMaxima.size(); ++idx) {
				if (tmpMinima[idx + 1] < rowMaxima[pos]) {
					continue;
				}
				rowMinima.push_back(tmpMinima[idx]);
				++pos;
			}
		}
	}

	const auto deBruijn = deBruijnSequence(3, 4);
	std::array<double, 81> frequencyMap{ 0.0 };

	const auto encodeSequence = [&](std::size_t offset) {
		constexpr int base  = 3;
		constexpr int base2 = base * base;
		constexpr int base3 = base2 * base;
		return base3 * deBruijn[offset] + base2 * deBruijn[offset + 1] + base * deBruijn[offset + 2] + deBruijn[offset + 3];
	};

	for (std::size_t i = 0; i < 61; ++i) {
		frequencyMap[encodeSequence(i)] = 7.5 + 14.0 * static_cast<double>(i);
	}

	const auto result = reconstructPoints(maxima, colorLabels, leftProjection, rightProjection, std::span<const double>(frequencyMap), bounds, rgb);

	std::cout << "Reconstructed points: " << result.points.size() << '\n';

	saveTxt("../Data/my_result/result.txt", result);
	savePcd("../Data/my_result/result.pcd", result);
	savePly("../Data/my_result/result.ply", result);

	return 0;
}
