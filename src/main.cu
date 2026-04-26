#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::ostringstream error;                                              \
      error << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "     \
            << cudaGetErrorString(status);                                   \
      throw std::runtime_error(error.str());                                 \
    }                                                                        \
  } while (0)

struct Image {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> pixels;
};

struct Options {
  fs::path input_dir;
  fs::path output_dir;
  int threshold = 88;
  int max_images = 0;
  int warmup = 1;
};

struct ImageStats {
  std::string name;
  int width = 0;
  int height = 0;
  int edge_pixels = 0;
  double edge_ratio = 0.0;
  float gpu_ms = 0.0f;
};

__device__ int ClampInt(int value, int low, int high) {
  return value < low ? low : (value > high ? high : value);
}

__device__ int AbsInt(int value) { return value < 0 ? -value : value; }

__global__ void GaussianBlur3x3Kernel(const std::uint8_t* input,
                                      std::uint8_t* blurred, int width,
                                      int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const int xm1 = ClampInt(x - 1, 0, width - 1);
  const int xp1 = ClampInt(x + 1, 0, width - 1);
  const int ym1 = ClampInt(y - 1, 0, height - 1);
  const int yp1 = ClampInt(y + 1, 0, height - 1);

  const int sum = input[ym1 * width + xm1] + 2 * input[ym1 * width + x] +
                  input[ym1 * width + xp1] + 2 * input[y * width + xm1] +
                  4 * input[y * width + x] + 2 * input[y * width + xp1] +
                  input[yp1 * width + xm1] + 2 * input[yp1 * width + x] +
                  input[yp1 * width + xp1];
  blurred[y * width + x] = static_cast<std::uint8_t>(sum / 16);
}

__global__ void SobelThresholdKernel(const std::uint8_t* blurred,
                                     std::uint8_t* output, int width,
                                     int height, int threshold) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
    output[y * width + x] = 0;
    return;
  }

  const int top_left = blurred[(y - 1) * width + (x - 1)];
  const int top = blurred[(y - 1) * width + x];
  const int top_right = blurred[(y - 1) * width + (x + 1)];
  const int left = blurred[y * width + (x - 1)];
  const int right = blurred[y * width + (x + 1)];
  const int bottom_left = blurred[(y + 1) * width + (x - 1)];
  const int bottom = blurred[(y + 1) * width + x];
  const int bottom_right = blurred[(y + 1) * width + (x + 1)];

  const int gx = -top_left + top_right - 2 * left + 2 * right - bottom_left +
                 bottom_right;
  const int gy = -top_left - 2 * top - top_right + bottom_left + 2 * bottom +
                 bottom_right;
  const int magnitude = ClampInt(AbsInt(gx) + AbsInt(gy), 0, 255);
  output[y * width + x] = magnitude >= threshold ? 255 : 0;
}

void PrintUsage(const char* program) {
  std::cerr << "Usage: " << program
            << " --input <input_dir> --output <output_dir> [options]\n\n"
            << "Options:\n"
            << "  --threshold <0-255>   Edge threshold. Default: 88\n"
            << "  --max-images <n>      Process at most n images. Default: all\n"
            << "  --warmup <n>          Warmup GPU launches. Default: 1\n"
            << "  --help                Show this message\n";
}

int ParseInt(const std::string& value, const std::string& name) {
  try {
    size_t parsed = 0;
    const int result = std::stoi(value, &parsed);
    if (parsed != value.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return result;
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid integer for " + name + ": " + value);
  }
}

Options ParseArgs(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (arg == "--input") {
      options.input_dir = require_value(arg);
    } else if (arg == "--output") {
      options.output_dir = require_value(arg);
    } else if (arg == "--threshold") {
      options.threshold = ParseInt(require_value(arg), arg);
    } else if (arg == "--max-images") {
      options.max_images = ParseInt(require_value(arg), arg);
    } else if (arg == "--warmup") {
      options.warmup = ParseInt(require_value(arg), arg);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (options.input_dir.empty() || options.output_dir.empty()) {
    throw std::runtime_error("--input and --output are required");
  }
  if (options.threshold < 0 || options.threshold > 255) {
    throw std::runtime_error("--threshold must be between 0 and 255");
  }
  if (options.max_images < 0 || options.warmup < 0) {
    throw std::runtime_error("--max-images and --warmup cannot be negative");
  }
  return options;
}

std::string ReadToken(std::istream& input) {
  std::string token;
  while (input >> token) {
    if (!token.empty() && token[0] == '#') {
      input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }
    return token;
  }
  throw std::runtime_error("Unexpected end of PGM header");
}

Image ReadPgm(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Could not open input image: " + path.string());
  }

  const std::string magic = ReadToken(file);
  if (magic != "P5") {
    throw std::runtime_error("Unsupported image format in " + path.string() +
                             "; expected binary PGM P5");
  }

  Image image;
  image.width = ParseInt(ReadToken(file), "PGM width");
  image.height = ParseInt(ReadToken(file), "PGM height");
  const int max_value = ParseInt(ReadToken(file), "PGM max value");
  if (image.width <= 0 || image.height <= 0 || max_value != 255) {
    throw std::runtime_error("Invalid PGM header in " + path.string());
  }

  file.get();
  const size_t pixel_count =
      static_cast<size_t>(image.width) * static_cast<size_t>(image.height);
  image.pixels.resize(pixel_count);
  file.read(reinterpret_cast<char*>(image.pixels.data()),
            static_cast<std::streamsize>(pixel_count));
  if (file.gcount() != static_cast<std::streamsize>(pixel_count)) {
    throw std::runtime_error("PGM file ended early: " + path.string());
  }
  return image;
}

void WritePgm(const fs::path& path, const Image& image) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Could not open output image: " + path.string());
  }
  file << "P5\n" << image.width << " " << image.height << "\n255\n";
  file.write(reinterpret_cast<const char*>(image.pixels.data()),
             static_cast<std::streamsize>(image.pixels.size()));
}

std::vector<fs::path> ListPgmFiles(const fs::path& input_dir) {
  if (!fs::exists(input_dir)) {
    throw std::runtime_error("Input directory does not exist: " +
                             input_dir.string());
  }

  std::vector<fs::path> files;
  for (const fs::directory_entry& entry : fs::directory_iterator(input_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    std::string extension = entry.path().extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (extension == ".pgm") {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

ImageStats ProcessImageOnGpu(const fs::path& input_path,
                             const fs::path& output_path,
                             const Options& options) {
  const Image input = ReadPgm(input_path);
  Image output;
  output.width = input.width;
  output.height = input.height;
  output.pixels.resize(input.pixels.size());

  const size_t bytes = input.pixels.size() * sizeof(std::uint8_t);
  std::uint8_t* device_input = nullptr;
  std::uint8_t* device_blurred = nullptr;
  std::uint8_t* device_output = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  CUDA_CHECK(cudaMalloc(&device_input, bytes));
  CUDA_CHECK(cudaMalloc(&device_blurred, bytes));
  CUDA_CHECK(cudaMalloc(&device_output, bytes));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaMemcpy(device_input, input.pixels.data(), bytes,
                        cudaMemcpyHostToDevice));

  const dim3 block(16, 16);
  const dim3 grid((input.width + block.x - 1) / block.x,
                  (input.height + block.y - 1) / block.y);

  for (int i = 0; i < options.warmup; ++i) {
    GaussianBlur3x3Kernel<<<grid, block>>>(device_input, device_blurred,
                                           input.width, input.height);
    SobelThresholdKernel<<<grid, block>>>(device_blurred, device_output,
                                          input.width, input.height,
                                          options.threshold);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  GaussianBlur3x3Kernel<<<grid, block>>>(device_input, device_blurred,
                                         input.width, input.height);
  SobelThresholdKernel<<<grid, block>>>(device_blurred, device_output,
                                        input.width, input.height,
                                        options.threshold);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float gpu_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
  CUDA_CHECK(cudaMemcpy(output.pixels.data(), device_output, bytes,
                        cudaMemcpyDeviceToHost));

  int edge_pixels = 0;
  for (const std::uint8_t pixel : output.pixels) {
    if (pixel == 255) {
      ++edge_pixels;
    }
  }

  WritePgm(output_path, output);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_blurred));
  CUDA_CHECK(cudaFree(device_output));

  ImageStats stats;
  stats.name = input_path.filename().string();
  stats.width = input.width;
  stats.height = input.height;
  stats.edge_pixels = edge_pixels;
  stats.edge_ratio =
      static_cast<double>(edge_pixels) / static_cast<double>(output.pixels.size());
  stats.gpu_ms = gpu_ms;
  return stats;
}

void WriteSummaryCsv(const fs::path& path, const std::vector<ImageStats>& stats) {
  std::ofstream csv(path);
  if (!csv) {
    throw std::runtime_error("Could not write summary CSV: " + path.string());
  }
  csv << "image,width,height,pixels,edge_pixels,edge_ratio,gpu_ms\n";
  csv << std::fixed << std::setprecision(6);
  for (const ImageStats& row : stats) {
    const int pixels = row.width * row.height;
    csv << row.name << "," << row.width << "," << row.height << "," << pixels
        << "," << row.edge_pixels << "," << row.edge_ratio << ","
        << row.gpu_ms << "\n";
  }
}

void PrintDeviceInfo() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::cout << "GPU device: " << properties.name << "\n"
            << "Compute capability: " << properties.major << "."
            << properties.minor << "\n"
            << "Multiprocessors: " << properties.multiProcessorCount << "\n";
}

int main(int argc, char** argv) {
  try {
    const Options options = ParseArgs(argc, argv);
    fs::create_directories(options.output_dir);

    std::vector<fs::path> files = ListPgmFiles(options.input_dir);
    if (options.max_images > 0 &&
        static_cast<size_t>(options.max_images) < files.size()) {
      files.resize(static_cast<size_t>(options.max_images));
    }
    if (files.empty()) {
      throw std::runtime_error("No .pgm files found in " +
                               options.input_dir.string());
    }

    PrintDeviceInfo();
    std::cout << "Input directory: " << options.input_dir << "\n"
              << "Output directory: " << options.output_dir << "\n"
              << "Threshold: " << options.threshold << "\n"
              << "Warmup launches per image: " << options.warmup << "\n"
              << "Images scheduled: " << files.size() << "\n";

    std::vector<ImageStats> stats;
    stats.reserve(files.size());
    const auto wall_start = std::chrono::steady_clock::now();

    for (const fs::path& input_path : files) {
      const fs::path output_path = options.output_dir / input_path.filename();
      ImageStats row = ProcessImageOnGpu(input_path, output_path, options);
      std::cout << "processed " << row.name << " (" << row.width << "x"
                << row.height << "), gpu_ms=" << std::fixed
                << std::setprecision(4) << row.gpu_ms
                << ", edge_ratio=" << std::setprecision(4) << row.edge_ratio
                << "\n";
      stats.push_back(row);
    }

    const auto wall_stop = std::chrono::steady_clock::now();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
    const fs::path summary_path = options.output_dir / "summary.csv";
    WriteSummaryCsv(summary_path, stats);

    long long total_pixels = 0;
    double total_gpu_ms = 0.0;
    for (const ImageStats& row : stats) {
      total_pixels += static_cast<long long>(row.width) * row.height;
      total_gpu_ms += row.gpu_ms;
    }

    std::cout << "\nProcessed images: " << stats.size() << "\n"
              << "Total pixels: " << total_pixels << "\n"
              << "Accumulated kernel time (ms): " << std::fixed
              << std::setprecision(4) << total_gpu_ms << "\n"
              << "Wall time including I/O (ms): " << std::setprecision(4)
              << wall_ms << "\n"
              << "Summary CSV: " << summary_path << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    PrintUsage(argv[0]);
    return 1;
  }
}
