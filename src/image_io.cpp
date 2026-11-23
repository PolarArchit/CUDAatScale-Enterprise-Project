// Copyright 2026 Enterprise CUDA Team. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "image_io.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace cuda_scale {

namespace {

// Helper to skip Netpbm comments ('#') and whitespace.
void SkipCommentsAndWhitespace(std::ifstream& infile) {
  char ch;
  while (infile.get(ch)) {
    if (ch == '#') {
      // Skip the rest of this comment line.
      std::string comment;
      std::getline(infile, comment);
    } else if (!std::isspace(static_cast<unsigned char>(ch))) {
      infile.unget();
      break;
    }
  }
}

}  // namespace

bool ReadPGM(const std::string& filepath, Image* image) {
  if (!image) return false;

  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "[ImageIO] Error opening PGM file: " << filepath << std::endl;
    return false;
  }

  std::string magic;
  file >> magic;
  if (magic != "P5") {
    std::cerr << "[ImageIO] Unsupported PGM format (expected P5, got " << magic
              << ") in " << filepath << std::endl;
    return false;
  }

  SkipCommentsAndWhitespace(file);
  int width = 0;
  file >> width;

  SkipCommentsAndWhitespace(file);
  int height = 0;
  file >> height;

  SkipCommentsAndWhitespace(file);
  int max_val = 0;
  file >> max_val;

  // Single whitespace/newline after max_val before binary raster data.
  file.get();

  if (width <= 0 || height <= 0 || max_val <= 0 || max_val > 255) {
    std::cerr << "[ImageIO] Invalid PGM dimensions or bit depth in: " << filepath
              << " (" << width << "x" << height << ", max=" << max_val << ")"
              << std::endl;
    return false;
  }

  image->Allocate(width, height, 1);
  file.read(reinterpret_cast<char*>(image->data.data()), image->SizeInBytes());

  if (!file) {
    std::cerr << "[ImageIO] Incomplete binary data in: " << filepath << std::endl;
    return false;
  }

  return true;
}

bool WritePGM(const std::string& filepath, const Image& image) {
  if (!image.IsValid() || image.channels != 1) {
    std::cerr << "[ImageIO] Invalid image for PGM write: channels="
              << image.channels << ", size=" << image.width << "x"
              << image.height << std::endl;
    return false;
  }

  std::ofstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "[ImageIO] Error opening output file: " << filepath << std::endl;
    return false;
  }

  file << "P5\n"
       << "# Created by Enterprise CUDA Image Processing Pipeline\n"
       << image.width << " " << image.height << "\n"
       << "255\n";

  file.write(reinterpret_cast<const char*>(image.data.data()),
             image.SizeInBytes());

  return file.good();
}

bool ReadPPM(const std::string& filepath, Image* image) {
  if (!image) return false;

  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "[ImageIO] Error opening PPM file: " << filepath << std::endl;
    return false;
  }

  std::string magic;
  file >> magic;
  if (magic != "P6") {
    std::cerr << "[ImageIO] Unsupported PPM format (expected P6, got " << magic
              << ") in " << filepath << std::endl;
    return false;
  }

  SkipCommentsAndWhitespace(file);
  int width = 0;
  file >> width;

  SkipCommentsAndWhitespace(file);
  int height = 0;
  file >> height;

  SkipCommentsAndWhitespace(file);
  int max_val = 0;
  file >> max_val;

  file.get();

  if (width <= 0 || height <= 0 || max_val <= 0 || max_val > 255) {
    std::cerr << "[ImageIO] Invalid PPM dimensions or bit depth in: " << filepath
              << std::endl;
    return false;
  }

  image->Allocate(width, height, 3);
  file.read(reinterpret_cast<char*>(image->data.data()), image->SizeInBytes());

  if (!file) {
    std::cerr << "[ImageIO] Incomplete binary data in PPM: " << filepath
              << std::endl;
    return false;
  }

  return true;
}

bool WritePPM(const std::string& filepath, const Image& image) {
  if (!image.IsValid() || image.channels != 3) {
    std::cerr << "[ImageIO] Invalid image for PPM write: channels="
              << image.channels << std::endl;
    return false;
  }

  std::ofstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "[ImageIO] Error opening output file: " << filepath << std::endl;
    return false;
  }

  file << "P6\n"
       << "# Created by Enterprise CUDA Image Processing Pipeline\n"
       << image.width << " " << image.height << "\n"
       << "255\n";

  file.write(reinterpret_cast<const char*>(image.data.data()),
             image.SizeInBytes());

  return file.good();
}

bool LoadImageFile(const std::string& filepath, Image* image) {
  std::filesystem::path p(filepath);
  std::string ext = p.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

  if (ext == ".pgm") {
    return ReadPGM(filepath, image);
  } else if (ext == ".ppm") {
    return ReadPPM(filepath, image);
  }

  // Attempt reading first 2 bytes to check magic
  std::ifstream probe(filepath, std::ios::binary);
  if (!probe.is_open()) return false;
  std::string magic;
  probe >> magic;
  probe.close();

  if (magic == "P5") {
    return ReadPGM(filepath, image);
  } else if (magic == "P6") {
    return ReadPPM(filepath, image);
  }

  std::cerr << "[ImageIO] Unrecognized format for file: " << filepath << std::endl;
  return false;
}

bool SaveImageFile(const std::string& filepath, const Image& image) {
  std::filesystem::path p(filepath);
  std::string ext = p.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

  if (ext == ".pgm" || image.channels == 1) {
    return WritePGM(filepath, image);
  } else {
    return WritePPM(filepath, image);
  }
}

std::vector<std::string> GetImageFilesInDirectory(const std::string& dir_path) {
  std::vector<std::string> files;
  if (!std::filesystem::exists(dir_path)) {
    return files;
  }

  for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
    if (entry.is_regular_file()) {
      std::string ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".pgm" || ext == ".ppm") {
        files.push_back(entry.path().string());
      }
    }
  }

  std::sort(files.begin(), files.end());
  return files;
}

}  // namespace cuda_scale
