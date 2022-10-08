#include <cstddef>
#include <memory>
#include <iostream>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include "detect.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

/* #define STB_IMAGE_WRITE_IMPLEMENTATION */
/* #include "stb_image_write.h" */

/* stbi_write_png("sky.png", width, height, channels, img, width * channels); */
/* stbi_write_jpg("sky2.jpg", width, height, channels, img, 100); */
/* stbi_image_free(img); */


// Usage: ./mandel
int main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    std::string mode = "CPU";

    if (argc < 3) {
        std::cout << "Minimum two images are needed\n";
        return 1;
    }

    int width, height, channels;
    unsigned char *ref_image = stbi_load(argv[1], &width, &height, &channels, 0);
    unsigned char *obj_image = stbi_load(argv[1], &width, &height, &channels, 0);

    std::cout << "Reference image: " << argv[1] << " | " <<  height << "x" << width << "x" << channels << "\n";
    std::cout << "Object image: " << argv[1] << " | " <<  height << "x" << width << "x" << channels << "\n";

    stbi_image_free(ref_image);
    stbi_image_free(obj_image);

  // Create buffer
  /* constexpr int kRGBASize = 4; */
  /* int stride = width * kRGBASize; */
  /* auto buffer = std::make_unique<std::byte[]>(height * stride); */

  // Rendering
  /* spdlog::info("Runnging {} mode with (w={},h={},niter={}).", mode, width, height, niter); */
  /* spdlog::info("Runnging {} mode with.", mode); */
  /* if (mode == "CPU") */
  /* { */
  /*   detect_cpu(reinterpret_cast<char*>(buffer.get()), */
  /*           reinterpret_cast<char*>(buffer.get()), */
  /*           width, height, stride); */
  /* } */
  /* else if (mode == "GPU") */
  /* { */
  /*   detect_gpu(reinterpret_cast<char*>(buffer.get()), reinterpret_cast<char*>(buffer.get()), width, height, stride); */
  /* } */

  // Save
  /* write_png(buffer.get(), width, height, stride, filename.c_str()); */
  /* spdlog::info("Output saved in {}.", filename); */
}

