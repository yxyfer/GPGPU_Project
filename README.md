# GPGPU - Simple Object Detection

![Alt Text](https://github.com/othmamo/GPGPU_Project/blob/main/images/output_gifs/penguin_fast_detection.gif)

## Authors
  - Moustapha Diop
  - Mathieu Rivier
  - Othman Elbaz
  - Lucas Pinot
  
## Running the project:

### Running a video (in images format):
To run a a video file (that is given in a folder) as images use:

- your images have to be named `thumb{num}.jpg` (or change the name in the command bellow)

```sh
make build
make gpu_run path=../images/{your folder}/thumb*.jpg
make clean # to clean and delete the build folder
```

### Running the program once (with our test images):

```sh
make build
make gpu_test
make clean # to clean and delete the build folder
```

### Running the program once (with our test images) (CPU):

```sh
make build
make cpu_test
make clean # to clean and delete the build folder
```

### Running the benchmarks (CPU):

```sh
make build
make bench_cpu
make clean # to clean and delete the build folder
```

### Running the benchmarks (GPU):

```sh
make build
make bench_gpu
make clean # to clean and delete the build folder
```

## Project Architecture

### Basics:

The code is located in the `src` folder while the header files are in the `includes folder`.

The `paper.pdf` is the report that represents the final output and results of this project.

```sh
.
├── src
├── includes
├── images
├── visu
├── Makefile
└── paper.pdf

```

### Code Architecture
```sh
src
├── CMakeLists.txt
├── bench
│   ├── bench_cpu.cpp
│   └── bench_gpu.cpp
├── cpu
│   ├── CMakeLists.txt
│   ├── bbox_cpu.cpp
│   ├── blur_cpu.cpp
│   ├── detect_obj_cpu.cpp
│   ├── main_cpu.cpp
│   ├── opening.cpp
│   └── threshold
│       ├── connexe_components_cpu.cpp
│       ├── otsu_cpu.cpp
│       └── threshold_cpu.cpp
├── gpu
│   ├── CMakeLists.txt
│   ├── bbox_gpu.cu
│   ├── blurr_gpu.cu
│   ├── detect_obj_gpu.cu
│   ├── difference_gpu.cu
│   ├── gray_scale_gpu.cu
│   ├── helpers_gpu.cu
│   ├── main_gpu.cpp
│   ├── opening_closing_gpu.cu
│   ├── threshold
│   │   ├── connexe_components_gpu.cu
│   │   ├── otsu_gpu.cu
│   │   └── threshold_gpu.cu
│   └── utils_gpu.cu
├── helpers_images.cpp
└── struct_utils.cpp
```

# Different Outputs of the Program:


![Alt Text](https://github.com/othmamo/GPGPU_Project/blob/main/images/output_gifs/fast_detection.gif)
