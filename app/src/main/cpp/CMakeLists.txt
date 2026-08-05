cmake_minimum_required(VERSION 3.22.1)
project(muscriptor_k8v8 LANGUAGES CXX)

set(ORT_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/ort_headers")
foreach(required_header
    onnxruntime_c_api.h
    onnxruntime_ep_c_api.h
    onnxruntime_cxx_api.h
    onnxruntime_cxx_inline.h
    onnxruntime_float16.h)
  if(NOT EXISTS "${ORT_HEADERS}/${required_header}")
    message(FATAL_ERROR
      "Missing ${required_header}. Run tools/fetch_ort_headers.sh before Gradle.")
  endif()
endforeach()

add_library(muscriptor_k8v8 SHARED k8v8_attention.cc)
target_include_directories(muscriptor_k8v8 PRIVATE "${ORT_HEADERS}")
target_compile_features(muscriptor_k8v8 PRIVATE cxx_std_20)
target_compile_options(muscriptor_k8v8 PRIVATE
  -O3
  -fno-math-errno
  -Wall
  -Wextra
  -Wno-unused-parameter)

find_library(log_lib log)
target_link_libraries(muscriptor_k8v8 PRIVATE ${log_lib})
