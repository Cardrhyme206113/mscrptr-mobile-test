#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <jni.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

constexpr const char* kDomain = "dev.cardrhyme.kv";
constexpr const char* kOpName = "K8V8Attention";
constexpr int kMaxSupportedCache = 4096;

inline int8_t quantize_s8(float value, float inverse_scale) {
  const int rounded = static_cast<int>(std::nearbyint(value * inverse_scale));
  return static_cast<int8_t>(std::clamp(rounded, -127, 127));
}

inline float half_to_float(uint16_t bits) {
#if defined(__aarch64__)
  __fp16 value;
  std::memcpy(&value, &bits, sizeof(value));
  return static_cast<float>(value);
#else
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  uint32_t exponent = (bits >> 10) & 0x1fu;
  uint32_t mantissa = bits & 0x03ffu;
  uint32_t output;
  if (exponent == 0) {
    if (mantissa == 0) {
      output = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ffu;
      output = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    output = sign | 0x7f800000u | (mantissa << 13);
  } else {
    output = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  float value;
  std::memcpy(&value, &output, sizeof(value));
  return value;
#endif
}

inline int32_t dot_s8_64(const int8_t* a, const int8_t* b) {
#if defined(__ARM_FEATURE_DOTPROD)
  int32x4_t acc = vdupq_n_s32(0);
  acc = vdotq_s32(acc, vld1q_s8(a), vld1q_s8(b));
  acc = vdotq_s32(acc, vld1q_s8(a + 16), vld1q_s8(b + 16));
  acc = vdotq_s32(acc, vld1q_s8(a + 32), vld1q_s8(b + 32));
  acc = vdotq_s32(acc, vld1q_s8(a + 48), vld1q_s8(b + 48));
  return vaddvq_s32(acc);
#else
  int32_t result = 0;
  for (int index = 0; index < 64; ++index) {
    result += static_cast<int32_t>(a[index]) * static_cast<int32_t>(b[index]);
  }
  return result;
#endif
}

struct K8V8Kernel {
  explicit K8V8Kernel(const OrtApi& api, const OrtKernelInfo* info) : api_(api) {
    int64_t value = 0;
    OrtStatus* status = api_.KernelInfoGetAttribute_int64(info, "num_heads", &value);
    if (status != nullptr) {
      api_.ReleaseStatus(status);
      throw std::runtime_error("K8V8Attention is missing num_heads");
    }
    heads_ = static_cast<int>(value);
    status = api_.KernelInfoGetAttribute_int64(info, "head_size", &value);
    if (status != nullptr) {
      api_.ReleaseStatus(status);
      throw std::runtime_error("K8V8Attention is missing head_size");
    }
    head_size_ = static_cast<int>(value);
    if (heads_ <= 0 || head_size_ != 64) {
      throw std::runtime_error("K8V8Attention currently requires 64-dimensional heads");
    }
    scores_.resize(static_cast<size_t>(heads_) * kMaxSupportedCache);
  }

  OrtStatusPtr ComputeV2(OrtKernelContext* raw_context) noexcept {
    try {
      Compute(raw_context);
      return nullptr;
    } catch (const std::exception& error) {
      return api_.CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
    } catch (...) {
      return api_.CreateStatus(ORT_RUNTIME_EXCEPTION, "Unknown K8V8Attention error");
    }
  }

  void Compute(OrtKernelContext* raw_context) {
    Ort::KernelContext context(raw_context);
    const auto qkv_value = context.GetInput(0);
    const auto key_value = context.GetInput(1);
    const auto key_scale_value = context.GetInput(2);
    const auto value_value = context.GetInput(3);
    const auto value_scale_value = context.GetInput(4);
    const auto total_length_value = context.GetInput(6);

    const std::vector<int64_t> qkv_shape = qkv_value.GetTensorTypeAndShapeInfo().GetShape();
    const std::vector<int64_t> cache_shape = key_value.GetTensorTypeAndShapeInfo().GetShape();
    const std::vector<int64_t> scale_shape = key_scale_value.GetTensorTypeAndShapeInfo().GetShape();
    if (qkv_shape.size() != 3 || qkv_shape[0] != 1 || qkv_shape[1] != 1) {
      throw std::runtime_error("K8V8Attention is incremental-only and expects QKV shape [1,1,*]");
    }
    if (qkv_shape[2] != static_cast<int64_t>(heads_ * head_size_ * 3)) {
      throw std::runtime_error("Unexpected packed QKV width");
    }
    if (cache_shape.size() != 4 || cache_shape[0] != 1 || cache_shape[1] != heads_ ||
        cache_shape[3] != head_size_) {
      throw std::runtime_error("Unexpected K8V8 cache shape");
    }
    if (scale_shape.size() != 3 || scale_shape[0] != 1 || scale_shape[1] != heads_ ||
        scale_shape[2] != cache_shape[2]) {
      throw std::runtime_error("Unexpected K8V8 scale shape");
    }

    const int cache_length = static_cast<int>(cache_shape[2]);
    const int total_length = total_length_value.GetTensorData<int32_t>()[0];
    const int new_position = total_length - 1;
    if (total_length <= 0 || total_length > cache_length || total_length > kMaxSupportedCache) {
      throw std::runtime_error("K8V8 total sequence length is outside the cache");
    }

    const float* qkv = qkv_value.GetTensorData<float>();
    const int8_t* key_input = key_value.GetTensorData<int8_t>();
    const float* key_scale_input = key_scale_value.GetTensorData<float>();
    const int8_t* value_input = value_value.GetTensorData<int8_t>();
    const float* value_scale_input = value_scale_value.GetTensorData<float>();

    auto attention_output = context.GetOutput(0, {1, 1, static_cast<int64_t>(heads_ * head_size_)});
    auto key_output_value = context.GetOutput(1, cache_shape);
    auto key_scale_output_value = context.GetOutput(2, scale_shape);
    auto value_output_value = context.GetOutput(3, cache_shape);
    auto value_scale_output_value = context.GetOutput(4, scale_shape);

    float* attention = attention_output.GetTensorMutableData<float>();
    int8_t* key_cache = key_output_value.GetTensorMutableData<int8_t>();
    float* key_scales = key_scale_output_value.GetTensorMutableData<float>();
    int8_t* value_cache = value_output_value.GetTensorMutableData<int8_t>();
    float* value_scales = value_scale_output_value.GetTensorMutableData<float>();

    const size_t cache_elements = static_cast<size_t>(heads_) * cache_length * head_size_;
    const size_t scale_elements = static_cast<size_t>(heads_) * cache_length;
    if (key_cache != key_input) std::memcpy(key_cache, key_input, cache_elements);
    if (value_cache != value_input) std::memcpy(value_cache, value_input, cache_elements);
    if (key_scales != key_scale_input) {
      std::memcpy(key_scales, key_scale_input, scale_elements * sizeof(float));
    }
    if (value_scales != value_scale_input) {
      std::memcpy(value_scales, value_scale_input, scale_elements * sizeof(float));
    }

    Work work{
        this,
        qkv,
        key_cache,
        key_scales,
        value_cache,
        value_scales,
        attention,
        cache_length,
        total_length,
        new_position,
    };
    context.ParallelFor(&K8V8Kernel::RunHead, static_cast<size_t>(heads_), 4, &work);
  }

 private:
  struct Work {
    K8V8Kernel* kernel;
    const float* qkv;
    int8_t* key_cache;
    float* key_scales;
    int8_t* value_cache;
    float* value_scales;
    float* attention;
    int cache_length;
    int total_length;
    int new_position;
  };

  static void RunHead(void* opaque, size_t head_index) {
    static_cast<Work*>(opaque)->kernel->ComputeHead(*static_cast<Work*>(opaque),
                                                    static_cast<int>(head_index));
  }

  void ComputeHead(const Work& work, int head) {
    const int width = heads_ * head_size_;
    const float* query = work.qkv + head * head_size_;
    const float* new_key = work.qkv + width + head * head_size_;
    const float* new_value = work.qkv + width * 2 + head * head_size_;

    const size_t cache_head_base = static_cast<size_t>(head) * work.cache_length * head_size_;
    const size_t scale_head_base = static_cast<size_t>(head) * work.cache_length;
    int8_t* key_slot = work.key_cache + cache_head_base +
                       static_cast<size_t>(work.new_position) * head_size_;
    int8_t* value_slot = work.value_cache + cache_head_base +
                         static_cast<size_t>(work.new_position) * head_size_;

    float key_max = 0.0f;
    float value_max = 0.0f;
    float query_max = 0.0f;
    for (int dim = 0; dim < head_size_; ++dim) {
      query_max = std::max(query_max, std::abs(query[dim]));
      key_max = std::max(key_max, std::abs(new_key[dim]));
      value_max = std::max(value_max, std::abs(new_value[dim]));
    }
    const float query_scale = std::max(query_max / 127.0f, 1.0e-8f);
    const float key_scale = std::max(key_max / 127.0f, 1.0e-8f);
    const float value_scale = std::max(value_max / 127.0f, 1.0e-8f);
    const float query_inverse = 1.0f / query_scale;
    const float key_inverse = 1.0f / key_scale;
    const float value_inverse = 1.0f / value_scale;

    alignas(16) int8_t query_q[64];
    for (int dim = 0; dim < head_size_; ++dim) {
      query_q[dim] = quantize_s8(query[dim], query_inverse);
      key_slot[dim] = quantize_s8(new_key[dim], key_inverse);
      value_slot[dim] = quantize_s8(new_value[dim], value_inverse);
    }
    work.key_scales[scale_head_base + work.new_position] = key_scale;
    work.value_scales[scale_head_base + work.new_position] = value_scale;

    float* scores = scores_.data() + static_cast<size_t>(head) * kMaxSupportedCache;
    float maximum = -std::numeric_limits<float>::infinity();
    constexpr float attention_scale = 0.125f;  // 1 / sqrt(64)
    for (int position = 0; position < work.total_length; ++position) {
      const int8_t* key = work.key_cache + cache_head_base +
                          static_cast<size_t>(position) * head_size_;
      const int32_t dot = dot_s8_64(query_q, key);
      const float score = static_cast<float>(dot) * query_scale *
                          work.key_scales[scale_head_base + position] * attention_scale;
      scores[position] = score;
      maximum = std::max(maximum, score);
    }

    float denominator = 0.0f;
    for (int position = 0; position < work.total_length; ++position) {
      const float probability = std::exp(scores[position] - maximum);
      scores[position] = probability;
      denominator += probability;
    }
    const float inverse_denominator = 1.0f / std::max(denominator, 1.0e-20f);

    float* output = work.attention + head * head_size_;
    std::fill(output, output + head_size_, 0.0f);
    for (int position = 0; position < work.total_length; ++position) {
      const float factor = scores[position] * inverse_denominator *
                           work.value_scales[scale_head_base + position];
      const int8_t* value = work.value_cache + cache_head_base +
                            static_cast<size_t>(position) * head_size_;
#if defined(__clang__)
#pragma clang loop vectorize(enable)
#endif
      for (int dim = 0; dim < head_size_; ++dim) {
        output[dim] += factor * static_cast<float>(value[dim]);
      }
    }
  }

  const OrtApi& api_;
  int heads_ = 0;
  int head_size_ = 0;
  std::vector<float> scores_;
};

struct K8V8CustomOp : Ort::CustomOpBase<K8V8CustomOp, K8V8Kernel, true> {
  K8V8CustomOp() {
    OrtCustomOp::GetMayInplace = [](int** input_indices, int** output_indices) -> size_t {
      *input_indices = new int[4]{1, 2, 3, 4};
      *output_indices = new int[4]{1, 2, 3, 4};
      return 4;
    };
    OrtCustomOp::ReleaseMayInplace = [](int* input_indices, int* output_indices) {
      delete[] input_indices;
      delete[] output_indices;
    };
  }

  const char* GetName() const { return kOpName; }
  size_t GetInputTypeCount() const { return 7; }
  size_t GetOutputTypeCount() const { return 5; }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    static constexpr ONNXTensorElementDataType types[] = {
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
    };
    return types[index];
  }

  ONNXTensorElementDataType GetOutputType(size_t index) const {
    static constexpr ONNXTensorElementDataType types[] = {
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    };
    return types[index];
  }

  OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info,
                              void** kernel) const noexcept {
    try {
      *kernel = new K8V8Kernel(api, info);
      return nullptr;
    } catch (const std::exception& error) {
      return api.CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
    } catch (...) {
      return api.CreateStatus(ORT_RUNTIME_EXCEPTION, "Unknown K8V8 kernel creation error");
    }
  }
};

K8V8CustomOp g_k8v8_op;
std::mutex g_domain_mutex;
std::vector<Ort::CustomOpDomain> g_domains;

}  // namespace

extern "C" ORT_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* api_base) {
  const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
  Ort::InitApi(api);
  try {
    Ort::CustomOpDomain domain(kDomain);
    domain.Add(&g_k8v8_op);
    Ort::UnownedSessionOptions(options).Add(domain);
    std::lock_guard<std::mutex> lock(g_domain_mutex);
    g_domains.push_back(std::move(domain));
    return nullptr;
  } catch (const std::exception& error) {
    return api->CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
  } catch (...) {
    return api->CreateStatus(ORT_RUNTIME_EXCEPTION, "Unknown custom-op registration error");
  }
}

extern "C" JNIEXPORT void JNICALL
Java_dev_cardrhyme_muscriptormobile_K8V8Native_quantizeFp16Caches(
    JNIEnv* env, jclass, jobjectArray source_buffers, jobjectArray destination_buffers,
    jobjectArray scale_buffers, jint source_length, jint destination_length, jint positions,
    jint heads, jint head_size) {
  const jsize tensor_count = env->GetArrayLength(source_buffers);
  if (tensor_count != env->GetArrayLength(destination_buffers) ||
      tensor_count != env->GetArrayLength(scale_buffers) || source_length <= 0 ||
      destination_length < positions || positions <= 0 || heads <= 0 || head_size != 64) {
    jclass exception = env->FindClass("java/lang/IllegalArgumentException");
    env->ThrowNew(exception, "Invalid K8V8 cache quantization buffers or dimensions");
    return;
  }

  struct TensorPointers {
    const uint16_t* source;
    int8_t* destination;
    float* scales;
  };
  std::vector<TensorPointers> tensors(static_cast<size_t>(tensor_count));
  for (jsize index = 0; index < tensor_count; ++index) {
    jobject source = env->GetObjectArrayElement(source_buffers, index);
    jobject destination = env->GetObjectArrayElement(destination_buffers, index);
    jobject scales = env->GetObjectArrayElement(scale_buffers, index);
    tensors[static_cast<size_t>(index)] = {
        static_cast<const uint16_t*>(env->GetDirectBufferAddress(source)),
        static_cast<int8_t*>(env->GetDirectBufferAddress(destination)),
        static_cast<float*>(env->GetDirectBufferAddress(scales)),
    };
    env->DeleteLocalRef(source);
    env->DeleteLocalRef(destination);
    env->DeleteLocalRef(scales);
    if (tensors[static_cast<size_t>(index)].source == nullptr ||
        tensors[static_cast<size_t>(index)].destination == nullptr ||
        tensors[static_cast<size_t>(index)].scales == nullptr) {
      jclass exception = env->FindClass("java/lang/IllegalArgumentException");
      env->ThrowNew(exception, "K8V8 quantization requires direct ByteBuffers");
      return;
    }
  }

  const size_t total_tasks = static_cast<size_t>(tensor_count) * heads;
  std::atomic<size_t> next_task{0};
  const unsigned detected = std::max(1u, std::thread::hardware_concurrency());
  const unsigned worker_count = std::min<unsigned>(8u, std::min<unsigned>(detected, total_tasks));
  std::vector<std::thread> workers;
  workers.reserve(worker_count);

  auto worker = [&]() {
    while (true) {
      const size_t task = next_task.fetch_add(1, std::memory_order_relaxed);
      if (task >= total_tasks) break;
      const size_t tensor_index = task / static_cast<size_t>(heads);
      const int head = static_cast<int>(task % static_cast<size_t>(heads));
      const TensorPointers& tensor = tensors[tensor_index];
      const size_t source_head_base = static_cast<size_t>(head) * source_length * head_size;
      const size_t destination_head_base = static_cast<size_t>(head) * destination_length * head_size;
      const size_t scale_head_base = static_cast<size_t>(head) * destination_length;

      for (int position = 0; position < positions; ++position) {
        const uint16_t* source = tensor.source + source_head_base +
                                 static_cast<size_t>(position) * head_size;
        int8_t* destination = tensor.destination + destination_head_base +
                              static_cast<size_t>(position) * head_size;
        float maximum = 0.0f;
        for (int dim = 0; dim < head_size; ++dim) {
          maximum = std::max(maximum, std::abs(half_to_float(source[dim])));
        }
        const float scale = std::max(maximum / 127.0f, 1.0e-8f);
        const float inverse = 1.0f / scale;
        tensor.scales[scale_head_base + position] = scale;
        for (int dim = 0; dim < head_size; ++dim) {
          destination[dim] = quantize_s8(half_to_float(source[dim]), inverse);
        }
      }
    }
  };

  for (unsigned index = 0; index < worker_count; ++index) workers.emplace_back(worker);
  for (std::thread& thread : workers) thread.join();
}
