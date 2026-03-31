#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "src/model.h"
#include "src/tensor_ggml.hpp"
#include "src/unet.hpp"

namespace {

constexpr const char* kUNetPrefix          = "model.diffusion_model";
constexpr int kImageSize                   = 256;
constexpr int kLatentDownsampleFactor      = 8;
constexpr int kLatentChannels              = 4;
constexpr int kContextTokens               = 77;
constexpr int kContextDim                  = 768;
constexpr int kBatchSize                   = 1;
constexpr int kWarmupSteps                 = 1;
constexpr int kMeasuredSteps               = 2;
constexpr float kWeightInitScale           = 0.02f;
constexpr float kInputInitScale            = 1.0f;
constexpr unsigned int kRandomSeed         = 1337U;

bool ends_with(const std::string& value, const char* suffix) {
    const size_t suffix_len = std::char_traits<char>::length(suffix);
    return value.size() >= suffix_len &&
           value.compare(value.size() - suffix_len, suffix_len, suffix) == 0;
}

template <typename TFloat>
std::vector<TFloat> make_random_values(int64_t count,
                                       std::mt19937& rng,
                                       float min_value,
                                       float max_value);

template <>
std::vector<float> make_random_values<float>(int64_t count,
                                             std::mt19937& rng,
                                             float min_value,
                                             float max_value) {
    std::uniform_real_distribution<float> dist(min_value, max_value);
    std::vector<float> values(static_cast<size_t>(count));
    for (float& value : values) {
        value = dist(rng);
    }
    return values;
}

template <>
std::vector<ggml_fp16_t> make_random_values<ggml_fp16_t>(int64_t count,
                                                         std::mt19937& rng,
                                                         float min_value,
                                                         float max_value) {
    std::uniform_real_distribution<float> dist(min_value, max_value);
    std::vector<ggml_fp16_t> values(static_cast<size_t>(count));
    for (ggml_fp16_t& value : values) {
        value = ggml_fp32_to_fp16(dist(rng));
    }
    return values;
}

String2TensorStorage make_fp16_weight_storage_map() {
    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
    params.mem_buffer       = nullptr;
    params.no_alloc         = true;

    ggml_context* ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    UnetModelBlock unet(VERSION_SD1);
    unet.init(ctx, {}, kUNetPrefix);

    std::map<std::string, ggml_tensor*> tensors;
    unet.get_param_tensors(tensors, kUNetPrefix);

    String2TensorStorage tensor_storage_map;
    for (const auto& entry : tensors) {
        const std::string& name = entry.first;
        const ggml_tensor* tensor = entry.second;
        if (ends_with(name, "weight") && ggml_n_dims(tensor) >= 2) {
            TensorStorage tensor_storage;
            tensor_storage.name          = name;
            tensor_storage.expected_type = GGML_TYPE_F16;
            tensor_storage_map[name]     = tensor_storage;
        }
    }

    ggml_free(ctx);
    return tensor_storage_map;
}

void fill_backend_tensor_random(ggml_tensor* tensor,
                                std::mt19937& rng,
                                float min_value,
                                float max_value) {
    const int64_t numel = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F16) {
        auto values = make_random_values<ggml_fp16_t>(numel, rng, min_value, max_value);
        ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
        return;
    }
    if (tensor->type == GGML_TYPE_F32) {
        auto values = make_random_values<float>(numel, rng, min_value, max_value);
        ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
        return;
    }

    std::fprintf(stderr, "Unsupported parameter tensor type: %s\n", ggml_type_name(tensor->type));
    std::exit(EXIT_FAILURE);
}

double checksum_tensor(const ggml_tensor* tensor) {
    const int64_t numel = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> values(static_cast<size_t>(numel));
        ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
        return std::accumulate(values.begin(), values.end(), 0.0, [](double acc, ggml_fp16_t value) {
            return acc + ggml_fp16_to_fp32(value);
        });
    }
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> values(static_cast<size_t>(numel));
        ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
        return std::accumulate(values.begin(), values.end(), 0.0);
    }

    std::fprintf(stderr, "Unsupported output tensor type: %s\n", ggml_type_name(tensor->type));
    std::exit(EXIT_FAILURE);
}

class FP16UNetPerfRunner : public GGMLRunner {
public:
    explicit FP16UNetPerfRunner(ggml_backend_t backend, const String2TensorStorage& tensor_storage_map)
        : GGMLRunner(backend, false), unet(VERSION_SD1, tensor_storage_map) {
        unet.init(params_ctx, tensor_storage_map, kUNetPrefix);
    }

    std::string get_desc() override {
        return "unet-perf";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix = kUNetPrefix) {
        unet.get_param_tensors(tensors, prefix);
    }

    ggml_cgraph* build_graph(const sd::Tensor<ggml_fp16_t>& x,
                             const sd::Tensor<float>& timesteps,
                             const sd::Tensor<ggml_fp16_t>& context) {
        ggml_cgraph* gf = new_graph_custom(UNET_GRAPH_SIZE);

        ggml_tensor* x_tensor         = make_input(x);
        ggml_tensor* timesteps_tensor = make_input(timesteps);
        ggml_tensor* context_tensor   = make_input(context);

        auto runner_ctx = get_context();
        ggml_tensor* out = unet.forward(&runner_ctx,
                                        x_tensor,
                                        timesteps_tensor,
                                        context_tensor,
                                        nullptr,
                                        nullptr,
                                        static_cast<int>(x_tensor->ne[3]));

        ggml_build_forward_expand(gf, out);
        return gf;
    }

    const ggml_tensor* run(int n_threads,
                           const sd::Tensor<ggml_fp16_t>& x,
                           const sd::Tensor<float>& timesteps,
                           const sd::Tensor<ggml_fp16_t>& context) {
        auto get_graph = [&]() {
            return build_graph(x, timesteps, context);
        };

        auto result = GGMLRunner::compute<float>(get_graph, n_threads, false, true);
        GGML_ASSERT(result.has_value() == false);

        const ggml_tensor* output = ggml_get_tensor(compute_ctx, final_result_name.c_str());
        GGML_ASSERT(output != nullptr);
        return output;
    }

private:
    UnetModelBlock unet;
};

}  // namespace

int main() {
    const int latent_size = kImageSize / kLatentDownsampleFactor;
    const int n_threads = std::max(1u, std::thread::hardware_concurrency());

    auto tensor_storage_map = make_fp16_weight_storage_map();

    ggml_backend_t backend = ggml_backend_cpu_init();
    GGML_ASSERT(backend != nullptr);

    {
        FP16UNetPerfRunner runner(backend, tensor_storage_map);
        runner.set_flash_attention_enabled(true);

        if (!runner.alloc_params_buffer()) {
            std::fprintf(stderr, "Failed to allocate UNet parameter buffer.\n");
            ggml_backend_free(backend);
            return EXIT_FAILURE;
        }

        std::map<std::string, ggml_tensor*> param_tensors;
        runner.get_param_tensors(param_tensors);

        std::mt19937 weights_rng(kRandomSeed);
        for (const auto& entry : param_tensors) {
            fill_backend_tensor_random(entry.second, weights_rng, -kWeightInitScale, kWeightInitScale);
        }

        sd::Tensor<ggml_fp16_t> latent({latent_size, latent_size, kLatentChannels, kBatchSize});
        sd::Tensor<ggml_fp16_t> context({kContextDim, kContextTokens, kBatchSize});
        sd::Tensor<float> timesteps({kBatchSize});

        std::mt19937 input_rng(kRandomSeed + 1U);
        latent.values()  = make_random_values<ggml_fp16_t>(latent.numel(), input_rng, -kInputInitScale, kInputInitScale);
        context.values() = make_random_values<ggml_fp16_t>(context.numel(), input_rng, -kInputInitScale, kInputInitScale);
        timesteps.values().assign(static_cast<size_t>(kBatchSize), 999.0f);

        std::printf("Stable Diffusion 1.5 UNet perf test\n");
        std::printf("image_size=%dx%d latent_shape=[%d,%d,%d,%d] context_shape=[%d,%d,%d] threads=%d params=%zu params_buffer_mb=%.2f\n",
                    kImageSize,
                    kImageSize,
                    latent_size,
                    latent_size,
                    kLatentChannels,
                    kBatchSize,
                    kContextDim,
                    kContextTokens,
                    kBatchSize,
                    n_threads,
                    param_tensors.size(),
                    runner.get_params_buffer_size() / (1024.0 * 1024.0));

        for (int step = 0; step < kWarmupSteps; ++step) {
            const int64_t t0 = ggml_time_us();
            const ggml_tensor* output = runner.run(n_threads, latent, timesteps, context);
            const int64_t t1 = ggml_time_us();
            std::printf("warmup_step_%d duration_ms=%.3f output_type=%s output_shape=[%lld,%lld,%lld,%lld] checksum=%.6f\n",
                        step + 1,
                        (t1 - t0) / 1000.0,
                        ggml_type_name(output->type),
                        output->ne[0],
                        output->ne[1],
                        output->ne[2],
                        output->ne[3],
                        checksum_tensor(output));
        }

        std::vector<double> measured_ms;
        measured_ms.reserve(kMeasuredSteps);
        for (int step = 0; step < kMeasuredSteps; ++step) {
            const int64_t t0 = ggml_time_us();
            const ggml_tensor* output = runner.run(n_threads, latent, timesteps, context);
            const int64_t t1 = ggml_time_us();
            const double elapsed_ms = (t1 - t0) / 1000.0;
            measured_ms.push_back(elapsed_ms);

            std::printf("measured_step_%d duration_ms=%.3f output_type=%s output_shape=[%lld,%lld,%lld,%lld] checksum=%.6f\n",
                        step + 1,
                        elapsed_ms,
                        ggml_type_name(output->type),
                        output->ne[0],
                        output->ne[1],
                        output->ne[2],
                        output->ne[3],
                        checksum_tensor(output));
        }

        const double avg_ms = std::accumulate(measured_ms.begin(), measured_ms.end(), 0.0) / measured_ms.size();
        std::printf("measured_steps=%d average_duration_ms=%.3f\n", kMeasuredSteps, avg_ms);
    }

    ggml_backend_free(backend);
    return EXIT_SUCCESS;
}
