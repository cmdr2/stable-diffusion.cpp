#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <linux/memfd.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "ggml-cpu.h"
#include "ggml.h"
#include "stable-diffusion.h"

#include "src/auto_encoder_kl.hpp"
#include "src/conditioner.hpp"
#include "src/diffusion_model.hpp"

namespace {

struct TensorSpec {
    std::string name;
    std::vector<int64_t> shape;
    uint64_t element_count = 0;
    uint64_t data_offset   = 0;
};

struct TempFile {
    std::string path;

    explicit TempFile(const char* name_template) {
        std::vector<char> path_buf(std::strlen(name_template) + 1);
        std::memcpy(path_buf.data(), name_template, path_buf.size());

        const int fd = ::mkstemp(path_buf.data());
        if (fd < 0) {
            return;
        }

        ::close(fd);
        path = path_buf.data();
    }

    ~TempFile() {
        if (!path.empty()) {
            ::unlink(path.c_str());
        }
    }
};

int fail(const char* message) {
    std::fprintf(stderr, "%s\n", message);
    return 1;
}

bool write_all(int fd, const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    size_t written    = 0;
    while (written < size) {
        const ssize_t rc = ::write(fd, bytes + written, size - written);
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        written += static_cast<size_t>(rc);
    }
    return true;
}

bool write_u64_le(int fd, uint64_t value) {
    uint8_t bytes[8];
    for (int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<uint8_t>((value >> (8 * i)) & 0xffU);
    }
    return write_all(fd, bytes, sizeof(bytes));
}

std::vector<int64_t> to_safetensors_shape(const ggml_tensor* tensor) {
    std::vector<int64_t> shape;
    int dims = ggml_n_dims(tensor);
    if (dims <= 0) {
        dims = 1;
    }
    for (int i = dims - 1; i >= 0; --i) {
        shape.push_back(tensor->ne[i]);
    }
    return shape;
}

std::vector<TensorSpec> collect_sd_1_5_tensor_specs() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        return {};
    }

    String2TensorStorage empty_storage;
    std::map<std::string, std::string> empty_embeddings;
    std::map<std::string, ggml_tensor*> tensors;
    std::vector<TensorSpec> specs;

    {
        FrozenCLIPEmbedderWithCustomWords conditioner(backend,
                                                      false,
                                                      empty_storage,
                                                      empty_embeddings,
                                                      VERSION_SD1);
        UNetModel diffusion_model(backend, false, empty_storage, VERSION_SD1);
        AutoEncoderKL vae(backend,
                          false,
                          empty_storage,
                          "first_stage_model",
                          true,
                          false,
                          VERSION_SD1);

        conditioner.get_param_tensors(tensors);
        diffusion_model.get_param_tensors(tensors);
        vae.get_param_tensors(tensors, "first_stage_model");

        specs.reserve(tensors.size());

        uint64_t current_offset = 0;
        for (const auto& [name, tensor] : tensors) {
            TensorSpec spec;
            spec.name          = name;
            spec.shape         = to_safetensors_shape(tensor);
            spec.element_count = static_cast<uint64_t>(ggml_nelements(tensor));
            spec.data_offset   = current_offset;
            current_offset += spec.element_count * sizeof(ggml_fp16_t);
            specs.push_back(std::move(spec));
        }
    }

    ggml_backend_free(backend);

    return specs;
}

std::string build_safetensors_header(const std::vector<TensorSpec>& specs) {
    std::ostringstream header;
    header << '{';
    for (size_t i = 0; i < specs.size(); ++i) {
        const auto& spec = specs[i];
        if (i > 0) {
            header << ',';
        }
        header << '"' << spec.name << "\":{\"dtype\":\"F16\",\"shape\":[";
        for (size_t dim = 0; dim < spec.shape.size(); ++dim) {
            if (dim > 0) {
                header << ',';
            }
            header << spec.shape[dim];
        }
        header << "],\"data_offsets\":[" << spec.data_offset << ','
               << (spec.data_offset + spec.element_count * sizeof(ggml_fp16_t)) << "]}";
    }
    header << '}';
    return header.str();
}

int create_memfd(const char* name) {
    const long fd = ::syscall(SYS_memfd_create, name, MFD_CLOEXEC);
    return static_cast<int>(fd);
}

bool write_dummy_weights_file(int fd, const std::vector<TensorSpec>& specs) {
    const std::string header = build_safetensors_header(specs);
    if (!write_u64_le(fd, static_cast<uint64_t>(header.size()))) {
        return false;
    }
    if (!write_all(fd, header.data(), header.size())) {
        return false;
    }

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
    std::vector<ggml_fp16_t> chunk(1U << 16);

    for (const auto& spec : specs) {
        uint64_t remaining = spec.element_count;
        while (remaining > 0) {
            const size_t n = static_cast<size_t>(std::min<uint64_t>(remaining, chunk.size()));
            for (size_t i = 0; i < n; ++i) {
                chunk[i] = ggml_fp32_to_fp16(dist(rng));
            }
            if (!write_all(fd, chunk.data(), n * sizeof(chunk[0]))) {
                return false;
            }
            remaining -= n;
        }
    }

    return ::lseek(fd, 0, SEEK_SET) >= 0;
}

bool run_sd_1_5_smoke_test(const std::string& model_path) {
    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path            = model_path.c_str();
    ctx_params.n_threads             = 1;
    ctx_params.wtype                 = SD_TYPE_COUNT;
    ctx_params.rng_type              = CPU_RNG;
    ctx_params.sampler_rng_type      = CPU_RNG;
    ctx_params.enable_mmap           = false;
    ctx_params.offload_params_to_cpu = false;

    sd_ctx_t* ctx = new_sd_ctx(&ctx_params);
    if (ctx == nullptr) {
        return false;
    }

    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);
    gen_params.prompt                       = "test";
    gen_params.width                        = 64;
    gen_params.height                       = 64;
    gen_params.seed                         = 1234;
    gen_params.sample_params.sample_steps   = 1;
    gen_params.sample_params.sample_method  = EULER_A_SAMPLE_METHOD;
    gen_params.sample_params.scheduler      = DISCRETE_SCHEDULER;
    gen_params.sample_params.guidance.txt_cfg = 1.0f;
    gen_params.sample_params.guidance.img_cfg = 1.0f;

    sd_image_t* images = generate_image(ctx, &gen_params);

    free_sd_ctx(ctx);

    if (images == nullptr) {
        return false;
    }

    const bool valid_shape = images[0].data != nullptr &&
                             images[0].width == 64 &&
                             images[0].height == 64 &&
                             images[0].channel == 3;

    free(images[0].data);
    free(images);

    return valid_shape;
}

}  // namespace

int main() {
    const std::vector<TensorSpec> specs = collect_sd_1_5_tensor_specs();
    if (specs.empty()) {
        return fail("failed to collect SD 1.5 tensor specs");
    }

    const int fd = create_memfd("sd-1-5-dummy-model");
    if (fd < 0) {
        return fail("failed to create in-memory model file");
    }

    if (!write_dummy_weights_file(fd, specs)) {
        ::close(fd);
        return fail("failed to write dummy weights");
    }

    const std::string model_path = "/proc/self/fd/" + std::to_string(fd);

    constexpr std::array<sd_type_t, 3> kOutputTypes = {
        SD_TYPE_F16,
        SD_TYPE_Q3_K,
        SD_TYPE_Q5_K,
    };

    for (sd_type_t output_type : kOutputTypes) {
        TempFile gguf_file("/tmp/test-sd-1-5-XXXXXX");
        if (gguf_file.path.empty()) {
            ::close(fd);
            return fail("failed to create temporary GGUF file");
        }

        if (!convert(model_path.c_str(), nullptr, gguf_file.path.c_str(), output_type, nullptr, false)) {
            ::close(fd);
            std::fprintf(stderr, "failed to convert dummy model to %s\n", sd_type_name(output_type));
            return 1;
        }

        if (!run_sd_1_5_smoke_test(gguf_file.path)) {
            ::close(fd);
            std::fprintf(stderr, "image generation failed for %s model\n", sd_type_name(output_type));
            return 1;
        }
    }

    ::close(fd);

    return 0;
}
