#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-aarch64.h"
#include "ggml-impl.h"
#include <cctype>
#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <sys/mman.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <memory>
#include <algorithm>


#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>
#endif

// ggml-backend interface

#ifdef GGML_USE_CPU_HBM

// buffer type HBM

#include <hbwmalloc.h>

static const char * ggml_backend_cpu_hbm_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_HBM";

    GGML_UNUSED(buft);
}

static void ggml_backend_cpu_hbm_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    hbw_free(buffer->context);
}

static ggml_backend_buffer_t ggml_backend_cpu_hbm_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr;
    int result = hbw_posix_memalign(&ptr, ggml_backend_cpu_buffer_type_get_alignment(buft), size);
    if (result != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        return NULL;
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cpu_hbm_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cpu_hbm_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_hbm = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cpu_hbm_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_hbm_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .context  = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type_hbm;
}
#endif

// buffer type AARCH64

static void ggml_backend_cpu_aarch64_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *)ggml_aarch64_get_optimal_repack_type(tensor); // NOLINT

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_aarch64_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    enum ggml_type repack_type = (enum ggml_type)(intptr_t)tensor->extra;

    ggml_aarch64_repack_tensor(tensor, repack_type, data, size);

    GGML_UNUSED(buffer);
}

static const char * ggml_backend_cpu_aarch64_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_AARCH64";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_aarch64_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    if (buffer == NULL) {
        return NULL;
    }

    buffer->buft = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_aarch64_buffer_init_tensor;
    buffer->iface.set_tensor = ggml_backend_cpu_aarch64_buffer_set_tensor;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cpu_aarch64_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_aarch64 = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cpu_aarch64_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_aarch64_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ NULL,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type_aarch64;
}

bool ggml_backend_cpu_buft_is_aarch64(ggml_backend_buffer_type_t buft) {
    return buft == ggml_backend_cpu_aarch64_buffer_type();
}

static ggml_backend_buffer_type_t * ggml_backend_cpu_get_extra_bufts(ggml_backend_dev_t device) {
    static std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#ifdef GGML_USE_CPU_HBM
        bufts.push_back(ggml_backend_cpu_hbm_buffer_type());
#endif

#ifdef GGML_USE_CPU_AARCH64
        bufts.push_back(ggml_backend_cpu_aarch64_buffer_type());
#endif

        bufts.push_back(NULL);

        return bufts;
    }();

    return bufts.data();

    GGML_UNUSED(device);
}

// CPU backend - backend (stream)

struct ggml_backend_cpu_context {
    int                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    void *              abort_callback_data;
};

static const char * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    return "CPU";

    GGML_UNUSED(backend);
}

static void ggml_backend_cpu_free(ggml_backend_t backend) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

struct ggml_backend_plan_cpu {
    struct ggml_cplan cplan;
    struct ggml_cgraph cgraph;
};

static ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_backend_plan_cpu * cpu_plan = new ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = new uint8_t[cpu_plan->cplan.work_size];
        if (cpu_plan->cplan.work_data == NULL) {
            delete cpu_plan;
            return NULL;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;

    return cpu_plan;
}

static void ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    return ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

struct ggml_compute_params {
    // ith = thread index, nth = number of threads
    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;

    struct ggml_threadpool * threadpool;
};

struct layer_prefetch_status {
    int layer_index;
    bool is_dynamic_layer;
    std::atomic<bool> prefetched;   // 用原子变量记录此层是否已预取完成
    std::vector<std::tuple<size_t, size_t>> layeroff_fragment;
};

struct prefetch_ctx {
    int total_layers;
    struct ggml_cgraph *cgraph;
    std::map<std::string, std::unique_ptr<layer_prefetch_status>> layer_status;
    std::vector<std::string> sorted_layers;
};

static std::string get_layer_name(struct ggml_tensor * node)
{
    std::string layer_name;
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        struct ggml_tensor * src = node->src[j];
        if (src == NULL) {
            continue;
        }
        std::string node_name(src->name);
        
        size_t dot_pos = node_name.find('.');
        if (node_name.find("blk") == 0 && dot_pos != std::string::npos) {
            size_t second_dot_pos = node_name.find('.', dot_pos + 1);
            if (second_dot_pos != std::string::npos) {
                layer_name = node_name.substr(0, second_dot_pos);
                break;
            }
        } else if (node_name.find("output_norm") != std::string::npos) { 
            layer_name = "output_norm";
            break;
        } else if (node_name.find("output") != std::string::npos) {
            layer_name = "output_weight";
            break;
        } else if (node_name.find("token_embd") != std::string::npos) {
            layer_name = "token_embd";
            break;
        } else {
            continue;
        }
    }
    return layer_name;
}

static void de_prefetch_weights(struct prefetch_ctx *ctx, struct layer_prefetch_status *free_layer)
{
    long page_size = sysconf(_SC_PAGESIZE);
    void *mmap_addr = ctx->cgraph->mmap_addr;

    for (auto & tensor_flag : free_layer->layeroff_fragment) {

        size_t free_start_off = std::get<0>(tensor_flag);
        size_t free_end_off = std::get<1>(tensor_flag);
        size_t aligned_start = ((free_start_off + page_size - 1) / page_size) * page_size;   // 向上取整
        size_t aligned_end = (free_end_off / page_size) * page_size;                         // 向下取整
        size_t length = aligned_end - aligned_start;
        // if (munmap((char *)mmap_addr + aligned_start, length) != 0) {
        //     perror("munmap");
        // }
        // printf("deprefetch weights index:%d start_off:%lu end_off:%lu size:%.2f MB\n", free_layer->layer_index, free_start_off, free_end_off, (free_end_off-free_start_off)/1024.0/1024);
        if (posix_madvise(static_cast<char*>(mmap_addr) + aligned_start, length, MADV_PAGEOUT)) {
            fprintf(stderr, "warning: posix_madvise(.., MADV_PAGEOUT) failed: %s\n", strerror(errno));
        }
    }
}

typedef struct {
    int fd;
    size_t prefetch_start_off;
    void *addr;
    size_t length;
    std::string name;
} thread_args_t;

void *prefetch_thread(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    void * new_addr;

    new_addr = mmap(args->addr,
                    args->length, 
                    PROT_READ, 
                    MAP_SHARED | MAP_FIXED | MAP_POPULATE, 
                    args->fd, 
                    args->prefetch_start_off);

    if (new_addr == MAP_FAILED) {
        perror("mmap (remap)");
    }

    // if (posix_madvise(args->addr, args->length, MADV_POPULATE_READ)) {
    //     perror("madvise");
    // }

    return NULL;
}

static int prefetch_weights(const struct ggml_cgraph * cgraph, const std::string &cur_layer_name, size_t start_off, size_t end_off)
{
    long page_size = sysconf(_SC_PAGESIZE);
    void *mmap_addr = cgraph->mmap_addr;
    int fd = cgraph->fd;

    size_t prefetch_start_off = start_off;
    size_t prefetch_end_off = end_off;

    prefetch_start_off       &= ~(page_size - 1);                                     
    prefetch_end_off          = (prefetch_end_off + page_size - 1) & ~(page_size - 1);  
    size_t prefetch_size      = prefetch_end_off - prefetch_start_off;

    // can be adjust
    const int num_threads = 1;
    pthread_t threads[num_threads];
    thread_args_t args[num_threads];

    size_t chunk_size = prefetch_size / num_threads;
    if (chunk_size % page_size != 0) {
        chunk_size = ((chunk_size + page_size - 1) / page_size) * page_size;
    }

    for (int i = 0; i < num_threads; i++) {
        size_t thread_start_off = prefetch_start_off + i * chunk_size;
        size_t thread_end_off = thread_start_off + chunk_size;

        if (thread_end_off > prefetch_end_off || i == num_threads - 1) {
            thread_end_off = prefetch_end_off; 
        }

        args[i].addr   = (char*)mmap_addr + thread_start_off;
        args[i].length = thread_end_off - thread_start_off;
        args[i].name = cur_layer_name;
        args[i].fd = fd;
        args[i].prefetch_start_off = thread_start_off;

        if (pthread_create(&threads[i], NULL, prefetch_thread, &args[i]) != 0) {
            fprintf(stderr, "pthread_create failed for thread %d\n", i);
            args[i].length = 0;
        }
    }

    for (int i = 0; i < num_threads; i++) {
        if (args[i].length > 0) {
            pthread_join(threads[i], NULL);
        }
    }
    return 1;
}

static int one_thread_prefetch_weights(const struct ggml_cgraph * cgraph, size_t start_off, size_t end_off)
{
    long page_size = sysconf(_SC_PAGESIZE);
    void *mmap_addr = cgraph->mmap_addr;
    int fd = cgraph->fd;

    size_t prefetch_start_off = start_off;
    size_t prefetch_end_off   = end_off;
    prefetch_start_off       &= ~(page_size - 1);                                      
    prefetch_end_off          = (prefetch_end_off + page_size - 1) & ~(page_size - 1);  
    size_t prefetch_size      = prefetch_end_off - prefetch_start_off;

    void * new_addr;
    new_addr = mmap((char *)mmap_addr + prefetch_start_off, 
                    prefetch_size, 
                    PROT_READ, 
                    MAP_SHARED | MAP_FIXED | MAP_POPULATE, 
                    fd, 
                    prefetch_start_off);
                    
    if (new_addr == MAP_FAILED) {
        perror("mmap (remap)");
    }

    return 1;
}

void notify_deprefetch_weights(void *ctx, const struct ggml_cgraph *cgraph, int node_n)
{
    struct ggml_tensor *node;
    std::string cur_layer_name;
    struct prefetch_ctx *prefetch = static_cast<struct prefetch_ctx *>(ctx);

    for (int j = node_n; j >= 0; j--) {
        node = cgraph->nodes[j];
        cur_layer_name = get_layer_name(node);
        if (!cur_layer_name.empty()) {
            break;
        }
    }
    
    auto it = prefetch->layer_status.find(cur_layer_name);
    if (it == prefetch->layer_status.end()) {
        return;
    }
    it->second->prefetched.store(false, std::memory_order_release);
}

void wait_prefetch_weights(void *ctx, const struct ggml_cgraph *cgraph, int node_n)
{
    struct ggml_tensor *node;
    std::string cur_layer_name;
    struct prefetch_ctx *prefetch = static_cast<struct prefetch_ctx *>(ctx);
    
    for (int j = node_n; j < cgraph->n_nodes; j++) {
        node = cgraph->nodes[j];
        cur_layer_name = get_layer_name(node);
        if (!cur_layer_name.empty()) {
            break;
        }
    }
    
    auto it = prefetch->layer_status.find(cur_layer_name);
    if (it == prefetch->layer_status.end()) {
        return;
    }

    while (!it->second->prefetched.load(std::memory_order_acquire)) {

    }
}

void *prefetch_thread_main(void * arg)  
{
    struct prefetch_ctx *ctx = static_cast<struct prefetch_ctx *>(arg);
    struct ggml_cgraph *cgraph = ctx->cgraph;

    int w;
    memcpy(&w, (char*)cgraph->offline_planning_mapping + sizeof(int), sizeof(int));
    int dynamic_window_size = w;

    int cur_prefetch_size = 0;

    std::vector<std::string> prefetched_layers;

    size_t cur_index = 0;
    while(cur_index < ctx->sorted_layers.size()) {
        std::string cur_layer_name = ctx->sorted_layers[cur_index];
        auto layer_ctx = ctx->layer_status.find(cur_layer_name);
        if (layer_ctx == ctx->layer_status.end()) {
            printf("no layer info\n");
            cur_index++;
            continue;
        }
        struct layer_prefetch_status *layer_status = layer_ctx->second.get();
        if (!layer_status->is_dynamic_layer) {
            cur_index++;
            continue;
        }
        if (cur_prefetch_size < dynamic_window_size) {
            for (auto &tensor_frag : layer_status->layeroff_fragment) {
                size_t start_off = std::get<0>(tensor_frag);
                size_t end_off   = std::get<1>(tensor_frag);
                prefetch_weights(cgraph, cur_layer_name, start_off, end_off);
            }
            layer_status->prefetched.store(true, std::memory_order_release);
            prefetched_layers.push_back(cur_layer_name);
            cur_prefetch_size++;
            cur_index++;
        } else { // prefetch window size full

            while (true) {
                for (auto it = prefetched_layers.begin(); it != prefetched_layers.end(); ) {
                    auto free_layer = ctx->layer_status.find(*it);
                    if (free_layer == ctx->layer_status.end()) {
                        ++it;
                        continue;
                    }
                    if (!free_layer->second->prefetched.load(std::memory_order_acquire)) {
                        de_prefetch_weights(ctx, free_layer->second.get());
                        it = prefetched_layers.erase(it);
                        cur_prefetch_size--;
                    } else {
                        ++it;
                    }
                }
                if (cur_prefetch_size < dynamic_window_size) {
                    break;
                }
            }
        }

    }

    while (!prefetched_layers.empty()) {
        for (auto it = prefetched_layers.begin(); it != prefetched_layers.end(); ) {
            auto layer_it = ctx->layer_status.find(*it);
            if (layer_it == ctx->layer_status.end()) {
                ++it;
                continue;
            }
            if (!layer_it->second->prefetched.load(std::memory_order_acquire)) {
                de_prefetch_weights(ctx, layer_it->second.get());
                it = prefetched_layers.erase(it);
            } else {
                ++it;
            }
        }
    }

    delete ctx;
    return nullptr;
}

pthread_t create_prefetch_thread(void *ctx)
{
    struct prefetch_ctx *prefetch = static_cast<struct prefetch_ctx *>(ctx);

    pthread_t thread_id;
    pthread_create(&thread_id, NULL, prefetch_thread_main, prefetch);
    return thread_id;
}


void *create_prefetch_ctx(struct ggml_cgraph *cgraph)
{
    auto layer_offset = reinterpret_cast<std::map<std::string, std::vector<std::tuple<size_t, size_t, int>>> *>(cgraph->layer_weights_off);
    int layer_size = layer_offset->size();
    struct prefetch_ctx *prefetch = new prefetch_ctx;
    prefetch->total_layers = layer_size;
    prefetch->cgraph = cgraph;

    for (const auto& entry : *layer_offset) {
        const std::string &layer_name = entry.first;
        std::unique_ptr<layer_prefetch_status> data(new layer_prefetch_status());

        data->prefetched.store(false, std::memory_order_relaxed);
        data->is_dynamic_layer = false;
        for (const auto &layer_frag : entry.second) {
            size_t start_off = std::get<0>(layer_frag);
            size_t end_off   = std::get<1>(layer_frag);
            int layer_index  = std::get<2>(layer_frag);
            data->layer_index = layer_index;
            data->layeroff_fragment.emplace_back(std::make_tuple(start_off, end_off));
        }
        prefetch->layer_status.emplace(layer_name, std::move(data));
        prefetch->sorted_layers.emplace_back(layer_name);
    }

    auto cmp = [](const std::string &a, const std::string &b) {
        auto getRank = [](const std::string &key) -> std::pair<int, int> {
            if (key == "token_embd") {
                return {0, 0};
            } else if (key.size() >= 4 && key.compare(0, 4, "blk.") == 0) {
                int num = std::atoi(key.substr(4).c_str());
                return {1, num};
            } else if (key == "output_norm") {
                return {2, 0};
            } else if (key == "output_weight") {
                return {3, 0};
            }
            return {4, 0};
        };

        auto rankA = getRank(a);
        auto rankB = getRank(b);
        if (rankA.first != rankB.first)
            return rankA.first < rankB.first;
        else
            return rankA.second < rankB.second;
    };

    std::sort(prefetch->sorted_layers.begin(), prefetch->sorted_layers.end(), cmp);

    return static_cast<void*>(prefetch);
}

void sync_prefetch_weights(void *ctx, int node_n)
{
    struct prefetch_ctx *prefetch_ctx = reinterpret_cast<struct prefetch_ctx *>(ctx);
    struct ggml_cgraph *cgraph = prefetch_ctx->cgraph;

    std::string cur_layer_name;
    for (int j = node_n; j < cgraph->n_nodes; j++) {
        struct ggml_tensor *node_weight = cgraph->nodes[j];
        cur_layer_name = get_layer_name(node_weight);
        if (cur_layer_name.empty()) {  
            continue;
        } else {
            break;
        }
    }
    auto it = prefetch_ctx->layer_status.find(cur_layer_name);
    struct layer_prefetch_status *layer_status = it->second.get();

    size_t layer_size = 0;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (auto &tensor_frag : layer_status->layeroff_fragment) {
        size_t start_off = std::get<0>(tensor_frag);
        size_t end_off   = std::get<1>(tensor_frag);

        layer_size += (end_off-start_off);

        prefetch_weights(cgraph, cur_layer_name, start_off, end_off);

    }
    gettimeofday(&end, NULL);
    long mtime = (end.tv_sec - start.tv_sec)*1000 + (end.tv_usec - start.tv_usec)/1000;
    layer_size = layer_size / 1024 / 1024;
    long io_speed = (long)(layer_size/(mtime/1000.0));

    char buf[256];
    int len = snprintf(buf, sizeof(buf),"prefetch index:blk.%d, prefetch time:%ld, layer size:%lu MB, IO Speed:%lu MB/s\n",
                        layer_status->layer_index,
                        mtime,
                        layer_size,
                        io_speed);
    if (len > 0) {
        if (write(cgraph->offline_logfd, buf, len) != len) {
            perror("write to offline_logfd failed");
        }
    }
}

void sync_deprefetch_weights(void *ctx, int node_n)
{
    struct ggml_tensor *node;
    std::string free_layer_name;
    struct prefetch_ctx *prefetch = static_cast<struct prefetch_ctx *>(ctx);
    struct ggml_cgraph *cgraph = prefetch->cgraph;
    for (int j = node_n; j >= 0; j--) {
        node = cgraph->nodes[j];
        free_layer_name = get_layer_name(node);
        if (!free_layer_name.empty()) {
            break;
        }
    }
    
    auto it = prefetch->layer_status.find(free_layer_name);
    if (it == prefetch->layer_status.end()) {
        perror("sync_deprefetch_weights not find free layer name\n");
        return;
    }
    struct timeval deprefetch_start, deprefetch_end;
    gettimeofday(&deprefetch_start, NULL);
    struct layer_prefetch_status *layer_status = it->second.get();
    de_prefetch_weights(prefetch, layer_status);
    gettimeofday(&deprefetch_end, NULL);
    long deprefetch_mtime = (deprefetch_end.tv_sec - deprefetch_start.tv_sec) * 1000 + 
                            (deprefetch_end.tv_usec - deprefetch_start.tv_usec)/1000;
    char buf[256];
    int len = snprintf(buf, sizeof(buf),"release index:blk.%d, release time:%ld ms\n", 
                        layer_status->layer_index, deprefetch_mtime);
    if (len > 0) {
        if (write(cgraph->offline_logfd, buf, len) != len) {
            perror("write to offline_logfd failed");
        }
    }
}

void prefetch_resident_layer_weights(void *ctx)
{
    struct prefetch_ctx *prefetch_ctx = reinterpret_cast<struct prefetch_ctx *>(ctx);
    struct ggml_cgraph *cgraph = prefetch_ctx->cgraph;
    std::map<std::string, std::unique_ptr<layer_prefetch_status>> *layer_status = &prefetch_ctx->layer_status;
    
    static bool prefetch_done  = false;

    int k;
    memcpy(&k, cgraph->offline_planning_mapping, sizeof(int));
    int dynamic_layer_entrance = k;

    for (auto &layer : *layer_status) {
        struct layer_prefetch_status *status = layer.second.get();

        int layer_index  = status->layer_index;
        if (layer_index >= dynamic_layer_entrance) {
            status->is_dynamic_layer = true;
            continue;
        }
        status->is_dynamic_layer = false;
        if (!prefetch_done && (!(layer_index == -1 && !cgraph->prefetch_input))) {
            for (auto &tensor_frag : status->layeroff_fragment) {
                size_t start_off = std::get<0>(tensor_frag);
                size_t end_off = std::get<1>(tensor_frag);
                one_thread_prefetch_weights(cgraph, start_off, end_off);
                // printf("0-prefetch weights name:%s start_off:%lu end_off:%lu size:%.2f MB\n", layer.first.c_str(), start_off, end_off, (end_off-start_off)/1024.0/1024);
            }
            status->prefetched.store(true, std::memory_order_relaxed);
        }
    }
    prefetch_done = true;   
}

int judge_dynamic_layer(void *ctx, const struct ggml_cgraph *cgraph, int node_n)
{
    struct prefetch_ctx *prefetch_ctx = reinterpret_cast<struct prefetch_ctx *>(ctx);

    std::string cur_layer_name;
    for (int j = node_n; j < cgraph->n_nodes; j++) {
        struct ggml_tensor *node_weight = cgraph->nodes[j];
        cur_layer_name = get_layer_name(node_weight);
        if (cur_layer_name.empty()) {  // 当前节点没有权重
            continue;
        } else {
            break;
        }
    }
    auto it = prefetch_ctx->layer_status.find(cur_layer_name);
    struct layer_prefetch_status *layer_status = it->second.get();
    if (layer_status->is_dynamic_layer) {
        return 1;
    }
    return 0;
}

static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    if (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = new uint8_t[cplan.work_size];
        if (cpu_ctx->work_data == NULL) {
            cpu_ctx->work_size = 0;
            return GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;

    return ggml_graph_compute(cgraph, &cplan);
}

static const struct ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static ggml_guid_t ggml_backend_cpu_guid(void) {
    static ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    return &guid;
}

ggml_backend_t ggml_backend_cpu_init(void) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    struct ggml_backend_cpu_context * ctx = new ggml_backend_cpu_context;
    if (ctx == NULL) {
        return NULL;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = NULL;
    ctx->work_data           = NULL;
    ctx->work_size           = 0;
    ctx->abort_callback      = NULL;
    ctx->abort_callback_data = NULL;

    ggml_backend_t cpu_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_cpu_guid(),
        /* .interface = */ ggml_backend_cpu_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context   = */ ctx,
    };

    if (cpu_backend == NULL) {
        delete ctx;
        return NULL;
    }

    return cpu_backend;
}

bool ggml_backend_is_cpu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

void ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;

    if (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

// CPU backend - device

struct ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        if (!sysctlbyname("machdep.cpu.brand_string", NULL, &len, NULL, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, NULL, 0); // NOLINT
        }
#elif defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) {
                if (strncmp(buf, "model name", 10) == 0) {
                    char * p = strchr(buf, ':');
                    if (p) {
                        p++;
                        while (std::isspace(*p)) {
                            p++;
                        }
                        while (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        break;
                    }
                }
            }
            fclose(f);
        }
#elif defined(_WIN32)
        HKEY hKey;
        if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        KEY_READ,
                        &hKey) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            if (RegQueryValueExA(hKey,
                                TEXT("ProcessorNameString"),
                                NULL,
                                NULL,
                                NULL,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                if (RegQueryValueExA(hKey,
                                    TEXT("ProcessorNameString"),
                                    NULL,
                                    NULL,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    if (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            RegCloseKey(hKey);
        }
#endif
    }
};

static const char * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    return "CPU";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO
    *free = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_cpu_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_cpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (src0 && src0->buffer && ggml_backend_cpu_buft_is_aarch64(src0->buffer->buft)) {
        if (op->op != GGML_OP_MUL_MAT || src0->type != GGML_TYPE_Q4_0 || ggml_aarch64_get_optimal_repack_type(src0) == GGML_TYPE_Q4_0) {
            return false;
        }
    }

    for (int i = 1; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_cpu_buft_is_aarch64(op->src[i]->buffer->buft)) {
            return false;
        }
    }

    switch (op->op) {
        case GGML_OP_CPY:
            return
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        case GGML_OP_MUL_MAT:
            return src1->type == GGML_TYPE_F32 || src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
        case GGML_OP_ROPE_BACK:
            return op->src[2] == NULL && (op->op_params[2] & 4) == 0;
        case GGML_OP_IM2COL_BACK:
            return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32;
        case GGML_OP_OUT_PROD:
            return (src0->type == GGML_TYPE_F32 || ggml_is_quantized(src0->type)) && src1->type == GGML_TYPE_F32;
        default:
            return true;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft) || ggml_backend_cpu_buft_is_aarch64(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// CPU backend - backend (reg)

static const char * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    return "CPU";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_cpu_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_cpu_device_context ctx;
    static ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    return &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
static ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        if (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        if (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        if (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        if (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        if (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        if (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        if (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        if (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        if (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        if (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        if (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        if (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        if (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        if (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        if (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        if (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        if (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        if (ggml_cpu_get_sve_cnt() > 0) {
            static std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        if (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        if (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        if (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        if (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
        // TODO: rename this
    #ifdef GGML_USE_CPU_AARCH64
        features.push_back({ "AARCH64_REPACK", "1" });
    #endif

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *)ggml_backend_cpu_set_n_threads;
    }
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        return (void *)ggml_backend_cpu_get_extra_bufts;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cpu_get_features;
    }
    if (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return (void *)ggml_backend_cpu_set_abort_callback;
    }
    if (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        return (void *)ggml_numa_init;
    }
    if (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        return (void *)ggml_is_numa;
    }

    // threadpool - TODO:  move to ggml-base
    if (strcmp(name, "ggml_threadpool_new") == 0) {
        return (void *)ggml_threadpool_new;
    }
    if (strcmp(name, "ggml_threadpool_free") == 0) {
        return (void *)ggml_threadpool_free;
    }
    if (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        return (void *)ggml_backend_cpu_set_threadpool;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(void) {
    // init CPU feature detection
    ggml_cpu_init();

    static struct ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
