#define _DARWIN_C_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define NUM_EXPERTS 512
#define DEFAULT_MODEL_PATH "/Users/anemll/Models/flash_mlx_4bit"
#define MAX_SPLIT 32

typedef enum {
    MODE_4BIT = 0,
    MODE_2BIT = 1,
    MODE_Q3   = 2,
} BenchMode;

typedef struct {
    off_t offset;
    size_t size;
} BenchTask;

typedef struct {
    int fd;
    BenchTask *tasks;
    int num_tasks;
    size_t max_task_size;
    size_t bytes_read;
    uint64_t checksum;
    atomic_int next_task;
} BenchWork;

typedef struct {
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t work_ready;
    pthread_cond_t work_done;
    int id;
    int active_generation;
    int completed_generation;
    int stop;
    BenchWork *work;
    size_t thread_bytes;
    uint64_t checksum;
} BenchWorker;

typedef struct {
    BenchWorker *workers;
    int num_workers;
    int generation;
} BenchPool;

typedef struct {
    const char *model_path;
    const char *file_path;
    BenchMode mode;
    int layer;
    int threads;
    int split;
    int experts;
    int warmup_passes;
    int timed_passes;
    int contiguous;
    uint32_t seed;
} BenchConfig;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "\n"
            "Warm-page-cache pread microbenchmark for packed expert files.\n"
            "\n"
            "Options:\n"
            "  --model PATH        Model directory (default: %s)\n"
            "  --file PATH         Benchmark this exact layer file (overrides --model/mode/layer)\n"
            "  --layer N           Layer file index to use (default: 0)\n"
            "  --threads N         Worker threads (default: 8)\n"
            "  --split N           Split each expert into N page-aligned pread chunks (default: 1)\n"
            "  --experts N         Unique experts per pass, max 512 (default: 128)\n"
            "  --warmup N          Warmup passes before timing (default: 2)\n"
            "  --passes N          Timed passes to average (default: 5)\n"
            "  --seed N            Shuffle seed for expert selection (default: 1234)\n"
            "  --contiguous        Use expert ids 0..N-1 instead of shuffled sparse ids\n"
            "  --2bit              Use packed_experts_2bit/\n"
            "  --q3-experts        Use packed_experts_Q3/\n"
            "  -h, --help          Show this help\n",
            prog, DEFAULT_MODEL_PATH);
}

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static void shuffle_ints(int *values, int count, uint32_t seed) {
    uint32_t state = seed ? seed : 1u;
    for (int i = count - 1; i > 0; --i) {
        uint32_t r = xorshift32(&state);
        int j = (int)(r % (uint32_t)(i + 1));
        int tmp = values[i];
        values[i] = values[j];
        values[j] = tmp;
    }
}

static const char *mode_dir(BenchMode mode) {
    switch (mode) {
        case MODE_2BIT: return "packed_experts_2bit";
        case MODE_Q3: return "packed_experts_Q3";
        case MODE_4BIT:
        default: return "packed_experts";
    }
}

static const char *mode_label(BenchMode mode) {
    switch (mode) {
        case MODE_2BIT: return "2-bit";
        case MODE_Q3: return "Q3-GGUF";
        case MODE_4BIT:
        default: return "4-bit";
    }
}

static void resolve_layer_file(char *out, size_t out_cap, const BenchConfig *cfg) {
    if (cfg->file_path && cfg->file_path[0] != '\0') {
        snprintf(out, out_cap, "%s", cfg->file_path);
        return;
    }
    snprintf(out, out_cap, "%s/%s/layer_%02d.bin",
             cfg->model_path, mode_dir(cfg->mode), cfg->layer);
}

static int effective_split(size_t expert_size, long page_size, int requested_split) {
    if (requested_split < 1) requested_split = 1;
    if (requested_split > MAX_SPLIT) requested_split = MAX_SPLIT;
    if (page_size <= 0) return 1;
    if (expert_size == 0 || (expert_size % (size_t)page_size) != 0) return 1;
    size_t pages = expert_size / (size_t)page_size;
    if ((size_t)requested_split > pages) requested_split = (int)pages;
    if (requested_split < 1) requested_split = 1;
    return requested_split;
}

static size_t build_tasks(BenchTask *tasks,
                          const int *expert_ids,
                          int num_experts,
                          size_t expert_size,
                          int split,
                          long page_size) {
    size_t total_bytes = 0;
    size_t total_pages = (split > 1 && page_size > 0) ? (expert_size / (size_t)page_size) : 0;
    for (int i = 0; i < num_experts; ++i) {
        size_t page_cursor = 0;
        for (int c = 0; c < split; ++c) {
            size_t chunk_off = 0;
            size_t chunk_sz = expert_size;
            if (split > 1) {
                size_t pages_this_chunk = total_pages / (size_t)split;
                if ((size_t)c < (total_pages % (size_t)split)) pages_this_chunk++;
                chunk_off = page_cursor * (size_t)page_size;
                chunk_sz = pages_this_chunk * (size_t)page_size;
                page_cursor += pages_this_chunk;
            }
            int task_idx = i * split + c;
            tasks[task_idx].offset = (off_t)expert_ids[i] * (off_t)expert_size + (off_t)chunk_off;
            tasks[task_idx].size = chunk_sz;
            total_bytes += chunk_sz;
        }
    }
    return total_bytes;
}

static void *bench_worker_main(void *arg) {
    BenchWorker *worker = (BenchWorker *)arg;
    void *scratch = NULL;
    size_t scratch_cap = 0;

    pthread_mutex_lock(&worker->mutex);
    while (!worker->stop) {
        while (worker->active_generation == worker->completed_generation && !worker->stop) {
            pthread_cond_wait(&worker->work_ready, &worker->mutex);
        }
        if (worker->stop) break;

        int my_generation = worker->active_generation;
        BenchWork *work = worker->work;
        pthread_mutex_unlock(&worker->mutex);

        if (work && work->max_task_size > 0 && work->max_task_size > scratch_cap) {
            if (scratch) free(scratch);
            if (posix_memalign(&scratch, 16384, work->max_task_size) != 0) {
                scratch = NULL;
                scratch_cap = 0;
            } else {
                scratch_cap = work->max_task_size;
            }
        }

        size_t local_bytes = 0;
        uint64_t local_checksum = 0;
        if (work && scratch) {
            while (1) {
                int idx = atomic_fetch_add_explicit(&work->next_task, 1, memory_order_relaxed);
                if (idx >= work->num_tasks) break;
                BenchTask *task = &work->tasks[idx];
                ssize_t nr = pread(work->fd, scratch, task->size, task->offset);
                if (nr != (ssize_t)task->size) {
                    fprintf(stderr,
                            "[cachebench] pread failed at task %d offset=%lld size=%zu: %s\n",
                            idx, (long long)task->offset, task->size,
                            nr < 0 ? strerror(errno) : "short read");
                    exit(1);
                }
                local_bytes += (size_t)nr;
                local_checksum ^= ((const uint8_t *)scratch)[0];
                local_checksum ^= ((const uint8_t *)scratch)[nr - 1];
            }
        }

        pthread_mutex_lock(&worker->mutex);
        worker->thread_bytes = local_bytes;
        worker->checksum = local_checksum;
        worker->completed_generation = my_generation;
        pthread_cond_signal(&worker->work_done);
        pthread_mutex_unlock(&worker->mutex);
        pthread_mutex_lock(&worker->mutex);
    }
    pthread_mutex_unlock(&worker->mutex);

    if (scratch) free(scratch);
    return NULL;
}

static void pool_init(BenchPool *pool, int num_workers) {
    memset(pool, 0, sizeof(*pool));
    pool->num_workers = num_workers;
    pool->workers = calloc((size_t)num_workers, sizeof(BenchWorker));
    if (!pool->workers) {
        fprintf(stderr, "[cachebench] failed to allocate worker pool\n");
        exit(1);
    }
    for (int i = 0; i < num_workers; ++i) {
        BenchWorker *w = &pool->workers[i];
        w->id = i;
        pthread_mutex_init(&w->mutex, NULL);
        pthread_cond_init(&w->work_ready, NULL);
        pthread_cond_init(&w->work_done, NULL);
        if (pthread_create(&w->thread, NULL, bench_worker_main, w) != 0) {
            fprintf(stderr, "[cachebench] failed to create worker %d\n", i);
            exit(1);
        }
    }
}

static uint64_t pool_run(BenchPool *pool, BenchWork *work) {
    atomic_store_explicit(&work->next_task, 0, memory_order_relaxed);
    pool->generation++;

    for (int i = 0; i < pool->num_workers; ++i) {
        BenchWorker *w = &pool->workers[i];
        pthread_mutex_lock(&w->mutex);
        w->work = work;
        w->thread_bytes = 0;
        w->checksum = 0;
        w->active_generation = pool->generation;
        pthread_cond_signal(&w->work_ready);
        pthread_mutex_unlock(&w->mutex);
    }

    uint64_t checksum = 0;
    for (int i = 0; i < pool->num_workers; ++i) {
        BenchWorker *w = &pool->workers[i];
        pthread_mutex_lock(&w->mutex);
        while (w->completed_generation < pool->generation) {
            pthread_cond_wait(&w->work_done, &w->mutex);
        }
        checksum ^= w->checksum;
        pthread_mutex_unlock(&w->mutex);
    }
    return checksum;
}

static void pool_destroy(BenchPool *pool) {
    if (!pool->workers) return;
    for (int i = 0; i < pool->num_workers; ++i) {
        BenchWorker *w = &pool->workers[i];
        pthread_mutex_lock(&w->mutex);
        w->stop = 1;
        pthread_cond_signal(&w->work_ready);
        pthread_mutex_unlock(&w->mutex);
    }
    for (int i = 0; i < pool->num_workers; ++i) {
        BenchWorker *w = &pool->workers[i];
        pthread_join(w->thread, NULL);
        pthread_mutex_destroy(&w->mutex);
        pthread_cond_destroy(&w->work_ready);
        pthread_cond_destroy(&w->work_done);
    }
    free(pool->workers);
    pool->workers = NULL;
}

static void parse_args(BenchConfig *cfg, int argc, char **argv) {
    *cfg = (BenchConfig){
        .model_path = getenv("FLASH_MOE_MODEL") ? getenv("FLASH_MOE_MODEL") : DEFAULT_MODEL_PATH,
        .file_path = NULL,
        .mode = MODE_4BIT,
        .layer = 0,
        .threads = 8,
        .split = 1,
        .experts = 128,
        .warmup_passes = 2,
        .timed_passes = 5,
        .contiguous = 0,
        .seed = 1234u,
    };

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--model") && i + 1 < argc) {
            cfg->model_path = argv[++i];
        } else if (!strcmp(arg, "--file") && i + 1 < argc) {
            cfg->file_path = argv[++i];
        } else if (!strcmp(arg, "--layer") && i + 1 < argc) {
            cfg->layer = atoi(argv[++i]);
        } else if (!strcmp(arg, "--threads") && i + 1 < argc) {
            cfg->threads = atoi(argv[++i]);
        } else if (!strcmp(arg, "--split") && i + 1 < argc) {
            cfg->split = atoi(argv[++i]);
        } else if (!strcmp(arg, "--experts") && i + 1 < argc) {
            cfg->experts = atoi(argv[++i]);
        } else if (!strcmp(arg, "--warmup") && i + 1 < argc) {
            cfg->warmup_passes = atoi(argv[++i]);
        } else if (!strcmp(arg, "--passes") && i + 1 < argc) {
            cfg->timed_passes = atoi(argv[++i]);
        } else if (!strcmp(arg, "--seed") && i + 1 < argc) {
            cfg->seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(arg, "--contiguous")) {
            cfg->contiguous = 1;
        } else if (!strcmp(arg, "--2bit")) {
            cfg->mode = MODE_2BIT;
        } else if (!strcmp(arg, "--q3-experts")) {
            cfg->mode = MODE_Q3;
        } else if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", arg);
            usage(argv[0]);
            exit(1);
        }
    }

    if (cfg->threads < 1) cfg->threads = 1;
    if (cfg->experts < 1) cfg->experts = 1;
    if (cfg->experts > NUM_EXPERTS) cfg->experts = NUM_EXPERTS;
    if (cfg->warmup_passes < 0) cfg->warmup_passes = 0;
    if (cfg->timed_passes < 1) cfg->timed_passes = 1;
    if (cfg->split < 1) cfg->split = 1;
    if (cfg->split > MAX_SPLIT) cfg->split = MAX_SPLIT;
}

int main(int argc, char **argv) {
    BenchConfig cfg;
    parse_args(&cfg, argc, argv);

    char path[4096];
    resolve_layer_file(path, sizeof(path), &cfg);

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[cachebench] failed to open %s: %s\n", path, strerror(errno));
        return 1;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "[cachebench] fstat failed for %s: %s\n", path, strerror(errno));
        close(fd);
        return 1;
    }
    if (st.st_size <= 0 || (st.st_size % NUM_EXPERTS) != 0) {
        fprintf(stderr,
                "[cachebench] unexpected layer file size for %s: %" PRIdMAX " bytes\n",
                path, (intmax_t)st.st_size);
        close(fd);
        return 1;
    }

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) page_size = 16384;

    size_t expert_size = (size_t)st.st_size / NUM_EXPERTS;
    int split = effective_split(expert_size, page_size, cfg.split);
    size_t max_chunk_size = expert_size;
    if (split > 1) {
        size_t total_pages = expert_size / (size_t)page_size;
        size_t pages_this_chunk = total_pages / (size_t)split;
        if (total_pages % (size_t)split) pages_this_chunk++;
        max_chunk_size = pages_this_chunk * (size_t)page_size;
    }

    int expert_ids[NUM_EXPERTS];
    for (int i = 0; i < NUM_EXPERTS; ++i) expert_ids[i] = i;
    if (!cfg.contiguous) shuffle_ints(expert_ids, NUM_EXPERTS, cfg.seed);

    BenchTask *tasks = calloc((size_t)cfg.experts * (size_t)split, sizeof(BenchTask));
    if (!tasks) {
        fprintf(stderr, "[cachebench] failed to allocate tasks\n");
        close(fd);
        return 1;
    }
    size_t bytes_per_pass = build_tasks(tasks, expert_ids, cfg.experts, expert_size, split, page_size);

    BenchWork work = {
        .fd = fd,
        .tasks = tasks,
        .num_tasks = cfg.experts * split,
        .max_task_size = max_chunk_size,
        .bytes_read = bytes_per_pass,
        .checksum = 0,
    };
    atomic_init(&work.next_task, 0);

    BenchPool pool;
    pool_init(&pool, cfg.threads);

    printf("[cachebench] file=%s\n", path);
    printf("[cachebench] mode=%s layer=%d experts=%d/%d threads=%d split=%d page=%ld bytes\n",
           mode_label(cfg.mode), cfg.layer, cfg.experts, NUM_EXPERTS, cfg.threads, split, page_size);
    printf("[cachebench] expert_size=%zu bytes (%.2f MiB) pass_bytes=%zu (%.2f MiB)\n",
           expert_size, (double)expert_size / (1024.0 * 1024.0),
           bytes_per_pass, (double)bytes_per_pass / (1024.0 * 1024.0));
    printf("[cachebench] pattern=%s seed=%u warmup=%d passes=%d\n",
           cfg.contiguous ? "contiguous" : "shuffled", cfg.seed,
           cfg.warmup_passes, cfg.timed_passes);

    for (int i = 0; i < cfg.warmup_passes; ++i) {
        (void)pool_run(&pool, &work);
    }

    double total_s = 0.0;
    double best_s = 1e30;
    double worst_s = 0.0;
    uint64_t checksum = 0;
    for (int pass = 0; pass < cfg.timed_passes; ++pass) {
        double t0 = now_s();
        checksum ^= pool_run(&pool, &work);
        double dt = now_s() - t0;
        if (dt < best_s) best_s = dt;
        if (dt > worst_s) worst_s = dt;
        total_s += dt;

        double gib_s = ((double)bytes_per_pass / (1024.0 * 1024.0 * 1024.0)) / dt;
        double gb_s = ((double)bytes_per_pass / 1e9) / dt;
        double us_per_expert = dt * 1e6 / (double)cfg.experts;
        printf("[pass %d/%d] %.3f ms | %.2f GiB/s | %.2f GB/s | %.1f us/expert\n",
               pass + 1, cfg.timed_passes, dt * 1000.0, gib_s, gb_s, us_per_expert);
    }

    double avg_s = total_s / (double)cfg.timed_passes;
    double avg_gib_s = ((double)bytes_per_pass / (1024.0 * 1024.0 * 1024.0)) / avg_s;
    double avg_gb_s = ((double)bytes_per_pass / 1e9) / avg_s;
    double avg_us_per_expert = avg_s * 1e6 / (double)cfg.experts;

    printf("[result] avg=%.3f ms | %.2f GiB/s | %.2f GB/s | %.1f us/expert\n",
           avg_s * 1000.0, avg_gib_s, avg_gb_s, avg_us_per_expert);
    printf("[result] best=%.3f ms | %.2f GiB/s | %.2f GB/s\n",
           best_s * 1000.0,
           ((double)bytes_per_pass / (1024.0 * 1024.0 * 1024.0)) / best_s,
           ((double)bytes_per_pass / 1e9) / best_s);
    printf("[result] worst=%.3f ms | %.2f GiB/s | %.2f GB/s\n",
           worst_s * 1000.0,
           ((double)bytes_per_pass / (1024.0 * 1024.0 * 1024.0)) / worst_s,
           ((double)bytes_per_pass / 1e9) / worst_s);
    printf("[result] checksum=%" PRIu64 "\n", checksum);

    pool_destroy(&pool);
    free(tasks);
    close(fd);
    return 0;
}
