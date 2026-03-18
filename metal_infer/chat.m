/*
 * chat.m — Interactive TUI chat for Qwen3.5-397B via Metal inference
 *
 * Multi-turn conversation with streaming token output.
 * Reuses the full inference engine from infer.m.
 *
 * Build:  make chat
 * Run:    ./chat [--2bit] [--k N] [--cache-entries N] [--malloc-cache N]
 */

// Pull in all of infer.m except its main()
#define CHAT_MODE
#include "infer.m"

// ============================================================================
// Chat-specific constants
// ============================================================================

// Qwen3 chat template special tokens
#define IM_START_TOKEN  248006  // <|im_start|>
#define IM_END_TOKEN    248007  // <|im_end|>
#define NEWLINE_TOKEN   198     // \n

// Maximum conversation length in tokens
#define MAX_CONV_TOKENS 1048576  // 1M context — 397B full model, KV grows on demand
// Maximum input line length
#define MAX_INPUT_LINE  4096
// Maximum tokens per response
#define MAX_RESPONSE_TOKENS 1024

// ============================================================================
// Token history for multi-turn conversation
// ============================================================================

typedef struct {
    uint32_t *tokens;  // heap-allocated [MAX_CONV_TOKENS]
    int count;         // total tokens in conversation so far
    int processed;     // tokens already fed through the model
} ConversationState;

static void conv_init(ConversationState *conv) {
    conv->tokens = calloc(MAX_CONV_TOKENS, sizeof(uint32_t));
    conv->count = 0;
    conv->processed = 0;
}

static void conv_reset(ConversationState *conv) {
    conv->count = 0;
    conv->processed = 0;
}

static int conv_append(ConversationState *conv, uint32_t token) {
    if (conv->count >= MAX_CONV_TOKENS) return -1;
    conv->tokens[conv->count++] = token;
    return 0;
}

static int conv_append_tokens(ConversationState *conv, const uint32_t *toks, int n) {
    for (int i = 0; i < n; i++) {
        if (conv_append(conv, toks[i]) < 0) return -1;
    }
    return 0;
}

// ============================================================================
// Encode user message via external Python helper
// Returns token count written to out[], or -1 on error.
// ============================================================================

static int encode_user_message(const char *text, uint32_t *out, int max_out, int is_first_turn) {
    // Write the text to a temp file to avoid shell escaping issues
    const char *text_path = "/tmp/chat_input_text.txt";
    const char *tok_path = "/tmp/chat_input_tokens.bin";

    FILE *tf = fopen(text_path, "w");
    if (!tf) return -1;
    // System prompt on first turn enables thinking mode
    if (is_first_turn) {
        fprintf(tf, "<|im_start|>system\nYou are a helpful assistant. /think\nKeep your thinking concise and focused — aim for under 300 words in your <think> block before responding.<|im_end|>\n");
    }
    fprintf(tf, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", text);
    fclose(tf);

    // Call encode_prompt.py reading from file
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
             "python3 metal_infer/encode_prompt.py "
             "\"$(cat %s)\" -o %s 2>/dev/null",
             text_path, tok_path);
    int rc = system(cmd);
    if (rc != 0) {
        // Try from working directory
        snprintf(cmd, sizeof(cmd),
                 "python3 encode_prompt.py "
                 "\"$(cat %s)\" -o %s 2>/dev/null",
                 text_path, tok_path);
        rc = system(cmd);
    }
    if (rc != 0) return -1;

    // Read binary token file
    FILE *f = fopen(tok_path, "rb");
    if (!f) return -1;

    uint32_t count;
    if (fread(&count, 4, 1, f) != 1) { fclose(f); return -1; }
    if ((int)count > max_out) { fclose(f); return -1; }

    int n = (int)count;
    if (fread(out, 4, n, f) != (size_t)n) { fclose(f); return -1; }
    fclose(f);
    return n;
}

// ============================================================================
// Print usage
// ============================================================================

static void chat_print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --model PATH         Model path\n");
    printf("  --weights PATH       model_weights.bin path\n");
    printf("  --manifest PATH      model_weights.json path\n");
    printf("  --vocab PATH         vocab.bin path\n");
    printf("  --k N                Active experts per layer (default: 4)\n");
    printf("  --cache-entries N    Expert LRU cache size (default: 2500)\n");
    printf("  --malloc-cache N     Malloc expert cache entries\n");
    printf("  --2bit               Use 2-bit quantized experts\n");
    printf("  --think-budget N     Max thinking tokens (default: 2048, 0=unlimited)\n");
    printf("  --help               This message\n");
}

// ============================================================================
// Main: interactive chat loop
// ============================================================================

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *model_path = MODEL_PATH_DEFAULT;
        const char *weights_path = NULL;
        const char *manifest_path = NULL;
        const char *vocab_path = NULL;
        int K = 4;
        int cache_entries = 0;  // trust OS page cache
        int malloc_cache_entries = 0;

        static struct option long_options[] = {
            {"model",         required_argument, 0, 'm'},
            {"weights",       required_argument, 0, 'w'},
            {"manifest",      required_argument, 0, 'j'},
            {"vocab",         required_argument, 0, 'v'},
            {"k",             required_argument, 0, 'k'},
            {"cache-entries",  required_argument, 0, 'C'},
            {"malloc-cache",   required_argument, 0, 'M'},
            {"2bit",          no_argument,       0, '2'},
            {"think-budget",  required_argument, 0, 'B'},
            {"help",          no_argument,       0, 'h'},
            {0, 0, 0, 0}
        };

        int c;
        while ((c = getopt_long(argc, argv, "m:w:j:v:k:C:M:B:2h", long_options, NULL)) != -1) {
            switch (c) {
                case 'm': model_path = optarg; break;
                case 'w': weights_path = optarg; break;
                case 'j': manifest_path = optarg; break;
                case 'v': vocab_path = optarg; break;
                case 'k': K = atoi(optarg); break;
                case 'C': cache_entries = atoi(optarg); break;
                case 'M': malloc_cache_entries = atoi(optarg); break;
                case '2': g_use_2bit = 1; break;
                case 'B': g_think_budget = atoi(optarg); break;
                case 'h': chat_print_usage(argv[0]); return 0;
                default:  chat_print_usage(argv[0]); return 1;
            }
        }

        // Build default paths (try metal_infer/ prefix then cwd)
        char default_weights[1024], default_manifest[1024], default_vocab[1024];

        if (!weights_path) {
            snprintf(default_weights, sizeof(default_weights), "metal_infer/model_weights.bin");
            if (access(default_weights, R_OK) != 0)
                snprintf(default_weights, sizeof(default_weights), "model_weights.bin");
            weights_path = default_weights;
        }
        if (!manifest_path) {
            snprintf(default_manifest, sizeof(default_manifest), "metal_infer/model_weights.json");
            if (access(default_manifest, R_OK) != 0)
                snprintf(default_manifest, sizeof(default_manifest), "model_weights.json");
            manifest_path = default_manifest;
        }
        if (!vocab_path) {
            snprintf(default_vocab, sizeof(default_vocab), "metal_infer/vocab.bin");
            if (access(default_vocab, R_OK) != 0)
                snprintf(default_vocab, sizeof(default_vocab), "vocab.bin");
            vocab_path = default_vocab;
        }

        // Auto-detect: try 2-bit first if not explicitly set
        if (!g_use_2bit) {
            char probe[1024];
            snprintf(probe, sizeof(probe), "%s/packed_experts_2bit/layer_00.bin", model_path);
            int probe_fd = open(probe, O_RDONLY);
            if (probe_fd >= 0) {
                close(probe_fd);
                snprintf(probe, sizeof(probe), "%s/packed_experts/layer_00.bin", model_path);
                int probe4 = open(probe, O_RDONLY);
                if (probe4 < 0) {
                    g_use_2bit = 1;
                } else {
                    close(probe4);
                }
            }
        }

        // ---- Header ----
        printf("==================================================\n");
        printf("  Qwen3.5-397B-A17B Chat (Metal Inference Engine)\n");
        printf("==================================================\n");
        printf("  Model:   %s\n", model_path);
        printf("  Experts: K=%d, %s quantization\n", K, g_use_2bit ? "2-bit" : "4-bit");
        if (malloc_cache_entries > 0) {
            printf("  Cache:   malloc %d entries\n", malloc_cache_entries);
        } else {
            printf("  Cache:   %d entries\n", cache_entries);
        }
        printf("  Context: %d tokens max\n", MAX_CONV_TOKENS);
        printf("\n  Commands: /quit /exit /clear\n");
        printf("==================================================\n\n");
        printf("Loading model...\n");

        double t0 = now_ms();

        // ---- Initialize Metal ----
        g_metal = metal_setup();
        if (!g_metal) {
            fprintf(stderr, "WARNING: Metal init failed, falling back to CPU\n");
        }

        // ---- Initialize I/O thread pool ----
        io_pool_init();

        // ---- Initialize caches ----
        if (malloc_cache_entries > 0) {
            g_malloc_cache = malloc_cache_init(malloc_cache_entries,
                g_metal ? g_metal->device : MTLCreateSystemDefaultDevice());
            cache_entries = 0;
        }
        if (cache_entries > 0 && g_metal) {
            g_expert_cache = expert_cache_new(g_metal->device, cache_entries);
        }

        // ---- Load weights ----
        WeightFile *wf = open_weights(weights_path, manifest_path);
        if (!wf) { fprintf(stderr, "ERROR: Failed to load weights\n"); return 1; }
        if (g_metal) metal_set_weights(g_metal, wf->data, wf->size);

        // ---- Load vocabulary ----
        Vocabulary *vocab = load_vocab(vocab_path);
        if (!vocab) { fprintf(stderr, "ERROR: Failed to load vocabulary\n"); return 1; }

        // ---- Open expert layer files ----
        int layer_fds[NUM_LAYERS];
        void *layer_mmaps[NUM_LAYERS];
        size_t layer_mmap_sizes[NUM_LAYERS];
        int expert_layers_available = 0;

        for (int i = 0; i < NUM_LAYERS; i++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s/layer_%02d.bin", model_path,
                     g_use_2bit ? "packed_experts_2bit" : "packed_experts", i);
            layer_fds[i] = open(path, O_RDONLY);
            if (layer_fds[i] >= 0 && g_use_2bit) fcntl(layer_fds[i], F_NOCACHE, 1);
            layer_mmaps[i] = MAP_FAILED;
            layer_mmap_sizes[i] = 0;
            if (layer_fds[i] >= 0) {
                expert_layers_available++;
                struct stat st;
                if (fstat(layer_fds[i], &st) == 0 && st.st_size > 0) {
                    layer_mmaps[i] = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, layer_fds[i], 0);
                    if (layer_mmaps[i] != MAP_FAILED) {
                        layer_mmap_sizes[i] = st.st_size;
                        madvise(layer_mmaps[i], st.st_size, MADV_RANDOM);
                    }
                }
            }
        }
        printf("[experts] %d/%d layer files available\n", expert_layers_available, NUM_LAYERS);

        // Warm page cache
        if (expert_layers_available > 0) {
            for (int i = 0; i < NUM_LAYERS; i++) {
                if (layer_fds[i] >= 0) {
                    char dummy[4096];
                    pread(layer_fds[i], dummy, sizeof(dummy), 0);
                }
            }
        }

        // ---- Allocate per-layer state ----
        void **layer_states = calloc(NUM_LAYERS, sizeof(void *));
        KVCache **kv_caches = calloc(NUM_LAYERS, sizeof(KVCache *));

        for (int i = 0; i < NUM_LAYERS; i++) {
            int is_full = ((i + 1) % FULL_ATTN_INTERVAL == 0);
            if (is_full)
                kv_caches[i] = kv_cache_new();
            else
                layer_states[i] = linear_attn_state_new();
        }

        // ---- Working buffers ----
        float *hidden = calloc(HIDDEN_DIM, sizeof(float));
        float *logits = calloc(VOCAB_SIZE, sizeof(float));
        uint16_t *final_norm_w = get_tensor_ptr(wf, "model.norm.weight");

        // ---- Conversation state ----
        ConversationState conv;
        conv_init(&conv);
        int pos = 0;  // RoPE position counter

        reset_delta_net_state();

        // ---- Cache system prompt: prefill once, snapshot state ----
        // Encode system prompt tokens
        const char *sys_prompt =
            "<|im_start|>system\nYou are a helpful assistant. /think\n"
            "Keep your thinking concise and focused — aim for under 300 words "
            "in your <think> block before responding.<|im_end|>\n";
        init_tokenizer();
        uint32_t sys_ids[256];
        int sys_n = bpe_encode(&g_tokenizer, sys_prompt, sys_ids, 256);
        if (sys_n > 0) {
            fprintf(stderr, "[chat] Pre-caching system prompt (%d tokens)...\n", sys_n);
            // Batch embed
            float *sys_embeds = malloc((size_t)sys_n * HIDDEN_DIM * sizeof(float));
            for (int i = 0; i < sys_n; i++)
                embed_lookup(wf, sys_ids[i], sys_embeds + (size_t)i * HIDDEN_DIM);
            // Prefill with discard for intermediate tokens
            for (int i = 0; i < sys_n; i++) {
                memcpy(hidden, sys_embeds + (size_t)i * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));
                for (int layer = 0; layer < NUM_LAYERS; layer++) {
                    int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        K, layer_fds[layer]);
                }
                if (i < sys_n - 1)
                    discard_deferred_experts();
                else
                    complete_deferred_experts();
                pos++;
            }
            free(sys_embeds);
            // Add system tokens to conversation history
            for (int i = 0; i < sys_n; i++) conv_append(&conv, sys_ids[i]);
            conv.processed = conv.count;
            fprintf(stderr, "[chat] System prompt cached (pos=%d)\n", pos);
        }

        // Snapshot state after system prompt for /clear restoration
        int sys_prompt_pos = pos;
        int sys_prompt_conv_count = conv.count;
        size_t kv_dim = NUM_KV_HEADS * HEAD_DIM;
        size_t conv_state_sz = (CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM * sizeof(float);
        size_t ssm_state_sz = LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float);

        // KV cache snapshots
        float *kv_k_snaps[NUM_LAYERS], *kv_v_snaps[NUM_LAYERS];
        int kv_lens[NUM_LAYERS];
        memset(kv_k_snaps, 0, sizeof(kv_k_snaps));
        memset(kv_v_snaps, 0, sizeof(kv_v_snaps));
        for (int i = 0; i < NUM_LAYERS; i++) {
            if (kv_caches[i] && kv_caches[i]->len > 0) {
                size_t sz = kv_caches[i]->len * kv_dim * sizeof(float);
                kv_k_snaps[i] = malloc(sz);
                kv_v_snaps[i] = malloc(sz);
                memcpy(kv_k_snaps[i], kv_caches[i]->k_cache, sz);
                memcpy(kv_v_snaps[i], kv_caches[i]->v_cache, sz);
                kv_lens[i] = kv_caches[i]->len;
            }
        }
        // Linear attention snapshots
        float *la_conv_snaps[NUM_LAYERS], *la_ssm_snaps[NUM_LAYERS];
        memset(la_conv_snaps, 0, sizeof(la_conv_snaps));
        memset(la_ssm_snaps, 0, sizeof(la_ssm_snaps));
        for (int i = 0; i < NUM_LAYERS; i++) {
            if (layer_states[i]) {
                LinearAttnState *s = (LinearAttnState *)layer_states[i];
                la_conv_snaps[i] = malloc(conv_state_sz);
                la_ssm_snaps[i] = malloc(ssm_state_sz);
                memcpy(la_conv_snaps[i], s->conv_state, conv_state_sz);
                memcpy(la_ssm_snaps[i], s->ssm_state, ssm_state_sz);
            }
        }

        double t_init = now_ms();
        printf("\nModel loaded in %.1f s. Ready to chat.\n\n", (t_init - t0) / 1000.0);

        // ============================================================
        // Interactive chat loop
        // ============================================================

        char input_line[MAX_INPUT_LINE];

        for (;;) {
            // Prompt
            printf("> ");
            fflush(stdout);

            // Read user input
            if (!fgets(input_line, sizeof(input_line), stdin)) {
                printf("\n");
                break;  // EOF
            }

            // Strip trailing newline
            size_t len = strlen(input_line);
            while (len > 0 && (input_line[len-1] == '\n' || input_line[len-1] == '\r'))
                input_line[--len] = '\0';

            // Skip empty lines
            if (len == 0) continue;

            // Handle commands
            if (strcmp(input_line, "/quit") == 0 || strcmp(input_line, "/exit") == 0) {
                printf("Goodbye.\n");
                break;
            }

            if (strcmp(input_line, "/clear") == 0) {
                // Restore to system prompt snapshot (not zero)
                conv_reset(&conv);
                // Re-add system prompt tokens to conversation
                for (int i = 0; i < sys_prompt_conv_count && i < sys_n; i++)
                    conv_append(&conv, sys_ids[i]);
                conv.processed = sys_prompt_conv_count;
                pos = sys_prompt_pos;
                // Restore KV caches
                for (int i = 0; i < NUM_LAYERS; i++) {
                    if (kv_caches[i] && kv_k_snaps[i]) {
                        size_t sz = kv_lens[i] * kv_dim * sizeof(float);
                        memcpy(kv_caches[i]->k_cache, kv_k_snaps[i], sz);
                        memcpy(kv_caches[i]->v_cache, kv_v_snaps[i], sz);
                        kv_caches[i]->len = kv_lens[i];
                    } else if (kv_caches[i]) {
                        kv_caches[i]->len = 0;
                    }
                    if (layer_states[i] && la_conv_snaps[i]) {
                        LinearAttnState *s = (LinearAttnState *)layer_states[i];
                        memcpy(s->conv_state, la_conv_snaps[i], conv_state_sz);
                        memcpy(s->ssm_state, la_ssm_snaps[i], ssm_state_sz);
                    } else if (layer_states[i]) {
                        LinearAttnState *s = (LinearAttnState *)layer_states[i];
                        memset(s->conv_state, 0, conv_state_sz);
                        memset(s->ssm_state, 0, ssm_state_sz);
                    }
                }
                reset_delta_net_state();
                printf("[cleared] Conversation reset to system prompt.\n\n");
                continue;
            }

            // ---- Encode user message to tokens ----
            uint32_t *msg_tokens = malloc(MAX_CONV_TOKENS * sizeof(uint32_t));
            // System prompt is pre-cached — never include it in user turns
            int n_msg = encode_user_message(input_line, msg_tokens, MAX_CONV_TOKENS - conv.count, 0);
            if (n_msg < 0) {
                fprintf(stderr, "[error] Failed to tokenize input. Is encode_prompt.py available?\n\n");
                free(msg_tokens);
                continue;
            }

            // Check context overflow
            if (conv.count + n_msg + MAX_RESPONSE_TOKENS > MAX_CONV_TOKENS) {
                fprintf(stderr, "[error] Context full (%d tokens). Use /clear to reset.\n\n",
                        conv.count + n_msg);
                free(msg_tokens);
                continue;
            }

            // Append user tokens to conversation
            conv_append_tokens(&conv, msg_tokens, n_msg);
            free(msg_tokens);

            // ---- Prefill: process all new tokens ----
            double t_prefill = now_ms();
            int prefill_count = conv.count - conv.processed;

            // Batch prefill: pre-embed all tokens, use discard for intermediates
            float *chat_embed_batch = NULL;
            if (prefill_count > 1) {
                chat_embed_batch = malloc((size_t)prefill_count * HIDDEN_DIM * sizeof(float));
                for (int i = 0; i < prefill_count; i++) {
                    embed_lookup(wf, conv.tokens[conv.processed + i],
                                 chat_embed_batch + (size_t)i * HIDDEN_DIM);
                }
            }
            // Intermediate prefill tokens: discard last-layer expert output
            for (int i = conv.processed; i < conv.count - 1; i++) {
                int bi = i - conv.processed;
                if (chat_embed_batch) {
                    memcpy(hidden, chat_embed_batch + (size_t)bi * HIDDEN_DIM,
                           HIDDEN_DIM * sizeof(float));
                } else {
                    embed_lookup(wf, conv.tokens[i], hidden);
                }

                for (int layer = 0; layer < NUM_LAYERS; layer++) {
                    int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        K, layer_fds[layer]);
                }
                discard_deferred_experts();
                pos++;

                fprintf(stderr, "  [prefill] %d/%d\r", i - conv.processed + 1, prefill_count);
            }
            // Last prefill token: full completion (need hidden for logits)
            {
                int last_i = conv.count - 1;
                int bi = last_i - conv.processed;
                if (chat_embed_batch) {
                    memcpy(hidden, chat_embed_batch + (size_t)bi * HIDDEN_DIM,
                           HIDDEN_DIM * sizeof(float));
                } else {
                    embed_lookup(wf, conv.tokens[last_i], hidden);
                }

                for (int layer = 0; layer < NUM_LAYERS; layer++) {
                    int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        K, layer_fds[layer]);
                }
                complete_deferred_experts();
                pos++;
            }
            if (chat_embed_batch) { free(chat_embed_batch); chat_embed_batch = NULL; }
            conv.processed = conv.count;

            double prefill_ms = now_ms() - t_prefill;
            fprintf(stderr, "  [prefill] %d tokens in %.1f ms (%.1f tok/s)    \n",
                    prefill_count, prefill_ms,
                    prefill_count > 0 ? prefill_count * 1000.0 / prefill_ms : 0.0);

            // ---- Final norm + LM head for first generated token ----
            if (final_norm_w) {
                float *normed = malloc(HIDDEN_DIM * sizeof(float));
                cpu_rms_norm(hidden, final_norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);
                memcpy(hidden, normed, HIDDEN_DIM * sizeof(float));
                free(normed);
            }
            lm_head_forward(wf, hidden, logits);
            int next_token = cpu_argmax(logits, VOCAB_SIZE);

            // ---- Stream generation ----
            double t_gen_start = now_ms();
            int gen_count = 0;
            int in_think = 0;
            int think_tokens = 0;

            printf("\n");
            for (int gen = 0; gen < MAX_RESPONSE_TOKENS; gen++) {
                // Check EOS
                if (next_token == EOS_TOKEN_1 || next_token == EOS_TOKEN_2) break;
                if (next_token == IM_END_TOKEN) break;

                // Think budget enforcement
                if (next_token == THINK_START_TOKEN) in_think = 1;
                if (next_token == THINK_END_TOKEN) in_think = 0;
                if (in_think) think_tokens++;

                // Check context overflow
                if (conv.count + 1 >= MAX_CONV_TOKENS) {
                    fprintf(stderr, "\n[warning] Context limit reached.\n");
                    break;
                }

                // Print token immediately
                printf("%s", decode_token(vocab, next_token));
                fflush(stdout);

                // Track in conversation
                conv_append(&conv, (uint32_t)next_token);
                conv.processed = conv.count;
                gen_count++;

                // Generate next token
                embed_lookup(wf, next_token, hidden);

                for (int layer = 0; layer < NUM_LAYERS; layer++) {
                    int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        K, layer_fds[layer]);
                }
                complete_deferred_experts();
                pos++;

                if (final_norm_w) {
                    float *normed = malloc(HIDDEN_DIM * sizeof(float));
                    cpu_rms_norm(hidden, final_norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);
                    memcpy(hidden, normed, HIDDEN_DIM * sizeof(float));
                    free(normed);
                }
                lm_head_forward(wf, hidden, logits);
                next_token = cpu_argmax(logits, VOCAB_SIZE);

                // Think budget: force end thinking if over budget
                if (in_think && g_think_budget > 0 && think_tokens >= g_think_budget) {
                    next_token = THINK_END_TOKEN;
                    in_think = 0;
                }
            }

            // Append EOS to conversation history
            conv_append(&conv, (uint32_t)IM_END_TOKEN);
            conv.processed = conv.count;

            double gen_ms = now_ms() - t_gen_start;
            double tok_s = gen_count > 0 ? gen_count * 1000.0 / gen_ms : 0.0;

            printf("\n\n[%d tokens, %.2f tok/s, %.1fs]\n\n",
                   gen_count, tok_s, gen_ms / 1000.0);
        }

        // ---- Cleanup ----
        io_pool_shutdown();
        if (g_malloc_cache) { malloc_cache_free(g_malloc_cache); g_malloc_cache = NULL; }
        if (g_expert_cache) { expert_cache_free(g_expert_cache); g_expert_cache = NULL; }
        for (int i = 0; i < NUM_LAYERS; i++) {
            if (kv_caches[i]) kv_cache_free(kv_caches[i]);
            if (layer_states[i]) linear_attn_state_free(layer_states[i]);
            if (layer_mmaps[i] != MAP_FAILED) munmap(layer_mmaps[i], layer_mmap_sizes[i]);
            if (layer_fds[i] >= 0) close(layer_fds[i]);
        }
        free(layer_states);
        free(kv_caches);
        free(hidden);
        free(logits);

        return 0;
    }
}
