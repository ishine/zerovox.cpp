#pragma once

#include <cinttypes>
#include <string>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

namespace ZeroVOX
{

    #define HPARAM_MAX_SEQ_LEN              "zerovox-resnet-fs2-styletts.max_seq_len"
    #define HPARAM_EMB_DIM                  "zerovox-resnet-fs2-styletts.emb_dim"
    #define HPARAM_PUNCT_EMB_DIM            "zerovox-resnet-fs2-styletts.punct_emb_dim"
    #define HPARAM_DECODER_N_HEAD           "zerovox-resnet-fs2-styletts.decoder.n_head"
    #define HPARAM_CONV_FILTER_SIZE         "zerovox-resnet-fs2-styletts.decoder.conv_filter_size"
    #define HPARAM_CONV_KERNEL_SIZE_0       "zerovox-resnet-fs2-styletts.decoder.conv_kernel_size.0"
    #define HPARAM_CONV_KERNEL_SIZE_1       "zerovox-resnet-fs2-styletts.decoder.conv_kernel_size.1"

    #define HPARAM_ENCODER_LAYER            "zerovox-resnet-fs2-styletts.encoder.layer"
    #define HPARAM_ENCODER_HEAD             "zerovox-resnet-fs2-styletts.encoder.head"
    #define HPARAM_ENCODER_VP_FILTER_SIZE   "zerovox-resnet-fs2-styletts.encoder.vp_filter_size"
    #define HPARAM_ENCODER_VP_KERNEL_SIZE   "zerovox-resnet-fs2-styletts.encoder.vp_kernel_size"
    #define HPARAM_ENCODER_VE_N_BINS        "zerovox-resnet-fs2-styletts.encoder.ve_n_bins"

    // #define HPARAM_STATS_ENERGY_MAX         "zerovox-resnet-fs2-styletts.stats.energy_max"
    // #define HPARAM_STATS_ENERGY_MIN         "zerovox-resnet-fs2-styletts.stats.energy_min"
    // #define HPARAM_STATS_PITCH_MAX          "zerovox-resnet-fs2-styletts.stats.pitch_max"
    // #define HPARAM_STATS_PITCH_MIN          "zerovox-resnet-fs2-styletts.stats.pitch_min"

    const int NUM_PHONEMES   = 154;
    const int NUM_PUNCTS     =   6;
    const int MAX_N_PHONEMES =  11;

    struct zerovox_hparams
    {
        uint32_t max_seq_len;
        uint32_t emb_dim;
        uint32_t punct_emb_dim;

        uint32_t decoder_n_head;
        uint32_t conv_filter_size;
        uint32_t conv_kernel_size[2];

        uint32_t encoder_layer;
        uint32_t encoder_head;
        uint32_t encoder_vp_filter_size;
        uint32_t encoder_vp_kernel_size;
        uint32_t encoder_ve_n_bins;
    };

    class MultiHeadAttention
    {
        public:
            MultiHeadAttention(ggml_context &ctx, int layidx, uint32_t n_head, uint32_t d_k, uint32_t d_v);

            struct ggml_tensor *graph(struct ggml_cgraph *gf, ggml_context *ctx, struct ggml_tensor *q, struct ggml_tensor *k, struct ggml_tensor *v);

        private:

            struct ggml_tensor *w_ks_b;
            struct ggml_tensor *w_ks_w;

            struct ggml_tensor *w_qs_b;
            struct ggml_tensor *w_qs_w;

            struct ggml_tensor *w_vs_b;
            struct ggml_tensor *w_vs_w;

            struct ggml_tensor *layer_norm_b;
            struct ggml_tensor *layer_norm_w;

            struct ggml_tensor *fc_b;
            struct ggml_tensor *fc_w;

            int   layidx, n_head, d_k, d_v;
            float temperature;
    };

    class PositionwiseFeedForward
    {
        public:
            PositionwiseFeedForward(ggml_context &ctx, int layidx, uint32_t d_in, uint32_t d_hid, uint32_t kernel_size[2]);

            struct ggml_tensor *graph(struct ggml_cgraph *gf, ggml_context *ctx, struct ggml_tensor *x);

        private:
            struct ggml_tensor *layer_norm_b;
            struct ggml_tensor *layer_norm_w;
            struct ggml_tensor *w_1_b;
            struct ggml_tensor *w_1_w;
            struct ggml_tensor *w_2_b;
            struct ggml_tensor *w_2_w;

            int layidx;
            uint32_t d_in;
            uint32_t d_hid;
            uint32_t kernel_size[2];

    };

    class FFTBlock
    {
        public:

            FFTBlock(ggml_context &ctx, int layidx, uint32_t d_model, uint32_t n_head, uint32_t d_k, uint32_t d_v, uint32_t d_inner, uint32_t kernel_size[2]);

            struct ggml_tensor * graph(struct ggml_cgraph *gf, ggml_context *ctx, struct ggml_tensor *enc_input);

        private:
            MultiHeadAttention slf_attn;
            PositionwiseFeedForward pos_ffn;
    };

    class Encoder
    {
        public:

            Encoder(ggml_context &ctx, uint32_t max_n_phonemes, uint32_t embed_dim, uint32_t encoder_layer, uint32_t encoder_head, uint32_t conv_filter_size, uint32_t conv_kernel_size[2], uint32_t punct_embed_dim);
            ~Encoder();

            struct ggml_tensor * graph(struct ggml_cgraph *gf, ggml_context *ctx, struct ggml_tensor *src_seq, struct ggml_tensor *puncts);

        private:

            struct ggml_tensor *src_word_emb_w;
            struct ggml_tensor *src_punct_embed_w;
            struct ggml_tensor *sinusoid_encoding_table;// (1501, 528)

            FFTBlock **layer_stack;

            uint32_t max_n_phonemes;
            uint32_t embed_dim;
            uint32_t punct_embed_dim;
            uint32_t encoder_layer;
    };

    class VariancePredictor
    {
        public:

            VariancePredictor(ggml_context &ctx, const char *prefix, uint32_t vp_kernel_size);
            //~VariancePredictor();

            struct ggml_tensor *graph(struct ggml_cgraph *gf, ggml_context *ctx, struct ggml_tensor *encoder_output);

        private:

            struct ggml_tensor *conv1d_1_conv_b;
            struct ggml_tensor *conv1d_1_conv_w;
            struct ggml_tensor *conv1d_2_conv_b;
            struct ggml_tensor *conv1d_2_conv_w;

            struct ggml_tensor *layer_norm_1_b;
            struct ggml_tensor *layer_norm_1_w;
            struct ggml_tensor *layer_norm_2_b;
            struct ggml_tensor *layer_norm_2_w;

            struct ggml_tensor *linear_layer_b;
            struct ggml_tensor *linear_layer_w;

            uint32_t vp_kernel_size;
    };

#if 0
    class VarianceAdaptor
    {
        public:

            VarianceAdaptor(ggml_context &ctx,
                            int    emb_size,
                            int    vp_filter_size,
                            int    vp_kernel_size,
                            int    ve_n_bins,
                            int    max_seq_len
                            );
            //~VarianceAdaptor();

            struct ggml_tensor * graph(struct ggml_cgraph *gf, ggml_context *ctx, struct ggml_tensor *x,
                                    struct ggml_tensor *pitch_min, struct ggml_tensor *pitch_range);

            int               ve_n_bins;

        private:

            VariancePredictor duration_predictor;
            VariancePredictor pitch_predictor;
            VariancePredictor energy_predictor;

            struct ggml_tensor *pitch_embedding_w;
            struct ggml_tensor *energy_embedding_w;

            int max_seq_len;
            int emb_size;
    };
#endif

    class FS2Encoder
    {
        public:

            FS2Encoder(ggml_context &ctx,
                           uint32_t      max_n_phonemes,
                           uint32_t      embed_dim,
                           uint32_t      punct_embed_dim,
                           uint32_t      encoder_layer,
                           uint32_t      encoder_head,
                           uint32_t      conv_filter_size,
                           uint32_t      conv_kernel_size[2],
                           uint32_t      vp_kernel_size,
                           uint32_t      ve_n_bins,
                           uint32_t      max_seq_len);

        private:

            Encoder             encoder;

            VariancePredictor   duration_predictor;
            VariancePredictor   pitch_predictor;
            VariancePredictor   energy_predictor;

            struct ggml_tensor *pitch_embedding_w;
            struct ggml_tensor *energy_embedding_w;

            int                 max_seq_len;
            int                 ve_n_bins;

            // graph

            struct ggml_cgraph *gf;
            struct ggml_tensor *src_seq;
            struct ggml_tensor *puncts;
            struct ggml_tensor *style_embed;
            // struct ggml_tensor *pitch_min;
            // struct ggml_tensor *pitch_range;
    };

    class ZeroVOXModel
    {
        public:
            ZeroVOXModel(const std::string & fname);
            ~ZeroVOXModel();

        private:

            zerovox_hparams        hparams;

            FS2Encoder            *encoder;

            ggml_backend_t         backend;
            ggml_backend_buffer_t  buf_w;
            struct ggml_context   *ctx_w;
    };


    // utils

    #define die(msg)          do { fputs("error: " msg "\n", stderr);                exit(1); } while (0)
    #define die_fmt(fmt, ...) do { fprintf(stderr, "error: " fmt "\n", __VA_ARGS__); exit(1); } while (0)

    struct ggml_tensor *checked_get_tensor(struct ggml_context *ctx, const char *name);
    void tensor_dbg(struct ggml_cgraph *gf, struct ggml_context *ctx, struct ggml_tensor *x, const char *name);
    void print_tensor(const char* title, struct ggml_tensor * t, int n = 3);

    #define GGUF_GET_KEY(ctx, dst, func, type, req, key) \
    { \
        const std::string skey(key); \
        const int kid = gguf_find_key(ctx, skey.c_str()); \
        if (kid >= 0) { \
            enum gguf_type ktype = gguf_get_kv_type(ctx, kid); \
            if (ktype != (type)) { \
                die_fmt("key %s has wrong type: %s", skey.c_str(), gguf_type_name(ktype)); \
            } \
            (dst) = func(ctx, kid); \
        } else if (req) { \
            die_fmt("key not found in model: %s", skey.c_str()); \
        } \
    }

}