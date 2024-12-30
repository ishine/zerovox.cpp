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

    #define HPARAM_AUDIO_NUM_MELS           "zerovox-resnet-fs2-styletts.audio.num_mels"

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

        uint32_t audio_num_mels;
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

    class FS2Encoder
    {
        public:

            FS2Encoder(ggml_context   &ctx_w,
                       ggml_backend_t  backend,
                       uint32_t        max_n_phonemes,
                       uint32_t        embed_dim,
                       uint32_t        punct_embed_dim,
                       uint32_t        encoder_layer,
                       uint32_t        encoder_head,
                       uint32_t        conv_filter_size,
                       uint32_t        conv_kernel_size[2],
                       uint32_t        vp_kernel_size,
                       uint32_t        ve_n_bins,
                       uint32_t        max_seq_len);
            ~FS2Encoder();

            uint32_t eval(const int32_t *src_seq_data, const int32_t *puncts_data, const float *style_embed_data, uint32_t num_phonemes, float *x);

        private:

            Encoder             encoder;

            VariancePredictor   duration_predictor;
            VariancePredictor   pitch_predictor;
            VariancePredictor   energy_predictor;

            struct ggml_tensor *pitch_embedding_w;
            struct ggml_tensor *energy_embedding_w;

            uint32_t            max_n_phonemes;
            uint32_t            embed_dim;
            uint32_t            punct_embed_dim;
            uint32_t            max_seq_len;
            uint32_t            ve_n_bins;

            // graph
            struct ggml_cgraph *gf;
            ggml_backend_t      backend;
            ggml_gallocr_t      alloc;

            // inputs
            struct ggml_tensor *src_seq; 
            struct ggml_tensor *puncts; 
            struct ggml_tensor *style_embed; 

            // outputs
            struct ggml_tensor *features;
            struct ggml_tensor *log_duration_prediction;
    };

    class ResBlk1d
    {
        public:
            ResBlk1d(ggml_context  &ctx_w,
                     int            idx,
                     uint32_t       dim_in,
                     uint32_t       dim_out);

            struct ggml_tensor *graph(struct ggml_cgraph *gf, ggml_context *ctx, struct ggml_tensor *x);

        private:

            uint32_t       dim_in;
            uint32_t       dim_out;
            bool           learned_sc;

            struct ggml_tensor *conv1_b;
            // struct ggml_tensor *conv1_w_g;
            // struct ggml_tensor *conv1_w_v;
            struct ggml_tensor *conv1_w;
            // struct ggml_tensor *conv1x1_w_g;
            // struct ggml_tensor *conv1x1_w_v;
            struct ggml_tensor *conv1x1_w;
            struct ggml_tensor *conv2_b;
            // struct ggml_tensor *conv2_w_g;
            // struct ggml_tensor *conv2_w_v;
            struct ggml_tensor *conv2_w;
            struct ggml_tensor *norm1_b;
            struct ggml_tensor *norm1_w;
            struct ggml_tensor *norm2_b;
            struct ggml_tensor *norm2_w;
    };

    class AdaIN1d
    {
        public:

            AdaIN1d(ggml_context   &ctx_w,
                    ggml_backend_t  backend,
                    int             idx0,
                    int             idx1,
                    uint32_t        style_dim,
                    uint32_t        num_features);

            struct ggml_tensor *graph(struct ggml_cgraph *gf, ggml_context *ctx, struct ggml_tensor *x, struct ggml_tensor *s, struct ggml_tensor *one);

        private:
            uint32_t            style_dim;
            uint32_t            num_features;

            struct ggml_tensor *fc_b;
            struct ggml_tensor *fc_w;
    };


    class AdainResBlk1d
    {
        public:

            AdainResBlk1d(ggml_context   &ctx_w,
                          ggml_backend_t  backend,
                          int             idx,
                          uint32_t        dim_in,
                          uint32_t        dim_out,
                          uint32_t        style_dim);

            struct ggml_tensor *graph(struct ggml_cgraph *gf, ggml_context *ctx, struct ggml_tensor *x, struct ggml_tensor *s, struct ggml_tensor *one);

        private:

            uint32_t        dim_in;
            uint32_t        dim_out;
            uint32_t        style_dim;
            bool            learned_sc;

            AdaIN1d         norm1;
            AdaIN1d         norm2;

            struct ggml_tensor *conv1_b;
            struct ggml_tensor *conv1_w;
            struct ggml_tensor *conv1x1_w;
            struct ggml_tensor *conv2_b;
            struct ggml_tensor *conv2_w;
    };

    class StyleTTSDecoder
    {
        public:

            StyleTTSDecoder(ggml_context   &ctx_w,
                            ggml_backend_t  backend,
                            uint32_t        max_seq_len,
                            uint32_t        dim_in,
                            uint32_t        style_dim,
                            uint32_t        residual_dim,
                            uint32_t        dim_out);
            ~StyleTTSDecoder();

            void eval(float *enc_seq_data, float *spk_emb_data);

        private:

            uint32_t            max_seq_len;
            uint32_t            style_dim;
            uint32_t            dim_in;

            ResBlk1d            encode0, encode1;

            struct ggml_tensor *asr_res_0_b;
            struct ggml_tensor *asr_res_0_w;
            struct ggml_tensor *asr_res_1_b;
            struct ggml_tensor *asr_res_1_w;

            AdainResBlk1d       decode0;

            // graph
            struct ggml_cgraph *gf;
            ggml_backend_t      backend;
            ggml_gallocr_t      alloc;

            // inputs
            struct ggml_tensor *enc_seq; 
            struct ggml_tensor *spk_emb; 
            struct ggml_tensor *one; 

            // output
            struct ggml_tensor *x;
    };

    class ZeroVOXModel
    {
        public:
            ZeroVOXModel(const std::string & fname);
            ~ZeroVOXModel();

            void eval(void);

        private:

            zerovox_hparams        hparams;

            FS2Encoder            *encoder;
            StyleTTSDecoder       *decoder;

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