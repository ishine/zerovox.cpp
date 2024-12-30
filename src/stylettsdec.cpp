// original python source:
// https://github.com/yl4579/StyleTTS
// by Aaron (Yinghao) Li (Columbia University) under MIT license

#include <cstdio>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <stdexcept>

#include "zerovox.h"

namespace ZeroVOX
{

    ResBlk1d::ResBlk1d(ggml_context  &ctx_w,
                       int            idx,
                       uint32_t       dim_in,
                       uint32_t       dim_out)
    {
        this->dim_in = dim_in;
        this->dim_out = dim_out;

        this->learned_sc = dim_in != dim_out;

        char namebuf[GGML_MAX_NAME];


        // std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.encode.%d.conv1.w_g", idx);
        // conv1_w_g = checked_get_tensor(&ctx_w, namebuf);
        // std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.encode.%d.conv1.w_v", idx);
        // conv1_w_v = checked_get_tensor(&ctx_w, namebuf);
        std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.encode.%d.conv1.w", idx);
        conv1_w = checked_get_tensor(&ctx_w, namebuf);
        std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.encode.%d.conv1.b", idx);
        conv1_b = checked_get_tensor(&ctx_w, namebuf);

        if (learned_sc)
        {
            // std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.encode.%d.conv1x1.w_g", idx);
            // conv1x1_w_g = checked_get_tensor(&ctx_w, namebuf);
            // std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.encode.%d.conv1x1.w_v", idx);
            // conv1x1_w_v = checked_get_tensor(&ctx_w, namebuf);
            std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.encode.%d.conv1x1.w", idx);
            conv1x1_w = checked_get_tensor(&ctx_w, namebuf);
        }

        std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.encode.%d.conv2.b", idx);
        conv2_b = checked_get_tensor(&ctx_w, namebuf);

        // std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.encode.%d.conv2.w_g", idx);
        // conv2_w_g = checked_get_tensor(&ctx_w, namebuf);
        // std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.encode.%d.conv2.w_v", idx);
        // conv2_w_v = checked_get_tensor(&ctx_w, namebuf);
        std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.encode.%d.conv2.w", idx);
        conv2_w = checked_get_tensor(&ctx_w, namebuf);

        std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.encode.%d.norm1.w", idx);
        norm1_w = checked_get_tensor(&ctx_w, namebuf);
        std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.encode.%d.norm1.b", idx);
        norm1_b = checked_get_tensor(&ctx_w, namebuf);

        std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.encode.%d.norm2.w", idx);
        norm2_w = checked_get_tensor(&ctx_w, namebuf);
        std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.encode.%d.norm2.b", idx);
        norm2_b = checked_get_tensor(&ctx_w, namebuf);
    }

#if 0
    ggml_tensor* instance_normalization(ggml_context *ctx, ggml_tensor *x)
    {
        int64_t num_features = x->ne[0]; // Assuming the input tensor is in the format (num_features, ...)
        int64_t num_samples = 1;
        int n_dims = ggml_n_dims(x);
        for (int i = 1; i < n_dims; ++i)
        {
            num_samples *= x->ne[i];
        }

        // Calculate mean and variance
        ggml_tensor *mean = ggml_mean(ctx, x);
        ggml_tensor *variance = ggml_sqr(ctx, ggml_sub(ctx, x, mean));
        variance = ggml_mean(ctx, variance);

        // Add epsilon for numerical stability
        const float epsilon = 1e-5f;
        variance = ggml_add(ctx, variance, ggml_new_f32(ctx, epsilon));

        // Compute the inverse standard deviation
        ggml_tensor *inv_std_dev = ggml_sqrt(ctx, variance);
        inv_std_dev = ggml_recip(ctx, inv_std_dev);

        // Normalize the input tensor
        ggml_tensor *normalized_x = ggml_sub(ctx, x, mean);
        normalized_x = ggml_mul(ctx, normalized_x, inv_std_dev);

        return normalized_x;
    }
#endif

    struct ggml_tensor *ResBlk1d::graph([[maybe_unused]] struct ggml_cgraph *gf, [[maybe_unused]] ggml_context *ctx, struct ggml_tensor *x)
    {
        //x = self._shortcut(x) + self._residual(x)

        x = ggml_cont(ctx, x);

        struct ggml_tensor *y = x;

        if (learned_sc)
        {
            //y = ggml_cont(ctx, ggml_transpose(ctx, y));
            struct ggml_tensor *kernel = conv1x1_w; //  ggml_cont(ctx, ggml_transpose(ctx, conv1x1_w));
            y = ggml_conv_1d(ctx,
                             kernel,   // convolution kernel
                             y,           // data
                             1,           // stride
                             0,           // padding
                             1);          // dilation
        }

        // y += self._residual(x)

        // self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
        // x = self.norm1(x)
        //x = ggml_cont(ctx, ggml_transpose(ctx, x));
        x = ggml_norm(ctx, x, 1e-5);
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
        //x = ggml_mul(ctx, norm1_w, x);
        x = ggml_mul(ctx, x, norm1_w);
        x = ggml_add(ctx, x, norm1_b);

        x = ggml_cont(ctx, ggml_transpose(ctx, x));

        // actv=nn.LeakyReLU(0.2)
        // x = self.actv(x)
        x = ggml_leaky_relu(ctx, x, 0.2, /*inplace=*/true);

        // self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_in, 3, 1, 1))
        // x = self.conv1(x)
        x = ggml_conv_1d(ctx,
                         conv1_w,     // convolution kernel
                         x,           // data
                         1,           // stride
                         1,           // padding
                         1);          // dilation
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
        x = ggml_add(ctx, x, ggml_repeat(ctx, conv1_b, x));
        x = ggml_cont(ctx, ggml_transpose(ctx, x));

        // x = self.norm2(x)
        x = ggml_norm(ctx, x, 1e-5);
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
        x = ggml_mul(ctx, x, norm2_w);
        x = ggml_add(ctx, x, norm2_b);

        x = ggml_cont(ctx, ggml_transpose(ctx, x));

        // actv=nn.LeakyReLU(0.2)
        // x = self.actv(x)
        x = ggml_leaky_relu(ctx, x, 0.2, /*inplace=*/true);
        
        // self.conv2 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        // x = self.conv2(x)
        x = ggml_conv_1d(ctx,
                         conv2_w,     // convolution kernel
                         x,           // data
                         1,           // stride
                         1,           // padding
                         1);          // dilation
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
        x = ggml_add(ctx, x, ggml_repeat(ctx, conv2_b, x));
        x = ggml_cont(ctx, ggml_transpose(ctx, x));

        //x = self._shortcut(x) + self._residual(x)
        y = ggml_add(ctx, x, y);

        //return x / math.sqrt(2)  # unit variance
        y = ggml_scale_inplace(ctx, y, 1.0/sqrt(2.0));

        return y;
    }

    StyleTTSDecoder::StyleTTSDecoder(ggml_context   &ctx_w,
                                     [[maybe_unused]]ggml_backend_t  backend,
                                     uint32_t        max_seq_len,
                                     uint32_t        dim_in,
                                     uint32_t        style_dim,
                                     [[maybe_unused]]uint32_t        residual_dim,
                                     [[maybe_unused]]uint32_t        dim_out) :
        encode0(ctx_w, 0, dim_in, dim_in*2),
        encode1(ctx_w, 1, dim_in*2, dim_in*2)
    {
        this->max_seq_len = max_seq_len;
        this->style_dim   = style_dim;
        this->dim_in      = dim_in;

        //uint32_t bottleneck_dim = dim_in * 2;
        //        self.encode = nn.Sequential(ResBlk1d(dim_in, self.bottleneck_dim, normalize=True),
        //                                    ResBlk1d(self.bottleneck_dim, self.bottleneck_dim, normalize=True))

        this->backend      = backend;

        alloc              = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

        // pitch_embedding_w  = checked_get_tensor(&ctx_w, "_pe._var_adapt.pitch_embedding.w");
        // energy_embedding_w = checked_get_tensor(&ctx_w, "_pe._var_adapt.energy_embedding.w");

        // build graph

        //struct ggml_cgraph * zerovox_graph(const zerovox_model & model, int max_n_phonemes) {

        //const auto & hparams = model.hparams;

        // FIXME: size
        static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params params = {
            /*.mem_size   =*/ buf_size,
            /*.mem_buffer =*/ buf.data(),
            /*.no_alloc   =*/ true,
        };

        struct ggml_context *ctx = ggml_init(params);

        gf = ggml_new_graph(ctx);

        // x = self.src_word_emb(src_seq) # [115, 528]
        enc_seq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim_in, max_seq_len); 
        ggml_set_name(enc_seq, "enc_seq");
        ggml_set_input(enc_seq);

        spk_emb  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, style_dim); 
        ggml_set_name(spk_emb, "spk_emb");
        ggml_set_input(spk_emb);

        enc_seq = ggml_transpose(ctx, enc_seq); // [528, 115]

        x = encode0.graph(gf, ctx, enc_seq);
        x = encode1.graph(gf, ctx, x);
        tensor_dbg (gf, ctx, x, "dbg");

        // FIXME
        // spk_emb = spk_emb.squeeze(1)
        // ...

        ggml_set_name(x, "x");
        ggml_set_output(x);
        ggml_build_forward_expand(gf, x);

        if (!ggml_gallocr_alloc_graph(alloc, gf))
            throw std::runtime_error("ggml_gallocr_alloc_graph() failed");
    }

    StyleTTSDecoder::~StyleTTSDecoder()
    {
        if (alloc)
            ggml_gallocr_free(alloc);
    }

    void StyleTTSDecoder::eval(float *enc_seq_data, [[maybe_unused]]float *spk_emb_data)
    {
        ggml_backend_tensor_set(enc_seq, enc_seq_data, 0, max_seq_len*dim_in*sizeof(float));
        //ggml_backend_tensor_set(spk_emb, spk_emb_data, 0, style_dim*sizeof(float));

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("ggml_backend_graph_compute() failed");

        struct ggml_tensor *dbg = ggml_graph_get_tensor(gf, "dbg");
        print_tensor("dbg", dbg, 3);
    }
}