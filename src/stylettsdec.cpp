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

    AdaIN1d::AdaIN1d(ggml_context   &ctx_w,
                     [[maybe_unused]]ggml_backend_t  backend,
                     int             idx0,
                     int             idx1,
                     uint32_t        style_dim,
                     uint32_t        num_features)
    {
        char namebuf[GGML_MAX_NAME];

        this->style_dim    = style_dim;
        this->num_features = num_features;

        // _mel_decoder.decode.0.norm1.fc.b: [2240,   1,   1,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.decode.%d.norm%d.fc.b", idx0, idx1);
        fc_b = checked_get_tensor(&ctx_w, namebuf);
        // _mel_decoder.decode.0.norm1.fc.w: [528, 2240,   1,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.decode.%d.norm%d.fc.w", idx0, idx1);
        fc_w = checked_get_tensor(&ctx_w, namebuf);
    }

    struct ggml_tensor *AdaIN1d::graph([[maybe_unused]]struct ggml_cgraph *gf, ggml_context *ctx, struct ggml_tensor *x, struct ggml_tensor *s, struct ggml_tensor *one)
    {
        struct ggml_tensor *out = x;

        // h = self.fc(s)

        struct ggml_tensor *h = s;
        h = ggml_mul_mat(ctx, fc_w, h);
        h = ggml_add(ctx, h, fc_b);

        // h = h.view(h.size(0), h.size(1), 1)
        //     h.shape torch.Size([1, 2240]) -> h.shape torch.Size([1, 2240, 1])
        // gamma, beta = torch.chunk(h, chunks=2, dim=1)
        //     gamma [1, 1120, 1] beta [1, 1120, 1]

        struct ggml_tensor *gamma = ggml_view_1d(ctx, h, h->ne[0]/2, 0);
        struct ggml_tensor *beta  = ggml_view_1d(ctx, h, h->ne[0]/2, gamma->nb[1]);

        gamma = ggml_add_inplace(ctx, gamma, ggml_repeat(ctx, one, gamma));

        x = ggml_norm_inplace(ctx, x, 1e-5);

        // return (1 + gamma) * self.norm(x) + beta
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
        out = ggml_mul(ctx, x, gamma);
        out = ggml_add_inplace (ctx, out, beta);
        out = ggml_cont(ctx, ggml_transpose(ctx, out));

        return out;
    }


    AdainResBlk1d::AdainResBlk1d(ggml_context   &ctx_w,
                                 [[maybe_unused]]ggml_backend_t  backend,
                                 int             idx,
                                 uint32_t        dim_in,
                                 uint32_t        dim_out,
                                 uint32_t        style_dim):
        norm1(ctx_w, backend, idx, 1, style_dim, dim_in),
        norm2(ctx_w, backend, idx, 2, style_dim, dim_out)
    {
        this->dim_in    = dim_in;
        this->dim_out   = dim_out;
        this->style_dim = style_dim;

        learned_sc = dim_in != dim_out;

        char namebuf[GGML_MAX_NAME];

        // _mel_decoder.decode.0.conv1.b : [1056,   1,   1,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.decode.%d.conv1.b", idx);
        conv1_b = checked_get_tensor(&ctx_w, namebuf);
        // _mel_decoder.decode.0.conv1.w : [  3, 1120, 1056,   1] f16
        std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.decode.%d.conv1.w", idx);
        conv1_w = checked_get_tensor(&ctx_w, namebuf);

        if (learned_sc)
        {
            // _mel_decoder.decode.0.conv1x1.w: [  1, 1120, 1056,   1] f16
            std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.decode.%d.conv1x1.w", idx);
            conv1x1_w = checked_get_tensor(&ctx_w, namebuf);
        }

        // _mel_decoder.decode.0.conv2.b : [1056,   1,   1,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.decode.%d.conv2.b", idx);
        conv2_b = checked_get_tensor(&ctx_w, namebuf);
        // _mel_decoder.decode.0.conv2.w : [  3, 1056, 1056,   1] f16
        std::snprintf(namebuf, GGML_MAX_NAME, "_mel_decoder.decode.%d.conv2.w", idx);
        conv2_w = checked_get_tensor(&ctx_w, namebuf);
    }

    struct ggml_tensor *AdainResBlk1d::graph(struct ggml_cgraph *gf, ggml_context *ctx, struct ggml_tensor *x, struct ggml_tensor *s, struct ggml_tensor *one)
    {
        struct ggml_tensor *out = ggml_cpy(ctx, x, ggml_dup_tensor(ctx, x));

        //out = self._residual(x, s)
        
        //out = self.norm1(out, s)
        //self.norm1 = AdaIN1d(style_dim, dim_in)
        out = norm1.graph(gf, ctx, out, s, one);

        // out = self.actv(out)
        out = ggml_leaky_relu(ctx, out, 0.2, true);

        // out = self.conv1(out)
        // self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        out = ggml_conv_1d(ctx,
                           conv1_w, // convolution kernel
                           out,     // data
                           1,       // stride
                           1,       // padding
                           1);      // dilation
        out = ggml_cont(ctx, ggml_transpose(ctx, out));
        out = ggml_add_inplace (ctx, out, ggml_repeat(ctx, conv1_b, out));
        out = ggml_cont(ctx, ggml_transpose(ctx, out));

        // out = self.norm2(out, s)
        out = norm2.graph(gf, ctx, out, s, one);

        // out = self.actv(out)
        out = ggml_leaky_relu(ctx, out, 0.2, true);

        // out = self.conv2(out)
        out = ggml_conv_1d(ctx,
                           conv2_w, // convolution kernel
                           out,     // data
                           1,       // stride
                           1,       // padding
                           1);      // dilation
        out = ggml_cont(ctx, ggml_transpose(ctx, out));
        out = ggml_add_inplace (ctx, out, ggml_repeat(ctx, conv2_b, out));
        out = ggml_cont(ctx, ggml_transpose(ctx, out));

        // out = (out + self._shortcut(x)) / math.sqrt(2)

        if (learned_sc)
        {
            struct ggml_tensor *sc = x;
            sc = ggml_conv_1d(ctx,
                              conv1x1_w,    // convolution kernel
                              sc,           // data
                              1,            // stride
                              0,            // padding
                              1);           // dilation
            out = ggml_add_inplace(ctx, sc, out);
        }
        out = ggml_scale_inplace(ctx, out, 1/sqrt(2.0));

        tensor_dbg (gf, ctx, out, "dbg");


        return out;
    }

    StyleTTSDecoder::StyleTTSDecoder(ggml_context   &ctx_w,
                                     ggml_backend_t  backend,
                                     uint32_t        max_seq_len,
                                     uint32_t        dim_in,
                                     uint32_t        style_dim,
                                     uint32_t        residual_dim,
                                     [[maybe_unused]]uint32_t        dim_out) :
        encode0(ctx_w, 0, dim_in, dim_in*2),
        encode1(ctx_w, 1, dim_in*2, dim_in*2),
        decode0(ctx_w, backend, 0, dim_in * 2 + residual_dim, dim_in*2, style_dim)
    {
        this->max_seq_len = max_seq_len;
        this->style_dim   = style_dim;
        this->dim_in      = dim_in;

        //uint32_t bottleneck_dim = dim_in * 2;
        //        self.encode = nn.Sequential(ResBlk1d(dim_in, self.bottleneck_dim, normalize=True),
        //                                    ResBlk1d(self.bottleneck_dim, self.bottleneck_dim, normalize=True))

        this->backend      = backend;

        alloc              = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

        asr_res_0_b  = checked_get_tensor(&ctx_w, "_mel_decoder.asr_res.0.b");
        asr_res_0_w  = checked_get_tensor(&ctx_w, "_mel_decoder.asr_res.0.w");
        asr_res_1_b  = checked_get_tensor(&ctx_w, "_mel_decoder.asr_res.1.b");
        asr_res_1_w  = checked_get_tensor(&ctx_w, "_mel_decoder.asr_res.1.w");

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

        one  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1); 
        ggml_set_name(one, "one");
        ggml_set_input(one);

        struct ggml_tensor *enc_seq_t = ggml_cont(ctx, ggml_transpose(ctx, enc_seq)); // [528, 115]

        x = encode0.graph(gf, ctx, enc_seq_t);
        x = encode1.graph(gf, ctx, x);

        // self.asr_res = nn.Sequential(
        //    weight_norm(nn.Conv1d(dim_in, residual_dim, kernel_size=1)),
        //    nn.InstanceNorm1d(residual_dim, affine=True)
        // )
        // asr_res = self.asr_res(enc_seq)

        struct ggml_tensor *asr_res = ggml_conv_1d(ctx,
                                                   asr_res_0_w, // convolution kernel
                                                   enc_seq_t,   // data
                                                   1,           // stride
                                                   0,           // padding
                                                   1);          // dilation
        asr_res = ggml_cont(ctx, ggml_transpose(ctx, asr_res));
        asr_res = ggml_add (ctx, asr_res, ggml_repeat(ctx, asr_res_0_b, asr_res));
        asr_res = ggml_cont(ctx, ggml_transpose(ctx, asr_res));

        asr_res = ggml_norm(ctx, asr_res, 1e-5);
        asr_res = ggml_cont(ctx, ggml_transpose(ctx, asr_res));
        asr_res = ggml_mul(ctx, asr_res, asr_res_1_w);
        asr_res = ggml_add(ctx, asr_res, asr_res_1_b);
        asr_res = ggml_cont(ctx, ggml_transpose(ctx, asr_res));

        // x = torch.cat([x, asr_res], axis=1)
        // [1120, 115] <- [1056, 115] . [64, 115]
        struct ggml_tensor *x_cat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, max_seq_len, x->ne[1] + asr_res->ne[1]);
        struct ggml_tensor *x_cat_lo = ggml_view_2d(ctx, x_cat, max_seq_len, x->ne[1], x_cat->nb[1], 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, x, x_cat_lo));
        struct ggml_tensor *x_cat_hi = ggml_view_2d(ctx, x_cat, max_seq_len, asr_res->ne[1], x_cat->nb[1], x->nb[2]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, asr_res, x_cat_hi));

        // struct ggml_tensor *ones = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, gamma->ne[0]);
        // float one = 1.0;
        // for (int64_t i = 0; i < ones->ne[0]; ++i) 
        //     ggml_backend_tensor_set(ones, &one, sizeof(float)*i, sizeof(float));



        // self.decode.append(AdainResBlk1d(self.bottleneck_dim + residual_dim, self.bottleneck_dim, style_dim))
        // x = block(x, spk_emb)
        x = decode0.graph(gf, ctx, x_cat, spk_emb, one);



        // self.decode.append(AdainResBlk1d(self.bottleneck_dim + residual_dim, self.bottleneck_dim, style_dim))
        // self.decode.append(AdainResBlk1d(self.bottleneck_dim + residual_dim, dim_in, style_dim, upsample=True))
        // self.decode.append(AdainResBlk1d(dim_in, dim_in, style_dim))
        // self.decode.append(AdainResBlk1d(dim_in, dim_in, style_dim))




        //bool res = true;




        // for block in self.decode:
        //     if res:
        //         x = torch.cat([x, asr_res], axis=1)




        //     x = block(x, spk_emb)
        //     if block.upsample_type != "none":
        //         res = False
                
        // x = self.to_out(x)



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
        ggml_backend_tensor_set(spk_emb, spk_emb_data, 0, style_dim*sizeof(float));

        float f = 1.0;
        ggml_backend_tensor_set(one, &f, 0, sizeof(float));

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("ggml_backend_graph_compute() failed");

        struct ggml_tensor *dbg;

        dbg = ggml_graph_get_tensor(gf, "dbg");
        print_tensor("dbg", dbg, 3);

        dbg = ggml_graph_get_tensor(gf, "x");
        print_tensor("x", dbg, 3);
    }
}