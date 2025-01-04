//
// FastSpeech 2 Encoder
//
// original python code borrowed (under MIT license) from Chung-Ming Chien's implementation of FastSpeech2
//
// https://github.com/ming024/FastSpeech2

#include <cstdio>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <stdexcept>

#include "zerovox.h"

namespace ZeroVOX
{

    MultiHeadAttention::MultiHeadAttention(ggml_context &ctx, int layidx, uint32_t n_head, uint32_t d_k, uint32_t d_v)
    {
        char namebuf[GGML_MAX_NAME];

        this->layidx = layidx;
        this->n_head = n_head;
        this->d_k    = d_k;
        this->d_v    = d_v;

        // _pe._enc.laystk.0.slf_attn.w_ks.b: [528,   1,   1,   1] f32
        // _pe._enc.laystk.0.slf_attn.w_ks.w: [528, 528,   1,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "_pe._enc.laystk.%d.slf_attn.w_ks.b", layidx);
        w_ks_b = checked_get_tensor(&ctx, namebuf);
        std::snprintf(namebuf, GGML_MAX_NAME, "_pe._enc.laystk.%d.slf_attn.w_ks.w", layidx);
        w_ks_w = checked_get_tensor(&ctx, namebuf);

        // _pe._enc.laystk.0.slf_attn.w_qs.b: [528,   1,   1,   1] f32
        // _pe._enc.laystk.0.slf_attn.w_qs.w: [528, 528,   1,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "_pe._enc.laystk.%d.slf_attn.w_qs.b", layidx);
        w_qs_b = checked_get_tensor(&ctx, namebuf);
        std::snprintf(namebuf, GGML_MAX_NAME, "_pe._enc.laystk.%d.slf_attn.w_qs.w", layidx);
        w_qs_w = checked_get_tensor(&ctx, namebuf);

        // _pe._enc.laystk.0.slf_attn.w_vs.b: [528,   1,   1,   1] f32
        // _pe._enc.laystk.0.slf_attn.w_vs.w: [528, 528,   1,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "_pe._enc.laystk.%d.slf_attn.w_vs.b", layidx);
        w_vs_b = checked_get_tensor(&ctx, namebuf);
        std::snprintf(namebuf, GGML_MAX_NAME, "_pe._enc.laystk.%d.slf_attn.w_vs.w", layidx);
        w_vs_w = checked_get_tensor(&ctx, namebuf);

        // _pe._enc.laystk.0.slf_attn.fc.b: [528,   1,   1,   1] f32
        // _pe._enc.laystk.0.slf_attn.fc.w: [528, 528,   1,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "_pe._enc.laystk.%d.slf_attn.fc.b", layidx);
        fc_b = checked_get_tensor(&ctx, namebuf);
        std::snprintf(namebuf, GGML_MAX_NAME, "_pe._enc.laystk.%d.slf_attn.fc.w", layidx);
        fc_w = checked_get_tensor(&ctx, namebuf);

        // _pe._enc.laystk.0.slf_attn.layer_norm.b: [528,   1,   1,   1] f32
        // _pe._enc.laystk.0.slf_attn.layer_norm.w: [528,   1,   1,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "_pe._enc.laystk.%d.slf_attn.layer_norm.b", layidx);
        layer_norm_b = checked_get_tensor(&ctx, namebuf);
        std::snprintf(namebuf, GGML_MAX_NAME, "_pe._enc.laystk.%d.slf_attn.layer_norm.w", layidx);
        layer_norm_w = checked_get_tensor(&ctx, namebuf);

        //std::snprintf(namebuf, GGML_MAX_NAME, "_pe._enc.laystk.%d.slf_attn.temperature", layidx);
        //temperature = checked_get_tensor(&ctx, namebuf);
        temperature = pow(d_k, 0.5);
        // struct ggml_tensor *temperature = ggml_new_tensor_1d(&ctx, layer_norm_w->type, 1);
        // ggml_backend_tensor_set(temperature, &t, 0, sizeof(t));
    }

    struct ggml_tensor *MultiHeadAttention::graph([[maybe_unused]] struct ggml_cgraph *gf, ggml_context *ctx, struct ggml_tensor *q, struct ggml_tensor *k, struct ggml_tensor *v)
    {
        struct ggml_tensor *residual = q;

        // self.w_qs = nn.Linear(d_model, n_head * d_k)
        // q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        q = ggml_mul_mat(ctx, w_qs_w, q);
        q = ggml_add(ctx, q, w_qs_b);
        q = ggml_view_3d(ctx, q, d_k, n_head, q->ne[1], q->nb[0]*d_k, q->nb[0]*d_k*n_head, 0);

        // k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        k = ggml_mul_mat(ctx, w_ks_w, k);
        k = ggml_add(ctx, k, w_ks_b);
        k = ggml_view_3d(ctx, k, d_k, n_head, k->ne[1], k->nb[0]*d_k, k->nb[0]*d_k*n_head, 0);

        // v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        v = ggml_mul_mat(ctx, w_vs_w, v);
        v = ggml_add(ctx, v, w_vs_b);
        v = ggml_view_3d(ctx, v, d_v, n_head, v->ne[1], v->nb[0]*d_v, v->nb[0]*d_v*n_head, 0);

        // q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        // k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        // v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

        //self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        // output, attn = self.attention(q, k, v, mask=mask)

        //attn = torch.bmm(q, k.transpose(1, 2))
        //struct ggml_tensor *attn = ggml_mul_mat (ctx, q, ggml_permute(ctx, k, 0, 2, 1, 3));
        struct ggml_tensor *attn = ggml_mul_mat (ctx, q, k);
        attn = ggml_cont(ctx, ggml_permute(ctx, attn, 1, 0, 2, 3));
        
        //attn = attn / self.temperature
        attn = ggml_scale(ctx, attn, 1.0/temperature);

        //attn = self.softmax(attn)
        attn = ggml_soft_max(ctx, attn);

        //output = torch.bmm(attn, v)
        v    = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
        struct ggml_tensor *output = ggml_mul_mat (ctx, attn, v);

        // output = output.view(n_head, sz_b, len_q, d_v)
        // output = (
        //     output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        // )  # b x lq x (n*dv)

        output = ggml_view_2d(ctx, output, output->ne[0], output->ne[1]*output->ne[2], output->nb[1], 0);
        output = ggml_permute(ctx, output, 1, 0, 2, 3);
        output = ggml_cont(ctx, output);

        // self.fc = nn.Linear(n_head * d_v, d_model)
        // output = self.fc(output)
        output = ggml_mul_mat(ctx, fc_w, output);
        output = ggml_add(ctx, output, fc_b);

        // self.layer_norm = nn.LayerNorm(d_model)
        // output = self.layer_norm(output + residual)
        output = ggml_norm(ctx, ggml_add(ctx, output, residual), 1e-5);
        output = ggml_add(ctx,
                    ggml_mul(ctx,
                        ggml_repeat(ctx, layer_norm_w, output),
                        output),
                    ggml_repeat(ctx, layer_norm_b, output));

        return output;
    }

    PositionwiseFeedForward::PositionwiseFeedForward(ggml_context &ctx, int layidx, uint32_t d_in, uint32_t d_hid, uint32_t kernel_size[2])
    {
        char namebuf[GGML_MAX_NAME];

        this->layidx         = layidx;
        this->d_in           = d_in;
        this->d_hid          = d_hid;
        this->kernel_size[0] = kernel_size[0];
        this->kernel_size[1] = kernel_size[1];

        // _pe._enc.laystk.0.pos_ffn.layer_norm.b: [528,   1,   1,   1] f32
        // _pe._enc.laystk.0.pos_ffn.layer_norm.w: [528,   1,   1,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "_pe._enc.laystk.%d.pos_ffn.layer_norm.b", layidx);
        layer_norm_b = checked_get_tensor(&ctx, namebuf);
        std::snprintf(namebuf, GGML_MAX_NAME, "_pe._enc.laystk.%d.pos_ffn.layer_norm.w", layidx);
        layer_norm_w = checked_get_tensor(&ctx, namebuf);

        // _pe._enc.laystk.0.pos_ffn.w_1.b: [1024,   1,   1,   1] f32
        // _pe._enc.laystk.0.pos_ffn.w_1.w: [  9, 528, 1024,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "_pe._enc.laystk.%d.pos_ffn.w_1.b", layidx);
        w_1_b = checked_get_tensor(&ctx, namebuf);
        std::snprintf(namebuf, GGML_MAX_NAME, "_pe._enc.laystk.%d.pos_ffn.w_1.w", layidx);
        w_1_w = checked_get_tensor(&ctx, namebuf);

        // _pe._enc.laystk.0.pos_ffn.w_2.b: [528,   1,   1,   1] f32
        // _pe._enc.laystk.0.pos_ffn.w_2.w: [  1, 1024, 528,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "_pe._enc.laystk.%d.pos_ffn.w_2.b", layidx);
        w_2_b = checked_get_tensor(&ctx, namebuf);
        std::snprintf(namebuf, GGML_MAX_NAME, "_pe._enc.laystk.%d.pos_ffn.w_2.w", layidx);
        w_2_w = checked_get_tensor(&ctx, namebuf);
    }

    struct ggml_tensor *PositionwiseFeedForward::graph([[maybe_unused]]struct ggml_cgraph *gf, ggml_context *ctx, struct ggml_tensor *x)
    {
        // residual = x
        struct ggml_tensor *residual = x;

        // output = x.transpose(1, 2)
        struct ggml_tensor *output = ggml_cont(ctx, ggml_transpose(ctx, x));

        // self.w_1 = nn.Conv1d(
        //     d_in,
        //     d_hid,
        //     kernel_size=kernel_size[0],
        //     padding=(kernel_size[0] - 1) // 2,
        // )

        //output = self.w_1(output);
        output = ggml_conv_1d(ctx,
                            w_1_w,                     // convolution kernel
                            output,                    // data
                            1,                         // stride
                            (kernel_size[0] - 1) / 2,  // padding
                            1);                        // dilation
        output = ggml_cont(ctx, ggml_transpose(ctx, output));
        output = ggml_add(ctx, output, 
                        ggml_repeat(ctx, w_1_b, output));

        output = ggml_cont(ctx, ggml_transpose(ctx, output));

        // output = F.relu(output)
        output = ggml_relu(ctx, output);

        //output = self.w_2(output)
        output = ggml_conv_1d(ctx,
                            w_2_w,                     // convolution kernel
                            output,                    // data
                            1,                         // stride
                            (kernel_size[1] - 1) / 2,  // padding
                            1);                        // dilation
        output = ggml_cont(ctx, ggml_transpose(ctx, output));
        output = ggml_add(ctx, output, 
                        ggml_repeat(ctx, w_2_b, output));

        // output = output.transpose(1, 2)

        // self.layer_norm = nn.LayerNorm(d_in)
        // output = self.layer_norm(output + residual)
        output = ggml_norm(ctx, ggml_add(ctx, output, residual), 1e-5);
        output = ggml_add(ctx,
                    ggml_mul(ctx,
                        ggml_repeat(ctx, layer_norm_w, output),
                        output),
                    ggml_repeat(ctx, layer_norm_b, output));

        return output;
    }

    FFTBlock::FFTBlock(ggml_context &ctx, int layidx, uint32_t d_model, uint32_t n_head, uint32_t d_k, uint32_t d_v, uint32_t d_inner, uint32_t kernel_size[2]) :
        slf_attn(ctx, layidx, n_head, d_k, d_v),
        pos_ffn(ctx, layidx, d_model, d_inner, kernel_size)
    {
    }

    struct ggml_tensor *FFTBlock::graph(struct ggml_cgraph *gf, ggml_context *ctx, struct ggml_tensor *enc_input)
    {
        struct ggml_tensor *enc_output = slf_attn.graph(gf, ctx, enc_input, enc_input, enc_input);

        enc_output = pos_ffn.graph(gf, ctx, enc_output);

        return enc_output;
    }

    Encoder::Encoder(ggml_context &ctx, uint32_t max_n_phonemes, uint32_t embed_dim, uint32_t encoder_layer, uint32_t encoder_head, uint32_t conv_filter_size, uint32_t conv_kernel_size[2], uint32_t punct_embed_dim)
    {
        this->max_n_phonemes = max_n_phonemes;
        this->embed_dim = embed_dim;
        this->punct_embed_dim = punct_embed_dim;
        this->encoder_layer = encoder_layer;

        // _mel_decoder.laystk.0.slf_attn.w_qs.b: (528,)
        // _mel_decoder.laystk.0.slf_attn.w_qs.w: (528, 528)
        // _pe._enc.src_word_emb.w: (155, 512)

        src_word_emb_w = checked_get_tensor(&ctx, "_pe._enc.src_word_emb.w");

        // _pe._enc.punct_embed.w
        src_punct_embed_w = checked_get_tensor(&ctx, "_pe._enc.punct_embed.w");

        sinusoid_encoding_table = checked_get_tensor(&ctx, "sinusoid_encoding_table");

        // self.layer_stack = nn.ModuleList(
        //     [
        //         FFTBlock(
        //             self.d_model, encoder_head, d_k, d_v, d_inner, kernel_size, spk_emb_size=0, scln=False, dropout=dropout
        //         )
        //         for _ in range(encoder_layer)
        //     ]
        // )

        uint32_t encoder_hidden = embed_dim + punct_embed_dim;
        uint32_t d_k = encoder_hidden / encoder_head ;
        uint32_t d_v = encoder_hidden / encoder_head ;
        uint32_t d_inner = conv_filter_size;

        layer_stack = new FFTBlock*[encoder_layer];

        for (uint32_t i=0; i<encoder_layer; i++)
            layer_stack[i] = new FFTBlock (ctx, i, encoder_hidden, encoder_head, d_k, d_v, d_inner, conv_kernel_size);
    }

    Encoder::~Encoder()
    {
        delete layer_stack;
    }


    struct ggml_tensor *Encoder::graph(struct ggml_cgraph *gf, ggml_context *ctx, struct ggml_tensor *src_seq, struct ggml_tensor *puncts)
    {
        int emb_size = embed_dim + punct_embed_dim;
        // // dec_hidden = emb_size

        // int n_head = hparams.decoder_n_head;
        // int d_model = emb_size;
        // int d_k = emb_size / n_head;
        // int d_v = emb_size / n_head;
        // int spk_emb_size = emb_size;
        // // int max_seq_len = hparams.max_seq_len;

        // printf("nhead=%d, d_model=%d, d_k=%d, d_v=%d, spk_emb_size=%d, num_phonemes=%d, num_puncts=%d, n_phonemes=%d\n",
        //        n_head, d_model, d_k, d_v, spk_emb_size, NUM_PHONEMES, NUM_PUNCTS, max_n_phonemes);

        // src_word_emb = nn.Embedding(symbols.num_phonemes + 1, embed_dim, padding_idx=0)
        // x = self.src_word_emb(src_seq) # [16, 126, 128]
        struct ggml_tensor *x = ggml_get_rows(ctx, src_word_emb_w, src_seq); // [emb_dim, max_n_phonemes]

        // x_punct = self.punct_embed(puncts) # [16, 126, 16]
        struct ggml_tensor *x_punct = ggml_get_rows(ctx, src_punct_embed_w, puncts); // [punct_emb_dim, max_n_phonemes]

        // x = torch.cat((x, x_punct), 2) # [16, 126, 144]
        struct ggml_tensor *cur = ggml_new_tensor_2d(ctx, x->type, emb_size, max_n_phonemes);
        struct ggml_tensor *cur_lo = ggml_view_2d(ctx, cur, embed_dim, max_n_phonemes, cur->nb[1], 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, x, cur_lo));
        struct ggml_tensor * cur_hi = ggml_view_2d(ctx, cur, punct_embed_dim, max_n_phonemes, cur->nb[1], ggml_element_size(cur) * embed_dim);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, x_punct, cur_hi));
        
        // enc_output = x + self.position_enc[ :, :max_len, :].expand(batch_size, -1, -1)
        // self.position_enc.shape                                             torch.Size([1, 1501, 528])
        // self.position_enc[ :, :max_len, :].shape                            torch.Size([1, 11, 528])
        // self.position_enc[ :, :max_len, :].expand(batch_size, -1, -1).shape torch.Size([16, 11, 528])
        // (1501, 528) -> (11, 528)
        struct ggml_tensor *posenc = ggml_view_2d(ctx, sinusoid_encoding_table, emb_size, max_n_phonemes, sinusoid_encoding_table->nb[1], 0);
        struct ggml_tensor *enc_output = ggml_add(ctx, cur, posenc);

        // for enc_layer in self.layer_stack:
        //     enc_output, enc_slf_attn = enc_layer(
        //         enc_output, spk_emb=None, mask=mask, slf_attn_mask=slf_attn_mask
        //     )

        for (uint32_t i=0; i<encoder_layer; i++)
            //int i = 0; // FIXME
            enc_output = layer_stack[i]->graph(gf, ctx, enc_output);

        return enc_output;
    }

    VariancePredictor::VariancePredictor(ggml_context &ctx, const char *prefix, uint32_t vp_kernel_size)
    {
        char namebuf[GGML_MAX_NAME];

        this->vp_kernel_size = vp_kernel_size;

        // _pe._var_adapt.duration_predictor.conv_layer.conv1d_1.conv.b: [256,   1,   1,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "%s.conv_layer.conv1d_1.conv.b", prefix);
        conv1d_1_conv_b = checked_get_tensor(&ctx, namebuf);

        // _pe._var_adapt.duration_predictor.conv_layer.conv1d_1.conv.w: [  3, 528, 256,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "%s.conv_layer.conv1d_1.conv.w", prefix);
        conv1d_1_conv_w = checked_get_tensor(&ctx, namebuf);

        // _pe._var_adapt.duration_predictor.conv_layer.conv1d_2.conv.b: [256,   1,   1,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "%s.conv_layer.conv1d_2.conv.b", prefix);
        conv1d_2_conv_b = checked_get_tensor(&ctx, namebuf);

        // _pe._var_adapt.duration_predictor.conv_layer.conv1d_2.conv.w: [  3, 256, 256,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "%s.conv_layer.conv1d_2.conv.w", prefix);
        conv1d_2_conv_w = checked_get_tensor(&ctx, namebuf);

        // _pe._var_adapt.duration_predictor.conv_layer.layer_norm_1.b: [256,   1,   1,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "%s.conv_layer.layer_norm_1.b", prefix);
        layer_norm_1_b = checked_get_tensor(&ctx, namebuf);

        // _pe._var_adapt.duration_predictor.conv_layer.layer_norm_1.w: [256,   1,   1,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "%s.conv_layer.layer_norm_1.w", prefix);
        layer_norm_1_w = checked_get_tensor(&ctx, namebuf);

        // _pe._var_adapt.duration_predictor.conv_layer.layer_norm_2.b: [256,   1,   1,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "%s.conv_layer.layer_norm_2.b", prefix);
        layer_norm_2_b = checked_get_tensor(&ctx, namebuf);

        // _pe._var_adapt.duration_predictor.conv_layer.layer_norm_2.w: [256,   1,   1,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "%s.conv_layer.layer_norm_2.w", prefix);
        layer_norm_2_w = checked_get_tensor(&ctx, namebuf);

        // _pe._var_adapt.duration_predictor.linear_layer.b: [  1,   1,   1,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "%s.linear_layer.b", prefix);
        linear_layer_b = checked_get_tensor(&ctx, namebuf);

        // _pe._var_adapt.duration_predictor.linear_layer.w: [256,   1,   1,   1] f32
        std::snprintf(namebuf, GGML_MAX_NAME, "%s.linear_layer.w", prefix);
        linear_layer_w = checked_get_tensor(&ctx, namebuf);

    }

    struct ggml_tensor *VariancePredictor::graph([[maybe_unused]] struct ggml_cgraph *gf, ggml_context *ctx, struct ggml_tensor *encoder_output)
    {
        struct ggml_tensor *out=encoder_output;

        // out = self.conv_layer(encoder_output)

        out = ggml_cont(ctx, ggml_transpose(ctx, out));
        //struct ggml_tensor *w = ggml_cont(ctx, ggml_transpose(ctx, conv1d_1_conv_w));
        out = ggml_conv_1d(ctx,
                        conv1d_1_conv_w,           // convolution kernel
                        out,                       // data
                        1,                         // stride
                        (vp_kernel_size - 1) / 2,  // padding
                        1);                        // dilation
        out = ggml_cont(ctx, ggml_transpose(ctx, out));
        out = ggml_add(ctx, out, 
                    ggml_repeat(ctx, conv1d_1_conv_b, out));

        out = ggml_relu(ctx, out);
        out = ggml_norm(ctx, out, 1e-5);
        out = ggml_add(ctx,
                ggml_mul(ctx,
                    ggml_repeat(ctx, layer_norm_1_w, out),
                    out),
                ggml_repeat(ctx, layer_norm_1_b, out));

        out = ggml_cont(ctx, ggml_transpose(ctx, out));
        out = ggml_conv_1d(ctx,
                        conv1d_2_conv_w,           // convolution kernel
                        out,                       // data
                        1,                         // stride
                        1,                         // padding
                        1);                        // dilation
        out = ggml_cont(ctx, ggml_transpose(ctx, out));
        out = ggml_add(ctx, out, 
                    ggml_repeat(ctx, conv1d_2_conv_b, out));

        out = ggml_relu(ctx, out);
        out = ggml_norm(ctx, out, 1e-5);
        out = ggml_add(ctx,
                ggml_mul(ctx,
                    ggml_repeat(ctx, layer_norm_2_w, out),
                    out),
                ggml_repeat(ctx, layer_norm_2_b, out));

        //out = ggml_cont(ctx, ggml_transpose(ctx, out));

        // out = self.linear_layer(out)
        out = ggml_mul_mat(ctx, out, linear_layer_w);
        out = ggml_add(ctx, out, linear_layer_b);

        // out = out.squeeze(-1)

        return out;
    }

    static void ggml_zv_mul_clamp_to_i32(struct ggml_tensor *dst, [[maybe_unused]]const struct ggml_tensor *a, const struct ggml_tensor *src, int ith, int nth, void * userdata)
    {
        GGML_ASSERT(userdata);
        GGML_ASSERT(ggml_are_same_shape(dst, src));
        GGML_ASSERT(ggml_is_contiguous(dst));
        GGML_ASSERT(ggml_is_contiguous(src));

        //FS2Encoder *self = (FS2Encoder *)userdata;
        intptr_t ve_n_bins = (intptr_t)userdata;

        const float *src_data = ggml_get_data_f32(src);
        int32_t *dst_data = (int32_t *)ggml_get_data(dst);

        const int ne = (int)ggml_nelements(dst);
        const int dr = (ne + nth - 1) / nth;
        const int ie0 = dr * ith;
        const int ie1 = std::min(ie0 + dr, ne);

        int bin_max = ve_n_bins-1;

        for (int i = ie0; i < ie1; ++i)
        {
            float x = src_data[i];

            //torch.clamp(torch.round(prediction*(self._ve_n_bins-1)).long(), min=0, max=self._ve_n_bins-1
            x = x * bin_max;
            int32_t y = (int32_t)(x+0.5);
            if (y<0) y = 0;
            if (y>bin_max) y = bin_max;

            dst_data[i] = y;
        }
    }


    FS2Encoder::FS2Encoder(ggml_context   &ctx_w,
                           ggml_backend_t backend,
                           uint32_t       max_n_phonemes,
                           uint32_t       embed_dim,
                           uint32_t       punct_embed_dim,
                           uint32_t       encoder_layer,
                           uint32_t       encoder_head,
                           uint32_t       conv_filter_size,
                           uint32_t       conv_kernel_size[2],
                           uint32_t       vp_kernel_size,
                           uint32_t       ve_n_bins,
                           uint32_t       max_seq_len) :
        encoder(ctx_w, max_n_phonemes, embed_dim, encoder_layer, encoder_head, conv_filter_size, conv_kernel_size, punct_embed_dim),
        duration_predictor(ctx_w, "_pe._var_adapt.duration_predictor", vp_kernel_size),
        pitch_predictor(ctx_w, "_pe._var_adapt.pitch_predictor", vp_kernel_size),
        energy_predictor(ctx_w, "_pe._var_adapt.engy_pred", vp_kernel_size)
    {
        this->max_seq_len      = max_seq_len;
        this->ve_n_bins        = ve_n_bins;
        this->max_n_phonemes   = max_n_phonemes;
        this->embed_dim        = embed_dim;
        this->punct_embed_dim  = punct_embed_dim;

        this->backend          = backend;

        alloc              = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

        pitch_embedding_w  = checked_get_tensor(&ctx_w, "_pe._var_adapt.pitch_embedding.w");
        energy_embedding_w = checked_get_tensor(&ctx_w, "_pe._var_adapt.energy_embedding.w");

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

        int emb_size = embed_dim + punct_embed_dim;

        // _pe._enc.position_enc: (1, 1501, 528)
        // _pe._enc.punct_embed.w: (7, 16)
        // _pe._enc.src_word_emb.w: (155, 512)

        gf = ggml_new_graph(ctx);

        // x = self.src_word_emb(src_seq) # [16, 126, 128]
        src_seq = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_n_phonemes); 
        ggml_set_name(src_seq, "src_seq");
        ggml_set_input(src_seq);

        puncts  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_n_phonemes); 
        ggml_set_name(puncts, "puncts");
        ggml_set_input(puncts);

        style_embed  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, emb_size); 
        ggml_set_name(style_embed, "style_embed");
        ggml_set_input(style_embed);

        // features = self._encoder(src_seq=phoneme, puncts=puncts, mask=phoneme_mask, return_attns=False)
        features = encoder.graph(gf, ctx, src_seq, puncts);

        // se = style_embed.expand_as(features)
        struct ggml_tensor *se = ggml_repeat(ctx, style_embed, features);
        // features = features + se
        features = ggml_add (ctx, features, se);

        // log_duration_prediction = self.duration_predictor(features, src_mask)
        log_duration_prediction = duration_predictor.graph(gf, ctx, features);
        ggml_set_name(log_duration_prediction, "duration");
        ggml_set_output(log_duration_prediction);
        ggml_build_forward_expand(gf, log_duration_prediction);

        // pitch_prediction = self.pitch_predictor(features, mask)
        struct ggml_tensor *pitch_prediction = pitch_predictor.graph(gf, ctx, features);

        // embedding = self.pitch_embedding(torch.clamp(torch.round(prediction*(self._ve_n_bins-1)).long(), min=0, max=self._ve_n_bins-1))
        struct ggml_tensor *buckets = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, pitch_prediction->ne[0]);
        buckets = ggml_map_custom2_inplace(ctx, buckets, pitch_prediction, ggml_zv_mul_clamp_to_i32, GGML_N_TASKS_MAX, (void*)(intptr_t)ve_n_bins);
        struct ggml_tensor *pitch_embedding = ggml_get_rows(ctx, pitch_embedding_w, buckets);
    
        // features = features + pitch_embedding
        features = ggml_add(ctx, features, pitch_embedding);

        // energy_prediction = self.energy_predictor(x, mask)
        struct ggml_tensor *energy_prediction = energy_predictor.graph(gf, ctx, features);
        buckets = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, energy_prediction->ne[0]);
        buckets = ggml_map_custom2_inplace(ctx, buckets, energy_prediction, ggml_zv_mul_clamp_to_i32, GGML_N_TASKS_MAX, (void*)(intptr_t)ve_n_bins);
        struct ggml_tensor *energy_embedding = ggml_get_rows(ctx, energy_embedding_w, buckets);

        // features = features + energy_embedding
        features = ggml_add(ctx, features, energy_embedding);

        ggml_set_name(features, "features");
        ggml_set_output(features);
        ggml_build_forward_expand(gf, features);

        if (!ggml_gallocr_alloc_graph(alloc, gf))
            throw std::runtime_error("ggml_gallocr_alloc_graph() failed");
    }

    FS2Encoder::~FS2Encoder()
    {
        if (alloc)
            ggml_gallocr_free(alloc);
    }

    uint32_t FS2Encoder::eval(const int32_t *src_seq_data, const int32_t *puncts_data, const float *style_embed_data, uint32_t num_phonemes, float *x)
    {
        int emb_size = embed_dim + punct_embed_dim;

        ggml_backend_tensor_set(src_seq, src_seq_data, 0, max_n_phonemes*sizeof(int32_t));
        ggml_backend_tensor_set(puncts, puncts_data, 0, max_n_phonemes*sizeof(int32_t));
        ggml_backend_tensor_set(style_embed, style_embed_data, 0, emb_size*sizeof(float));

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("ggml_backend_graph_compute() failed");

        //struct ggml_tensor *x = ggml_graph_get_tensor(gf, "x");
        //print_tensor("x", x, 6);

        //struct ggml_tensor *x_punct = ggml_graph_get_tensor(gf, "x_punct");
        //print_tensor("x_punct", x_punct, 11);

        // length regulator: expand features to final mel seq len

        off_t xoff=0;
        std::memset(x, 0, max_seq_len*emb_size*sizeof(float));

        GGML_ASSERT(ggml_is_contiguous(features));
        const float *feat_data = ggml_get_data_f32(features);

        GGML_ASSERT(ggml_is_contiguous(log_duration_prediction));
        const float *dur_data = ggml_get_data_f32(log_duration_prediction);

        for (uint32_t i=0; i<num_phonemes; i++)
        {
            float dur = exp(dur_data[i])-1.0;
            int32_t duration_runded = (int32_t) (dur+0.5);
            if (duration_runded<0)
                continue;

            //printf("duration #%5d: %10.3f -> %10.3f -> %d\n", i, dur_data[i], dur, duration_runded);

            // repeat feature duration_runded times
            for (int32_t r=0; r<duration_runded; r++)
            {
                std::memcpy(&x[xoff*emb_size], &feat_data[i*emb_size], emb_size*sizeof(float));
                xoff += 1;
                if (xoff >= max_seq_len)
                    break;
            }
            if (xoff >= max_seq_len)
                break;
        }

        // for (off_t xo=0; xo<xoff; xo++)
        // {
        //     for (off_t e=0; e<5; e++)
        //     {
        //         printf("%g ", x[xo*emb_size+e]);
        //     }
        //     printf("\n");
        // }

        //printf ("length regulator done. xoff=%ld\n", xoff);

        return (uint32_t) xoff;

    }

}