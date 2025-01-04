// original python source:
//
// Parallel WaveGAN implementation with Pytorch
// https://github.com/kan-bayashi/ParallelWaveGAN
// by Tomoki Hayashi (Nagoya University) under MIT license
//
// This code is based on https://github.com/jik876/hifi-gan.
// by Jungil Kong (SK Telecom) under MIT license


#include <cstdio>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <stdexcept>

#include "zerovox.h"

namespace ZeroVOX
{
    static struct ggml_tensor *conv_transpose1d(ggml_cgraph  *gf,
                                                struct ggml_context *ctx,
                                                struct ggml_context *ctx_w,
                                                struct ggml_tensor  *x,
                                                int idx,
                                                int64_t stride,
                                                int64_t padding,
                                                int64_t output_padding,
                                                int64_t dilation)
    {
        char namebuf[GGML_MAX_NAME];

        std::snprintf(namebuf, GGML_MAX_NAME, "_meldec.upsamples.%d.1.w", idx);
        struct ggml_tensor *kernel = checked_get_tensor(ctx_w, namebuf);
        int64_t kernel_size = kernel->ne[0];

        std::snprintf(namebuf, GGML_MAX_NAME, "_meldec.upsamples.%d.1.b", idx);
        struct ggml_tensor *bias = checked_get_tensor(ctx_w, namebuf);

        int64_t in_channels = x->ne[1];
        int64_t input_length = x->ne[0];

        // Step 1: Upsampling (insert zeros between elements based on stride)
        int64_t upsampled_length = (input_length - 1) * stride + 1;
        // add input padding
        int64_t off = (dilation * (kernel_size - 1) - padding);
        int64_t padded_length = upsampled_length + 2 * off + output_padding;

        struct ggml_tensor *upsampled = ggml_new_tensor_2d(ctx, x->type, in_channels, padded_length);

        //upsampled[:, :, ::stride] = x
        struct ggml_tensor *upsampled_view = ggml_view_2d(ctx, upsampled, in_channels, input_length, stride*upsampled->nb[1], off*upsampled->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_transpose(ctx, x), upsampled_view));

        upsampled = ggml_cont(ctx, ggml_transpose(ctx, upsampled));

        // output = torch.nn.functional.conv1d(padded, flipped_weight, bias=None, stride=1, padding=0)

        struct ggml_tensor *y = ggml_conv_1d(ctx,
                                             kernel,      // convolution kernel
                                             upsampled,   // data
                                             1,           // stride
                                             0,           // padding
                                             1);          // dilation
        y = ggml_cont(ctx, ggml_transpose(ctx, y));
        y = ggml_add (ctx, y, ggml_repeat(ctx, bias, y));
        y = ggml_cont(ctx, ggml_transpose(ctx, y));

        return y;
    }


    static struct ggml_tensor *HiFiGANResidualBlock([[maybe_unused]]ggml_cgraph  *gf,
                                                    struct ggml_context *ctx,
                                                    [[maybe_unused]]struct ggml_context *ctx_w,
                                                    struct ggml_tensor  *x,
                                                    [[maybe_unused]]int                  idx,
                                                    [[maybe_unused]]int                  num_dilations,
                                                    [[maybe_unused]]const int64_t       *dilations)
    {
        [[maybe_unused]]char namebuf[GGML_MAX_NAME];

        // _meldec.blocks.0.convs1.0.1.b : [256,   1,   1,   1] f32
        // _meldec.blocks.0.convs1.0.1.w : [  3, 256, 256,   1] f16
        // _meldec.blocks.0.convs1.1.1.b : [256,   1,   1,   1] f32
        // _meldec.blocks.0.convs1.1.1.w : [  3, 256, 256,   1] f16
        // _meldec.blocks.0.convs1.2.1.b : [256,   1,   1,   1] f32
        // _meldec.blocks.0.convs1.2.1.w : [  3, 256, 256,   1] f16
        // _meldec.blocks.0.convs2.0.1.b : [256,   1,   1,   1] f32
        // _meldec.blocks.0.convs2.0.1.w : [  3, 256, 256,   1] f16
        // _meldec.blocks.0.convs2.1.1.b : [256,   1,   1,   1] f32
        // _meldec.blocks.0.convs2.1.1.w : [  3, 256, 256,   1] f16
        // _meldec.blocks.0.convs2.2.1.b : [256,   1,   1,   1] f32
        // _meldec.blocks.0.convs2.2.1.w : [  3, 256, 256,   1] f16

        struct ggml_tensor *y = x;
        
        for (int dil_idx = 0; dil_idx<num_dilations; dil_idx++)
        {
            // for idx in range(len(self.convs1)):
            // for dilation in dilations:
            //   self.convs1 += [
            //       torch.nn.Sequential(
            //           getattr(torch.nn, nonlinear_activation)(
            //               **nonlinear_activation_params
            //           ),
            struct ggml_tensor *xt = ggml_leaky_relu(ctx, y, 0.1, /*inplace=*/false);

            //           torch.nn.Conv1d(
            //               channels,
            //               channels,
            //               kernel_size,
            //               1,
            //               dilation=dilation,
            //               bias=bias,
            //               padding=(kernel_size - 1) // 2 * dilation,
            //           ),
            //       )
            //   ]
            //     xt = self.convs1[idx](x)

            std::snprintf(namebuf, GGML_MAX_NAME, "_meldec.blocks.%d.convs1.%d.1.w", idx, dil_idx);
            struct ggml_tensor *kernel = checked_get_tensor(ctx_w, namebuf);
            int64_t kernel_size = kernel->ne[0];

            std::snprintf(namebuf, GGML_MAX_NAME, "_meldec.blocks.%d.convs1.%d.1.b", idx, dil_idx);
            struct ggml_tensor *bias = checked_get_tensor(ctx_w, namebuf);

            int64_t dilation = dilations[dil_idx];

            xt = ggml_conv_1d(ctx,
                              kernel,                            // convolution kernel
                              xt,                                // data
                              1,                                 // stride
                              (kernel_size - 1) / 2 * dilation,  // padding
                              dilation);                         // dilation
            xt = ggml_cont(ctx, ggml_transpose(ctx, xt));
            xt = ggml_add (ctx, xt, ggml_repeat(ctx, bias, xt));
            xt = ggml_cont(ctx, ggml_transpose(ctx, xt));

            // if self.use_additional_convs:
            //     xt = self.convs2[idx](xt)

            // self.convs2 += [
            //     torch.nn.Sequential(
            //         getattr(torch.nn, nonlinear_activation)(
            //             **nonlinear_activation_params
            //         ),
            xt = ggml_leaky_relu(ctx, xt, 0.1, /*inplace=*/true);
            //         torch.nn.Conv1d(
            //             channels,
            //             channels,
            //             kernel_size,
            //             dilation=1,
            //             bias=bias,
            //             padding=(kernel_size - 1) // 2,
            //         ),
            //     )
            // ]

            std::snprintf(namebuf, GGML_MAX_NAME, "_meldec.blocks.%d.convs2.%d.1.w", idx, dil_idx);
            kernel = checked_get_tensor(ctx_w, namebuf);
            kernel_size = kernel->ne[0];

            std::snprintf(namebuf, GGML_MAX_NAME, "_meldec.blocks.%d.convs2.%d.1.b", idx, dil_idx);
            bias = checked_get_tensor(ctx_w, namebuf);

            xt = ggml_conv_1d(ctx,
                            kernel,                            // convolution kernel
                            xt,                                // data
                            1,                                 // stride
                            (kernel_size - 1) / 2,             // padding
                            1);                                // dilation
            xt = ggml_cont(ctx, ggml_transpose(ctx, xt));
            xt = ggml_add (ctx, xt, ggml_repeat(ctx, bias, xt));
            xt = ggml_cont(ctx, ggml_transpose(ctx, xt));

            //     x = xt + x

            y = ggml_add(ctx, y, xt);
        }
        return y;

    }

    HiFiGAN::HiFiGAN(ggml_context   &ctx_w,
                     ggml_backend_t  backend,
                     uint32_t        max_seq_len,
                     uint32_t        in_channels,
                     uint32_t        hop_size,
                     uint32_t        kernel_size,
                     int             num_upsamples,
                     const int      *upsample_scales,
                     int             num_resblocks,
                     int             num_resblock_dilations,
                     const int64_t  *resblock_dilations)
    {
        this->backend               = backend;
        this->max_seq_len           = max_seq_len;
        this->in_channels           = in_channels;
        this->hop_size              = hop_size;
        this->num_upsamples         = num_upsamples,
        this->num_resblocks         = num_resblocks;

        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

        mean  = checked_get_tensor(&ctx_w, "hifigan.mean");
        scale = checked_get_tensor(&ctx_w, "hifigan.scale");

        input_conv_b = checked_get_tensor(&ctx_w, "_meldec.input_conv.b");
        input_conv_w = checked_get_tensor(&ctx_w, "_meldec.input_conv.w");

        // _meldec.output_conv.1.b       : [  1,   1,   1,   1] f32
        // _meldec.output_conv.1.w       : [  7,  32,   1,   1] f16

        output_conv_b = checked_get_tensor(&ctx_w, "_meldec.output_conv.1.b");
        output_conv_w = checked_get_tensor(&ctx_w, "_meldec.output_conv.1.w");

        // build graph

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

        mel = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_channels, max_seq_len);
        ggml_set_name(mel, "mel");
        ggml_set_input(mel);

        // mel_x = (mel - self._meldec.mean) / self._meldec.scale

        struct ggml_tensor *mel_x = ggml_sub(ctx, mel, mean);
        mel_x = ggml_div_inplace(ctx, mel_x, scale);

        // mel = mel.transpose(1, 2)
        mel_x = ggml_cont(ctx, ggml_transpose(ctx, mel_x));

        // wav = self._meldec(c=mel)

        // c = self.input_conv(c)
        // self.input_conv = torch.nn.Conv1d(
        //     in_channels,
        //     channels,
        //     kernel_size,
        //     bias=bias,
        //     padding=(kernel_size - 1) // 2,
        struct ggml_tensor *c = ggml_conv_1d(ctx,
                                            input_conv_w,               // convolution kernel
                                            mel_x,                 // data
                                            1,                     // stride
                                            (kernel_size - 1) / 2, // padding
                                            1);                    // dilation
        c = ggml_cont(ctx, ggml_transpose(ctx, c));
        c = ggml_add(ctx, c, ggml_repeat(ctx, input_conv_b, c));
        c = ggml_cont(ctx, ggml_transpose(ctx, c));


        // for i in range(len(upsample_kernel_sizes)):
        for (int i = 0; i<num_upsamples; i++)
        {

            // self.upsamples += [
            //     torch.nn.Sequential(
            //         getattr(torch.nn, nonlinear_activation)(
            //             **nonlinear_activation_params
            //         ),

            // nonlinear_activation="LeakyReLU",
            // nonlinear_activation_params={"negative_slope": 0.1},

            c = ggml_leaky_relu(ctx, c, 0.1, /*inplace=*/true);

            //             torch.nn.ConvTranspose1d(
            //                 channels // (2**i),
            //                 channels // (2 ** (i + 1)),
            //                 upsample_kernel_sizes[i],
            //                 upsample_scales[i],
            //                 padding=upsample_scales[i] // 2 + upsample_scales[i] % 2,
            //                 output_padding=upsample_scales[i] % 2,
            //                 bias=bias,
            //             ),

            c = conv_transpose1d(gf, ctx, &ctx_w, c, /*idx=*/i,
                                 /*stride=*/upsample_scales[i],
                                 /*padding=*/upsample_scales[i] / 2 + upsample_scales[i] % 2,
                                 /*output_padding=*/upsample_scales[i] % 2,
                                 /*dilation=*/1);

            // cs = 0.0  # initialize
            struct ggml_tensor *cs = nullptr;
            // for j in range(self.num_blocks):
            for (int j=0; j<num_resblocks; j++)
            {
                //     cs += self.blocks[i * self.num_blocks + j](c)
                struct ggml_tensor *c_ = HiFiGANResidualBlock(gf, ctx, &ctx_w, c, /*idx=*/ i * num_resblocks + j,
                                                              num_resblock_dilations, &resblock_dilations[j*num_resblock_dilations]);
                if (cs)
                    cs = ggml_add(ctx, cs, c_);
                else
                    cs = c_;
            }

            // c = cs / self.num_blocks

            c = ggml_scale_inplace(ctx, cs, 1.0 / (float)num_resblocks);
        }

        // c = self.output_conv(c)
        // self.output_conv = torch.nn.Sequential(
        //     # NOTE(kan-bayashi): follow official implementation but why
        //     #   using different slope parameter here? (0.1 vs. 0.01)
        //     torch.nn.LeakyReLU(),

        c = ggml_leaky_relu(ctx, c, 1e-2, /*inplace=*/true);

        //     torch.nn.Conv1d(
        //         channels // (2 ** (i + 1)),
        //         out_channels,
        //         kernel_size,
        //         bias=bias,
        //         padding=(kernel_size - 1) // 2,
        //     ),

        c = ggml_conv_1d(ctx,
                         output_conv_w,         // convolution kernel
                         c,                     // data
                         1,                     // stride
                         (kernel_size - 1) / 2, // padding
                         1);                    // dilation
        c = ggml_cont(ctx, ggml_transpose(ctx, c));
        c = ggml_add(ctx, c, ggml_repeat(ctx, output_conv_b, c));
        c = ggml_cont(ctx, ggml_transpose(ctx, c));

        //     torch.nn.Tanh(),
        c = ggml_tanh_inplace(ctx, c);

        tensor_dbg (gf, ctx, c, "dbg");

        ggml_set_name(c, "x");
        ggml_set_output(c);
        ggml_build_forward_expand(gf, c);

        if (!ggml_gallocr_alloc_graph(alloc, gf))
            throw std::runtime_error("ggml_gallocr_alloc_graph() failed");

    }

    void HiFiGAN::eval(const float *mel_data, float *wav)
    {
        ggml_backend_tensor_set(mel, mel_data, 0, max_seq_len*in_channels*sizeof(float));

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("ggml_backend_graph_compute() failed");

        ggml_tensor *dbg;
        ggml_tensor *x;

        dbg = ggml_graph_get_tensor(gf, "dbg");
        print_tensor("dbg", dbg, 3);

        x = ggml_graph_get_tensor(gf, "x");
        print_tensor("x", x, 3);

        const float *x_data = ggml_get_data_f32(x);
        memcpy(wav, x_data, sizeof(float)*max_seq_len*hop_size);
 
    }
}
