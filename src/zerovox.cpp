#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <string>
#include <vector>
#include <format>
#include <cassert>
#include <cmath>

#include "zerovox.h"

const std::string g_gguf_filename = "medium-ldec.gguf";

namespace ZeroVOX
{

    ZeroVOXModel::ZeroVOXModel(const std::string & fname)
    {
        encoder = nullptr;

        ctx_w = nullptr;
        struct gguf_init_params params = {
            /*.no_alloc   =*/ true,
            /*.ctx        =*/ &ctx_w,
        };

        struct gguf_context * ctx_gguf = gguf_init_from_file(fname.c_str(), params);
        if (!ctx_gguf)
            throw std::runtime_error("gguf_init_from_file() failed");

        // extract hyperparams

        GGUF_GET_KEY(ctx_gguf, hparams.max_seq_len        , gguf_get_val_u32 , GGUF_TYPE_UINT32 , true, HPARAM_MAX_SEQ_LEN);

        GGUF_GET_KEY(ctx_gguf, hparams.emb_dim            , gguf_get_val_u32 , GGUF_TYPE_UINT32 , true, HPARAM_EMB_DIM);
        GGUF_GET_KEY(ctx_gguf, hparams.punct_emb_dim      , gguf_get_val_u32 , GGUF_TYPE_UINT32 , true, HPARAM_PUNCT_EMB_DIM);
        GGUF_GET_KEY(ctx_gguf, hparams.decoder_n_head     , gguf_get_val_u32 , GGUF_TYPE_UINT32 , true, HPARAM_DECODER_N_HEAD);
        GGUF_GET_KEY(ctx_gguf, hparams.conv_filter_size   , gguf_get_val_u32 , GGUF_TYPE_UINT32 , true, HPARAM_CONV_FILTER_SIZE);
        GGUF_GET_KEY(ctx_gguf, hparams.conv_kernel_size[0], gguf_get_val_u32 , GGUF_TYPE_UINT32 , true, HPARAM_CONV_KERNEL_SIZE_0);
        GGUF_GET_KEY(ctx_gguf, hparams.conv_kernel_size[1], gguf_get_val_u32 , GGUF_TYPE_UINT32 , true, HPARAM_CONV_KERNEL_SIZE_1);

        GGUF_GET_KEY(ctx_gguf, hparams.encoder_layer            , gguf_get_val_u32 , GGUF_TYPE_UINT32 , true, HPARAM_ENCODER_LAYER);
        GGUF_GET_KEY(ctx_gguf, hparams.encoder_head             , gguf_get_val_u32 , GGUF_TYPE_UINT32 , true, HPARAM_ENCODER_HEAD);
        GGUF_GET_KEY(ctx_gguf, hparams.encoder_vp_filter_size   , gguf_get_val_u32 , GGUF_TYPE_UINT32 , true, HPARAM_ENCODER_VP_FILTER_SIZE);
        GGUF_GET_KEY(ctx_gguf, hparams.encoder_vp_kernel_size   , gguf_get_val_u32 , GGUF_TYPE_UINT32 , true, HPARAM_ENCODER_VP_KERNEL_SIZE);
        GGUF_GET_KEY(ctx_gguf, hparams.encoder_ve_n_bins        , gguf_get_val_u32 , GGUF_TYPE_UINT32 , true, HPARAM_ENCODER_VE_N_BINS);

        // Initialize a backend
        backend = nullptr;

        #ifdef GGML_USE_CUDA
            fprintf(stderr, "%s: using CUDA backend\n", __func__);
            model.backend = ggml_backend_cuda_init(0); // init device 0
            if (!model.backend) {
                fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
            }
        #endif
        // if there aren't GPU Backends fallback to CPU backend
        if (!backend) {
            fprintf(stderr, "%s: using CPU backend\n", __func__);
            backend = ggml_backend_cpu_init();
        }

        buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
        if (!buf_w)
        {
            gguf_free(ctx_gguf);
            throw std::runtime_error("ggml_backend_alloc_ctx_tensors() failed");
        }

        FILE * f = fopen(fname.c_str(), "rb");
        if (!f) {
            gguf_free(ctx_gguf);
            throw std::runtime_error("fopen() failed");
        }

        encoder = new FS2Encoder(*ctx_w,
                                 MAX_N_PHONEMES,
                                 hparams.emb_dim,
                                 hparams.punct_emb_dim,
                                 hparams.encoder_layer,
                                 hparams.encoder_head,
                                 hparams.conv_filter_size,
                                 hparams.conv_kernel_size,
                                 hparams.encoder_vp_kernel_size,
                                 hparams.encoder_ve_n_bins,
                                 hparams.max_seq_len);


        const int n_tensors = gguf_get_n_tensors(ctx_gguf);

        for (int i = 0; i < n_tensors; i++)
        {
            const char * name = gguf_get_tensor_name(ctx_gguf, i);
            struct ggml_tensor * tensor = ggml_get_tensor(ctx_w, name);
            size_t offs = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);

            printf("%-30s: [%3ld, %3ld, %3ld, %3ld] %s\n",
                    name,
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                    ggml_type_name(tensor->type));

            std::vector<uint8_t> buf(ggml_nbytes(tensor));
            if (fseek(f, offs, SEEK_SET) != 0)
            {
                gguf_free(ctx_gguf);
                fclose(f);
                throw std::runtime_error("fseek() failed");
            }

            if (fread(buf.data(), 1, buf.size(), f) != buf.size())
            {
                gguf_free(ctx_gguf);
                fclose(f);
                throw std::runtime_error("fread() failed");
            }

            ggml_backend_tensor_set(tensor, buf.data(), 0, buf.size());

            // print_tensor(name, tensor, 10);

        }

        fclose(f);

        gguf_free(ctx_gguf);
    }

    ZeroVOXModel::~ZeroVOXModel()
    {
        if (encoder)
            delete encoder;
        ggml_backend_buffer_free(buf_w);
        ggml_backend_free(backend);
        ggml_free(ctx_w);
    }

}

#if 0
static bool zerovox_eval(zerovox_model &model)
{
    static ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    ggml_cgraph *gf = zerovox_graph(model, MAX_N_PHONEMES);

    if (!ggml_gallocr_alloc_graph(alloc, gf))
    {
        fprintf(stderr, "%s: ggml_gallocr_alloc_graph() failed\n", __func__);
        return false;
    }
    
    //std::vector<float> w((NUM_PHONEMES+1)*model.hparams.emb_dim);
    //ggml_backend_tensor_get(model.src_word_emb_w, w.data(), 0, (NUM_PHONEMES+1)*model.hparams.emb_dim*sizeof(float));

    struct ggml_tensor *src_seq     = ggml_graph_get_tensor(gf, "src_seq");
    struct ggml_tensor *puncts      = ggml_graph_get_tensor(gf, "puncts");
    struct ggml_tensor *style_embed = ggml_graph_get_tensor(gf, "style_embed");
    // struct ggml_tensor *pitch_min   = ggml_graph_get_tensor(gf, "pitch_min");
    // struct ggml_tensor *pitch_range = ggml_graph_get_tensor(gf, "pitch_range");
    struct ggml_tensor *y           = ggml_graph_get_tensor(gf, "y");

    // src_seq = [150, 115,  86,  60,  86,  38, 115,  87,  26,  86,  87]
    int32_t src_seq_data[MAX_N_PHONEMES] = {150, 115,  86,  60,  86,  38, 115,  87,  26,  86,  87};
    ggml_backend_tensor_set(src_seq, src_seq_data, 0, MAX_N_PHONEMES*sizeof(int32_t));

    // puncts = [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 3]
    int32_t puncts_data[MAX_N_PHONEMES] = {0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 3};
    ggml_backend_tensor_set(puncts, puncts_data, 0, MAX_N_PHONEMES*sizeof(int32_t));

    // float pitch_min_a[MAX_N_PHONEMES];
    // for (int i=0; i<MAX_N_PHONEMES; i++)
    //     pitch_min_a[i] = model.hparams.stats_pitch_min;
    // ggml_backend_tensor_set(pitch_min, &pitch_min_a, 0, MAX_N_PHONEMES*sizeof(float));

    // float pitch_range_a[MAX_N_PHONEMES];
    // for (int i=0; i<MAX_N_PHONEMES; i++)
    //     pitch_range_a[i] = model.hparams.stats_pitch_max - model.hparams.stats_pitch_min;
    // ggml_backend_tensor_set(pitch_range, &pitch_range_a, 0, MAX_N_PHONEMES*sizeof(float));

    float style_embed_data[528] = {
         6.4413e-02,  4.6879e-02, -2.6598e-02, -3.2998e-02, -2.7469e-03,
         2.0687e-02,  3.9076e-02,  1.6894e-05, -2.3216e-02,  5.8207e-02,
         4.0752e-02,  5.2596e-02,  3.3418e-02, -4.6485e-02, -8.8002e-04,
        -1.5139e-02,  6.4484e-02, -3.8063e-02,  1.2196e-02,  1.2775e-03,
         1.7364e-02,  4.3021e-02,  4.5563e-02, -1.0169e-01,  4.1363e-02,
         2.1426e-02,  2.6302e-02, -3.4999e-02,  8.9453e-03,  3.6006e-02,
         6.2847e-03, -1.2371e-02, -2.7606e-02, -2.6448e-02,  3.0774e-02,
        -3.7235e-02, -1.0069e-01, -3.0126e-02,  1.0730e-02, -4.0589e-02,
         3.3084e-02,  5.6684e-02,  1.6363e-01, -7.1753e-02, -5.3727e-02,
         1.2580e-03, -2.2575e-02, -3.0262e-02, -3.9339e-02, -2.3255e-02,
        -1.8450e-02, -3.2518e-03,  6.5418e-02, -1.4949e-02, -3.5673e-02,
        -4.0768e-02,  3.5910e-02,  3.8033e-02,  7.6466e-03,  6.4089e-02,
         3.8186e-02, -2.8315e-03, -6.6218e-02,  4.7510e-02,  6.2450e-02,
         1.3308e-02, -1.9381e-02,  2.5187e-02,  1.8493e-02, -1.7092e-02,
         3.1124e-02,  6.1219e-03,  2.3596e-02, -6.9266e-02, -1.3215e-02,
        -5.5941e-02, -2.0907e-02,  4.8014e-02, -1.5435e-02,  5.0603e-02,
         1.2983e-02,  4.4999e-02,  3.5882e-02, -1.0514e-02, -4.2758e-02,
        -2.1181e-02, -5.2432e-02, -4.4450e-02,  1.0488e-01,  3.6581e-03,
         4.9190e-03,  4.3898e-02,  1.8416e-02, -2.0955e-02, -3.0476e-02,
         3.2808e-03,  1.7472e-02,  2.1457e-02, -4.4757e-03, -2.2338e-02,
        -5.7156e-02, -1.1708e-02,  3.6488e-02, -3.7821e-02, -1.5225e-02,
        -2.5800e-02, -4.2393e-02, -5.8202e-02, -6.5573e-02, -2.1913e-02,
         2.9674e-02, -9.1968e-02, -2.1146e-02,  3.0766e-02,  5.7010e-02,
         3.7440e-02, -2.9700e-02,  5.5999e-03,  5.3362e-04, -3.2951e-02,
        -2.4883e-02, -2.4995e-02,  5.7880e-02,  4.9129e-02,  2.5473e-02,
         5.1472e-02,  1.0372e-03,  1.8515e-02, -1.6095e-02, -2.6175e-02,
        -4.0626e-03,  2.6469e-02,  1.8297e-02, -5.8756e-02, -2.3758e-02,
         4.4555e-02,  2.3268e-02,  2.5847e-02, -3.4271e-03,  2.4090e-02,
        -2.0315e-03,  3.7578e-02,  1.6186e-03, -2.7846e-03,  9.1695e-02,
         3.9006e-02,  1.6529e-03,  2.5806e-02, -1.5198e-02,  7.0474e-02,
         5.5976e-02,  2.7697e-02, -5.6932e-02, -6.9039e-02,  1.7611e-02,
        -5.2241e-02,  4.4882e-02,  3.4310e-02, -2.4360e-02, -2.3788e-03,
        -4.4939e-02,  4.3467e-03,  1.9399e-02, -3.1981e-02,  9.2626e-02,
        -7.2036e-02, -6.2976e-02,  1.8834e-02,  2.0321e-02,  4.8475e-03,
        -6.0823e-02, -4.2036e-02, -2.0489e-02,  5.1654e-02, -5.5813e-02,
        -5.0531e-03, -6.9612e-02, -3.5518e-02, -5.4424e-02, -1.5781e-02,
         8.8940e-03,  1.7808e-02,  9.6812e-03,  4.8326e-03, -1.0126e-02,
        -5.6048e-02, -2.3610e-02,  1.8102e-03,  1.2593e-02,  3.2506e-03,
        -3.1587e-03, -8.3154e-02,  8.2494e-02, -3.0046e-02, -2.0302e-02,
         2.1140e-02, -1.6413e-02, -6.1734e-04,  7.7558e-02, -3.8583e-02,
         4.3000e-02,  4.3590e-02,  6.2491e-02,  1.4449e-02, -7.4908e-02,
        -1.4065e-02, -3.3853e-03,  5.8812e-02, -1.2742e-02, -6.8500e-02,
         3.4716e-02, -3.2954e-02,  2.0460e-03, -1.1859e-02, -1.0502e-02,
         5.9386e-02, -7.5621e-03, -6.3917e-02,  4.9465e-02,  3.6628e-02,
         3.9381e-02,  3.0595e-02,  9.3308e-02, -4.7134e-02,  1.9396e-02,
         6.2322e-03,  3.1274e-03, -1.7853e-02, -2.7447e-02,  5.1133e-02,
         3.9239e-02,  1.7624e-02, -4.4946e-02, -6.0132e-03, -1.3930e-02,
        -8.4793e-02,  9.2985e-02,  9.8942e-03,  1.1648e-01, -1.4152e-02,
         3.2278e-02,  5.2247e-03, -1.4607e-02,  4.2390e-02,  5.9980e-02,
        -5.7770e-02,  4.8086e-02, -6.9875e-03, -2.3048e-02,  2.1690e-02,
         3.9874e-02,  1.1729e-02,  1.5177e-02,  6.4998e-02,  2.9886e-02,
         3.7317e-03, -9.6641e-02, -2.1629e-02,  2.8223e-03,  5.3406e-02,
         1.0433e-06, -1.1303e-02, -3.8274e-02,  1.4059e-02,  2.3294e-02,
         6.7201e-02, -7.6931e-02,  3.1107e-02, -6.4874e-02,  2.4931e-03,
        -4.6029e-02,  9.6119e-03,  3.6972e-02, -3.5562e-02, -5.0971e-02,
         3.9903e-03,  6.4043e-03, -4.2781e-02, -1.9898e-02, -9.5564e-02,
         5.1416e-02, -1.8215e-02, -4.7846e-03,  1.9780e-02, -1.5281e-03,
         5.1336e-02,  1.0279e-03, -3.4938e-02,  4.0660e-03, -7.3837e-03,
        -3.7284e-02,  1.8873e-02, -2.0185e-03, -7.7631e-02,  1.7847e-02,
        -1.2223e-02, -8.9148e-02,  6.7925e-03, -2.4165e-03, -5.5483e-02,
         4.3820e-02, -5.0662e-02, -5.8446e-02, -2.4010e-02, -2.0479e-02,
        -1.8235e-02, -4.5743e-02, -9.1451e-03,  4.8560e-02,  7.7167e-02,
         6.5621e-03, -4.1172e-02,  7.8471e-02, -6.7024e-02,  3.8501e-02,
        -4.0525e-03, -1.6528e-03,  3.7383e-02, -4.5214e-02,  1.3892e-03,
        -5.5247e-02, -3.1012e-02,  4.9840e-02, -4.5581e-03,  6.0740e-02,
         3.2317e-02,  2.5739e-02,  5.1442e-02,  8.2820e-03, -6.0750e-02,
        -8.1229e-02, -1.2485e-02, -3.0073e-02,  7.9099e-02, -3.0473e-02,
        -2.2765e-02,  7.8018e-03,  3.7624e-02,  4.0572e-02,  4.3503e-02,
         3.8614e-02,  2.7010e-03, -8.4070e-02, -4.2479e-02, -1.8932e-02,
         6.1075e-02, -1.9313e-02, -5.0625e-02, -4.7957e-02,  1.0107e-02,
        -6.7888e-02, -2.2787e-03, -1.4270e-01,  5.9217e-02,  5.5106e-02,
        -2.6081e-02,  2.9639e-02,  2.5789e-02,  2.2874e-02, -5.0503e-02,
         2.4235e-02, -2.8485e-03, -2.7239e-02,  2.4761e-03,  2.8014e-03,
         1.9715e-02,  1.9690e-02, -3.2433e-02, -8.9065e-02,  1.0432e-02,
        -3.3720e-02, -1.6476e-02,  8.6488e-02,  7.7011e-04, -8.4829e-02,
         3.3465e-02, -2.0689e-03,  1.0350e-01, -8.3425e-03, -1.7746e-02,
         5.4856e-02, -7.4056e-02, -1.0204e-01, -4.6861e-03,  1.0409e-02,
         5.3423e-02, -5.6020e-02,  3.4654e-02,  3.2104e-02, -7.9552e-02,
        -2.1279e-02,  3.8467e-02, -2.6825e-02, -8.6757e-02,  3.3008e-02,
        -3.2012e-02, -4.8109e-02,  1.3579e-02,  7.4214e-02, -1.8319e-02,
         3.0584e-02, -1.8285e-02,  1.0230e-02,  7.3613e-02,  2.0257e-02,
         5.9807e-02, -1.1903e-02, -3.3789e-02,  5.1037e-02, -2.3728e-02,
         2.1789e-02, -8.6820e-03,  3.7088e-02, -6.9707e-02,  4.6548e-02,
        -5.7936e-02, -9.2730e-03,  8.5789e-03,  1.6880e-02, -1.5902e-02,
         2.7376e-02,  3.1349e-02, -8.0938e-02,  2.9405e-02, -1.8493e-02,
         3.6997e-02, -8.2751e-02,  5.7971e-02, -2.7808e-03, -1.8418e-02,
        -3.2973e-02,  1.0583e-01, -6.7228e-02, -4.1803e-02,  2.5459e-02,
         2.2866e-02, -7.1230e-03, -9.4446e-02,  5.3193e-02, -4.6769e-02,
        -1.0804e-02,  1.9894e-02,  7.8438e-02,  1.0654e-02,  5.0386e-02,
         4.3779e-02,  2.3314e-02,  8.6451e-02, -7.0233e-02, -8.1798e-02,
        -5.0694e-03,  9.3770e-03,  4.0149e-02,  5.1429e-03,  2.0888e-04,
         1.4781e-02, -1.5224e-02, -1.0574e-02,  6.3091e-02,  1.8778e-02,
        -4.2481e-02,  4.6084e-02,  9.5147e-03, -4.4561e-02, -4.9554e-02,
        -3.4259e-02, -7.4075e-02,  4.1931e-02, -1.8020e-02, -2.6464e-02,
        -5.0500e-03,  3.5008e-02, -2.3535e-02, -9.8669e-02,  4.7585e-02,
        -1.1820e-02, -4.6142e-02,  1.2068e-02, -5.3374e-02,  6.2627e-03,
        -7.7500e-02, -3.8245e-02,  5.5429e-02, -6.0840e-02,  5.5183e-02,
         1.7353e-02, -1.0582e-03, -1.1141e-01,  1.3023e-02, -1.2360e-02,
         2.3702e-03,  6.2283e-02,  4.7141e-02,  1.9399e-02,  6.0652e-02,
        -7.4375e-02,  6.3541e-02, -3.2369e-02, -1.9760e-02, -2.6573e-02,
        -5.9831e-02, -5.8190e-02,  5.1493e-02, -1.0775e-02,  7.3810e-02,
        -2.2087e-02,  1.9343e-03,  6.0284e-02,  2.5526e-02,  6.9142e-02,
        -5.8340e-02,  2.0689e-02,  4.0069e-02,  4.8823e-03,  4.6079e-02,
         6.6860e-03, -9.4775e-03, -3.7937e-02, -2.9789e-03, -7.7826e-03,
         1.6546e-02,  1.2974e-02, -3.5167e-02, -3.6223e-02,  4.8041e-02,
         3.9814e-02, -2.8331e-04,  1.3723e-03
    };

    ggml_backend_tensor_set(style_embed, style_embed_data, 0, 528*sizeof(float));

    if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS)
    {
        fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        return false;
    }


    //struct ggml_tensor *x = ggml_graph_get_tensor(gf, "x");
    //print_tensor("x", x, 6);

    //struct ggml_tensor *x_punct = ggml_graph_get_tensor(gf, "x_punct");
    //print_tensor("x_punct", x_punct, 11);

    struct ggml_tensor *dbg = ggml_graph_get_tensor(gf, "dbg");
    print_tensor("dbg", dbg, 3);

    // struct ggml_tensor *dbg2 = ggml_graph_get_tensor(gf, "dbg2");
    // print_tensor("dbg2", dbg2, 11);

    print_tensor("y", y, 6);

    return true;
}
#endif

int main(void)
{

    ZeroVOX::ZeroVOXModel model(g_gguf_filename);


    //zerovox_eval(model);

    return 0;
}
