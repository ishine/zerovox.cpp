#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <string>
#include <vector>
#include <format>
#include <cassert>
#include <cmath>
#include <cstring>

#include <sndfile.h>

#include "zerovox.h"

const std::string g_gguf_filename = "medium-ldec.gguf";

namespace ZeroVOX
{

    ZeroVOXModel::ZeroVOXModel(const std::string & fname)
    {
        encoder      = nullptr;
        hidden_state = nullptr;
        mel          = nullptr;

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

        GGUF_GET_KEY(ctx_gguf, hparams.audio_sampling_rate      , gguf_get_val_u32 , GGUF_TYPE_UINT32 , true, HPARAM_AUDIO_SAMPLING_RATE);
        GGUF_GET_KEY(ctx_gguf, hparams.audio_num_mels           , gguf_get_val_u32 , GGUF_TYPE_UINT32 , true, HPARAM_AUDIO_NUM_MELS);
        GGUF_GET_KEY(ctx_gguf, hparams.audio_hop_size           , gguf_get_val_u32 , GGUF_TYPE_UINT32 , true, HPARAM_AUDIO_HOP_SIZE);

        
        //hparams.max_seq_len = 116;//FIXME: remove debug code

        // alloc buffers

        hidden_state = (float*) std::malloc(hparams.max_seq_len*(hparams.emb_dim+hparams.punct_emb_dim)*sizeof(float));
        if (!hidden_state)
            throw std::runtime_error("hidden_state malloc() failed");

        mel = (float*) std::malloc(hparams.max_seq_len*hparams.audio_num_mels*sizeof(float));
        if (!mel)
            throw std::runtime_error("mel malloc() failed");

        wav = (float*) std::malloc(hparams.max_seq_len*hparams.audio_hop_size*sizeof(float));
        if (!wav)
            throw std::runtime_error("wav malloc() failed");

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
                                 backend,
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

        uint32_t emb_size = hparams.emb_dim+hparams.punct_emb_dim;

        decoder = new StyleTTSDecoder(*ctx_w,
                                      backend,
                                      hparams.max_seq_len,
                                      /*dim_in=*/emb_size,
                                      /*style_dimm=*/emb_size,
                                      /*residual_dim=*/64,
                                      hparams.audio_num_mels);

        const int kernel_size                    = 7;
        const int num_upsamples                  = 4;
        int upsample_scales[num_upsamples]       = {5,5,4,3};
        const int num_resblocks                  = 3;
        const int num_resblock_dilations         = 3;
        int64_t resblock_dilations[num_resblocks*num_resblock_dilations] = { 1, 3, 5,
                                                                             1, 3, 5,
                                                                             1, 3, 5};

        meldec = new HiFiGAN(*ctx_w, backend, hparams.max_seq_len, hparams.audio_num_mels, hparams.audio_hop_size,
                             kernel_size,
                             num_upsamples, upsample_scales, num_resblocks, num_resblock_dilations, resblock_dilations );

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
        if (decoder)
            delete decoder;
        if (meldec)
            delete meldec;
        if (hidden_state)
            free(hidden_state);
        if (mel)
            free(mel);
        ggml_backend_buffer_free(buf_w);
        ggml_backend_free(backend);
        ggml_free(ctx_w);
    }

    void ZeroVOXModel::eval(void)
    {
        int num_phonemes = MAX_N_PHONEMES; // FIXME

        // Der Regenbogen ist ein atmosphärisch-optisches Phänomen, das als kreisbogenförmiges farbiges Lichtband in einem von der Sonne beschienenen Regenschauer erscheint.

        int32_t src_seq_data[MAX_N_PHONEMES] = {69, 26, 102, 117, 5, 73, 109, 81, 68, 83, 73, 109, 81, 60, 86, 87, 35, 81, 123, 87, 80, 105, 86, 72, 27, 117, 115, 118, 54, 84, 87, 115, 118, 109, 86, 72, 110, 81, 83, 80, 5, 81, 69, 0, 86, 34, 79, 86, 78, 117, 1, 86, 68, 83, 73, 109, 81, 72, 100, 102, 80, 115, 73, 109, 86, 72, 0, 102, 68, 115, 73, 109, 86, 79, 30, 97, 87, 68, 64, 81, 87, 60, 81, 35, 81, 109, 80, 72, 21, 81, 69, 26, 102, 95, 21, 81, 109, 68, 109, 118, 7, 81, 109, 81, 109, 81, 117, 5, 73, 109, 81, 118, 66, 102, 145, 102, 118, 1, 81, 87};
        int32_t puncts_data [MAX_N_PHONEMES] = {0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 2, 2, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 3};

        float style_embed_data[528] = {
         -3.3215e-02,  3.8201e-02, -5.9022e-02,  1.1678e-02,  3.6475e-02,
         -2.7423e-02, -6.1589e-04, -5.7507e-02, -8.3661e-02, -7.9317e-02,
         1.6819e-02,  6.2799e-05, -3.5430e-02,  2.8256e-02,  3.1210e-03,
         1.0054e-02,  8.5303e-02, -1.0629e-02, -8.0564e-02, -1.4030e-02,
         6.3730e-02, -8.6354e-03,  4.7948e-02, -1.9710e-02, -2.8996e-02,
        -6.5679e-02,  2.3563e-02, -9.5020e-03,  7.5294e-02, -4.4114e-02,
         6.2254e-02, -2.9038e-02,  4.8717e-02,  4.5398e-02, -5.0592e-03,
         5.8795e-02, -2.6262e-02,  1.8539e-02,  2.6485e-02,  3.8732e-02,
        -1.5195e-02, -2.4412e-02,  5.4493e-02, -2.6948e-02, -1.3855e-02,
         4.0775e-02, -3.5305e-02, -8.2256e-02,  2.3287e-02,  1.1727e-02,
         3.6795e-03, -3.4937e-02, -3.2590e-03,  3.8497e-04, -7.2939e-04,
        -3.1414e-02,  5.4537e-02,  2.5797e-02, -4.5980e-02, -1.4410e-03,
         3.8969e-02,  3.2946e-03,  3.1352e-02,  1.4927e-02, -2.4787e-02,
         3.0365e-02,  5.8112e-02,  4.0310e-02,  5.1757e-02, -9.5479e-03,
         5.6483e-03, -4.1380e-02, -3.9694e-02,  1.3546e-02, -4.7699e-02,
        -2.2485e-02, -4.7646e-02, -1.9125e-02, -4.4340e-02,  5.4521e-02,
        -4.6883e-02, -4.2803e-02, -5.0035e-02,  5.2024e-02, -8.6474e-02,
         5.6821e-02,  3.0837e-02,  1.9413e-02, -3.2098e-02, -2.3332e-02,
         3.7121e-02,  2.5287e-02,  9.8179e-03, -1.5703e-02,  3.2516e-02,
        -4.4404e-02, -1.9952e-02, -8.5996e-03,  4.3328e-03,  5.4122e-02,
         2.8946e-02, -3.0179e-03,  2.8949e-02, -2.5539e-02, -4.8855e-03,
        -1.1224e-02, -3.2807e-02,  4.4556e-02,  3.2388e-02, -2.2374e-02,
         9.1722e-02,  3.6068e-02,  1.7756e-02, -3.1594e-02, -9.2172e-02,
        -1.0658e-01, -4.1726e-02,  5.8524e-02, -1.9533e-02,  2.1717e-02,
         3.5877e-02, -4.0133e-02,  3.4485e-02,  7.7395e-02,  2.1210e-02,
         5.8594e-02, -1.9482e-02, -3.6054e-02, -1.1842e-02, -4.1441e-02,
         6.7316e-04, -1.1915e-02,  1.3289e-02, -2.2518e-02,  1.7770e-02,
        -7.8534e-03,  1.1667e-02,  2.9361e-02,  1.7638e-02,  1.1973e-02,
         1.7526e-02, -4.4542e-02, -3.3217e-02, -2.5015e-02, -5.4413e-02,
        -1.9724e-02, -2.5991e-03, -5.0686e-02, -2.7104e-02,  4.1738e-02,
        -2.0600e-02, -7.7754e-02, -1.1024e-02, -3.8237e-02,  4.5542e-02,
         5.2697e-02,  5.0668e-02, -9.0702e-04,  6.3776e-03, -2.7626e-02,
         2.8720e-02, -6.4390e-03,  1.3270e-02, -3.6621e-02, -4.2676e-02,
         1.7160e-02,  6.6821e-04, -9.1236e-03,  7.9707e-02, -3.1427e-02,
        -9.7707e-02,  3.0654e-02, -2.3645e-02, -6.7404e-02,  5.8089e-02,
        -1.5079e-02, -5.3696e-03,  1.9862e-01,  7.3243e-02,  7.9739e-02,
        -5.0053e-02,  3.1905e-05, -1.0611e-01,  3.2571e-02, -3.8418e-02,
         4.8736e-03,  5.1538e-02,  2.3183e-03,  3.4974e-02, -1.6590e-02,
         3.5530e-02,  1.9276e-02,  6.6471e-02, -3.4386e-02,  6.7165e-03,
         3.2527e-02, -8.9185e-03,  1.9203e-02,  2.1367e-02,  2.7984e-02,
        -2.2409e-02,  6.8348e-02, -1.2399e-03,  2.6317e-02,  6.3847e-02,
        -3.4924e-03, -2.2350e-02, -9.2317e-03,  4.9535e-02, -2.0052e-02,
         1.0053e-01,  7.5240e-03,  1.9086e-03,  1.5362e-02, -9.4972e-02,
         1.6310e-03,  3.5154e-02,  3.4412e-02,  4.0742e-02, -4.3078e-02,
         3.8023e-02,  7.8372e-03, -7.4794e-03,  5.0684e-02, -5.1240e-02,
         5.3390e-02, -2.7314e-02,  3.5919e-03,  2.5983e-02, -6.9249e-03,
        -6.0799e-02,  4.4554e-02, -3.2238e-02,  4.6624e-02, -2.6189e-02,
         2.3149e-02,  6.9735e-02, -8.9658e-03, -5.4266e-03, -7.9390e-03,
         1.8791e-02,  9.7308e-03, -1.0107e-02, -3.9234e-02, -3.2387e-03,
        -7.5655e-03, -1.9982e-02,  4.1388e-02, -5.6373e-02,  3.0449e-02,
         3.1917e-02, -1.2647e-03, -1.1675e-02, -3.0076e-02,  7.6463e-02,
        -1.8073e-02, -5.3291e-03, -1.7936e-02, -2.1800e-03,  5.7789e-02,
        -3.9775e-02,  3.9785e-02, -3.5088e-03,  6.1291e-03,  3.4373e-03,
         2.2329e-02,  4.7882e-02,  1.9970e-02, -5.9728e-02,  3.1290e-02,
        -1.5654e-03, -1.3179e-03, -8.5198e-03,  1.8759e-03,  4.7919e-03,
        -7.5008e-02, -1.5729e-03,  6.3809e-02, -2.1049e-02,  3.2666e-02,
        -5.2376e-02, -3.1811e-03,  7.8827e-02,  1.5260e-02,  4.1064e-02,
        -4.4569e-02,  2.8538e-02, -1.3009e-02,  1.9214e-02, -1.4053e-02,
         3.2405e-02, -2.9060e-02, -3.7925e-02,  2.5997e-02, -4.1312e-02,
        -1.1945e-02,  1.7964e-02, -7.7487e-02,  1.1780e-02, -3.5243e-02,
         4.3898e-02, -1.7044e-02,  3.9825e-02,  5.9645e-02, -6.9346e-02,
        -2.1006e-03,  2.1957e-02, -1.1927e-02,  2.8720e-02,  1.0882e-02,
        -3.5296e-02,  1.0871e-03,  3.1802e-02,  8.5042e-02,  3.3373e-02,
        -8.2192e-03, -6.0587e-02,  4.8460e-02,  3.3073e-02, -2.0976e-02,
         3.1006e-02,  5.0424e-02, -2.6720e-02,  1.7916e-02, -2.1062e-01,
        -4.6970e-02,  8.7889e-03,  1.2457e-02, -4.0161e-02,  1.0424e-02,
         3.4485e-02, -3.5720e-02,  2.2036e-02,  1.1039e-02,  3.0930e-02,
         9.6744e-04,  1.7452e-02, -6.0250e-02, -3.2940e-02,  3.9362e-02,
         4.8447e-02,  3.2648e-02,  1.8395e-02, -6.9633e-02, -9.9153e-02,
        -2.8067e-02, -4.6497e-03, -3.5246e-02, -1.1738e-02, -3.4535e-02,
        -5.7202e-02,  2.4374e-02,  8.9280e-03, -4.1428e-02, -1.9166e-02,
         1.9527e-02, -4.4860e-02,  4.5595e-02,  3.9441e-02,  6.5347e-02,
         5.7278e-02,  3.2119e-02, -2.4958e-02, -1.9082e-02, -3.5265e-02,
         9.5697e-03, -7.5254e-03, -4.3990e-02, -6.4076e-02,  8.1393e-02,
         7.4648e-03,  9.1873e-02, -1.4765e-02, -1.1455e-02,  3.0522e-02,
         2.1051e-02,  4.6780e-02,  1.3205e-02, -1.4833e-02, -5.4270e-02,
         9.8399e-03,  1.7885e-02,  9.3177e-03,  6.5344e-03, -6.9957e-02,
         1.5476e-02,  9.6923e-02,  9.0533e-03,  6.3101e-02,  2.1883e-02,
        -5.3379e-02, -2.1589e-02,  5.9184e-02, -1.3627e-02,  7.6371e-02,
         3.3433e-02, -4.0558e-02,  1.8679e-01,  2.9458e-02, -8.6836e-03,
        -3.8180e-03,  4.3705e-02,  9.7839e-03,  1.0383e-02,  6.7799e-02,
        -6.7745e-03, -2.3792e-03,  2.5210e-02,  5.6954e-02, -9.6462e-02,
        -1.7826e-02, -5.4665e-03,  7.2767e-03, -4.6537e-03,  1.3716e-02,
         7.3392e-02, -5.2109e-03, -5.1407e-03, -3.5027e-02,  1.1521e-03,
         2.3709e-03,  3.6950e-02,  5.8684e-03, -4.0941e-02, -6.3830e-02,
         3.7619e-02,  1.6036e-02,  5.2230e-02,  3.2547e-02,  7.6609e-04,
         6.2975e-02,  1.9834e-02, -4.9903e-02,  7.0185e-02,  1.4505e-02,
        -1.0321e-02, -9.5576e-03,  1.1040e-01,  2.6941e-02,  2.8073e-02,
        -3.9346e-02,  4.4460e-02, -4.5495e-02,  1.5063e-02,  1.1960e-02,
        -1.7319e-02,  4.8269e-02,  8.1929e-02, -3.4443e-03,  7.9763e-03,
        -7.1177e-02,  6.6055e-03,  2.3034e-02, -5.1478e-02, -1.4807e-03,
         5.9008e-02, -1.9507e-02, -2.8964e-02, -7.1374e-02,  4.6614e-02,
        -4.9382e-02,  1.4291e-02,  3.6436e-02, -1.6752e-02,  1.6439e-02,
        -2.7916e-02,  3.3630e-02, -5.3471e-02, -2.5925e-02,  9.5559e-02,
         1.0826e-02, -2.1186e-02, -2.6649e-02,  4.3056e-03,  9.9544e-03,
        -4.9696e-02,  4.2241e-02,  6.4465e-02, -1.7665e-02, -2.9195e-02,
         2.5760e-02, -7.6103e-02,  8.7177e-02, -2.2697e-03, -4.3329e-02,
         6.4015e-03,  2.9885e-03, -3.6598e-02, -1.1688e-02, -3.4011e-02,
        -4.5343e-03, -9.0722e-03,  2.6475e-03,  1.8297e-02,  5.6105e-02,
        -4.1172e-02,  1.4369e-02,  7.6279e-03, -8.7920e-03, -6.0066e-02,
        -1.6982e-02,  1.2943e-02,  6.9036e-02,  2.1832e-02, -4.8196e-03,
         4.1074e-02, -5.3367e-02, -1.2246e-02, -5.1704e-02, -2.7597e-04,
         2.8585e-03, -2.1111e-02, -2.4354e-02,  1.7635e-01, -1.2212e-02,
        -6.5809e-03,  2.1182e-02,  3.5522e-02,  3.8160e-02, -1.4582e-02,
        -5.0781e-02,  8.8718e-02, -2.7904e-02, -2.7633e-02, -3.6089e-02,
        -2.1236e-02,  1.8911e-01, -5.5733e-02
        };

        //ggml_backend_tensor_set(style_embed, style_embed_data, 0, 528*sizeof(float));

        // if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS)
        // {
        //     fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        //     return false;
        // }

        // run FastSpeech2 encoder

        encoder->eval(src_seq_data, puncts_data, style_embed_data, num_phonemes, hidden_state);

        // run StyleTTS decoder

        decoder->eval(hidden_state, style_embed_data, mel);

        // run HifiGAN

        meldec->eval(mel, wav);
    }

    bool ZeroVOXModel::write_wav_file(const std::string &fname)
    {

        // // Function to generate a sine wave for testing
        // std::vector<float> generateSineWave(double frequency, double duration, int sampleRate) {
        //     std::vector<float> samples;
        //     for (int i = 0; i < duration * sampleRate; ++i) {
        //         double time = static_cast<double>(i) / sampleRate;
        //         samples.push_back(static_cast<float>(sin(2.0 * M_PI * frequency * time)));
        //     }
        //     return samples;
        // }

        // --- Generate Sample Data (Sine Wave in this example) ---
        //std::vector<float> audioData = generateSineWave(frequency, duration, sampleRate);

        // --- libsndfile Setup ---
        SF_INFO sfinfo;
        sfinfo.samplerate = hparams.audio_sampling_rate;
        sfinfo.channels   = 1;
        sfinfo.format     = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

        // --- Open the WAV file for writing ---
        SNDFILE* outfile = sf_open(fname.c_str(), SFM_WRITE, &sfinfo);

        if (!outfile)
        {
            std::cerr << "Error opening " << fname << " : " << sf_strerror(NULL) << std::endl;
            return false;
        }

        // --- Write the float data to the WAV file ---
        sf_count_t num_frames = hparams.max_seq_len*hparams.audio_hop_size;
        sf_count_t frames_written = sf_write_float(outfile, wav, num_frames);

        if (frames_written != num_frames)
        {
            std::cerr << "Error writing to file " << fname << " .  Frames written: " << frames_written << " Expected: " << num_frames << std::endl;
            std::cerr << "Error: " << sf_strerror(outfile) << std::endl; // Get error from the file object
            sf_close(outfile);
            return false;
        }


        // --- Close the file ---
        if (sf_close(outfile) != 0)
        {
            std::cerr << "Error closing " << fname <<  " : " << sf_strerror(NULL) << std::endl;
            return false;
        }

        std::cout << "Successfully created " << fname << " with " << frames_written << " samples." << std::endl;

        return true;
    }
}



int main(void)
{

    ZeroVOX::ZeroVOXModel model(g_gguf_filename);

    model.eval();

    model.write_wav_file("foo.wav");

    return 0;
}
