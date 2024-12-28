#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <string>
#include <vector>
#include <format>
#include <cassert>
#include <cmath>
#include <cstring>

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

    void ZeroVOXModel::eval(void)
    {
        // static ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        // ggml_cgraph *gf = zerovox_graph(model, MAX_N_PHONEMES);

        // if (!ggml_gallocr_alloc_graph(alloc, gf))
        // {
        //     fprintf(stderr, "%s: ggml_gallocr_alloc_graph() failed\n", __func__);
        //     return false;
        // }
        
        //std::vector<float> w((NUM_PHONEMES+1)*model.hparams.emb_dim);
        //ggml_backend_tensor_get(model.src_word_emb_w, w.data(), 0, (NUM_PHONEMES+1)*model.hparams.emb_dim*sizeof(float));

        // struct ggml_tensor *src_seq     = ggml_graph_get_tensor(gf, "src_seq");
        // struct ggml_tensor *puncts      = ggml_graph_get_tensor(gf, "puncts");
        // struct ggml_tensor *style_embed = ggml_graph_get_tensor(gf, "style_embed");
        // // struct ggml_tensor *pitch_min   = ggml_graph_get_tensor(gf, "pitch_min");
        // // struct ggml_tensor *pitch_range = ggml_graph_get_tensor(gf, "pitch_range");
        // struct ggml_tensor *y           = ggml_graph_get_tensor(gf, "y");

        int num_phonemes = 11;

        // src_seq = [150, 115,  86,  60,  86,  38, 115,  87,  26,  86,  87]
        int32_t src_seq_data[MAX_N_PHONEMES] = {150, 115,  86,  60,  86,  38, 115,  87,  26,  86,  87};
        //ggml_backend_tensor_set(src_seq, src_seq_data, 0, MAX_N_PHONEMES*sizeof(int32_t));

        // puncts = [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 3]
        int32_t puncts_data[MAX_N_PHONEMES] = {0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 3};
        //ggml_backend_tensor_set(puncts, puncts_data, 0, MAX_N_PHONEMES*sizeof(int32_t));

        // float pitch_min_a[MAX_N_PHONEMES];
        // for (int i=0; i<MAX_N_PHONEMES; i++)
        //     pitch_min_a[i] = model.hparams.stats_pitch_min;
        // ggml_backend_tensor_set(pitch_min, &pitch_min_a, 0, MAX_N_PHONEMES*sizeof(float));

        // float pitch_range_a[MAX_N_PHONEMES];
        // for (int i=0; i<MAX_N_PHONEMES; i++)
        //     pitch_range_a[i] = model.hparams.stats_pitch_max - model.hparams.stats_pitch_min;
        // ggml_backend_tensor_set(pitch_range, &pitch_range_a, 0, MAX_N_PHONEMES*sizeof(float));

        float style_embed_data[528] = {
         7.3659e-02,  5.0266e-04, -6.5428e-03, -3.7471e-02, -1.0563e-03,
         6.9590e-02, -1.0326e-03,  1.7677e-02, -9.3176e-03,  5.8936e-02,
        -1.0843e-02,  7.9101e-02,  7.6831e-03, -3.2428e-02,  7.5209e-03,
         2.7240e-02,  5.7505e-02, -2.0784e-02,  2.1862e-02, -6.8526e-03,
         1.4014e-02,  2.2559e-02,  2.9803e-02, -7.5510e-02,  2.4085e-02,
        -4.3732e-03,  5.0991e-02, -2.5234e-02,  6.5645e-03, -2.0487e-02,
        -2.9009e-03,  1.5339e-02, -3.3771e-02, -1.6031e-02,  1.7801e-02,
        -2.7948e-02, -9.8711e-02,  2.3475e-03,  4.5506e-02, -2.3937e-02,
         2.6321e-02,  6.1904e-02,  1.2196e-01, -8.8925e-02, -8.5295e-03,
         7.6742e-03, -1.7869e-02, -3.1901e-02, -6.7269e-02,  1.3832e-03,
        -1.2114e-03,  3.8728e-02,  4.7201e-02,  2.8416e-02, -4.3200e-02,
        -4.8024e-02,  4.4020e-02,  2.9262e-02, -1.8148e-02,  3.0653e-02,
         3.3095e-02,  8.7608e-03, -1.7734e-02,  3.9198e-02,  5.4806e-02,
         1.0569e-02, -1.7970e-02,  7.6060e-02, -1.0480e-02, -2.1298e-02,
         1.2409e-02,  8.7465e-03,  1.2928e-02, -6.2916e-02, -9.8409e-03,
        -2.5053e-02,  1.0192e-02,  5.0692e-02, -4.6819e-02,  3.5347e-02,
        -3.6800e-03,  3.3029e-02,  3.3573e-02,  1.1007e-02, -1.3887e-02,
         7.5855e-03, -7.8048e-02, -2.9146e-02,  7.4297e-02, -2.5245e-03,
         1.1319e-03,  6.1587e-02,  1.2090e-02, -2.6567e-03, -2.8753e-02,
        -1.6253e-02,  2.4569e-02,  1.8101e-02, -1.0581e-02, -3.0767e-02,
        -3.8904e-02, -4.1255e-02,  3.8281e-02, -1.9447e-02, -1.7183e-02,
        -2.6157e-02,  8.6538e-03, -2.2790e-02, -3.8058e-02, -3.0478e-02,
         1.7347e-02, -6.5259e-02, -5.5485e-02,  2.9537e-02,  4.7108e-02,
         8.0406e-03, -5.6096e-02,  3.1028e-02, -3.3126e-03, -4.4283e-02,
        -3.6299e-03,  1.8594e-02,  3.5091e-02,  4.7711e-02,  1.4424e-02,
         6.3097e-02, -2.9126e-02,  3.2912e-02, -3.0261e-02, -4.1327e-02,
        -2.9079e-02,  4.2970e-02, -4.9886e-03, -1.6280e-03, -3.1329e-02,
         2.5877e-02,  8.2097e-03,  2.9592e-02, -2.5918e-02,  4.5382e-02,
        -7.4126e-03,  1.6643e-02, -1.7109e-02, -3.0812e-02,  1.0357e-01,
         4.1917e-02,  1.1606e-02,  1.5964e-02,  6.3962e-02,  4.8968e-02,
         3.6265e-02,  6.6040e-02, -8.2431e-02, -6.2185e-02, -1.0214e-04,
        -7.0343e-02,  3.4672e-02,  2.2465e-02, -4.2347e-02, -1.2055e-02,
        -9.3694e-03,  1.9636e-02,  2.1955e-02, -6.7462e-02,  8.7874e-02,
        -8.1950e-02, -6.4222e-02,  4.8626e-02,  1.6362e-02, -1.4027e-02,
        -3.9972e-02, -3.7761e-02, -2.6421e-03,  6.1635e-02, -7.3240e-02,
        -2.5963e-02, -4.2246e-02, -3.6840e-02, -2.2311e-02,  4.7500e-02,
         3.1948e-02,  1.0754e-02,  3.5694e-02, -1.6924e-02, -3.5907e-02,
        -4.8971e-02, -4.4917e-02, -2.4257e-02,  4.0181e-02,  6.9334e-03,
         6.3546e-03, -7.3680e-02,  4.3863e-02, -4.1697e-02, -7.5610e-02,
         5.6283e-02, -1.5105e-02, -2.0589e-03,  4.9693e-02, -3.4776e-02,
         1.0781e-02,  6.4383e-02,  3.3948e-02,  1.1891e-02, -5.7819e-02,
        -2.0503e-02,  4.1314e-02,  4.3428e-02, -5.8559e-02, -6.2019e-02,
         4.0894e-02,  5.2956e-03,  2.8838e-03,  2.1549e-02,  1.0729e-02,
         6.2147e-02,  3.4747e-02, -1.0820e-01,  7.4249e-02, -3.0443e-02,
        -2.2190e-02,  4.1751e-02,  1.1029e-01, -1.4219e-02,  1.1669e-02,
         4.2097e-02, -2.8579e-02, -2.0263e-02,  5.5132e-03,  8.7324e-03,
         1.6081e-02,  2.8152e-02, -1.7053e-02, -2.0992e-02, -5.0242e-02,
        -2.7748e-02,  3.3896e-02,  2.2596e-02,  9.2270e-02, -8.2071e-03,
         1.0586e-02,  3.8054e-02, -2.4005e-02,  7.6061e-02,  6.3751e-02,
        -3.3300e-02,  4.9671e-02,  7.0344e-03, -3.7034e-03,  3.2764e-02,
         4.7955e-02,  4.9348e-02,  2.5320e-02,  1.3136e-02,  1.7242e-02,
        -2.1625e-02, -7.0697e-02,  1.6638e-03, -3.0637e-02,  6.6399e-02,
         3.6314e-02, -3.5447e-02,  4.5695e-03,  5.2071e-02,  5.3575e-02,
         6.1097e-02, -4.7511e-02,  4.3019e-02, -4.2027e-02,  2.2281e-02,
        -6.1882e-02,  2.5569e-02,  1.0726e-02, -3.8276e-02, -4.1588e-02,
        -9.9238e-03,  4.2918e-02, -7.7892e-02,  1.4573e-02, -4.0984e-02,
         7.9660e-02,  2.7177e-02, -9.0981e-04,  6.9981e-03,  1.1556e-02,
         4.2847e-02, -3.4945e-03,  1.4815e-02,  1.5982e-02, -2.8289e-02,
        -1.0511e-02,  1.2357e-02,  6.1703e-03, -7.3271e-02,  2.1268e-02,
        -1.3880e-02, -6.8653e-02,  1.0565e-02, -1.6328e-03, -4.7050e-02,
         4.8317e-02, -3.0465e-02, -1.9396e-02, -3.9026e-02, -2.0494e-02,
        -1.7990e-02, -3.2886e-02, -1.2596e-02,  5.7034e-02,  7.7916e-02,
         1.5011e-02, -4.0524e-02,  6.5738e-02, -6.4497e-02,  7.3282e-02,
        -9.3706e-03,  2.7256e-02,  4.7374e-02, -4.1460e-02,  3.9556e-02,
        -3.7284e-02,  2.3300e-02,  2.7274e-02, -5.8233e-03,  3.8058e-02,
         3.3855e-02,  6.7934e-02,  3.5258e-02, -7.0924e-03, -2.4355e-02,
        -4.7719e-02,  2.4226e-02, -2.6625e-02,  8.9272e-02, -3.4004e-02,
        -2.6442e-02,  4.5164e-02,  6.9057e-02, -1.8437e-02,  6.3769e-02,
         7.7940e-03,  3.6622e-03, -8.0939e-02, -8.7010e-02, -4.8476e-02,
         2.7678e-02, -1.5762e-02, -5.8475e-02, -6.1275e-02,  1.2637e-02,
        -5.2410e-02, -1.8670e-02, -1.4053e-01,  5.7134e-02,  7.0297e-02,
         1.8210e-03,  1.3602e-02,  2.3701e-02,  2.4422e-02, -3.7841e-02,
         2.4869e-02, -2.2182e-02,  3.9201e-03,  2.9589e-02, -1.3709e-02,
         3.0735e-02,  5.5844e-02, -3.0001e-02, -4.8189e-02, -1.2909e-02,
        -2.5347e-02, -2.0596e-02,  1.0211e-01,  3.0619e-02, -1.0197e-01,
         2.2304e-02, -8.3113e-03,  1.2583e-01,  1.9637e-02, -2.4393e-02,
         2.2151e-02, -7.5491e-02, -5.1172e-02,  1.1070e-02, -1.1944e-02,
         7.4198e-02, -3.9359e-02,  2.2818e-02,  3.8035e-02, -4.0181e-02,
        -1.3766e-02,  2.9062e-02, -1.0164e-02, -9.8718e-02,  2.1204e-02,
        -2.5231e-02, -4.4410e-02,  3.0807e-02,  6.7081e-02,  6.8380e-03,
        -1.5150e-02, -3.4259e-02, -1.4193e-02,  9.2370e-02, -2.1744e-02,
         7.5269e-02, -2.6060e-02, -2.1665e-02,  3.3039e-02, -4.3331e-02,
         1.7362e-02, -4.0898e-02,  5.2190e-02, -2.1288e-02,  4.2191e-02,
        -3.2984e-02,  2.4955e-02, -1.7306e-02,  2.7595e-02,  3.0643e-02,
        -1.7492e-02,  1.4456e-02, -1.2413e-01,  1.4219e-02, -2.6955e-02,
         6.3441e-02, -1.0650e-01,  4.8868e-02,  5.7556e-03, -6.3269e-03,
        -3.9281e-02,  8.7581e-02, -3.8899e-02, -5.8725e-02,  9.8057e-03,
         9.8030e-03, -1.6890e-02, -1.1993e-01,  1.6286e-02, -3.5781e-02,
         3.1365e-02,  1.8312e-02,  9.2168e-02,  9.3589e-03,  4.5563e-02,
         3.5109e-02,  5.5509e-02,  6.2787e-02, -6.6396e-02, -8.6968e-02,
        -6.1091e-03,  5.4935e-02, -1.0500e-02,  9.3638e-03, -4.9057e-02,
        -2.4767e-02, -3.6407e-03, -1.4540e-02,  7.0527e-02,  1.0632e-02,
        -4.4427e-02,  5.2730e-02,  4.5080e-03, -5.8011e-02, -5.7675e-02,
        -1.2262e-02, -4.3363e-02,  1.6011e-02, -6.2183e-03, -5.8607e-02,
         7.1235e-02,  3.9330e-02, -3.8231e-02, -1.0898e-01,  3.6655e-02,
        -2.4495e-02, -5.3603e-02, -2.7977e-03, -6.1488e-02,  1.3309e-02,
        -7.5936e-02, -8.1418e-02,  5.1106e-02, -3.1873e-02,  3.0707e-02,
         5.2765e-02, -2.2604e-02, -1.1471e-01,  2.8089e-02, -1.6024e-02,
        -2.1106e-02,  3.4020e-02, -2.3916e-02,  4.3647e-02,  1.0358e-01,
        -6.0562e-02,  4.1847e-02, -1.6619e-02, -4.5269e-02,  1.3921e-02,
        -3.8562e-02, -4.8185e-02,  7.0595e-02, -4.1297e-03,  7.8182e-02,
        -4.7511e-02, -1.3953e-03,  4.3699e-02,  9.5456e-03,  7.5895e-02,
        -4.8886e-02,  3.4786e-02,  5.2504e-02,  5.1594e-03,  2.7654e-02,
         6.1631e-03, -6.2448e-03, -2.5518e-02, -1.5373e-02, -2.0481e-02,
         1.0327e-02,  5.0946e-02, -2.7792e-02, -5.4474e-02,  2.3683e-02,
         4.6950e-02, -2.2749e-02, -3.4948e-02
        };

        //ggml_backend_tensor_set(style_embed, style_embed_data, 0, 528*sizeof(float));

        // if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS)
        // {
        //     fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        //     return false;
        // }

        float *x = (float*) std::malloc(hparams.max_seq_len*(hparams.emb_dim+hparams.punct_emb_dim)*sizeof(float));
        if (!x)
            throw std::runtime_error("x malloc() failed");

        encoder->eval(src_seq_data, puncts_data, style_embed_data, num_phonemes, x);


        //struct ggml_tensor *x = ggml_graph_get_tensor(gf, "x");
        //print_tensor("x", x, 6);

        //struct ggml_tensor *x_punct = ggml_graph_get_tensor(gf, "x_punct");
        //print_tensor("x_punct", x_punct, 11);

        // struct ggml_tensor *dbg = ggml_graph_get_tensor(gf, "dbg");
        // print_tensor("dbg", dbg, 3);

        // struct ggml_tensor *dbg2 = ggml_graph_get_tensor(gf, "dbg2");
        // print_tensor("dbg2", dbg2, 11);

        // print_tensor("y", y, 6);
        // return true;
    }
}

int main(void)
{

    ZeroVOX::ZeroVOXModel model(g_gguf_filename);

    model.eval();

    return 0;
}
