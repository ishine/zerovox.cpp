#include <vector>
#include <cassert>
#include <stdexcept>

#include "zerovox.h"

namespace ZeroVOX
{
    struct ggml_tensor *checked_get_tensor(struct ggml_context *ctx, const char *name)
    {
        struct ggml_tensor *tensor = ggml_get_tensor(ctx, name);
        if (!tensor) {
            fprintf(stderr, "%s: tensor '%s' not found\n", __func__, name);
            throw std::runtime_error("ggml_get_tensor() failed");
        }
        return tensor;
    }

    void tensor_dbg(struct ggml_cgraph *gf, struct ggml_context *ctx, struct ggml_tensor *x, const char *name)
    {
        int ndim = ggml_n_dims(x);

        struct ggml_tensor *x_dbg = nullptr;
        switch (ndim)
        {
            case 1:
                x_dbg = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, x->ne[0]);
                break;
            case 2:
                x_dbg = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1]);
                break;
            case 3:
                x_dbg = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], x->ne[2]);
                break;
            case 4:
                x_dbg = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
                break;
            default:
                assert(false);
        }

        ggml_build_forward_expand(gf, ggml_cpy(ctx, x, x_dbg));
        ggml_set_name(x_dbg, name);
    }

    static void _print_t_f32(const char* title, struct ggml_tensor * t, int n, int dim, off_t offset, uint8_t *data)
    {
        bool skip = false;
        int ndims = ggml_n_dims(t);

        for (int i=0; i<ndims-dim; i++)
            printf ("  ");
        printf ("[");

        if (dim)
            printf ("\n");

        for (int i = 0; i < t->ne[dim]; i++)
        {

            if (!skip && (i >= n) && (i < (t->ne[dim]-n)) )
            {
                skip = true;
                if (!dim)
                {
                    printf (" ... ");
                }
                else
                {
                    for (int j=0; j<=ndims-dim; j++)
                        printf ("  ");
                    printf ("...\n");
                }
            }

            if (!skip || (i >= (t->ne[dim]-n)))
            {
                if (!dim)
                {
                    float *fptr = (float *) (data + offset + i * t->nb[0]);
                    printf("%.5f ", *fptr);
                }
                else
                    _print_t_f32(title, t, n, dim-1, offset + i * t->nb[dim], data);
            }
        }

        if (dim)
        {
            for (int i=0; i<ndims-dim; i++)
                printf ("  ");
        }
        printf ("]\n");

    }

    void print_tensor(const char* title, struct ggml_tensor * t, int n/*= 3*/)
    {
        int ndims = ggml_n_dims(t);

        printf("%s [", title);
        
        for (int i=ndims-1; i>=0; i--)
            printf("%" PRId64 "%c ", t->ne[i], i>0 ? ',' : ' ');

        for (int i=0; i<ndims; i++)
            printf("ne[%d]=%" PRId64 " ", i, t->ne[i]);

        switch (t->type)
        {
            case GGML_TYPE_F16:
                printf ("] f16 = \n");
                break;
            case GGML_TYPE_F32:
                printf ("] f32 = \n");
                break;
            default:
                assert(false);
        }

        const int nb = ggml_nbytes(t);
        std::vector<float> res(nb / sizeof(float));
        float *data = res.data();
        ggml_backend_tensor_get(t, data, 0, nb);

        switch (t->type)
        {
            case GGML_TYPE_F32:
                _print_t_f32(title, t, n, ndims-1, 0, (uint8_t *) data);
                break;
            default:
                assert(false);
        }

        double sum = 0.0;
        for (int i = 0; i < ggml_nelements(t); i++) {
            sum += data[i];
        }
        printf("sum:  %f\n\n", sum);
    }
    
}
