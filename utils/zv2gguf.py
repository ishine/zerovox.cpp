#!/usr/bin/env python3

import glob
import os
import re
from pathlib import Path

import numpy as np
import torch
import yaml
import h5py

from gguf import GGUFWriter

# MODELPATH = Path('/home/guenter/projects/hal9000/ennis/zerovox/models/tts_de_zerovox_mini_1')
MODELPATH = Path('/home/guenter/projects/hal9000/ennis/zerovox/models/tts_en_de_zerovox_medium_2_styledec')
MODELARCH = 'zerovox-resnet-fs2-styletts'
HIFIGAN_PATH = Path('/home/guenter/projects/hal9000/ennis/zerovox/models/meldec-libritts-hifigan-v1')

OUT_MODEL_FN = "medium-ldec.gguf"

SHORTNAMES = {
    '_phoneme_encoder' : '_pe',
    '_encoder': '_enc',
    'layer_stack' : 'laystk',
    'weight' : 'w',
    '_variance_adaptor' : '_var_adapt',
    'energy_predictor' : 'engy_pred',
    'bias': 'b'
}

def shorten_tensor_name (long_name):

    sname = long_name

    for l, s in SHORTNAMES.items():
        sname = sname.replace(l, s)

    return sname

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)],
        dtype=np.float32
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return sinusoid_table

def main() -> None:

    # load HifiGAN

    print (f"loading HifiGAN stats from {HIFIGAN_PATH / 'stats.h5'} ...")
    hifigan_stats = {}
    with h5py.File(HIFIGAN_PATH / 'stats.h5', 'r') as f:
        #print (f"h5 keys: {f.keys()}")
        hifigan_stats = {'mean': f['mean'][:], 'scale': f['scale'][:]}
        #print (f"stats: {hifigan_stats}")

    print (f"loading HifiGAN checkpoint data from {HIFIGAN_PATH / 'checkpoint.pkl'} ...")
    hifigan = torch.load(HIFIGAN_PATH / 'checkpoint.pkl', weights_only=False)
    print (hifigan.keys())

    # load model config

    config_path = Path(MODELPATH / 'modelcfg.yaml')
    with open (config_path) as modelcfgf:
        cfg = yaml.load(modelcfgf, Loader=yaml.FullLoader)

    # load the latest checkpoint

    list_of_files = glob.glob(str(MODELPATH / 'checkpoints' / '*.ckpt'))
    ckpt_path = max(list_of_files, key=os.path.getctime)

    print (f"loading checkpoint data from {ckpt_path} ...")

    checkpoint = torch.load(ckpt_path, weights_only=False)

    state_dict = checkpoint['state_dict']

    # replace _meldec.*

    for key in list(checkpoint['state_dict']):
        if key.startswith('_meldec.'):
            print (f"removing {key}")
            del checkpoint['state_dict'][key]

    meldec_weights = hifigan['model']['generator']
    for key in meldec_weights:
        mdeckey = '_meldec.'+key
        print (f"adding meldec key {mdeckey}")
        checkpoint['state_dict'][mdeckey] = meldec_weights[key]

    keys = list(state_dict.keys())

    out_model_fn = OUT_MODEL_FN

    gguf_writer = GGUFWriter(out_model_fn, MODELARCH)

    # write hyperparams

    gguf_writer.add_uint32 (f'{MODELARCH}.max_seq_len', cfg['model']['max_seq_len'])

    gguf_writer.add_uint32 (f'{MODELARCH}.emb_dim', cfg['model']['emb_dim'])
    gguf_writer.add_uint32 (f'{MODELARCH}.punct_emb_dim', cfg['model']['punct_emb_dim'])

    gguf_writer.add_uint32 (f'{MODELARCH}.decoder.n_head', cfg['model']['decoder']['n_head'])

    gguf_writer.add_uint32 (f'{MODELARCH}.encoder.layer', cfg['model']['encoder']['fs2_layer'])
    gguf_writer.add_uint32 (f'{MODELARCH}.encoder.head', cfg['model']['encoder']['fs2_head'])

    gguf_writer.add_uint32 (f'{MODELARCH}.encoder.vp_filter_size', cfg['model']['encoder']['vp_filter_size'])
    gguf_writer.add_uint32 (f'{MODELARCH}.encoder.vp_kernel_size', cfg['model']['encoder']['vp_kernel_size'])
    # gguf_writer.add_bool   (f'{MODELARCH}.encoder.log_pitch_quant', cfg['model']['encoder']['ve_pitch_quantization'] != 'linear')
    # gguf_writer.add_bool   (f'{MODELARCH}.encoder.log_energy_quant', cfg['model']['encoder']['ve_energy_quantization'] != 'linear')
    gguf_writer.add_uint32 (f'{MODELARCH}.encoder.ve_n_bins', cfg['model']['encoder']['ve_n_bins'])

    gguf_writer.add_uint32 (f'{MODELARCH}.decoder.conv_filter_size', cfg['model']['decoder']['conv_filter_size'])
    gguf_writer.add_uint32 (f'{MODELARCH}.decoder.conv_kernel_size.0', cfg['model']['decoder']['conv_kernel_size'][0])
    gguf_writer.add_uint32 (f'{MODELARCH}.decoder.conv_kernel_size.1', cfg['model']['decoder']['conv_kernel_size'][1])

    gguf_writer.add_uint32 (f'{MODELARCH}.audio.sampling_rate', cfg['audio']['sampling_rate'])
    gguf_writer.add_uint32 (f'{MODELARCH}.audio.num_mels', cfg['audio']['num_mels'])
    gguf_writer.add_uint32 (f'{MODELARCH}.audio.hop_size', cfg['audio']['hop_size'])

    gguf_writer.add_tensor(f'hifigan.mean', hifigan_stats['mean'])
    gguf_writer.add_tensor(f'hifigan.scale', hifigan_stats['scale'])

    for key in sorted(keys):
        sname = shorten_tensor_name(key)

        tensor = state_dict[key].cpu().detach().numpy()

        #print(f"{sname}: {key}")
        print(f"{sname}: {tensor.shape}")

        if len(tensor.shape) == 0:
            print ("0-dim -> skip")
            continue

        if sname.endswith('pos_ffn.w_1.w'):
            tensor = tensor.astype(np.float16);
        if sname.endswith('pos_ffn.w_2.w'):
            tensor = tensor.astype(np.float16);
        if sname.endswith('conv.w'):
            tensor = tensor.astype(np.float16);

        # recompute original weights for torch.nn.utils.weight_norm:
        if key.endswith('weight_g'):
            #print (f"--> {key} : {tensor}")
            continue
        if key.endswith('weight_v'):
            gname = key.replace('.weight_v', '.weight_g')
            _g = state_dict[gname].cpu().detach()
            _v = state_dict[key].cpu().detach()
            # fixme tensor = _g * (_v / _v.norm(dim=0, keepdim=True))
            tensor = torch._weight_norm(_v, _g, 0)
            sname = key.replace('weight_v', 'w')

            # flip weight for deconvolutional layers
            if re.match(r'^_meldec.upsamples.[0-9].1.w$', sname):
                tensor = torch.flip(tensor, [2])  # Flip along the kernel dimension
                tensor = tensor.permute(1, 0, 2)  # Swap in_channels and out_channels

            tensor = tensor.numpy().astype(np.float16)

        gguf_writer.add_tensor(sname, tensor)

    st = get_sinusoid_encoding_table(cfg['model']['max_seq_len']+1, cfg['model']['emb_dim']+cfg['model']['punct_emb_dim'])
    gguf_writer.add_tensor('sinusoid_encoding_table', st)

    # pitch_min_a = np.array(
    #     [cfg['stats']['pitch_min'] for pos_i in range(n_position)],
    #     dtype=np.float32
    # )

    # d_k = (cfg['model']['emb_dim']+cfg['model']['punct_emb_dim']) // cfg['model']['encoder']['fs2_head']
    # temperature = np.float32(np.power(d_k, 0.5))
    # for i in range (cfg['model']['encoder']['fs2_layer']):
    #     name = f"_pe._enc.laystk.{i}.slf_attn.temperature"
    #     gguf_writer.add_tensor(name, temperature)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()

    print (f"{out_model_fn} written.")


if __name__ == '__main__':
    main()
