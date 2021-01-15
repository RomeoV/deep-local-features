from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils.generic_utils import Progbar
import os
import urllib3
from getpass import getpass

BASE_URL = "https://polybox.ethz.ch/remote.php/webdav/Shared/Deep%20Learning/"

# model train_shared_fe64.CorrespondenceEncoder
ENCODER = {
    "correspondence_encoder": "tb_logs/correspondence_encoder/version_5/checkpoints/epoch%3D37-step%3D5509.ckpt",
    "correspondence_encoder_lr1e3": "tb_logs/correspondence_encoder_lr1e3/version_0/checkpoints/epoch%3D44-step%3D6524.ckpt",
}

# models : attention_model.AttentionLayer, AttentionLayer2, MultiAttentionLayer, MultiAttentionLayer2
ATTENTION = {
    "d2enc" : "tb_logs/d2net_encoder3_lr5e4/version_0/checkpoints/epoch%3D25-step%3D3769.ckpt",

    "attention_model": "tb_logs/attention_model/version_9/checkpoints/epoch%3D44-step%3D6524.ckpt",
    "attention_model_delf_distinctiveness_loss": "tb_logs/attention_model_delf_distinctiveness_loss/version_1/checkpoints/epoch%3D4-step%3D586.ckpt",
    "attention_model_sum_distinctiveness_loss": "tb_logs/attention_model_sum_distinctiveness_loss/version_4/checkpoints/epoch%3D40-step%3D5808.ckpt",
    "cfe64_multi_attention_model2_d2netloss": "tb_logs/cfe64_multi_attention_model2_d2netloss/version_0/checkpoints/epoch%3D35-step%3D5219.ckpt",
    "cfe64_multi_attention_model2_d2netloss_backprop": "tb_logs/cfe64_multi_attention_model2_d2netloss_backprop/version_0/checkpoints/epoch%3D25-step%3D3769.ckpt",
    ##"cfe64_multi_attention_model2_distinctiveness+_loss": "tb_logs/cfe64_multi_attention_model2_distinctiveness%2B_loss/version_0/checkpoints/epoch%3D41-step%3D6089.ckpt",
    ##"cfe64_multi_attention_model2_distinctiveness+_lossN8_lambda1": "tb_logs/cfe64_multi_attention_model2_distinctiveness%2B_lossN8_lambda1/version_0/checkpoints/epoch%3D31-step%3D4639.ckpt",
    ##"cfe64_multi_attention_model2_distinctiveness+_lossN32_lambda1": "tb_logs/cfe64_multi_attention_model2_distinctiveness%2B_lossN32_lambda1/version_0/checkpoints/epoch%3D26-step%3D3914.ckpt",
    ##"cfe64_multi_attention_model2_distinctiveness+_lossN64_lambda1": "tb_logs/cfe64_multi_attention_model2_distinctiveness%2B_lossN64_lambda1/version_0/checkpoints/epoch%3D26-step%3D3914.ckpt",
    "cfe64_multi_attention_model_d2netloss": "tb_logs/cfe64_multi_attention_model_d2netloss/version_0/checkpoints/epoch%3D26-step%3D3914.ckpt",
    "cfe64_multi_attention_model_d2netloss_backprop": "tb_logs/cfe64_multi_attention_model_d2netloss_backprop/version_0/checkpoints/epoch%3D34-step%3D5074.ckpt",
    "cfe64_multi_attention_model_distinctiveness+_loss": "tb_logs/cfe64_multi_attention_model_distinctiveness%2B_loss/version_2/checkpoints/epoch%3D4-step%3D582.ckpt",
    "cfe64_multi_attention_model_distinctiveness+_lossN8_l1": "tb_logs/cfe64_multi_attention_model_distinctiveness%2B_lossN8_l1/version_0/checkpoints/epoch%3D36-step%3D5364.ckpt",
    "cfe64_multi_attention_model_distinctiveness+_lossN32_l1": "tb_logs/cfe64_multi_attention_model_distinctiveness%2B_lossN32_l1/version_0/checkpoints/epoch%3D31-step%3D4639.ckpt",
    "multi_attention_model_distinctiveness_loss": "tb_logs/multi_attention_model_distinctiveness_loss/version_0/checkpoints/epoch%3D58-step%3D8554.ckpt",

    ## something could be wrong with this
    "cfe64_multi_attention_model2_distinctiveness+_lossN16_lambda01_sm_lowmargin_SHARED" : "tb_logs/cfe64_multi_attention_model2_distinctiveness%2B_lossN16_lambda01_sm_lowmargin/version_0/checkpoints/epoch%3D19-step%3D2899.ckpt",

    "cfe64_multi_attention_model2_distinctiveness+_lossN16_lambda01_lowmargin" : "tb_logs/cfe64_multi_attention_model2_distinctiveness%2B_lossN16_lambda01_lowmargin/version_0/checkpoints/epoch%3D24-step%3D3624.ckpt",

    "cfe64_multi_attention_model2_distinctiveness+_lossN8_lambda01_sm_SHARED" : "tb_logs/cfe64_multi_attention_model2_distinctiveness%2B_lossN8_lambda01_sm/version_3/checkpoints/epoch%3D28-step%3D4204.ckpt", #also version 2
    "cfe64_multi_attention_model2_distinctiveness+_lossN8_lambda01_sm": "tb_logs/cfe64_multi_attention_model2_distinctiveness%2B_lossN8_lambda01_sm/version_4/checkpoints/epoch%3D38-step%3D5654.ckpt",##NOT SHARED

    "cfe64_multi_attention_model2_distinctiveness+_lossN8_lambda1" : "tb_logs/cfe64_multi_attention_model2_distinctiveness%2B_lossN8_lambda1/version_0/checkpoints/epoch%3D44-step%3D6524.ckpt",

    "cfe64_multi_attention_model2_distinctiveness+_lossN2_lambda05_sm_SHARED" : "tb_logs/cfe64_multi_attention_model2_distinctiveness%2B_lossN2_lambda05_sm/version_5/checkpoints/epoch%3D27-step%3D4059.ckpt", ##SHARED only5
    "cfe64_multi_attention_model2_distinctiveness+_lossN2_lambda05_sm" : "tb_logs/cfe64_multi_attention_model2_distinctiveness%2B_lossN2_lambda05_sm/version_4/checkpoints/epoch%3D43-step%3D6379.ckpt",

    "cfe64_multi_attention_model2_distinctiveness+_lossN64_lambda1" : "tb_logs/cfe64_multi_attention_model2_distinctiveness%2B_lossN64_lambda1/version_0/checkpoints/epoch%3D40-step%3D5944.ckpt",
    "cfe64_multi_attention_model2_distinctiveness+_lossN32_lambda1" : "tb_logs/cfe64_multi_attention_model2_distinctiveness%2B_lossN32_lambda1/version_0/checkpoints/epoch%3D40-step%3D5944.ckpt",
    "cfe64_multi_attention_model2_distinctiveness+_lossN16_lambda01_sm" : "tb_logs/cfe64_multi_attention_model2_distinctiveness%2B_lossN16_lambda01_sm/version_2/checkpoints/epoch%3D39-step%3D5799.ckpt",
    "cfe64_multi_attention_model2_distinctiveness+_loss" : "tb_logs/cfe64_multi_attention_model2_distinctiveness%2B_loss/version_0/checkpoints/epoch%3D41-step%3D6089.ckpt",

    # missing "cfe64_multi_attention_model2_distinctiveness+_lossN8_lambda05_sm" : "",
    # missing "cfe64_multi_attention_model2_distinctiveness+_lossN8_lambda01" : "",
    ##fix"cfe64_multi_attention_model2_distinctiveness+_lossN4_lambda1" : "tb_logs/cfe64_multi_attention_model2_distinctiveness%2B_lossN8_lambda1/version_0/checkpoints/epoch%3D44-step%3D6524.ckpt",

    "correspondence_encoder": "tb_logs/correspondence_encoder/version_5/checkpoints/epoch%3D37-step%3D5509.ckpt",
    "correspondence_encoder_lr1e3": "tb_logs/correspondence_encoder_lr1e3/version_0/checkpoints/epoch%3D44-step%3D6524.ckpt"

}
# names that end with


def get_encoder_ckpt(name):
    if not name in ENCODER:
        print(f"Encoder {name} is not available. Choose one from ")
        for e in ENCODER.keys():
            print(e)
        raise "Encoder not available"

    origin = f"{BASE_URL}/{ENCODER[name]}"
    fname = f"{name}.ckpt"

    return get_file(fname, origin)


def get_attention_ckpt(name, ask_overwrite=False):
    if not name in ATTENTION:
        print(f"Encoder {name} is not available. Choose one from ")
        for e in ATTENTION.keys():
            print(e)
        raise "Encoder not available"

    origin = f"{BASE_URL}/{ATTENTION[name]}"
    fname = f"{name}.ckpt"

    return get_file(fname, origin)


def get_encoder_ckpt(name, ask_overwrite=False):
    if not name in ENCODER:
        print(f"Encoder {name} is not available. Choose one from ")
        for e in ENCODER.keys():
            print(e)
        raise "Encoder not available"

    origin = f"{BASE_URL}/{ENCODER[name]}"
    fname = f"{name}.ckpt"

    return get_file(fname, origin)


def get_file(fname,
             origin,
             cache_subdir='datasets',
             cache_dir=None,
             ask_overwrite=False):
    """Downloads a file from a URL if it not already in the cache.
    Returns:
        Path to the downloaded file
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.deep_learning')
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.deep_learning')
    datadir = os.path.join(datadir_base, cache_subdir)
    os.makedirs(datadir, exist_ok=True)

    fpath = os.path.join(datadir, fname)

    download = False
    if not os.path.exists(fpath):
        download = True
    elif ask_overwrite:
        choice = input(f"Cached file {fpath} already exists. Overwrite? (y/N)")
        if choice.lower() == "y":
            download = True

    if download:
        print('Downloading data from', origin)
        uname = input("NETHZ username ")
        password = getpass()
        headers = urllib3.make_headers(basic_auth=f'{uname}:{password}')

        class ProgressTracker(object):
            # Maintain progbar for the lifetime of download.
            # This design was chosen for Python 2.7 compatibility.
            progbar = None

        def dl_progress(count, block_size, total_size):
            if ProgressTracker.progbar is None:
                if total_size == -1:
                    total_size = None
                ProgressTracker.progbar = Progbar(total_size)
            else:
                ProgressTracker.progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            urlretrieve(origin, fpath, dl_progress, headers=headers)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        ProgressTracker.progbar = None

    return fpath


def urlretrieve(url, filename, reporthook=None, data=None, headers=None):
    http = urllib3.PoolManager()
    r = http.request('GET', url, headers=headers, preload_content=False)
    if r.status != 200:
        raise Exception("Failed to fetch data")
    cl = int(r.headers['Content-Length'])

    count = 0
    with open(filename, 'wb') as fd:
        while True:
            data = r.read(4096)
            if not data:
                break
            fd.write(data)
            count += 1
            if reporthook is not None:
                reporthook(count, 4096, cl)
    r.release_conn()


if __name__ == "__main__":
    # uname = input("NETHZ username ")
    # password = getpass()
    # headers = urllib3.make_headers(basic_auth=f'{uname}:{password}')
    # urlretrieve(
    #     f'{BASE_URL}{ENCODER["correspondence_encoder"]}', "test.ckpt", headers=headers)

    print(
        get_file("encoder", f'{BASE_URL}{ENCODER["correspondence_encoder"]}'))
    #f = "https://polybox.ethz.ch/remote.php/webdav/Shared/Deep%20Learning/tb_logs/cfe64_multi_attention_model_distinctiveness%2B_loss/version_2/hparams.yaml"
    #print(get_file("encoder", f))
