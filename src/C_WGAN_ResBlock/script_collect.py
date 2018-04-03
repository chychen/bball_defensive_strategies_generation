#python result_collect.py --mode=9 --restore_path='wi_no_latent/checkpoints/model.ckpt-828981' --#folder_path='wi_no_latent' --gpus='0'

import os
def main():
    directory = 'wi_no_latent/checkpoints'
    for filename in os.listdir(directory):
        if filename.endswith(".index"):
            ckpt_path = os.path.join(directory, filename[:-6])
            os.system("python result_collect.py --mode=9 --restore_path=\'{}\' --folder_path=\'wi_no_latent\' --gpus=\'0\'".format(ckpt_path))
 

if __name__ == '__main__':
    main()