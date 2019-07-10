from StarGAN import StarGAN
import argparse
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Keras implementation of StarGAN"
    parser = argparse.ArgumentParser(description=desc)

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels on CelebA dataset)')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', help='dataset_name')
    parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'custom'])

    # Directories.
    parser.add_argument('--data_dir', type=str, default='data/celeba')
    parser.add_argument('--model_save_dir', type=str, default='models')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--result_dir', type=str, default='results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    # Custom image
    parser.add_argument('--custom_image_name', type=str, default='test.png')
    parser.add_argument('--custom_image_label', '--list2', nargs='+', type=int, default=[1, 0, 0, 1, 1])    

    return parser.parse_args()

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    gan = StarGAN(args)

    # build graph
    gan.build_model()

    if args.mode == 'train' :
        gan.train()
        print(" [*] Training finished!")

    if args.mode == 'test' :
        gan.test()
        print(" [*] Test finished!")

    if args.mode == 'custom' :
        gan.custom()
        print(" [*] Test on custom image finished!")        

if __name__ == '__main__':
    main()