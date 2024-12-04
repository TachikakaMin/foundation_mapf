import argparse

def get_args():
    """用于解析命令行参数
    """
    parser = argparse.ArgumentParser(description='UNet training')
    
    # training
    parser.add_argument('--seed', '-sd', metavar='S', type=int, default=1919180, help='seed')
    parser.add_argument('--epochs', '-ep', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', '-bs', dest='batch_size', metavar='B', 
                        type=int, default=512, help='Batch size')
    parser.add_argument('--learning_rate', '-lr', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-3,
                        help='weight decay')
    parser.add_argument('--load', '-l', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--dataset_path', '-dp', type=str, default="data/dataset.npz")
    parser.add_argument('--plot_interval', type=int, default=1, help='plot interval')
    parser.add_argument('--save_interval', type=int, default=20, help='save interval')
    parser.add_argument('--log_dir', type=str, default="runs", help='plot log')
    parser.add_argument('--comments', type=str, default="", help='comments')
    #MAPF settings
    parser.add_argument('--max_agent_num', '-na', metavar='NA', type=int, default=250, 
                        help='Max number of agents')
    parser.add_argument('--action_dim', '-ad', metavar='AD', type=int, default=5, help='Action types')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args.epochs)
    print(args.batch_size) 
    print(args.learning_rate) 
    print(args.amp)  