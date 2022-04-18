import argparse

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default = "Recon",
            help = "Name of experiment"
            )
    parser.add_argument("--train_dataset_file", default = "../dataset/train_dataset.h5", help = "File path of data")
    parser.add_argument("--train_dataset_info_file", default = "../dataset/train_dataset.csv", help = "File path of data")
    parser.add_argument("--valid_dataset_file", default = "../dataset/valid_dataset.h5", help = "File path of data")
    parser.add_argument("--valid_dataset_info_file", default = "../dataset/valid_dataset.csv", help = "File path of data")
    parser.add_argument("--test_dataset_file", default = "../dataset/test_dataset.h5", help = "File path of data")
    parser.add_argument("--test_dataset_info_file", default = "../dataset/test_dataset.csv", help = "File path of data")
    parser.add_argument("--base_dir",default = "../result", help = "Directory of log")
    # Dataset Parameters
    parser.add_argument("--dataset_name", default = "CNN", help = "Name of dataset", choices = ["CNN"])

    # Model Parameteers
    parser.add_argument("--model_name", default = "VGG", help = "Name of model", choices = ["VGG","ResNet"])
    parser.add_argument("--ckpt", default = None, help = "Path of pretrained model")
    parser.add_argument("--in_channels", default = 2, help = "Num of input channels")
    parser.add_argument("--out_channels", default = 3, help = "Num of output channels")

    # Train parameter
    parser.add_argument("--local_rank", type = int)
    parser.add_argument("--seed", default = 2022, type = int, help = "Random seed")
    parser.add_argument("--lr", default = 1.e-4, type = float, help = "Learning rate")
    parser.add_argument("--batch_size", default = 256, type = int, help = "Batch Size")
    parser.add_argument("--epoch", default = 20, type = int, help = "Epoch")
    parser.add_argument("--train_epoch_num", default = 10, type = int, help = "Max train iteration number")
    parser.add_argument("--valid_every_step", default = 10, type = int, help = "Max train iteration number")

    # evaluation parameter
    parser.add_argument("--eval_output", default = "../result/vgg/recon_nodp/eval_result.csv", help = "File path of evaluation output")
    return parser.parse_args()
