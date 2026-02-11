import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from utils import get_splits
import data_selection
import datasets

parser = argparse.ArgumentParser("Milestone 1")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='tinyimagenet', choices=['cifar10', 'cifar100', 'tinyimagenet'])
parser.add_argument('-j', '--workers', default=0, type=int,
                    help="number of data loading workers (default: 0)")
parser.add_argument('--batch-size', type=int, default=128)
# misc
parser.add_argument('--gpu', type=str, default='0') 
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
# openset
parser.add_argument('--known-class', type=int, default=40)  # mismatch ratio (@20% - 2/20/40 for the CIFAR10/CIFAR100/Tiny-ImageNet datasets, respectively)

args = parser.parse_args()

def main():
    seed = 1

    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    if use_gpu:
        print("Using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Using CPU")

    knownclass = get_splits(args.dataset, seed, args.known_class)  # ground truth labels
    print("Fraction of known classes amoung classes:", knownclass)

    print("Creating dataset: {}".format(args.dataset))
    
    # Create dataset
    dataset = datasets.create(
        name=args.dataset, known_class_=args.known_class, knownclass=knownclass,
        batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers
    )

    # Prepare for analysis
    trainset = dataset.trainset
    testset = dataset.testset
    trainloader= dataset.trainloader
    testloader = dataset.testloader
    dataset_name = args.dataset.upper()
    splits = [("training", trainset), ("test", testset)]
    dataloaders = [("training", trainloader), ("test", testloader)]

    # Save plots directory
    save_dir = f'./plots/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)

    # Generate analysis and plots
    data_selection.check_missingness(dataset_name, splits, dataloaders)
    data_selection.check_class_imbalance(dataset_name, splits)
    data_selection.check_duplicates(dataset_name, splits)
    data_selection.check_outliers(dataset_name, splits)
    data_selection.show_schema_and_examples(dataset_name, trainset)
    data_selection.compute_summary_stats(dataset_name, splits)
    data_selection.check_format_consistency(dataset_name, splits)
    data_selection.check_class_imbalance(dataset_name, splits)
    data_selection.generate_plots(dataset_name, splits, save_path=save_dir)


if __name__ == '__main__':
    main()