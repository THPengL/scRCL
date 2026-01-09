import os
import yaml
import argparse
import numpy as np
import warnings

warnings.simplefilter("ignore")


def main(config):
    from train import train

    # Training
    if config['train']:
        seeds = [5, 15, 25, 35, 45]
    else:
        seeds = [5]

    results = train(config, seeds)

    print(f"Dataset: {config['dataset']}")
    print(f"- ACC: {np.mean(results['acc']) * 100:.2f} ± {np.std(results['acc']) * 100:.2f}")
    print(f"- NMI: {np.mean(results['nmi']) * 100:.2f} ± {np.std(results['nmi']) * 100:.2f}")
    print(f"- ARI: {np.mean(results['ari']) * 100:.2f} ± {np.std(results['ari']) * 100:.2f}")
    print(f"- F1 : {np.mean(results['f1']) * 100:.2f} ± {np.std(results['f1']) * 100:.2f}")
    print(f"================= scRCL Training Completed ================")


if __name__ == '__main__':
    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default="Quake_Smart-seq2_Diaphragm",
                        choices=["Quake_Smart-seq2_Diaphragm", "Quake_Smart-seq2_Lung", "Quake_Smart-seq2_Trachea",
                                 "Quake_10x_Bladder", "Quake_10x_Limb_Muscle", "Quake_10x_Spleen",
                                 "li_tumor", "human_ESC", "Zeisel", "Baron_human"],
                        help='Single-cell RNA sequencing dataset name.')
    parser.add_argument('--lambda1', type=float, default=100.0, help='Loss balance parameter.')
    parser.add_argument('--lambda2', type=float, default=1.0, help='Loss balance parameter.')
    parser.add_argument('--hvg', type=int, default=2000, help='Number of highly variable genes.')

    args = parser.parse_args()

    datasets = {
        0: "Quake_Smart-seq2_Diaphragm", 1: "Quake_Smart-seq2_Lung", 2: "Quake_Smart-seq2_Trachea",
        3: "Quake_10x_Bladder", 4: "Quake_10x_Limb_Muscle", 5: "Quake_10x_Spleen",
        6: "li_tumor", 7: "human_ESC", 8: "Zeisel", 9: "Baron_human",
    }
    # Set environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if args.dataset in ["Quake_Smart-seq2_Diaphragm", "Quake_Smart-seq2_Lung", "Quake_Smart-seq2_Trachea",
                        "Quake_10x_Bladder", "Quake_10x_Limb_Muscle", "Quake_10x_Spleen",
                        "li_tumor", "human_ESC", "Zeisel", "Baron_human"]:
        config_file_name = f"./config/SC/{args.dataset}.yaml"
    else:
        print("Use the default parameter configuration.")
        config_file_name = f"./config/SC/init_config_sc.yaml"
    if os.path.exists(config_file_name):
        with open(config_file_name) as f:
            if hasattr(yaml, 'FullLoader'):
                config = yaml.load(f.read(), Loader=yaml.FullLoader)
            else:
                config = yaml.load(f.read())
    else:
        raise NotImplementedError(f"The config file '{config_file_name}' is not implemented.")

    config['train'] = True          # True: 5 seeds, False: 1 seed
    # config['showimg'] = False
    config['save_res'] = False
    config['save_path'] = f"./results/SC"

    # config['lambda1'] = args.lambda1
    # config['lambda2'] = args.lambda2

    if args.dataset in ["Quake_Smart-seq2_Diaphragm", "Quake_Smart-seq2_Lung", "Quake_Smart-seq2_Trachea",
                        "Quake_10x_Bladder", "Quake_10x_Limb_Muscle", "Quake_10x_Spleen",
                        "li_tumor", "human_ESC", "Zeisel", "Baron_human"]:
        config['task'] = "SC"

    main(config)
