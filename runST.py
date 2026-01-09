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
        seeds = [config['seed']]

    results = train(config, seeds)

    print(f"Dataset: {config['dataset']}")
    if len(seeds) > 1:
        print(f"- ACC: {np.mean(results['acc']) * 100:.2f} ± {np.std(results['acc']) * 100:.2f}")
        print(f"- NMI: {np.mean(results['nmi']) * 100:.2f} ± {np.std(results['nmi']) * 100:.2f}")
        print(f"- ARI: {np.mean(results['ari']) * 100:.2f} ± {np.std(results['ari']) * 100:.2f}")
        print(f"- F1 : {np.mean(results['f1']) * 100:.2f} ± {np.std(results['f1']) * 100:.2f}")
    else:
        print(f"- ACC: {results['acc'][0] * 100:.2f}")
        print(f"- NMI: {results['nmi'][0] * 100:.2f}")
        print(f"- ARI: {results['ari'][0] * 100:.2f}")
        print(f"- F1 : {results['f1'][0] * 100:.2f}")
    print(f"================= scRCL Training Completed ================")


if __name__ == '__main__':
    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default="151507",
                        choices=["151507", "151508", "151509", "151510", "151669", "151670",
                                 "151671", "151672", "151673", "151674", "151675", "151676",
                                 "Mouse_Brain_Anterior", "Human_Breast_Cancer", "Mouse_Embryo_E9.5"],
                        help='Spatial transcriptomics dataset name.')
    parser.add_argument('--lambda1', type=float, default=100.0, help='Loss balance parameter.')
    parser.add_argument('--lambda2', type=float, default=1.0, help='Loss balance parameter.')
    parser.add_argument('--hvg', type=int, default=3000, help='Number of highly variable genes.')

    args = parser.parse_args()

    datasets = {
        20: "Mouse_Brain_Anterior", 21: "Human_Breast_Cancer", 22: "Mouse_Embryo_E9.5",
        107: "151507", 108: "151508", 109: "151509", 110: "151510",
        169: "151669", 170: "151670", 171: "151671", 172: "151672",
        173: "151673", 174: "151674", 175: "151675", 176: "151676"
    }
    # Set environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if args.dataset in ["151507", "151508", "151509", "151510", "151669", "151670",
                        "151671", "151672", "151673", "151674", "151675", "151676"]:
        config_file_name = f"./config/ST/DLPFC/{args.dataset}.yaml"
    elif args.dataset in ["Mouse_Brain_Anterior", "Human_Breast_Cancer", "Mouse_Embryo_E9.5"]:
        config_file_name = f"./config/ST/{args.dataset}.yaml"
    else:
        print("Use the default parameter configuration.")
        config_file_name = f"./config/ST/init_config_st.yaml"
    if os.path.exists(config_file_name):
        with open(config_file_name) as f:
            if hasattr(yaml, 'FullLoader'):
                config = yaml.load(f.read(), Loader=yaml.FullLoader)
            else:
                config = yaml.load(f.read())
    else:
        raise NotImplementedError(f"The config file '{config_file_name}' is not implemented.")

    if config['dataset'] == "":
        config['dataset'] = args.dataset
    config['train'] = True          # True: 5 seeds, False: 1 seed
    config['showimg'] = False
    config['showpaga'] = False      # for DLPFC
    config['save_res'] = False
    config['save_path'] = f"./results/ST"
    config['fig_path'] = f"./figures/ST"

    # config['lambda1'] = args.lambda1
    # config['lambda2'] = args.lambda2

    if args.dataset in ["151507", "151508", "151509", "151510", "151669", "151670",
                        "151671", "151672", "151673", "151674", "151675", "151676",
                        "Mouse_Brain_Anterior", "Human_Breast_Cancer", "Mouse_Embryo_E9.5"]:
        config['task'] = "ST"

    main(config)
