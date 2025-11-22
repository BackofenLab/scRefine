from preprocessing.translate_data import  translate
from preprocessing.create_data import  preprocessing
import argparse
import os

if __name__ == "__main__":
    
    cmdline_parser = argparse.ArgumentParser('preprocess single cell dataset')


    cmdline_parser.add_argument('-d', '--dataset',
                                help='dataset',
                                required = True,
                                type=str)
    cmdline_parser.add_argument('-m', '--marker_genes',
                                default="../marker_genes.txt",
                                help='marker genes',
                                type=str)
    cmdline_parser.add_argument('-o', '--output_h5ad',
                                default="test_output.h5ad",
                                help='output',
                                type=str)
    cmdline_parser.add_argument('-n', '--npy_dataset',
                                default="test.npz",
                                help='output',
                                type=str)
    cmdline_parser.add_argument('-b', '--dataset_key',
                                default="batch",
                                help='dataset_key',
                                type=str)
    parser.add_argument("--obs_fields",nargs="+",type=str,help="list of observation fields", default=[])
    parser.add_argument("--layer_fields",nargs="+",type=str, help="list of layer fields", default=[])
                                
                                
                                
                                
    cmdline_parser.add_argument('--no_hvg', action=argparse.BooleanOptionalAction, default = False)
                                
    args, unknowns = cmdline_parser.parse_known_args()
    print(args.no_hvg)
    
    if not args.dataset_path.endswith(os.sep):
        args.dataset_path = args.dataset_path + os.sep
    
    
    preprocessing(marker_gene_list = args.marker_genes, dataset = args.dataset, full_name = args.output_h5ad, batch_key=args.dataset_key, no_hvg = args.no_hvg)
    
    
    obs_fields = args.obs_fields
    obs_fields.append(args.dataset_key)
    obs_fields = list(set(obs_fields))
    layer_fields = args.layer_fields
    
    translate(args.output_h5ad, args.npy_dataset, obs_fields, layer_fields)
