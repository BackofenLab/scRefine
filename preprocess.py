from preprocessing.translate_data import  translate
from preprocessing.create_data import  preprocessing
import argparse
import os

if __name__ == "__main__":
    
    cmdline_parser = argparse.ArgumentParser('preprocess single cell dataset')

    cmdline_parser.add_argument('-f', '--dataset_path',
                                default="../data/",
                                help='dataset_path',
                                required = True,
                                type=str)
    cmdline_parser.add_argument('-d', '--dataset',
                                help='dataset',
                                required = True,
                                type=str)
    cmdline_parser.add_argument('-m', '--marker_genes',
                                default="./preprocessing/marker_genes.txt",
                                help='marker genes',
                                type=str)
    cmdline_parser.add_argument('-o', '--output_h5ad',
                                default="test_output.h5ad",
                                help='output',
                                type=str)
    cmdline_parser.add_argument('-r', '--output_ref_h5ad',
                                default="test_ref.h5ad",
                                help='output',
                                type=str)
    cmdline_parser.add_argument('-q', '--output_query_h5ad',
                                default="test_query.h5ad",
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
    cmdline_parser.add_argument('--no_hvg', action=argparse.BooleanOptionalAction, default = False)
                                
    args, unknowns = cmdline_parser.parse_known_args()
    print(args.no_hvg)
    
    if not args.dataset_path.endswith(os.sep):
        args.dataset_path = args.dataset_path + os.sep
    
    
    preprocessing(marker_gene_list = args.marker_genes, data_directory = args.dataset_path, dataset = args.dataset, full_name = args.output_h5ad, ref_name = args.output_ref_h5ad, query_name = args.output_query_h5ad, batch_key=args.dataset_key, no_hvg = args.no_hvg)
    
    obs_fields = [args.dataset_key]
    layer_fields = []
    
    translate(args.dataset_path, args.output_h5ad, args.npy_dataset, obs_fields, layer_fields)
