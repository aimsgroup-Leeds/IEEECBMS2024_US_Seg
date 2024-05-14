# Making Cross_val files
import os
import argparse



def get_argparser():
    parser = argparse.ArgumentParser()
    return parser

def append_files(source_file1, source_file2, destination_file):
    with open(source_file1, 'r') as source1, open(source_file2, 'r') as source2:
        source_contents = source1.readlines() + source2.readlines()
    
    # Sort the filenames before writing them to the destination file
    source_contents.sort()
    
    with open(destination_file, 'w') as dest:
        dest.writelines(source_contents)
        
        
        
opts = get_argparser()

opts.data_root = r'/home/scs/Desktop/Eddy/BUSI'

append_files(os.path.join(opts.data_root, 'train.txt'),
             os.path.join(opts.data_root, 'val.txt'),
             os.path.join(opts.data_root, 'cross_val.txt'))