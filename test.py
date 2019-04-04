from process_file import process_file
import argparse


parser = argparse.ArgumentParser(description='Process IOB file, recognize NE and save the output to another IOB file.')
parser.add_argument('-i', required=True, metavar='PATH', help='input IOB file')

args = parser.parse_args()

if __name__ == '__main__':
    process_file(args.i, '/PolDeepNer/test_result.iob')
