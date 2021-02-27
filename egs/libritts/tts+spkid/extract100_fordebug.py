import argparse
import json
import codecs

def get_parser():
    parser = argparse.ArgumentParser(
        description='Extract N examples from a json file (mostly for debug purpose',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('json', type=str,
                        help='json file path')
    parser.add_argument('-N', '--num-example', default=100, type=int,
                        help='The number of examples to extract')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    j_extract = {'utts':{}}
    with codecs.open(args.json, 'r', encoding="utf-8") as f:
        j = json.load(f)

    for i in list(j['utts'].keys())[0:args.num_example]:
        j_extract['utts'][i] = j['utts'][i]

    jsonstring = json.dumps({'utts': j_extract['utts']}, indent=4, ensure_ascii=False,
                            sort_keys=True, separators=(',', ': '))
    print(jsonstring)
