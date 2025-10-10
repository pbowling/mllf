from mllf.rl.graph import Graph
from mllf.file_handling.write_bias_coeff import write_bias_inp_from_graph
import os

def _read_set_names(path):
    names = []
    with open(path, 'r', encoding='utf-8') as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            if not ln.startswith('set '):
                continue
            parts = ln.split('=')
            left = parts[0].strip()
            _, name = left.split(None, 1)
            names.append(name)
    return names

if __name__ == '__main__':
    out_path = os.path.join(os.path.dirname(__file__), '..', 'tmp_generated.inp')
    out_path = os.path.abspath(out_path)
    g = Graph(5)
    for i in range(5):
        for j in range(i+1, 5):
            g.set_edge(i, j, [0.0, 0.0, 0.0, 0.0])
    subs = [3, 4, 8, 8, 8]
    write_bias_inp_from_graph(g, out_path, sub_counts=subs)
    example = os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'rl', 'variables85.inp')
    example = os.path.abspath(example)
    gen_names = set(_read_set_names(out_path))
    ex_names = set(_read_set_names(example))
    only_in_ex = sorted(ex_names - gen_names)
    only_in_gen = sorted(gen_names - ex_names)
    print('generated count:', len(gen_names))
    print('example count:  ', len(ex_names))
    print('only in example ({}):'.format(len(only_in_ex)))
    print('\n'.join(only_in_ex[:100]))
    print('\nonly in generated ({}):'.format(len(only_in_gen)))
    print('\n'.join(only_in_gen[:100]))
    
    # Print a few lines of generated file
    print('\n--- generated file head ---')
    with open(out_path) as fh:
        for i,ln in enumerate(fh):
            if i>=40: break
            print(ln.rstrip())
