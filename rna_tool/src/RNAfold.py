import os
import subprocess
import pandas as pd

def encode_structure(structure, energy):
    # 根据规则编码
    encoding = []
    for char in structure:
        if char == '.':
            encoding.append(0)
        elif char == '(':
            encoding.append(-1)
        elif char == ')':
            encoding.append(1)
        elif char == '~':
            encoding.append(2)
        elif char == '+':
            encoding.append(-2)
    encoding.append(energy.strip('()'))
    return encoding

def parse_rna_fold_output(output_file):
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Error: {output_file} does not exist.")

    pattern = ('.', ',', '(', ')', '+', '~')
    encoded_data = []
    _outblock = False
    with open(output_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                # 跳过 FASTA 头部行
                _outblock = True
                continue
            if line.startswith(pattern) and _outblock:
                # 找到第三行结构并编码
                try:
                    _line_content = line.strip().split(' ')
                    if len(_line_content) == 2:
                        structure_line, _ene = line.strip().split(' ')
                    elif len(_line_content) == 3:
                        structure_line, _, _ene = line.strip().split(' ')
                    elif len(_line_content) == 4:
                        structure_line, _, _, _ene = line.strip().split(' ')
                    else:
                        print(f'Unexpected line length: {line}')
                    _outblock = False
                    encoded_data.append(encode_structure(structure_line, _ene))
                    continue
                except:
                    print()
                

    return encoded_data

def save_to_csv(data, output_csv):
    _d = pd.DataFrame(data)
    _d.to_csv(output_csv, header=False, index=False)

def RNAfold(fasta):
    with open('rna_predict.fasta', 'w') as f:
        for i, l in enumerate(fasta):
            f.write(f'> fasta_{i}\n{l}\n')
    out_file = 'rna_fold.out'
    out_csv = 'rna_fold.txt'
    command = f'RNAfold rna_predict.fasta -T 37 --salt 1.0 -d 2 --noLP > {out_file}'
    subprocess.run(command, shell=True, check=True)
    encoded_data = parse_rna_fold_output(out_file)
    save_to_csv(encoded_data, out_csv)
    subprocess.run("rm -- *ps", shell=True)

    file = 'rna_fold.txt'
    df = pd.read_csv(file, header=None)
    return df
