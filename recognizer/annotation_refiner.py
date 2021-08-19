from __future__ import print_function
import argparse
import os


def most_frequent(lt):
    counter = 0
    num = lt[0]

    for i in lt:
        curr_frequency = lt.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i
    return num


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../save/test/raw', help='')
parser.add_argument('--txt-path', type=str, default='labels', help='')
parser.add_argument('--save', type=str, default='refined_1', help='')
OPT = parser.parse_args()

txt_list = os.listdir(os.path.join(OPT.data, OPT.txt_path))

save_dir = os.path.join(OPT.data, OPT.save)
os.makedirs(save_dir, exist_ok=True)

tokenizer = ', '

'''
filename = 'BJW_1.MP4.txt'
id_counter = []
new_tokens_list = []
new_lines = []
mf = ''
name = filename.split('_')[0]

# check most frequent tracked face
with open(os.path.join(OPT.data, OPT.txt_path, filename), 'r') as f:
    for line in f.readlines():
        id_counter.append(line.split(', ')[1])
    mf = most_frequent(id_counter)

# delete mis_detected face, change id to initial
with open(os.path.join(OPT.data, OPT.txt_path, filename), 'r') as f:
    for line in f.readlines():
        if line.split(tokenizer)[1] == mf:
            tokens = line.split(tokenizer)
            tokens[1] = name
            new_tokens_list.append(tokens)  # tokenizer.join(tokens)

# check output frame
last_frame = int(new_tokens_list[0][0]) - 1
for tokens in new_tokens_list:
    now_frame = int(tokens[0])
    if now_frame - 1 != last_frame:
        now_frame = last_frame + 1
    tokens[0] = str(now_frame)
    last_frame = now_frame

for tokens in new_tokens_list:
    new_lines.append(tokenizer.join(tokens))

[print(line) for line in new_lines]
'''

for file_idx, file_name in enumerate(txt_list):
    save_path = os.path.join(save_dir, file_name)

    id_counter = []
    new_tokens_list = []
    new_lines = []
    mf = ''
    name = file_name.split('_')[0]

    # check most frequent tracked face
    with open(os.path.join(OPT.data, OPT.txt_path, file_name), 'r') as f:
        for line in f.readlines():
            id_counter.append(line.split(', ')[1])
        mf = most_frequent(id_counter)

    # delete mis_detected face, change id to initial
    with open(os.path.join(OPT.data, OPT.txt_path, file_name), 'r') as f:
        for line in f.readlines():
            if line.split(tokenizer)[1] == mf:
                tokens = line.split(tokenizer)
                tokens[1] = name
                new_tokens_list.append(tokens)  # tokenizer.join(tokens)

    # check output frame
    last_frame = int(new_tokens_list[0][0]) - 1
    for tokens in new_tokens_list:
        now_frame = int(tokens[0])
        if now_frame - 1 != last_frame:
            now_frame = last_frame + 1
        tokens[0] = str(now_frame)
        last_frame = now_frame

    for tokens in new_tokens_list:
        new_lines.append(tokenizer.join(tokens))

    with open(os.path.join(OPT.data, OPT.save, file_name), 'a') as f:
        for line in new_lines:
            f.write(line)

