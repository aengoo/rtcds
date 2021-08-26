class VideoLabel:
    def __init__(self, label_path):
        self.label_dict = {}
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '')
                if line:
                    tokens = list(line.split(', '))
                    tokens[2:] = [int(ax) for ax in tokens[2:]]
                    frame = int(tokens[0])
                    if frame in self.label_dict.keys():
                        self.label_dict[frame].append(tokens[1:])
                    else:
                        self.label_dict.update({frame: [tokens[1:]]})

    def get_gt_boxes(self, frame_idx: int):
        if frame_idx in self.label_dict.keys():
            return self.label_dict[frame_idx]
        else:
            return [[]]

