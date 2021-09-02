import argparse
import os

logging_exception = {'dlib', '.idea', '__pycache__'}


class Logger:
    def __init__(self, save_path, test_id=''):
        self.save_dir = '.'
        self.set_save_path(save_path, test_id)

    def set_save_path(self, save_path, test_id: str = ''):
        if test_id:
            self.save_dir = os.path.join(save_path, test_id)
        else:
            idx_list = [int(dn.split('_')[1]) if dn.startswith('test_') else -1 for dn in os.listdir(save_path)]
            test_id = str(max(idx_list) + 1).zfill(3)
            self.save_dir = os.path.join(save_path, 'test_' + test_id)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'txt'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'mat'), exist_ok=True)

    def log_codes(self, path_stack='.'):
        import shutil
        assert self.save_dir
        copy_path = os.path.join(self.save_dir, 'codes')
        list_excepted = set(os.listdir(path_stack)) - logging_exception
        for i in list_excepted:
            if os.path.isdir(os.path.join(path_stack, i)):
                self.log_codes(path_stack=os.path.join(path_stack, i))
                pass
            elif os.path.isfile(os.path.join(path_stack, i)):
                os.makedirs(os.path.join(copy_path, path_stack), exist_ok=True)
                shutil.copy(os.path.join(path_stack, i), os.path.join(copy_path, path_stack))

    def print_args(self, args, is_save=False):
        f = None
        if is_save:
            f = open(os.path.join(self.save_dir, 'txt', 'args.txt'), 'a')
        if args is argparse.Namespace:
            print(args, file=f if f else None)
        elif args is dict:
            [print(args, ':', args[arg], file=f if f else None) for arg in args.keys()]
        if f:
            f.close()

    def print_eval(self, counter, is_save=False):
        from utils.eval_util import Counter
        assert type(counter) is Counter

        f = None
        if is_save:
            f = open(os.path.join(self.save_dir, 'txt', 'evals.txt'), 'a')
            counter.save_csv(save_path=os.path.join(self.save_dir, 'mat', 'result.csv'))

        print(counter.get_mat(), file=f if f else None)
        print('Acc:', counter.get_accuracy(), file=f if f else None)
        print('Multi-Precision:', counter.get_multi_precision(), file=f if f else None)
        print('Multi-Recall:', counter.get_multi_recall(), file=f if f else None)
        print('F1-score:', counter.get_f1_score(), counter.get_accuracy(), file=f if f else None)
        if f:
            f.close()

    def print_timer(self, timer_name, timer, is_save=False):
        from utils.timer import Timer
        assert type(timer) is Timer

        f = None
        if is_save:
            f = open(os.path.join(self.save_dir, 'txt', 'time.txt'), 'a')
        print(timer_name, ':', timer.average_time, file=f if f else None)
        if f:
            f.close()