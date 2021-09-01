import os

logging_exception = {'dlib', '.idea', '__pycache__'}


class Logger:
    def __init__(self, save_path, test_id=''):
        self.save_dir = ''
        self.set_save_path(save_path, test_id)

    def set_save_path(self, save_path, test_id: str = ''):
        if test_id:
            self.save_dir = os.path.join(save_path, test_id)
        else:
            idx_list = [int(dn.split('_')[1]) if dn.startswith('test_') else -1 for dn in os.listdir(save_path)]
            test_id = str(max(idx_list) + 1).zfill(3)
            self.save_dir = os.path.join(save_path, 'test_' + test_id)
        os.makedirs(self.save_dir, exist_ok=True)

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



