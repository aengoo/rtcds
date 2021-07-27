matched_point = 10
empty_point = 1


class Face:
    def __init__(self, coord, name):
        self.name = name
        self.point = empty_point if name == '-' else matched_point
        self.last_coord = coord
        self.is_new = False

    def add_face(self, coord, name):
        self.is_new = False
        self.last_coord = coord
        if self.name == name:
            self.point += empty_point if name == '-' else matched_point
        else:
            tp = empty_point if name == '-' else matched_point
            if self.point - tp < 0:
                self.name = name
                self.point = abs(self.point - tp)
            else:
                self.point -= tp

    def get_name(self):
        return self.name

    def get_last_coord(self):
        return self.last_coord

    def get_conf_point(self):
        return self.point

    def reset(self):
        self.is_new = True

    def is_renewed(self):
        return not self.is_new
