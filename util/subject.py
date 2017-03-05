

class Subject(object):
    def __init__(self, p_id, group):
        self.p_id = p_id
        self.group = group
        self.image = None

    def __str__(self):
        return '<%s-%s>' % (self.group, self.p_id)

    def __repr__(self):
        return str(self)
