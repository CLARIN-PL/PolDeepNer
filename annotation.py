class Annotation:
    def __init__(self, annotation, sid, id):
        self.sentence_id = sid
        self.token_ids = [id]
        self.annotation = annotation.replace('-', '_')
        self.annotation = self.annotation.replace(' ', '')

    def add_id(self, id):
        self.token_ids.append(id)

    def __str__(self):
        return self.annotation

    def __eq__(self, other):
        return self.annotation == other.annotation and self.token_ids[0] == other.token_ids[0] and \
               self.token_ids[-1] == other.token_ids[-1] and self.sentence_id == other.sentence_id

    def __hash__(self):
        return hash(self.annotation + str(self.sentence_id) + str(self.token_ids[0]) + str(self.token_ids[-1]))

    @property
    def annotation_length(self):
        return self.token_ids[-1] - self.token_ids[0]
