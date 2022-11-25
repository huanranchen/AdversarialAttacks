from .PGD import PGD


class FGSM(PGD):
    def __init__(self, *args, **kwargs):
        kwargs['total_step'] = 1
        kwargs['random_start'] = False
        super(FGSM, self).__init__(*args, **kwargs)
