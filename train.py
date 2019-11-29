from utils import TrainOptions
from train import Trainer

# Load own predtrained model although model state dict changed
def load_my_state_dict(self, state_dict):

    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
             continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)


if __name__ == '__main__':
    options = TrainOptions().parse_args()
    trainer = Trainer(options)
    trainer.train()
