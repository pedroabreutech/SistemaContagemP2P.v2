from .p2pnet import build

# constrói o modelo P2PNet
# defina training como 'True' durante o treino
def build_model(args, training=False):
    return build(args, training)