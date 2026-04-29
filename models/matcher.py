
# Copyright (c) Facebook, Inc. e afiliadas. Todos os direitos reservados
"""
Em grande parte copiado/adaptado do DETR (https://github.com/facebookresearch/detr).
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher_Crowd(nn.Module):
    """Esta classe calcula uma associação entre os alvos e as predições da rede.

    Por motivos de eficiência, os alvos não incluem a classe `no_object`.
    Por isso, em geral, há mais predições do que alvos. Nesse caso, fazemos
    um matching 1-para-1 das melhores predições, enquanto as demais ficam
    sem correspondência (e são tratadas como não-objetos).
    """

    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        """Cria o matcher.

        Parâmetros:
            cost_class: peso relativo do objeto de primeiro plano.
            cost_point: peso relativo do erro L1 das coordenadas dos pontos no custo de matching.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Executa o matching.

        Parâmetros:
            outputs: dicionário com pelo menos:
                 "pred_logits": Tensor com dimensão [batch_size, num_queries, num_classes]
                                contendo os logits de classificação.
                 "points": Tensor com dimensão [batch_size, num_queries, 2]
                           contendo as coordenadas de pontos previstas.

            targets: lista de alvos (len(targets) = batch_size), em que cada alvo é um dicionário contendo:
                 "labels": Tensor com dimensão [num_target_points] (onde num_target_points é o número de objetos
                           de ground-truth no alvo) contendo os rótulos de classe.
                 "points": Tensor com dimensão [num_target_points, 2] contendo as coordenadas de pontos-alvo.

        Retorna:
            Uma lista de tamanho batch_size, contendo tuplas (index_i, index_j), em que:
                - index_i são os índices das predições selecionadas (em ordem)
                - index_j são os índices dos alvos correspondentes (em ordem)
            Para cada elemento do batch:
                len(index_i) = len(index_j) = min(num_queries, num_target_points)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Achatamos para calcular as matrizes de custo em batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_points = outputs["pred_points"].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Também concatena os rótulos e pontos alvo
        # tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_points = torch.cat([v["point"] for v in targets])

        # Calcula o custo de classificação. Diferente da loss, não usamos NLL,
        # e sim uma aproximação em 1 - proba[classe alvo].
        # O 1 é uma constante que não altera o matching, pode ser omitida.
        cost_class = -out_prob[:, tgt_ids]

        # Calcula o custo L2 entre pontos
        cost_point = torch.cdist(out_points, tgt_points, p=2)

        # Calcula o custo giou entre pontos

        # Matriz de custo final
        C = self.cost_point * cost_point + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["point"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher_crowd(args):
    return HungarianMatcher_Crowd(cost_class=args.set_cost_class, cost_point=args.set_cost_point)
