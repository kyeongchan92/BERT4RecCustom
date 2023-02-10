import torch


def recall(scores, labels, k):
    scores = scores
    labels = labels
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())).mean().cpu().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    scores = scores
    labels = labels
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
       hits = labels_float.gather(1, cut)
       metrics['Recall@%d' % k] = \
           (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float())
       dcg = (hits * weights.to(hits.device)).sum(1)
       idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
       ndcg = (dcg / idcg).mean()
       metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics


class PrintInputShape:
    flag = True

    @classmethod
    def print_shape(self, t):
        if len(t.size()) == 2:
            self.print_2d_shape(t)
        elif len(t.size()) == 3:
            self.print_3d_shape(t)
        else:
            print(f"over 3d")

    @classmethod
    def print_2d_shape(self, t, ws=''):
        if self.flag:
            t_trim = t.clone().detach()[:4]
            for line, _ in enumerate(t_trim, start=1):
                if line == 1:  # 첫 번째 줄
                    s = '^'
                    sample = f"{_[0]:5}, {_[1]:5}, ..., {_[-1]:5}"
                elif line == len(t_trim):  # 마지막 줄
                    s = 'v'
                    sample = f"{_[0]:5}, {_[1]:5}, ..., {_[-1]:5}"
                else:
                    s = '|'
                    sample = f"\t\t."
                print(f"{ws}{s} {sample}")

                if line == len(t_trim)//2:  # 중간에 숫자 끼워넣기
                    print(f"{ws}{t.size()[0]}")

            print(f"{ws}<-- {t.size()[1]} -->")

            self.flag = False

    @classmethod
    def print_3d_shape(self, t):
        if self.flag:
            print(f"Input's shape : {t.size()}")
            ws = ''
            print(f"{ws}\\")
            ws += ' '
            print(f"{ws}{t.size()[0]}")
            ws += ' '
            print(f"{ws}\\")

            print_2d_shape(self, t[-1], ws=ws)
            # for line, _ in enumerate(t[-1], start=1):
            #     if line == 1:
            #         s = '^'
            #         sample = f"{_[0]}, {_[1]}, ..., {_[-1]}"
            #     elif line == len(t[-1]):  # 마지막줄
            #         s = 'v'
            #         sample = f"{_[0]}, {_[1]}, ..., {_[-1]}"
            #     else:
            #         s = '|'
            #         sample = f"\t\t."
            #     print(f"{ws}{s} {sample}")

            #     if line == len(t[-1])//2:
            #         print(f"{t.size()[1]}")

            # print(f"{ws}<-- {t.size()[2]} -->")

        self.flag = False


