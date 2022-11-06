import time
import datetime
from collections import defaultdict, deque
import numpy as np

from utils.train_tools import *


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a window or the global series average. """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """ Warning: does not synchronize the deque! """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(header, total_time_str, total_time / len(iterable)))


class MultiLabelConfusionMatrix:
    def __init__(self, num_classes, device=torch.device("cpu"), normalized=False):
        if num_classes <= 1:
            raise ValueError("Argument num_classes needs to be > 1")

        self.num_classes = num_classes
        self._num_examples = 0
        self.normalized = normalized
        self._device = device
        self.confusion_matrix = torch.zeros(self.num_classes, 2, 2, dtype=torch.int64, device=self._device)

    def reset(self):
        self.confusion_matrix = torch.zeros(self.num_classes, 2, 2, dtype=torch.int64, device=self._device)
        self._num_examples = 0

    def update(self, output):
        # self._check_input(output)
        y_pred, y_true = output[0].detach(), output[1].detach().long()
        y_pred = torch.ge(y_pred, 0.0)

        self._num_examples += y_true.shape[0]
        y_reshaped = y_true.transpose(0, 1).reshape(self.num_classes, -1)
        y_pred_reshaped = y_pred.transpose(0, 1).reshape(self.num_classes, -1)

        y_total = y_reshaped.sum(dim=1)
        y_pred_total = y_pred_reshaped.sum(dim=1)

        tp = (y_reshaped * y_pred_reshaped).sum(dim=1)
        fp = y_pred_total - tp
        fn = y_total - tp
        tn = y_reshaped.shape[1] - tp - fp - fn

        self.confusion_matrix += torch.stack([tn, fp, fn, tp], dim=1).reshape(-1, 2, 2).to(self._device)

    def compute(self):
        if self._num_examples == 0:
            raise ValueError("Confusion matrix must have at least one example before it can be computed.")

        if self.normalized:
            conf = self.confusion_matrix.to(dtype=torch.float64)
            sums = conf.sum(dim=(1, 2))
            return conf / sums[:, None, None]

        return self.confusion_matrix

    def compute_precision_recall(self):
        torch.set_printoptions(precision=2)

        # Compute precision/recall for each classes
        positive_predictions = self.confusion_matrix[:, 1, 1] + self.confusion_matrix[:, 0, 1]
        positive_truths = self.confusion_matrix[:, 1, 0] + self.confusion_matrix[:, 1, 1]
        precision = self.confusion_matrix[:, 1, 1] / positive_predictions
        recall = self.confusion_matrix[:, 1, 1] / positive_truths

        mean_precision = precision.mean()
        mean_recall = recall.mean()

        # Compute global precision/recall
        global_precision = self.confusion_matrix[:, 1, 1].sum() / positive_predictions.sum()
        global_recall = self.confusion_matrix[:, 1, 1].sum() / positive_truths.sum()

        print(f'Precision for each classes: {precision} and their average: {mean_precision.item():.2f}')
        print(f'Recall for each classes: {recall} and their average: {mean_recall.item():.2f}')
        print(f'Global precision: {global_precision.item():.2f} and global recall: {global_recall.item():.2f}')

        metrics = {
            'mean_precision': mean_precision.item(),
            # 'precision': precision,
            'mean_recall': mean_recall.item(),
            # 'recall': recall,
            'global_precision': global_precision.item(),
            'global_recall': global_recall.item()
        }
        return metrics

    def _check_input(self, output) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        if y_pred.ndimension() < 2:
            raise ValueError(
                f"y_pred must at least have shape (batch_size, num_classes (currently set to {self.num_classes}), ...)"
            )

        if y.ndimension() < 2:
            raise ValueError(
                f"y must at least have shape (batch_size, num_classes (currently set to {self.num_classes}), ...)"
            )

        if y_pred.shape[0] != y.shape[0]:
            raise ValueError(f"y_pred and y have different batch size: {y_pred.shape[0]} vs {y.shape[0]}")

        if y_pred.shape[1] != self.num_classes:
            raise ValueError(f"y_pred does not have correct number of classes: {y_pred.shape[1]} vs {self.num_classes}")

        if y.shape[1] != self.num_classes:
            raise ValueError(f"y does not have correct number of classes: {y.shape[1]} vs {self.num_classes}")

        if y.shape != y_pred.shape:
            raise ValueError("y and y_pred shapes must match.")

        valid_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if y_pred.dtype not in valid_types:
            raise ValueError(f"y_pred must be of any type: {valid_types}")

        if y.dtype not in valid_types:
            raise ValueError(f"y must be of any type: {valid_types}")

        if not torch.equal(y_pred, y_pred ** 2):
            raise ValueError("y_pred must be a binary tensor")

        if not torch.equal(y, y ** 2):
            raise ValueError("y must be a binary tensor")


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    Arguments
    ---------
    ranks : zero-based ranks of positive images
    nres  : number of positive images
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    num_img_ranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0
    recall_step = 1. / nres

    for j in np.arange(num_img_ranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2.
    return ap


def compute_map(ranks, gnd, kappas=()):
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precision (map) only
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd struct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    mAP = 0.
    num_queries = len(gnd)  # number of queries
    aps = np.zeros(num_queries)
    pr = np.zeros(len(kappas))
    prs = np.zeros((num_queries, len(kappas)))
    nempty = 0

    for i in np.arange(num_queries):
        query_gnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if query_gnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            query_gnd_j = np.array(gnd[i]['junk'])
        except:
            query_gnd_j = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], query_gnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], query_gnd_j)]

        k = 0
        ij = 0
        if len(junk):
            # decrease positions of positives based on the number of junk images appearing before them
            ip = 0
            while ip < len(pos):
                while ij < len(junk) and pos[ip] > junk[ij]:
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(query_gnd))
        mAP = mAP + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1  # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    mAP = mAP / (num_queries - nempty)
    pr = pr / (num_queries - nempty)

    return mAP, aps, pr, prs
