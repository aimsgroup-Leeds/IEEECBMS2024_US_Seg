import numpy as np
from .metrics_seg import dice_score, jac_score, hausdorff_distance, recall, precision

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.dice_coeff = []
        self.jaccard_index = []
        self.hausdorff_distance = []
        self.precision = []
        self.recall = []
        self.avg_DC_perimage = []
        self.avg_JC_perimage = []
        self.avg_HD_perimage = []
        self.avg_prec_perimage = []
        self.avg_recall_perimage = []
        
    
    def update(self, label_trues, label_preds):
        
        
        dice_score_per_image = []
        Jac_index_per_image = []
        HD_per_image = []
        Prec_per_image = []
        Recall_per_image = []
        
        for lt, lp in zip(label_trues, label_preds):
            
            d_score = dice_score(lt, lp)
            j_score = jac_score(lt,lp)
            HD_score = hausdorff_distance(lt, lp)
            prec_val = precision(lt,lp)
            recall_val = recall(lt,lp)
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
            self.dice_coeff.append(d_score)
            self.jaccard_index.append(j_score)
            self.hausdorff_distance.append(HD_score)
            self.precision.append(prec_val)
            self.recall.append(recall_val)            
            
            dice_score_per_image.append(d_score)
            Jac_index_per_image.append(j_score)
            HD_per_image.append(HD_score)
            Prec_per_image.append(prec_val)
            Recall_per_image.append(recall_val)
            
        Avg_dice_coef_per_image = sum(dice_score_per_image) / len(dice_score_per_image)
        Avg_Jac_index_per_image = sum(Jac_index_per_image) / len(Jac_index_per_image)
        Avg_HD_per_image = sum(HD_per_image) / len(HD_per_image)
        Avg_Prec_per_image = sum(Prec_per_image) / len(Prec_per_image)
        Avg_Recall_per_image = sum(Recall_per_image) / len(Recall_per_image)
        
        
        self.avg_DC_perimage.append(Avg_dice_coef_per_image)
        self.avg_JC_perimage.append(Avg_Jac_index_per_image)
        self.avg_HD_perimage.append(Avg_HD_per_image)
        self.avg_prec_perimage.append(Avg_Prec_per_image)
        self.avg_recall_perimage.append(Avg_Recall_per_image)
            
            
            # # Calculate precision and recall
            # tp = self.confusion_matrix[0, 0]  # True Positives
            # fp = self.confusion_matrix[0, 1]  # False Positives
            # fn = self.confusion_matrix[1, 0]  # False Negatives
            # precision = tp / (tp + fp + 1e-5)
            # recall = tp / (tp + fn + 1e-5)
            # self.precision.append(precision)
            # self.recall.append(recall)

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        # string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self, Eval = False):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
       
        if Eval:
            
            Seg_Perf = np.argsort(self.avg_DC_perimage)
        
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        Avg_Dice_Coef = sum(self.dice_coeff) / len(self.dice_coeff)
        Avg_Jaccard_Coef = sum(self.jaccard_index) / len(self.jaccard_index)
        Avg_hausdorff_distance = sum(self.hausdorff_distance) / len(self.hausdorff_distance)
        Avg_Precision = sum(self.precision) / len(self.precision)
        Avg_Recall = sum(self.recall) / len(self.recall)
        Std_Dice_Coef = np.std(self.avg_DC_perimage)
        Std_Jaccard_Coef = np.std(self.avg_JC_perimage)
        Std_hausdorff_distance = np.std(self.avg_HD_perimage)
        Std_Precision = np.std(self.avg_prec_perimage)
        Std_Recall = np.std(self.avg_recall_perimage)
        
        results = {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
            "Avg Dice Coefficient": Avg_Dice_Coef,
            "Std Dice Coefficient": Std_Dice_Coef,
            "Avg Jaccard Index": Avg_Jaccard_Coef,
            "Std Jaccard Index": Std_Jaccard_Coef,
            "Avg hausdorff_distance": Avg_hausdorff_distance,
            "Std hausdorff_distance": Std_hausdorff_distance,
            "Avg Precision": Avg_Precision,
            "Std Precision": Std_Precision,
            "Avg Recall": Avg_Recall,
            "Std Recall": Std_Recall
        }
        
        if Eval:
            results["Seg_Performance Image_indices_ranked"] = Seg_Perf
            
        return results
            
            
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    """Computes average values"""

    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()

    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0] += val
            record[1] += 1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
