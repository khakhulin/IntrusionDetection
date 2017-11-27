from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

_f1 = make_scorer(f1_score, 'macro')

def classification_report(y_true, y_pred, labels=None, target_names=None,
                          sample_weight=None, digits=2):
  
    if labels is None:
        labels = np.unique(y_true)
    else:
        labels = np.asarray(labels)

    if target_names is not None and len(labels) != len(target_names):
        warnings.warn(
            "labels size, {0}, does not match size of target_names, {1}"
            .format(len(labels), len(target_names))
        )

    last_line_heading = 'avg / total'

    if target_names is None:
        target_names = [u'%s' % l for l in labels]
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None,
                                                  sample_weight=sample_weight)

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'
    rows = zip(target_names, p, r, f1, s)
    for row in rows:
        report += row_fmt.format(*row, width=width, digits=digits)

    report += u'\n'


    report += row_fmt.format(last_line_heading,
                             np.average(p, weights=s),
                             np.average(r, weights=s),
                             f1_score(y_true, y_pred,  labels=[0,1,2,3,4], average='macro'),
                             np.sum(s),
                             width=width, digits=digits)

    return report
