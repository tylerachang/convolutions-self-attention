"""
Parse eval GLUE scores from a directory.
"""

from __future__ import unicode_literals

import codecs
import argparse
import os

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    return parser

def main(input):
    suffixes = ["", "1", "2"]
    score_dicts = [{} for i in range(len(suffixes))]
    tasks = [("CoLA", "eval_results_cola.txt", "eval_mcc"),
             ("MNLI-m", "eval_results_mnli.txt", "eval_mnli/acc"),
             ("MNLI-mm", "eval_results_mnli-mm.txt", "eval_mnli-mm/acc"),
             ("MRPC", "eval_results_mrpc.txt", "eval_acc_and_f1"),
             ("QNLI", "eval_results_qnli.txt", "eval_acc"),
             ("QQP", "eval_results_qqp.txt", "eval_acc_and_f1"),
             ("RTE", "eval_results_rte.txt", "eval_acc"),
             ("SST-2", "eval_results_sst-2.txt", "eval_acc"),
             ("STS-B", "eval_results_sts-b.txt", "eval_corr")]
    for i in range(len(suffixes)):
        for task, eval_path, task_metric in tasks:
            task_dir_name = "MNLI" if "MNLI" in task else task
            full_eval_path = os.path.join(input, task_dir_name)
            full_eval_path = full_eval_path + suffixes[i]
            full_eval_path = os.path.join(full_eval_path, eval_path)
            try:
                infile = codecs.open(full_eval_path, 'rb', encoding='utf-8')
                for line in infile:
                    if task_metric in line:
                        score = float(line.split()[-1])
                        score = round(score*100, 1) # Round to one decimal place.
                        score_dicts[i][task] = score
                        break
                infile.close()
            except FileNotFoundError:
                score_dicts[i][task] = 'N/A'
    # Print results.
    print("GLUE EVAL SCORES:")
    for task, _ in score_dicts[0].items():
        print(task, end="\t")
    print('', end="\n")
    for score_dict in score_dicts:
        for _, score in score_dict.items():
            print(score, end="\t")
        print('', end="\n")

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.input)
