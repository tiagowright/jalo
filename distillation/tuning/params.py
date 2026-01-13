f_output = r'./distillation/tuning/params/'

misclassification_penalty = 100.0
stability_penalty = 1000.0
sparsity_penalty = 1.0


misclassification_penalty_ranges = [1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
stability_penalty_ranges = [1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
sparsity_penalty_ranges = [1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]

f_header = r'''
# output = './distillation/logs/simple.csv'

[default]
layout = 'sturdy'
hardware = 'ansi'
corpus = 'en'
depth = 7
samples_per_layer = 80
name_format = '{layout}__C_{misclassification_penalty}__L1_{stability_penalty}__L2_{sparsity_penalty}'

misclassification_penalty = 100.0
stability_penalty = 1000.0
sparsity_penalty = 1.0

'''

f_run = r'''
[[runs]]
# C_{misclassification_penalty}__L1_{stability_penalty}__L2_{sparsity_penalty}
misclassification_penalty = {misclassification_penalty}
stability_penalty = {stability_penalty}
sparsity_penalty = {sparsity_penalty}
'''

print(f_header)
for misclassification_penalty in misclassification_penalty_ranges:
    for stability_penalty in stability_penalty_ranges:
        for sparsity_penalty in sparsity_penalty_ranges:
            if sparsity_penalty > stability_penalty:
                continue
            print(f_run.format(misclassification_penalty=misclassification_penalty, stability_penalty=stability_penalty, sparsity_penalty=sparsity_penalty))

