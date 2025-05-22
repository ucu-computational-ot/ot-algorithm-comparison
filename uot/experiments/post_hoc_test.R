library('scmamp')

args <- commandArgs(trailingOnly = TRUE)
experiment_result_path <- args[1]

results <- read.csv(experiment_result_path, row.names = 1)

test.res <- postHocTest(data = results, test = 'friedman', correct = 'bergmann')

print(test.res)