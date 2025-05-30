library('scmamp')

args <- commandArgs(trailingOnly = TRUE)
experiment_result_path <- args[1]

results <- read.csv(experiment_result_path, row.names = 1, check.names = FALSE)


test.res <- postHocTest(data = results, test = 'friedman', correct = 'bergmann')

write.table(test.res$corrected.pval, row.names = TRUE, col.names = NA, quote = FALSE, file = "")

cat("\n")

write.table(colMeans(rankMatrix(results, decreasing = FALSE)), row.names = TRUE, col.names = NA, quote = FALSE, file = "")