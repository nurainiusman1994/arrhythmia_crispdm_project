# eda.R
library(ggplot2)
library(dplyr)
library(gridExtra)

eda_report <- function(df, outdir="artifacts") {
  dir.create(outdir, showWarnings = FALSE)
  # Basic summaries
  summary_stats <- summary(df)
  capture.output(summary_stats, file = file.path(outdir, "summary.txt"))
  # Missingness
  miss <- sapply(df, function(x) sum(is.na(x)))
  write.csv(as.data.frame(miss), file.path(outdir, "missingness.csv"))
  # Class distribution
  cls_dist <- table(df$class)
  write.csv(as.data.frame(cls_dist), file.path(outdir, "class_distribution.csv"))
  # Plot top numeric feature distributions and correlation matrix heatmap
  num_cols <- names(df)[sapply(df, is.numeric)]
  if (length(num_cols)>0) {
    p_list <- lapply(num_cols[1:min(6,length(num_cols))], function(col) {
      ggplot(df, aes_string(x=col)) + geom_histogram(bins=30) + ggtitle(col)
    })
    ggsave(file.path(outdir, "feature_histograms.png"), plot = marrangeGrob(p_list, nrow=2, ncol=3), width=12, height=8)
    # Correlation
    corr <- cor(df[,num_cols], use="pairwise.complete.obs")
    png(file.path(outdir, "correlation_heatmap.png"), width=800, height=800)
    heatmap(corr, symm = TRUE, main = "Correlation heatmap")
    dev.off()
    write.csv(as.data.frame(corr), file.path(outdir, "correlation_matrix.csv"))
  }
  return(list(summary = summary_stats, missing = miss, class_dist = cls_dist))
}
