# install_R_packages.R - run in R to install packages required for the pipeline
packages <- c("readr","mice","caret","DMwR","dplyr","ggplot2","gridExtra","e1071","randomForest","keras","tensorflow","pROC")
install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}
for (p in packages) install_if_missing(p)
# For keras/tensorflow, run:
# library(keras); install_keras()
