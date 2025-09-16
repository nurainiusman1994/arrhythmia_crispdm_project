# data_acquisition.R
library(readr)
load_data <- function(path = "arrhythmia.csv") {
  if (!file.exists(path)) {
    stop("Dataset file not found. Please place 'arrhythmia.csv' in the project root.")
  }
  df <- read_csv(path, show_col_types = FALSE)
  # Ensure target column named 'class' or 'target'
  if (!("class" %in% names(df))) {
    possible <- names(df)[tolower(names(df)) %in% c("target","diagnosis","label","classid")]
    if (length(possible)>0) names(df)[names(df)==possible[1]] <- "class"
  }
  if (!("class" %in% names(df))) stop("Target column not found. Please ensure the dataset contains a class/target column.")
  df$class <- as.factor(df$class)
  return(df)
}
