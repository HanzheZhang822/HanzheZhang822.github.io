# List of required packages
packages <- c("rpart", "rpart.plot", "ROCR", "ggplot2", 
              "randomForest", "gbm", "MASS", "Rtsne", "flexclust")

# Function to check and install missing packages
install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Apply function to each package in the list
sapply(packages, install_if_missing)

# Print completion message
cat("✅ All required packages are installed and loaded.\n")
