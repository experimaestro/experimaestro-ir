import os

# Prevent OpenMP duplicate library abort when both PyTorch and FAISS
# ship their own libomp (common on macOS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
