from skimage.measure import block_reduce
import numpy as np
from .thesis_logger import logger

def smallify_response(response, block_size=(10, 10)):
    small_data = np.array([[block_reduce(frame, block_size, np.mean) for frame in trial] for trial in response])
    logger.debug(f"Shape of small_data is {small_data.shape}")
    return small_data