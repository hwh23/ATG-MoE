import logging
import torch.distributed as dist
import coloredlogs
from tqdm import tqdm


# -------------------------------------------------------------
# 1) The RankZeroFilter
# -------------------------------------------------------------
class RankZeroFilter(logging.Filter):
    def __init__(self, name = "", rank_zero_only:bool=True):
        super().__init__(name)
        self.rank_zero_only = rank_zero_only
    
    def filter(self, record):
        # Always set a default rank if it doesn't exist.
        if not hasattr(record, 'rank'):
            record.rank = 'N/A'
        
        # If distributed is not available or not initialized, use a default.
        if not dist.is_available() or not dist.is_initialized():
            record.rank = 'N/A'
        else:
            record.rank = dist.get_rank()
        
        # Optionally, only log from rank 0 if desired.
        if self.rank_zero_only:
            return record.rank == 0 or record.rank == 'N/A'
        else:
            return True


# -------------------------------------------------------------
# 2) The TqdmLoggingHandler
# -------------------------------------------------------------
class TqdmLoggingHandler(logging.Handler):
    """
    A handler that writes log messages via tqdm, preventing the
    log output from breaking the progress bar.
    """
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)

class FixedWidthLevelFormatter(logging.Formatter):
    def format(self, record):
        padded_level = f"[{record.levelname}]"  # e.g., [INFO]
        record.padded_level = f"{padded_level:<10}"  # pad outside the brackets
        return super().format(record)
    
# -------------------------------------------------------------
# 3) The get_logger function.
# -------------------------------------------------------------
def get_logger(name: str, 
               print_rank: bool = False,
               rank_zero_only: bool = True, 
               level: str = "INFO") -> logging.Logger:
    """Creates a logger that:
       - uses coloredlogs for color formatting
       - uses TqdmLoggingHandler to avoid breaking tqdm progress bars
       - can optionally filter out non-rank-zero logs
    """
    logger = logging.getLogger(name)

    # Prevent re-initialization if the logger already has handlers.
    if logger.handlers:
        return logger

    # Set the desired level.
    logger.setLevel(level.upper())
    logger.propagate = False  # <--- prevent duplicate logging
    
    # Define format and date format.
    if print_rank:
        fmt = '[%(levelname)s] [rank: %(rank)s] %(message)s'
    else:
        fmt = '[%(levelname)s] %(message)s'  # Now applies padding to the full prefix
        
        
    datefmt = '%Y-%m-%d %H:%M:%S'
    
    # Custom level styles.
    level_styles = {
        'debug': {'color': 'green'},
        'info': {'color': 'blue'},
        'warning': {'color': 'yellow'},
        'error': {'color': 'red'},
        'critical': {'color': 'red', 'bold': True},
    }
    
    # 1) Install coloredlogs, which by default adds a StreamHandler.
    coloredlogs.install(
        level=level.upper(),
        logger=logger,
        fmt=fmt,
        datefmt=datefmt,
        level_styles=level_styles
    )
    
    # 2) Remove the handler that coloredlogs installed, so we don't
    #    double-print or break tqdm with a plain StreamHandler.
    for h in list(logger.handlers):
        if isinstance(h, logging.StreamHandler):
            logger.removeHandler(h)
    
    
    # 3) Create a TqdmLoggingHandler and set a colored formatter on it.
    tqdm_handler = TqdmLoggingHandler()
    tqdm_formatter = coloredlogs.ColoredFormatter(
        fmt=fmt,
        datefmt=datefmt,
        level_styles=level_styles
    )
    tqdm_handler.setFormatter(tqdm_formatter)
    logger.addHandler(tqdm_handler)

    # 4) Add the rank-zero filter so that only rank 0 logs (if desired).
    if print_rank:
        logger.addFilter(RankZeroFilter(rank_zero_only=rank_zero_only))
    
    return logger

def log_prefix(tag, addr=None, tag_width=9, addr_width=16):
    tag_str = f'[{tag}]'
    if addr:
        addr_str = f"{addr[0]}:{addr[1]}" if isinstance(addr, tuple) else str(addr)
        return f"{tag_str:<{tag_width}} {addr_str:<{addr_width}}"
    else:
        return f"{tag_str:<{tag_width}}"