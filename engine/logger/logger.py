import sys, logging
    
class ColorFormatter(logging.Formatter):
    RESET = "\033[0m"  
    COLORS = {     
        "TIME": "\033[92m",        
        "LOCATION": "\033[93m",   
        "INFO": "\033[37m",       
        "DEBUG": "\033[36m",      
        "WARNING": "\033[33m",    
        "ERROR": "\033[31m",      
        "CRITICAL": "\033[1;31m",  
    }    

    def format(self, record):  
        time_str = f"{self.COLORS['TIME']}{self.formatTime(record, self.datefmt)}{self.RESET}"
        location = f"{self.COLORS['LOCATION']}[{record.filename}:{record.funcName}:{record.lineno}]{self.RESET}"   
        level_color = self.COLORS.get(record.levelname, self.RESET)
        levelname = f"{level_color}{record.levelname}{self.RESET}"  
        message = f"{level_color}{record.getMessage()}{self.RESET}"

        return f"{time_str} {location} {levelname}: {message}"   
  
def get_logger(name=None, level=logging.INFO):
    """     
    Creates and returns a logger.
    :param name: The name of the logger
    :param log_file: Optional, the file to write logs to
    :param level: The log level
    :return: A logger object

    """     
    logger = logging.getLogger(name)
    logger.setLevel(level)     
    logger.propagate = False    
    
    if not logger.handlers:  
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)   
        formatter = ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter) 
        logger.addHandler(ch)
    
    return logger     

def test_logger():  
    logger = get_logger(__name__)   
    logger.info("info")
    logger.warning("info")   
    logger.error("info")    
