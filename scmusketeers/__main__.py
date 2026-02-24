from scmusketeers.arguments.runfile import get_runfile
from scmusketeers.run_musk import run_sc_musketeers
import logging
from importlib.metadata import version

logger = logging.getLogger("Sc-Musketeers")

def main_entry_point():
    # Set up logging
    logging.basicConfig(format="|--- %(levelname)-8s    %(message)s")
    logger.info(f"Sc-Musketeers {version('sc-musketeers')} started")

    # Get all arguments
    run_file = get_runfile()

    # Set logger level
    if run_file.debug:
        logger.setLevel(getattr(logging, "DEBUG"))
    else:
        logger.setLevel(getattr(logging, "INFO"))

    
    #run_file = get_default_param()
    logger.debug(run_file)
    run_sc_musketeers(run_file)

if __name__ == "__main__":
    main_entry_point()
