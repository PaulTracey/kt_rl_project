# main.py
# --------------------------------------------------------------
# main entry point
#
# Launches TAP.
#
# Used for:
# - Configuring Loguru logging (console + optional file logs)
# - Executing the main Pygame GUI module (ui/pygame_gui_grid.py)
# - Handling unhandled exceptions with logging and safe exit
# --------------------------------------------------------------
import sys
import runpy
from loguru import logger
from pathlib import Path



def configure_logging(debug=False):

    """Configure the Loguru logger for the project.

    Sets up:
      - Console logging (INFO level by default, DEBUG if enabled).
      - Optional file logging when debug is enabled:
        * debug.log (DEBUG level and above).
        * errors.log (ERROR level and above).
      - Log rotation (100 MB) and retention (20 files).

    Note:
      To log in other modules, import the global ``logger`` from ``loguru``.

    Args:
      debug: If True, enable verbose logging and file outputs.
    """

    # Check log directory exists and create it if not.
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Clear any existing logger
    logger.remove()

    # Log INFO level and above to the console
    logger.add(sys.stderr, level="INFO")

    # log DEBUG + to debug.log and separate run time errors in errors.log
    # set the log files size and how many to keep, can set retention to days such as "3 days"

    if debug:
        logger.add(log_dir / "debug.log", level="DEBUG", rotation="100 MB", retention=20,
                   format="{time} | {level} | {message}")
        logger.add(log_dir / "errors.log", level="ERROR", rotation="100 MB", retention=20,
                   format="{time} | {level} | {message}")
        logger.info("Debug logging enabled.")
    else:
        logger.info("Debug logging disabled for faster training.")

    # confirmation logs
    logger.info("Loguru logging started.")
    logger.info(f"Log files are being written to: {log_dir.resolve()}")

def run_game():
    # execute ui/pygame_gui_grid.py as if it were __main__
    runpy.run_module("ui.pygame_gui_grid", run_name="__main__")


if __name__ == "__main__":

    try:
        # start the logger
        configure_logging(debug=False)
        # Launch gui script using runpy
        runpy.run_module("ui.pygame_gui_grid", run_name="__main__")


    except Exception:
        # Log any unhandled exception and exit with error code 1
        logger.exception("An unhandled exception occurred:")
        sys.exit(1)