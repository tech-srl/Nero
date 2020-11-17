import logging

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = list(range(8))

#The background is set with 40 plus the number of the color, and the foreground with 30

#These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


def formatter_message(message, use_color=True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message

COLORS = {
    'WARNING': YELLOW,
    'INFO': GREEN,
    'DEBUG': BLUE,
    'CRITICAL': MAGENTA,
    'ERROR': RED
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    @staticmethod
    def color_text(text, color):
        return COLOR_SEQ % (30 + color) + text + RESET_SEQ

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = ColoredFormatter.color_text(levelname, COLORS[levelname])
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


FORMAT = "%(levelname)-19s | $BOLD%(name)s$RESET | %(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
COLOR_FORMAT = formatter_message(FORMAT, True)


def register_colored_root_logger(level):
    root_logger = logging.getLogger()
    root_logger.setLevel(level=level)
    console = logging.StreamHandler()
    console.setFormatter(ColoredFormatter(COLOR_FORMAT))
    root_logger.addHandler(console)

