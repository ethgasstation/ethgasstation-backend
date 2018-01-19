"""
    Output
    Handles output to stdout for now, will eventually use logging.
"""

from colored import fg, bg, attr
import time

class OutputException(Exception):
    pass

class Output(object):
    """A wrapper for EGS output to stdout, stderr, etc."""

    loglevel = None
    levels = [ 'off', 'fatal', 'error', 'warn', 'info', 'debug', 'trace' ]

    def __init__(self):
        self.set_loglevel('info')

    def set_loglevel(self, level):
        if not level.lower() in self.levels:
            raise Exception("Unsupported log level %d" % level)
        else:
            self.loglevel = self.levels.index(level.lower())

    def log(self, msg, level='info'):
        level = self._getlevel(level)
        if level <= self.loglevel:
            # TODO use stderr, stdout properly
            # TODO use logging
            print(msg)

    def warn(self, msg):
        msg = self._pad(msg)
        msg = "%s %s[WARN]  %s%s" % (self._logtime(), fg('yellow'), attr(0), msg)
        return self.log(msg, 'warn')

    def error(self, msg):
        msg = self._pad(msg)
        msg = "%s %s[ERROR] %s%s" % (self._logtime(), fg('red'), attr(0), msg)
        return self.log(msg, 'error')

    def info(self, msg):
        msg = self._pad(msg)
        msg = "%s %s[INFO]  %s%s" % (self._logtime(), fg('cyan'), attr(0), msg)
        return self.log(msg, 'info')

    def debug(self, msg):
        msg = self._pad(msg)
        msg = "%s %s[DEBUG] %s%s" % (self._logtime(), fg('magenta'), attr(0), msg)
        return self.log(msg, 'debug')

    def _logtime(self):
        return time.strftime('[%Y-%m-%d %H:%M:%S]')

    def _getlevel(self, level):
        level = level.lower()
        if not level in self.levels:
            raise OutputException("Unsupported level %s" % level)
        else:
            return self.levels.index(level)

    def _pad(self, string):
        """Pad trailing multilines"""
        lines = string.split("\n")
        skip = True
        for line in lines:
            if skip:
                skip = False
            line = "                           " + line
        return "\n".join(lines)

    def banner(self):
        """stupid easter eggs are fun"""
        print ("""      __  __                     __       __  _         
 ___ / /_/ /  ___ ____ ____ ___ / /____ _/ /_(_)__  ___ 
/ -_) __/ _ \/ _ `/ _ `(_-<(_-</ __/ _ `/ __/ / _ \/ _ \\
\__/\__/_//_/\_, /\_,_/___/___/\__/\_,_/\__/_/\___/_//_/
            /___/                                       
""")
