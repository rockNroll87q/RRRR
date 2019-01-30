#!/usr/bin/env python
'''
Author Met
07/11/17 - 10:10
'''
import datetime
import logging
from logging import handlers

HOST = 'smtp.gmail.com'
USER = 'fmriworldmap@gmail.com'
PWD = 'svanerasavardi'
FROM = '"APPLICATION ALERT" <fmriworldmap@gmail.com>'
TO = 'fmriworldmap@gmail.com'
SUBJECT = 'New Critical Event From [fMRI WorldMap]'
LOG_FILENAME = 'debug.log'


class TimeFilter(logging.Filter):
    def filter(self, record):
        try:
            last = self.last
        except AttributeError:
            last = record.relativeCreated

        delta = datetime.datetime.fromtimestamp(record.relativeCreated / 1000.0) - datetime.datetime.fromtimestamp(
            last / 1000.0)

        record.relative = '{0:.2f}'.format(delta.seconds + delta.microseconds / 1000000.0)

        self.last = record.relativeCreated
        return True


class TlsSMTPHandler(logging.handlers.SMTPHandler):
    def emit(self, record):
        """
        Emit a record.

        Format the record and send it to the specified addressees.
        """
        try:
            import smtplib
            import string # for tls add this line
            try:
                from email.utils import formatdate
            except ImportError:
                formatdate = self.date_time
            port = self.mailport
            if not port:
                port = smtplib.SMTP_PORT
            smtp = smtplib.SMTP(self.mailhost, port)
            msg = self.format(record)
            msg = "From: %s\r\nTo: %s\r\nSubject: %s\r\nDate: %s\r\n\r\n%s" % (
                            self.fromaddr,
                            string.join(self.toaddrs, ","),
                            self.getSubject(record),
                            formatdate(), msg)
            if self.username:
                smtp.ehlo() # for tls add this line
                smtp.starttls() # for tls add this line
                smtp.ehlo() # for tls add this line
                smtp.login(self.username, self.password)
            smtp.sendmail(self.fromaddr, self.toaddrs, msg)
            smtp.quit()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def initialzie_logger(console=True, file=True, email=True, log_name='debug.log'):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_name)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create an email handler with higher log level
    eh = TlsSMTPHandler((HOST, 587), FROM, TO, SUBJECT,(USER, PWD))
    eh.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(fmt="%(asctime)-15s (%(relative)ss): %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    eh.setFormatter(formatter)
    # Add time filer
    fh.addFilter(TimeFilter())
    ch.addFilter(TimeFilter())
    eh.addFilter(TimeFilter())
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.addHandler(eh)

    return logger
