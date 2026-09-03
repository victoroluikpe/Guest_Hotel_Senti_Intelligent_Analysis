import sys
import logging

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Get detailed error message including the file name and line number where the exception occurred. 
    """
    _,_, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filenameline-number = exc_tb.tb_lineno
    error_message = f"Error occurred in file: [{file_name}, at line: {line_number}]: {str(error)}"

    logging.error(error_message) # log the error message

    return error_message

class MyException(Exception):
    """
    custom exception class that provides detailed error messages.

    """

    def__init__(self, error_message: str, error_detail: sys):

    # calling the base class constructor
    super().__init__(error_message)

    self.error_message = error_message_detail(error_message, error_detail)

    def__str__(self):
    """
    return the string representation of the exception
    """
    return self.error_message