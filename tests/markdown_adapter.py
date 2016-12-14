"""
Software API adapter for markdown.py
This module provides a function based API to markdown.py
since markdown.py only provides a CLI.
"""
 
from subprocess import Popen, PIPE, STDOUT
from tempfile import NamedTemporaryFile
import os
 
def run_markdown(input_text):
    """
    The default method when we don't care which method to use.
    """
    return run_markdown_pipe(input_text)
 
def run_markdown_pipe(input_text):
    """
    Simulate: echo 'some input' | python markdown.py
    """
    pipe = Popen(['python', 'markdown.py'],
            stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    output = pipe.communicate(input=input_text)[0]
    return output
 
def run_markdown_file(input_text):
    """
    Simulate: python markdown.py fileName
    """
    temp_file = NamedTemporaryFile(delete=False)
    temp_file.write(input_text)
    temp_file.close()
    pipe = Popen(['python', 'markdown.py', temp_file.name],
            stdout=PIPE, stderr=STDOUT)
    output = pipe.communicate()[0]
    os.unlink(temp_file.name)
    return output
