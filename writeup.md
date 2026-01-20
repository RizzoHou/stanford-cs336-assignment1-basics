problem unicode1
---
1. '\x00'
2. its `repr` result is a string consisting of exactly 6 characters ("'\x00'"), while its true value is a character whose unicode encoding is 0 ('\x00').
3. this character seems to represent nothing when occuring in text.

problem unicode2
---
1. the string encoded with `utf-16` or `utf-32` can contain many zeros which lower the efficiency of tokenization while `utf-8` encoding can use 1-4 bytes to encode a character, which makes the data representation more efficient.
2. it's wrong because not all the characters are encoded by 1 byte, and also not all the reprensentation of one byte has its corresponding character under `utf-8` encoding. one example: 
```python
def wrongly_encode(bs):
    return "".join([bytes([b]).decode("utf-8") for b in bs])
wrongly_encode("你好".encode("utf-8"))
```
results:
```
Traceback (most recent call last):
  File "/Users/rizzohou/.vscode/extensions/ms-python.python-2026.0.0-darwin-arm64/python_files/python_server.py", line 134, in exec_user_input
    retval = callable_(user_input, user_globals)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 1, in <module>
  File "<string>", line 2, in wrongly_encode
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
```
3. example:
```python
b'\xff\xff'.decode("utf-8")
```
results
```
Traceback (most recent call last):
  File "/Users/rizzohou/.vscode/extensions/ms-python.python-2026.0.0-darwin-arm64/python_files/python_server.py", line 134, in exec_user_input
    retval = callable_(user_input, user_globals)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 1, in <module>
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
```
