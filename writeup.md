problem unicode1
---
1. '\x00'
2. its `repr` result is a string consisting of exactly 6 characters ("'\x00'"), while its true value is a character whose unicode encoding is 0 ('\x00').
3. this character seems to represent nothing when occuring in text.

