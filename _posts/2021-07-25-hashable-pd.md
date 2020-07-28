---
layout: post
title: Hashing Pandas DataFrame
---

I spent sometimes today solving an interesting problem on hashing Pandas DataFrame so I decided to share this in case anyone is running into similar problems.

### Problem settings: Memoization with `@functools.lru_cache`
I have a function that performs data preprocessing on the content of an input Pandas DataFrame. This function is expensive, yet it needs to be called multiple times in the pipeline on the same DataFrame. So to improve efficiency, I used Python's native @functools.lru_cache decorator. The pseudo-code is below.
```python
import functools
import pandas as pd

@functools.lru_cache(max_size=None)
def preprocess(df: pd.DataFrame, **kwargs):
    df = df.copy()
    return df.values
```

However, since `lru_cache` use a dictionary in the backend to maintain the cache, the following error is returned in the Traceback.
```
TypeError: 'DataFrame' objects are mutable, thus they cannot be hashed
```

### Add `__hash__` to DataFrame
After some digging, I found out that the root of this problem is due to DataFrame not containing a `__hash__` method. A few more Stack Overflow questions and this is what I came up with.
```python
import pandas as pd

pd.DataFrame.__hash__ = lambda self: int(
    hashlib.sha256(pd.util.hash_pandas_object(self, index=True).values).hexdigest(),
    base=16,
)
```

### Testing
Now that we have a way to hash Pandas DataFrame, let's make sure things are working as expected. Specifically two main conditions need to be satisfied:
- `hash(df_1) == hash(df_1.copy())`
- `hash(df_1) != hash(df_2)`

```python
import pandas as pd

pd.DataFrame.__hash__ = lambda self: int(
    hashlib.sha256(pd.util.hash_pandas_object(self, index=True).values).hexdigest(),
    base=16,
)

df_1 = pd.DataFrame({"a": [0,1,2], "b": [3,4,5]})
df_2 = df_1.copy()
assert hash(df_1) == hash(df_2)

df_2["b"] = [3,4,6]
assert hash(df_1) != hash(df_2)
```

As expected!