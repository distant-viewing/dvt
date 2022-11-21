# Development of the Distant Viewing Toolkit

To test the toolkit, use:

```
pytest --disable-warnings .
```

To send the PyPI, use (needs password):

```
python setup.py sdist
twine upload dist/*
```
