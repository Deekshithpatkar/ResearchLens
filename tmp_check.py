import importlib.util
import traceback
print('spec', importlib.util.find_spec('sentence_transformers'))
try:
    import sentence_transformers
    print('import ok', sentence_transformers.__version__)
except Exception as e:
    print('import failed')
    traceback.print_exc()
