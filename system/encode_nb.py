import base64, json
import os
nb = json.load(open(os.path.join(os.path.dirname(__file__), "plot_results.ipynb")))
encoded = base64.b64encode(json.dumps(nb).encode()).decode()
print(encoded)
