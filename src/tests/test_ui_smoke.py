import os
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("../ui/app.py", default_timeout=120).run()
print("exceptions:", [e.value for e in at.exception]); print("tabs:", [t.label for t in at.tabs])
print("errors:", [e.value for e in at.error]); print("warn:", [e.value for e in at.warning])
btn = [b for b in at.button if b.label == "Search"]
print("search buttons:", len(btn))
if btn:
    btn[0].click().run(); print("after search -> success:", [s.value for s in at.success], "warning:", [w.value for w in at.warning], "exc:", [e.value for e in at.exception])
