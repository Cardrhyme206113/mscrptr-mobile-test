from pathlib import Path

p = Path("app/src/main/java/com/muscript/pocket/MainActivity.java")
s = p.read_text()
old = "        super.onCreate(state);\n        enterImmersive();\n        modelName="
new = "        super.onCreate(state);\n        modelName="
assert s.count(old) == 1, "Expected early immersive call missing or changed"
s = s.replace(old, new)
old = "        setContentView(web);\n        web.loadUrl("
new = "        setContentView(web);\n        enterImmersive();\n        web.loadUrl("
assert s.count(old) == 1, "Expected content view call missing or changed"
s = s.replace(old, new)
old = "WindowInsetsController c=w.getInsetsController();"
new = "WindowInsetsController c=w.getDecorView().getWindowInsetsController();"
assert s.count(old) == 1, "Expected controller access missing or changed"
s = s.replace(old, new)
p.write_text(s)
assert s.index("setContentView(web);") < s.index("enterImmersive();")
print("Fixed lifecycle ordering and null-decor startup crash")
