["Niagara Falls", "Golden Gate Bridge", "Eiffel Tower","Grand Canyon","Lake Como","Masada","Edinburgh Castle","Victoria Memorial, Kolkata","Faisal Mosque", "Jurassic Coast"]
die rausgefiltert, nach diesen kriterien: klar visuell erkennbar (ikonische Struktur)
kein Metadaten-Artefakt (z. B. ETH-Bibliothek ❌)
kein zu allgemeiner Ort (ganze Städte/Regionen ❌)


Phase A — Baseline robustness (ohne robust training)
Train: clean
Test: clean + jede corruption in 2–3 Stärken
→ du bekommst eine Tabelle/Plot: Accuracy vs severity
Phase B — Robust training
Train: mix (z.B. 50% clean / 50% “lazy augmentations”)
Test wieder: clean + corruptions
→ Ziel: clean bleibt ~gleich, corruption-drop wird kleiner