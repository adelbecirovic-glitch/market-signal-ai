# Browser-Version veröffentlichen

Diese Version ist für Streamlit Community Cloud vorbereitet.
Nach der Veröffentlichung brauchst du auf deinem PC oder Handy kein Python mehr.

## 1. Kostenloses GitHub-Konto
Erstelle ein GitHub-Konto, falls du noch keines hast.

## 2. Neues Repository
Erstelle ein neues Repository, z. B. `market-signal-ai`.

## 3. Dateien hochladen
Lade den INHALT dieses Ordners in das Repository:
- app.py
- requirements.txt
- runtime.txt
- .streamlit/config.toml

## 4. Streamlit Community Cloud
Öffne Streamlit Community Cloud und melde dich mit GitHub an.
Wähle `Create app`, anschließend dein Repository und als Main file `app.py`.

## 5. Deploy
Klicke auf Deploy. Nach dem Build erhältst du eine Webadresse.
Diese kannst du auf Windows, Android, iPhone/iPad oder jedem anderen Gerät im Browser öffnen.

## Hinweise
- Marktdaten werden über yfinance geladen.
- Kostenlose Datenfeeds können verzögert oder zeitweise nicht verfügbar sein.
- Die Signale sind ein Analyse-Prototyp und keine Anlageberatung.
- Vor echtem Trading sollten Backtesting, Gebühren, Slippage und Paper Trading ergänzt werden.
