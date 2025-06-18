# Plan: Structured Logging Library

## Ziel
Umwandlung des structured logging Codes in eine wiederverwendbare Python Library mit flexibler Konfiguration.

## Hauptverbesserungen

### 1. Dependency Entkopplung
- Entfernung der hardcoded `app.config.settings` Abhängigkeit
- Einführung von Standardwerten und optionalen Konfigurationsparametern
- LOG_LEVEL als Parameter mit Fallback auf 'INFO'

### 2. Flexible Context-Felder
- `tenant_id` und `user_id` als vollständig optionale Felder behandeln
- Generisches Context-System für beliebige Felder
- Automatisches Filtern von None/leeren Werten

### 3. Package-Struktur
```
structured-logging/
├── src/structured_logging/
│   ├── __init__.py          # Public API
│   ├── formatter.py         # StructuredFormatter
│   ├── logger.py           # get_logger, log_with_context  
│   ├── context.py          # Context management
│   └── config.py           # Configuration handling
├── tests/
├── pyproject.toml          # Modern Python packaging
└── README.md              # Usage documentation
```

### 4. Konfiguration
- Flexible LoggerConfig Klasse
- Environment variable support
- Sinnvolle Defaults für alle Optionen

### 5. Distribution
- pyproject.toml für pip installation
- Semantic versioning
- Proper metadata und dependencies

### 6. Qualitätssicherung
- Unit tests für alle Komponenten
- Type hints durchgängig
- Docstrings für Public API
- Usage examples in README

## Implementierungsreihenfolge
1. Package-Struktur erstellen
2. Dependencies entkoppeln
3. Code in Module aufteilen
4. Konfigurationssystem implementieren
5. Tests schreiben
6. Dokumentation erstellen
7. Distribution setup

## Aktuelle Probleme im Code

### Hardcoded Dependencies
```python
from app.config import settings
logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
```
**Lösung:** Konfiguration über Parameter oder Environment Variables

### Inflexible Context-Felder
```python
context = {
    "ctx_request_id": request_id.get(""),
    "ctx_user_id": user_ctx.get("user_id"),
    "ctx_tenant_id": user_ctx.get("tenant_id"),
}
```
**Lösung:** Generisches Context-System mit optionalen Standard-Feldern

### Fehlende Package-Struktur
Einzelne Datei `logging.py` ohne Installationsmöglichkeit
**Lösung:** Proper Python Package mit setup.py/pyproject.toml

## Vorteile nach Refactoring

1. **Wiederverwendbarkeit:** Einfache Installation via pip
2. **Flexibilität:** Konfigurierbare Context-Felder 
3. **Wartbarkeit:** Modulare Struktur
4. **Testbarkeit:** Unit Tests für alle Komponenten
5. **Dokumentation:** Klare API und Usage-Beispiele