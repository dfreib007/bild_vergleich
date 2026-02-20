# SSIM Image Compare App (Streamlit)

Kleine POC-App zum Vergleich von zwei Bildern mit **OpenCV** und **SSIM** (scikit-image).
Für Cloud-Deployment wird `opencv-python-headless` verwendet.

## Features
- Modusauswahl: **Single Vergleich**, **Batch Modus**, **Interaktion Modus** oder **Übersicht**
- Upload von Referenzbild (Golden) und Testbild (PNG/JPG)
- Optionales Alignment per ECC (`cv2.findTransformECC`) oder ORB + Homography
- SSIM-Score (0..1)
- Diff-Map, Threshold-Maske, Overlay-Visualisierung
- Filterung kleiner Artefakte über Mindestfläche
- Optionale Bounding Boxes und Heatmap-Overlay
- Download von Overlay/Mask/Diff als PNG, Kennzahlen als CSV und Vergleichsreport als PDF
- Batch-Vergleich von zwei Ordnern mit sequenzieller Auswertung und CSV-Export
- Interaktion-Modus mit beidseitigem Vergleich (A→B und B→A) und User-Entscheidung
- Login/Authentifizierung mit Benutzerverwaltung (Admin + User) in SQLite
- Übersichtsmodus mit Grid aller Interaktionsergebnisse inkl. Bildspalten

## Voraussetzungen
- Python **3.9+**
- `pip`

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Start
```bash
streamlit run app.py
```

## Modi
- **Single Vergleich**:
  - Bestehender Upload-Workflow für ein Referenz- und ein Testbild inkl. Visuals und Downloads.
- **Batch Modus**:
  - Zwei lokale Ordnerpfade angeben (Referenz/Test).
  - Optional rekursive Suche in Unterordnern.
  - Dateien werden über identischen Dateinamen (nicht rekursiv) bzw. identischen relativen Pfad (rekursiv) gepaart.
  - Alle Paare werden nacheinander verglichen; Ergebnisliste + Batch-CSV stehen bereit.
  - Zusätzliche Vorschau für ein ausgewähltes Paar.
- **Interaktion Modus**:
  - Upload von Bild A und Bild B.
  - Vergleich in beide Richtungen (`A → B` und `B → A`) inkl. Visualisierung der Unterschiede.
  - User wählt anschließend das korrekte Bild (`Bild A` oder `Bild B`) aus.
  - Ergebnis wird in einer SQLite-Datenbank gespeichert.
- **Übersicht**:
  - Zeigt alle Ergebnisse aus dem Interaktion-Modus in einem Grid:
    - `ID`
    - `Bild A`
    - `Bild B`
    - `Difference A to B`
    - `Ausgewähltes Bild (als Bild)`
    - `Buchstabe des Ausgewählten Bildes`
    - `Timestamp`
  - Filter und Sortierung sind direkt im Tabellen-Header integriert (Excel-ähnlich, Floating Filter).
- **Admin** (nur für Admin-Benutzer sichtbar):
  - Benutzerliste
  - Benutzer hinzufügen (E-Mail + Passwort)
  - Passwort ändern
  - Benutzer aktiv/deaktivieren

## Parameter erklärt
- **Bilder automatisch ausrichten (Alignment)**:
  - Versucht das Testbild auf das Referenzbild zu registrieren.
  - **ECC (Affine)**: robust bei kleinen Translationen/Rotationen/Skalierung.
  - **ORB (Homography)**: feature-basiert, oft besser bei perspektivischen Unterschieden.
  - Hilfreich bei kleinem Shift, leichter Rotation oder minimaler Perspektivabweichung.
  - Bei Fehlschlag erfolgt ein Fallback ohne Alignment inkl. Warnung in der UI.

- **Abweichungsschwelle (0–255)**:
  - Arbeitet auf der invertierten SSIM-Diff-Map (hohe Werte = stärkere Abweichung).
  - Höherer Wert = strenger, weniger markierte Pixel.
  - Niedriger Wert = empfindlicher, mehr markierte Pixel.

- **Min. Defektfläche (px)**:
  - Unterdrückt kleine Regionen (Rauschen/Artefakte) anhand der Konturfläche.

- **Bounding Boxes anzeigen**:
  - Zeichnet Rechtecke um gefilterte Abweichungsregionen.

- **Heatmap Overlay anzeigen**:
  - Legt eine Farbkarte der Defektmaske über das Testbild.

- **Downloads**:
  - Exportiert `overlay.png`, `mask.png`, `ssim_diff.png`, `metrics.csv` und `vergleich_report.pdf`.
  - Der PDF-Report enthält Kennzahlen und eine kompakte Vergleichsübersicht (Referenz, Test, Diff, Maske, Overlay).

- **Batch CSV**:
  - Enthält pro Bildpaar Pfade, Status, SSIM, Regionsanzahl, Abweichungsfläche und Alignment-Status.

## Datenbank (Interaktion Modus)
- Datei: `interaktion_results.db` (im Projektverzeichnis)
- Tabellen: `interaction_results`, `users`
- Gespeicherte Felder:
  - Zeitstempel (`created_at`)
  - Referenzbild A (`reference_a_png`)
  - Vergleichsbild B (`comparison_b_png`)
  - Unterschied A→B (`difference_a_to_b_png`)
  - Unterschied B→A (`difference_b_to_a_png`)
  - Vom User gewähltes korrektes Bild (`selected_image`)
  - Benutzer, der ausgewählt hat (`selected_by`)

## Login / Bootstrap-Admin
- Beim ersten Start wird automatisch ein Admin-Benutzer angelegt.
- Default:
  - E-Mail: `admin@bildvergleich.local`
  - Passwort: `Admin1234!`
- Für Deployment solltest du die Bootstrap-Werte per Umgebungsvariablen setzen:
  - `APP_BOOTSTRAP_ADMIN_EMAIL`
  - `APP_BOOTSTRAP_ADMIN_PASSWORD`

## Größen- und Seitenverhältnis-Strategie
- Die App verwendet das Referenzbild als geometrische Basis.
- Wenn Größen unterschiedlich sind, wird das Testbild auf die Referenzgröße **resized**.
- Bei unterschiedlichem Seitenverhältnis warnt die UI, da Resize zu Verzerrungen führen kann.
- Für präzisere Ergebnisse sollten beide Bilder idealerweise mit gleicher Kamera-Geometrie aufgenommen werden.

## SSIM-Pipeline (kurz)
1. Upload und Decode in BGR
2. Testbild auf Referenzgröße bringen
3. Optionales Alignment (ECC oder ORB-Homography)
4. Graustufen + SSIM (`score, diff`)
5. `diff` von `[0..1]` nach `uint8 [0..255]`
6. Invertieren der Diff-Map für intuitives Thresholding
7. Threshold + Morphology (Open/Close)
8. Konturerkennung + Flächenfilter
9. Metriken + Overlay darstellen

## Hinweise zur Robustheit
- SSIM reagiert empfindlich auf Beleuchtungsänderungen.
- Alignment kann lokale Shifts kompensieren, hilft aber nicht bei starken nichtlinearen Verformungen.
- Für produktive Nutzung sind stabile Aufnahmebedingungen (Licht, Fokus, Perspektive) entscheidend.

## Wie erweitern?
- **Batch-Vergleich in Ordnern**: CLI/Streamlit-Workflow für Serienprüfung ganzer Bildsätze.
- **ROI-Masken**: Nur relevante Bildbereiche prüfen und störende Zonen ausblenden.
- **Reporting (CSV/JSON)**: Ergebnisse pro Bildpaar persistieren (Score, Fläche, Regionen, Zeitstempel).
- **Toleranzzonen**: Unterschiedliche Schwellwerte/Freigaben je Bildregion (z. B. kritische vs. unkritische Bereiche).
