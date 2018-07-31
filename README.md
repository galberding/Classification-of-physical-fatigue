# Klassifikation von Erschöpfung

Diese Repo enthält den Quellcode den ich im Laufe meiner Bachelorarbeit erstellt habe.

## Skripts

Hier sind alle Programme gesammelt. Wichtig sind vor allem filemapper.py und stats.py
filemapper kann benutzt werden um Videos(die hier aus Datenschutzgründen nicht vorhanden sind) zu schneiden und Spikes aus
den Herzdaten detektieren.

stats bietet die möglichkeit alle vorhandenen extrahierten Action Units
zu bearbeiten bzw. eine SVC zu trainieren.

## Database

Hier sind alle Bilder und Datensätze zu finden:
* AUs_by_classes enthalten die Werte der Action Units in jeder Klasse.
* heartrate_with_timestamps beinhaltet, (wer hätte das gedacht) die Herzdaten
* open_face_out beinhaltet die Action Units die von OpenFace aus den Action Units extrahiert wurden
* open_face_out_trial2 beinhaltet die Action Units die den Normalzustand der jeweiligen Person darstellen sollen
Wenn Zugriff auf das Techfaknetzwerk besteht, sind die Daten geordnet unter /vol/prt/analysis/ zu finden.

## Ausführen

Um die Programme auszuführen ist environment.py sehr wichtig.
Diese Programm erwartet, das das System eine globale Variable mit dem Namen "BA_PATH" bereitstellt. Diese muss den Pfad zum Ordner Klassifizierung_von_Erschoepfung beinhalten.

Anschließend sollte die Ausführung möglich sein.
Diese setzt jedoch folgendes vorraus:
* Python 3 mit:
    * skleran
    * pandas
    * cv2
    * matplotlib
    * multiprocessing
    * numpy
    * statsmodel
    * scipy

* OpenFace (https://github.com/TadasBaltrusaitis/OpenFace)
    * wird nur benötigt falls AUs aus videos extrahiert werden sollen
