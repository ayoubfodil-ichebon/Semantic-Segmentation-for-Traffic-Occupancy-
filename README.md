# CamVid Segmentation – UNet · VMamba‑UNet · SegNet · FCN

Comparaison de 4 architectures de segmentation sémantique sur le dataset **CamVid** (32 classes), avec une application à l’**estimation du taux d’occupation du trafic** (véhicules, piétons, 2‑roues…).

---

## 📦 Contenu du dépôt

- `src/` – code modulaire (dataset, modèles, métriques, entraînement, évaluation)
- `scripts/` – lanceurs pour entraîner et évaluer tous les modèles
- `config.yaml` – hyperparamètres centralisés
- `outputs/` – poids sauvegardés, courbes et figures (générés automatiquement)

---

## 🚀 Installation et exécution

```bash
git clone https://github.com/votre-nom/camvid_segmentation.git
cd camvid_segmentation
pip install -r requirements.txt
