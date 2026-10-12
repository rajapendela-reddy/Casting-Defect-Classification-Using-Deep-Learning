# Uploading the complete repository

Suggested repository name: **casting-defect-classification**.

Suggested description:

> PyTorch casting quality classification with a custom CNN, ResNet18, Grad-CAM, validation-based threshold selection, and verified Kaggle artifacts.

Suggested topics: `computer-vision`, `deep-learning`, `pytorch`, `resnet18`, `grad-cam`, `quality-inspection`, `image-classification`, `kaggle`.

## Recommended: Git or GitHub Desktop

1. Extract `Casting_Defect_Classification_GitHub.zip`.
2. Create an empty GitHub repository with the suggested name. Leave the automatic initialization options unchecked; the README and gitignore are already present.
3. Add the extracted **Casting_Defect_Classification folder's contents** so `README.md` sits at the repository root.
4. Commit and push using Git or GitHub Desktop. Verify that the README figures render and `.github/workflows/checks.yml` is present.

With Git, run from the extracted folder:

```bash
git init
git add .
git commit -m "Add casting defect classification workflow and original Kaggle results"
git branch -M main
```

Copy the exact remote URL shown by your new GitHub repository, add it using `git remote add origin`, then run `git push -u origin main`.

## Why browser-only upload is insufficient

The included `best_resnet18.pt` is **44,791,115 bytes**, approximately **42.72 MiB**. GitHub browser uploads are limited to **25 MiB per file**. Git pushes accept individual files up to **100 MiB**, so the original checkpoint can be committed through Git without splitting it.

Official guidance: [Adding files to a repository](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository) and [About large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

The current folder does not contain a Git remote or `.git` directory, and no external repository has been created automatically.

## Verification

```bash
python -m casting_src.verify --check-checkpoints
python -m pytest -q
```

Keep the original artifact directory and project-source acknowledgement. Generated runs, prediction outputs, local environments, and downloaded image data are intentionally ignored by Git. Review any new run before explicitly adding it.
