# THA4 Model Setup Guide

## Required Files Structure

Place your THA4 model files in `data/models/tha4/` with the following structure:

```
data/models/tha4/
├── character_model.yaml    # Configuration file (paths to other files)
├── character.png           # Character source image (RGBA, 512x512)
├── face_morpher.pt         # Face morphing model weights
└── body_morpher.pt         # Body morphing model weights
```

## Example character_model.yaml

Create a file named `character_model.yaml` with the following content:

```yaml
character_image_file_name: character.png
face_morpher_file_name: face_morpher.pt
body_morpher_file_name: body_morpher.pt
```

**Note:** All paths in the YAML are relative to the directory containing the YAML file.

## Alternative Structure

You can also organize files in subdirectories:

```
data/models/tha4/
├── my_character/
│   ├── character_model.yaml
│   ├── image.png
│   ├── face_morpher.pt
│   └── body_morpher.pt
```

Then update your YAML accordingly:

```yaml
character_image_file_name: image.png
face_morpher_file_name: face_morpher.pt
body_morpher_file_name: body_morpher.pt
```

## Usage in EasyVtuber

1. Place your files as shown above
2. Launch the EasyVtuber launcher
3. Select "THA4" from the "Model Select" dropdown
4. The system will automatically:
   - Load `character_model.yaml`
   - Read the PNG character image
   - Load both PT model files
   - Use the PNG image as the source for animation

## Differences from THA3

- **THA4**: Uses YAML to define character (image + models as a package)
- **THA3**: Uses separate image file selected by user + shared model files

The THA4 approach bundles character and model together, making it easier to 
switch between different characters with their trained models.
