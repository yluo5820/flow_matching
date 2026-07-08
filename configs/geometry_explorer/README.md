# Geometry Explorer Config Layout

Configs mirror the explorer selection hierarchy:

```text
configs/geometry_explorer/
  views/
    raw_pixels.yaml
  datasets/
    <family>/
      <variant>/
        dataset.yaml
        models/
          image_unet_ot.yaml
          image_unet_xpred.yaml
          ...
```

The matching runtime artifacts are written under:

```text
outputs/geometry_explorer/
  datasets/<family>/<variant>/
  model_runs/<family>/<variant>/<run_id>/
```

`fm-lab-explorer build-all` recursively discovers `dataset.yaml` files under
`configs/geometry_explorer/datasets` and applies the shared view config
`configs/geometry_explorer/views/raw_pixels.yaml`.

Synthetic object pose sweeps use the same registry layout:

```bash
fm-lab-explorer make-synthetic-object \
  --config configs/geometry_explorer/datasets/synthetic_object/cube_pose_360_dense/dataset.yaml

fm-lab-explorer build-view \
  --dataset synthetic_object/cube_pose_360_dense \
  --config configs/geometry_explorer/views/raw_pixels.yaml
```

Uniform look-at camera samples on the sphere use the same commands, for example:

```bash
fm-lab-explorer make-synthetic-object \
  --config configs/geometry_explorer/datasets/synthetic_object/cube_sphere_10k/dataset.yaml

fm-lab-explorer build-view \
  --dataset synthetic_object/cube_sphere_10k \
  --config configs/geometry_explorer/views/raw_pixels.yaml
```

Translation controls are generated for the abstract statue:

- `abstract_statue_translation_xy_10k`: fixed depth, camera-plane x/y shift.
- `abstract_statue_translation_z_1k`: centered depth sweep. This stays at 1k
  because 10k depth-only samples at 64x64 create many duplicate rasterized
  images and destabilize local ID estimators.
- `abstract_statue_translation_xyz_10k`: joint camera-plane x/y and depth shift.

Synthetic object metadata includes pose, camera, light, and background fields
for hover inspection and downstream analysis.
