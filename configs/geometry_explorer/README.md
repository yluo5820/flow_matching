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
