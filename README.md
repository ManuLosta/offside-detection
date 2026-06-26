# Offside Detection

Pipeline de visión artificial para detectar jugadores, clasificarlos por equipo,
estimar poses, calcular puntos de fuga de la cancha y marcar posibles jugadores
en offside.

## Uso

Ejecutar el pipeline completo:

```bash
uv run offside-detect data/input/match.jpg
```

Ejecutar solo la etapa final de offside usando salidas previas:

```bash
uv run offside-detect data/input/match.jpg --offside \
  --attacking-side right \
  --attacking-team-id 0 \
  --defending-team-id 1
```

La etapa de offside guarda:

- `data/output/offside/<imagen>_offside.json`
- `data/output/offside/<imagen>_offside.jpg`
