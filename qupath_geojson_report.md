# QuPath annotations & QuPath‑compatible GeoJSON
*(Report generated 2026‑01‑07)*

## Executive summary
- **Yes, you can generate annotations externally (e.g., in Python) and import them into QuPath** as **GeoJSON** via *File → Import objects from file*. QuPath also exports objects as GeoJSON, making a reliable round‑trip workflow possible.
- **Coordinates must be in pixel units** with **origin (0,0) at the top‑left of the full‑resolution image**.
- The most robust interchange format is a **GeoJSON FeatureCollection** whose **features** each include a valid GeoJSON `geometry` plus QuPath metadata in `properties` (notably `objectType`, and optionally `classification`, `measurements`, etc.).
- **Avoid NaN/Infinity** in `measurements` (QuPath has had a documented failure mode when importing GeoJSON containing `NaN` values).

## What QuPath means by “annotations”
QuPath represents drawn/derived objects (annotations, detections, etc.) as objects in an internal hierarchy (commonly referred to as *PathObjects*). GeoJSON export/import is effectively a way to serialize/deserialize these objects and their ROIs, classifications, and measurements.

## Evidence that QuPath imports/exports GeoJSON (GUI + scripting)
- QuPath’s command reference includes **Import objects from file** (“Import objects from GeoJSON or .qpdata files”) and **Export objects as GeoJSON**.
- QuPath’s “Exporting annotations” documentation describes GeoJSON export and explicitly notes that QuPath uses a **FeatureCollection** to preserve additional properties (e.g., classifications).

## Coordinate system requirements (critical)
QuPath’s shape export methods define coordinates:
- **in pixel units**
- with **origin (0,0) at the top-left**
- for the **full-resolution image**

Implication for Python pipelines:
- If you generated polygons on a **downsampled** representation (thumbnail/mask), **scale coordinates back to full‑resolution** before import.
- If you cropped/tilled, ensure you’ve accounted for any offsets relative to the full image origin.

## GeoJSON “container” structures QuPath writes (and is most likely to read)
QuPath’s scripting API explicitly supports exporting:
- a standard GeoJSON **FeatureCollection** (recommended for multiple objects)
- or a “simple JSON object/array” (legacy / compact form)

**Recommendation:** Always emit a **FeatureCollection** unless you have a strong reason not to.

## Per‑object (Feature) format: required vs recommended

### Required (GeoJSON)
Each object should be a GeoJSON Feature with:
- `type`: `"Feature"`
- `geometry`: a valid GeoJSON geometry (`Polygon`, `MultiPolygon`, `Point`, etc.)
- `properties`: a JSON object (can be empty, but QuPath-specific metadata goes here)

### Strongly recommended (QuPath compatibility & display)
In `properties`, include at least:
- `objectType`: commonly `"annotation"` (for editable annotation ROIs) or `"detection"` (for detection objects)

**Why this matters:** QuPath examples and bug reports show `objectType` being used to interpret objects during (de)serialization.

## QuPath-specific `properties` fields you’ll see in the wild

### `objectType`
Typical values:
- `"annotation"`
- `"detection"`
Other types exist in QuPath internally, but these are the most common for interchange.

### `classification`
Classification often appears as an object, e.g.
- `{"name": "Tumor", "colorRGB": -8245601}` (packed integer color)
Some exports also appear to use an RGB array form in community examples; the safest approach is to **round‑trip one exported file from your target QuPath version** and mirror exactly what it produces.

### `isLocked`
May appear for annotations (`false` / `true`). Usually safe to omit unless you specifically need it.

### `measurements`
Measurements can serialize in different ways depending on QuPath version and exporter:
- as an **empty list** when none are present (example in older QuPath exports)
- as an **object/map** `{ "Measurement name": 1.23, ... }`

**Important:** Do **not** include `NaN` values. QuPath has had a documented issue where GeoJSON containing `NaN` in `measurements` could not be imported.

### `id`
IDs vary across versions/contexts:
- a UUID-like string (often used in newer examples)
- a class-like identifier string (e.g., `"PathAnnotationObject"` in older exports)

**Best practice:** assign a UUID string per feature, but do not assume QuPath requires UUIDs.

## Geometry details & gotchas
### Polygons
- Use `Polygon` for single regions.
- Coordinate format is:
  - `coordinates: [ [ [x,y], [x,y], ... ] ]`
  - Outer ring first; additional rings represent holes (standard GeoJSON).
- **Close rings**: repeat the first point as the last point (recommended; QuPath exports commonly do this).

### MultiPolygons
Use `MultiPolygon` if you have disjoint regions for one logical object.

### Non-polygon shapes
QuPath exports ellipses as polygons (GeoJSON limitation noted in QuPath docs). If you need “true ellipse” semantics, you won’t get it through GeoJSON—approximate it with sufficient vertices.

## Minimal “usually works” QuPath annotation template
Use this as a starting point, then adapt to match your QuPath version’s exported schema (especially for color encoding).

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": "d0852662-6941-4506-bc90-cbda1c2fa7b0",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [44196, 21480],
            [44420, 21480],
            [44420, 21687],
            [44196, 21687],
            [44196, 21480]
          ]
        ]
      },
      "properties": {
        "objectType": "annotation",
        "classification": { "name": "Tumor", "colorRGB": -8245601 },
        "measurements": {}
      }
    }
  ]
}
```

Notes:
- If you don’t need measurements, omit `measurements` entirely.
- If you include measurements, ensure values are finite numbers (no NaN/Infinity).

## Recommended validation workflow (fastest path to “exact format spec”)
Because QuPath’s serialization details can vary across versions and exporters, the most reliable approach is:

1. In your target QuPath version, draw a simple annotation (e.g., a rectangle) and set its class.
2. Export it using **File → Export objects as GeoJSON**.
3. Treat that file as the “ground truth schema”:
   - Does it use `colorRGB` (packed int) or `color: [r,g,b]`?
   - Does it include `objectType`?
   - How are `measurements` represented?
4. Generate your Python GeoJSON to match those exact fields and conventions.
5. Import using **File → Import objects from file** and confirm geometry, class name, and color.

## Common failure modes (checklist)
- ❌ Wrong coordinate reference (downsampled coords, crop offsets not applied)
- ❌ Y-axis flipped (some libraries treat origin bottom-left; QuPath is top-left)
- ❌ Invalid GeoJSON polygon structure (missing nesting, rings not closed)
- ❌ `NaN` values in `measurements`
- ❌ Classification color encoding mismatch for your QuPath version (`colorRGB` vs RGB array)

## What I did *not* find as a stable “spec” (so you should round‑trip)
- A universally documented, version‑agnostic schema for encoding **z-slice/timepoint** information inside GeoJSON features. QuPath does provide UI commands to paste GeoJSON objects to the current plane, but the file-level convention isn’t clearly standardized in the public docs.

## Sources consulted (high signal)
- QuPath docs: command reference (Import objects from file; Export objects as GeoJSON)
- QuPath docs: exporting annotations (GeoJSON, coordinate system)
- QuPath GitHub issues: examples of GeoJSON features, `classification.colorRGB`, and the `NaN` measurements import failure
