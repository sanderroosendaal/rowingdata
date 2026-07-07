# Rowing Data Standard for Garmin FIT Format

**Version:** 1.1  
**Date:** May 14, 2026  
**Application ID:** `89e86158-6d47-5c98-9d46-7d29437f27b9` (UUID v5 from DNS:rowingdata)

## 1. Introduction

### 1.1 Purpose

This document defines a standard for encoding rowing-specific data within the Garmin FIT (Flexible and Interoperable Data Transfer) format. The standard enables interoperability between rowing devices, software platforms, and data analysis tools through consistent use of FIT developer fields.

### 1.2 Scope

This standard specifies:

- Developer field definitions for rowing-specific metrics
- Recording strategies for different device types
- Requirements for data consumers
- In-stroke curve data encoding
- Native FIT field usage for rowing

### 1.3 Target Audience

- Rowing device manufacturers (ergometers, smart oarlocks, GPS devices)
- Software developers (training platforms, analysis tools)
- Data platform operators

### 1.4 Application ID

All developer fields defined in this standard MUST use the **rowingdata application ID** to prevent namespace collisions.

**Format:** 16-byte UUID array as required by FIT SDK `DeveloperDataIdMessage`

**UUID:** `89e86158-6d47-5c98-9d46-7d29437f27b9`

This UUID v5 is deterministically generated from DNS namespace with name "rowingdata" (`uuid.uuid5(uuid.NAMESPACE_DNS, 'rowingdata')`), ensuring consistency across implementations. The FIT SDK requires the application_id field to be a 16-byte array representation of this UUID.

### 1.5 Conformance Language

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in RFC 2119.

## 2. Recording Strategies

### 2.1 Overview

Different device types generate FIT Record messages at different frequencies. Consumers MUST support both approaches:

### 2.2 Stroke-Boundary Recording

**Definition:** One Record message per completed stroke cycle.

**Characteristics:**
- Each Record represents a complete stroke
- Stroke-specific metrics (drive time, force, angles) describe that specific stroke
- GPS position MAY be interpolated or repeated across strokes
- REQUIRED when in-stroke curve data is present

**Typical devices:** Indoor ergometers, instrumented oarlocks, on-water measurement systems with stroke detection, rowing-specific data loggers

### 2.3 GPS-Update Recording

**Definition:** Record messages generated at GPS position updates.

**Characteristics:**
- Records generated when GPS data is available (typically ~1 Hz, but irregular)
- Stroke metrics MAY be averaged, interpolated, or represent the most recent stroke
- GPS position is precisely measured at record time
- CANNOT include in-stroke curve data (curves require stroke boundaries)
- `total_cycles` field MAY repeat across multiple records (same stroke) or skip values (multiple strokes between GPS updates)

**Typical devices:** GPS-enabled sports watches, multi-sport fitness devices, smartphone applications with GPS tracking

### 2.4 RecordingStrategy Metadata Field

To indicate the recording approach, producers MAY include the **RecordingStrategy** developer field (ID 10, UINT8) on the **Session message**.

| Value | Name | Description |
|-------|------|-------------|
| 0 | Unknown | Recording strategy unspecified (consumers MUST handle both approaches) |
| 1 | StrokeBoundary | One Record per stroke cycle |
| 2 | GPSUpdate | Records at GPS position updates |

**Message Type:** Session (one value per file)

**Base Type:** UINT8  
**Scale:** 1  
**Units:** (none)

When omitted or zero, consumers MUST NOT assume any particular strategy.

## 3. Consumer Requirements

### 3.1 Mandatory Requirements

Consumers MUST:

1. **Support both recording strategies** without requiring configuration
2. **Not assume 1:1 correspondence** between Record messages and strokes
3. **Not interpolate stroke-specific developer fields** between records (DriveLength, StrokeDriveTime, Catch, Finish, oarlock angles, etc.) - these describe discrete stroke events
4. **Detect stroke occurrences** by monitoring changes in the native `total_cycles` field:
   - When `total_cycles` changes between consecutive records, at least one stroke occurred
   - If change is >1, multiple strokes occurred but per-stroke data for intermediate strokes is unavailable
5. **Calculate stroke rate** from the native `cadence` field (strokes/min), not from record message frequency
6. **Handle missing developer fields gracefully** (all developer fields are optional)

### 3.2 Recommended Requirements

Consumers SHOULD:

1. Read `RecordingStrategy` from Session message when present for optimization
2. Validate that in-stroke curve data only appears with `RecordingStrategy=StrokeBoundary` (or Unknown)

### 3.3 Optional Behaviors

Consumers MAY:

1. Optimize parsing based on `RecordingStrategy` value (e.g., skip stroke detection in StrokeBoundary files)
2. Issue warnings when encountering unexpected combinations (e.g., GPS-update with in-stroke data)

## 4. Native FIT Field Mappings

Producers SHOULD use these native FIT fields for rowing data:

| FIT Field | Type | Usage | Notes |
|-----------|------|-------|-------|
| timestamp | UINT32 | Record timestamp | Milliseconds since Garmin epoch (1989-12-31 UTC) |
| distance | UINT32 | Cumulative distance | Meters, scale 100 |
| cadence | UINT8 | Stroke rate | Strokes per minute |
| heart_rate | UINT8 | Heart rate | Beats per minute, 0-255 |
| power | UINT16 | Average power | Watts, 0-65535 |
| enhanced_speed | UINT32 | Boat speed | Meters per second (from pace) |
| position_lat | SINT32 | Latitude | Semicircles |
| position_long | SINT32 | Longitude | Semicircles |
| total_cycles | UINT32 | Cumulative stroke count | MAY repeat (GPS-update) or increment by >1 |
| cycle_length16 | UINT16 | Stroke distance | Distance per stroke cycle, scale 100, max 655m |

## 5. Developer Field Specifications

### 5.1 Core Rowing Metrics (Record-Level)

| Field Name | ID | Base Type | Scale | Units | Definition | Typical Range |
|------------|----|-----------| ------|-------|------------|---------------|
| DriveLength | 0 | UINT16 | 100 | m | Distance traveled by handle along longitudinal axis during drive phase | 1.2-1.5 m |
| StrokeDriveTime | 1 | UINT16 | 1 | ms | Duration of drive phase | 300-600 ms |
| DragFactor | 2 | UINT16 | 1 | | Resistance setting (ergometer) | Device-specific |
| StrokeRecoveryTime | 3 | UINT16 | 1 | ms | Duration of recovery phase | 500-1500 ms |
| AverageDriveForceLbs | 4 | UINT16 | 10 | lbs | Average force during drive (deprecated) | - |
| PeakDriveForceLbs | 5 | UINT16 | 10 | lbs | Peak force during drive (deprecated) | - |
| AverageDriveForceN | 6 | UINT16 | 10 | N | Average force during drive phase | 200-600 N |
| PeakDriveForceN | 7 | UINT16 | 10 | N | Peak force during drive phase | 400-1200 N |
| AverageBoatSpeed | 8 | UINT16 | 100 | m/s | Average boat speed during stroke | 3-6 m/s |
| WorkoutState | 9 | UINT8 | 1 | | Rowing state indicator | See WorkoutState values |
| StrokeWork | 19 | UINT16 | 1 | J | Work done over full stroke cycle | 100-500 J |

**Notes:**

- **DriveLength**: For OTW rowing, projection of handle trajectory on longitudinal axis. For indoor, handle travel catch-to-finish.
- **Force fields**: Newtons (IDs 6-7) are RECOMMENDED. Pounds (IDs 4-5) retained for backward compatibility only.
- **StrokeWork**: Energy over complete stroke cycle (not drive-only). Equivalent to average power × stroke period.

### 5.2 Oarlock Metrics (Single, Record-Level)

| Field Name | ID | Base Type | Scale | Units | Definition |
|------------|----|-----------| ------|-------|------------|
| Catch | 11 | SINT16 | 10 | deg | Oar angle at catch (0° = perpendicular to boat) |
| Finish | 12 | SINT16 | 10 | deg | Oar angle at finish (0° = perpendicular to boat) |
| Slip | 13 | SINT16 | 10 | deg | Oar slip angle (early blade entry) |
| Wash | 14 | SINT16 | 10 | deg | Wash angle (late blade exit) |
| PeakForceAngle | 15 | SINT16 | 10 | deg | Oar angle at peak force (0° = perpendicular to boat) |
| EffectiveLength | 16 | UINT16 | 100 | m | Effective oar lever length (rigging) |
| PeakForcePositionNorm | 17 | UINT16 | 1 | | Normalized position of peak force along drive (0-10000) |
| PeakForcePositionAbs | 18 | UINT16 | 100 | m | Absolute handle position at peak force |

**Notes:**

- **Oar angles** (Catch, Finish, PeakForceAngle) use the rowing convention: **0° = oar perpendicular to the boat's longitudinal axis**. Negative values indicate the catch direction (oar blade toward bow, handle toward stern); positive values indicate the finish direction (oar blade toward stern, handle toward bow). This is standard oarlock/gateforce sensor convention.
- Summary fields (Catch, Finish, etc.) serve as representative values when per-side data unavailable
- **PeakForcePositionNorm**: Value in range 0-10000 representing ten-thousandths (0 = catch, 10000 = end of drive)
- **PeakForceAngle** vs **PeakForcePosition**: Angle is for oar angle sensors (OTW); Position is for handle position sensors (indoor)

### 5.3 Dual Oarlock Metrics (Record-Level)

When both port and starboard oarlocks are present, per-side metrics MAY be included:

| Field Name | ID | Base Type | Scale | Units | Side |
|------------|----|-----------| ------|-------|------|
| CatchPort | 200 | SINT16 | 10 | deg | Port |
| CatchStarboard | 201 | SINT16 | 10 | deg | Starboard |
| FinishPort | 202 | SINT16 | 10 | deg | Port |
| FinishStarboard | 203 | SINT16 | 10 | deg | Starboard |
| SlipPort | 204 | SINT16 | 10 | deg | Port |
| SlipStarboard | 205 | SINT16 | 10 | deg | Starboard |
| WashPort | 206 | SINT16 | 10 | deg | Port |
| WashStarboard | 207 | SINT16 | 10 | deg | Starboard |
| PeakForceAnglePort | 208 | SINT16 | 10 | deg | Port |
| PeakForceAngleStarboard | 209 | SINT16 | 10 | deg | Starboard |
| EffectiveLengthPort | 210 | UINT16 | 100 | m | Port |
| EffectiveLengthStarboard | 211 | UINT16 | 100 | m | Starboard |

**Per-side field rules:**

- Per-side fields SHOULD only be included when both port and starboard data are available
- When per-side fields are present, summary fields (IDs 11-16) SHOULD contain the average of port and starboard values
- When only one side is available, summary fields SHOULD contain that side's value
- Consumers implementing only partial support MAY ignore per-side fields and use summary fields

## 6. In-Stroke Curve Data

### 6.1 Overview

In-stroke curve data provides high-resolution force, acceleration, or angle measurements throughout a stroke cycle. This data enables detailed stroke analysis and technique visualization.

**Constraint:** In-stroke curve data MUST only appear when `RecordingStrategy=StrokeBoundary` (or Unknown with stroke-boundary semantics), as curves require stroke boundaries for interpretation.

### 6.2 Field ID Allocation

- **90-92**: Axis metadata (per-Record)
- **20-59**: Curve summary statistics (dynamic, per-curve-type)
- **60-255**: Curve arrays (dynamic, per-curve-type)

### 6.3 Axis Metadata Fields (Record-Level)

These fields define the X-axis interpretation for curve data on each Record:

| Field Name | ID | Base Type | Scale | Units | Definition |
|------------|----|-----------| ------|-------|------------|
| InstrokeAbscissaType | 90 | UINT8 | 1 | | X-axis semantics (enum) |
| InstrokeSampleInterval | 91 | UINT16 | 1 | (varies) | Sample spacing (interpretation depends on Type) |
| InstrokePointCount | 92 | UINT8 | 1 | | Number of points in curve arrays |

**InstrokeAbscissaType values:**

| Value | Name | InstrokeSampleInterval Meaning |
|-------|------|--------------------------------|
| 0 | UNKNOWN | Not specified (shape-only curves) |
| 1 | TIME_UNIFORM_MS | Milliseconds between samples |
| 2 | HANDLE_DISTANCE_UNIFORM_M | Meters between samples (scale as documented) |
| 3 | OAR_ANGLE_UNIFORM_DEG | Degrees between samples (scale as documented) |
| 4 | NORMALIZED_DRIVE_0_1 | Dimensionless step size (0-1 range) |

**Rules:**

- Axis metadata fields MUST appear together on each Record containing curve data
- For Type=TIME_UNIFORM_MS with known drive time: `InstrokeSampleInterval = drive_time_ms / (point_count - 1)`
- For Type=OAR_ANGLE_UNIFORM_DEG: Domain is [Catch, Finish] angles (from fields 11-12)
- Type=UNKNOWN indicates shape-only data (for pattern analysis, not absolute plotting)
- Producers SHOULD strive for consistency between axis metadata and stroke scalars, but consumers SHOULD NOT enforce strict validation

### 6.4 Standard Curve Types

Recommended curve type names:

| Curve Name | Typical Data | Units |
|------------|--------------|-------|
| HandleForceCurve | Handle force over stroke | Newtons |
| BoatAcceleratorCurve | Boat acceleration | m/s² |
| OarAngleVelocityCurve | Angular velocity of oar | deg/s |
| SeatCurve | Seat position | meters |

### 6.5 Curve Array Format

Curve data MUST be encoded as **UINT16** arrays (developer fields with array size > 1):

- **Maximum points per curve:** 127 (FIT limit: 255 bytes / 2 bytes per UINT16)
- **Field ID allocation:** Start at 60, increment per curve type
- **Encoding:** Unsigned 16-bit integers in range [0, 65535]
- **Data representation:** Values are clipped to [0, 65535] range; negative values not supported
- **Scale factor:** Use appropriate scale in field description to map physical units to UINT16 range

**Note on signed data:** FIT SDK developer fields with arrays only reliably support UINT16 base type. For force curves and other rowing metrics, values are naturally non-negative. If future curve types require negative values, implement offset transformation (e.g., add 32768) and document in field description.

### 6.6 Curve Summary Statistics

As an alternative or supplement to full curves, summary statistics MAY be provided:

- **Field ID allocation:** Start at 20, increment per curve and metric
- **Metrics:** q1, q2, q3, q4 (quartile variations), diff (change), maxpos, minpos (normalized positions)

### 6.7 Companion Files

For curves exceeding 127 points, producers MAY use companion JSON files:

- **Filename:** Same basename as FIT file with `.instroke.json` extension
- **Format:** JSON object with curve names as keys, arrays of per-stroke samples as values
- **Metadata:** Include `_rowingdata_instroke` object with version, abscissa type, point counts

## 7. Compliance Levels

Implementers MAY choose compliance levels based on their capabilities:

### Level 1: Minimal

- Native FIT fields only
- `RecordingStrategy` on Session (recommended)
- Supports at least one recording strategy

### Level 2: Standard

- Level 1 requirements
- Core rowing metrics (IDs 0-9, 19)
- Handles missing developer fields gracefully

### Level 3: Full

- Level 2 requirements
- Oarlock metrics (IDs 11-18)
- In-stroke axis metadata (IDs 90-92)
- At least one in-stroke curve type

### Level 4: Advanced

- Level 3 requirements
- Dual oarlock per-side metrics (IDs 200-211)
- Multiple in-stroke curve types
- Curve summary statistics

## 8. Data Quality and Validation

### 8.1 Value Ranges

Producers SHOULD clamp values to reasonable ranges:

- Force: 0-2000 N typical maximum
- DriveLength: 0.8-2.0 m
- Angles: -180 to +180 degrees
- Stroke rate: 10-60 strokes/min typical

### 8.2 Missing Data

- Missing or unavailable fields SHOULD be omitted from the file
- Producers MUST NOT emit fields with placeholder values (e.g., -1, 999) without documentation
- Zero values SHOULD indicate actual measurements of zero (not missing data)

### 8.3 Consistency

While strict validation is not enforced, producers SHOULD maintain internal consistency:

- Drive + recovery time ≈ stroke period (60 / cadence)
- In-stroke axis metadata consistent with stroke scalars
- Summary oarlock fields = average of per-side fields (when both present)

## 9. Field Registry

### 9.1 Reserved ID Ranges

| Range | Purpose | Status |
|-------|---------|--------|
| 0-19 | Core rowing metrics | Assigned |
| 20-59 | In-stroke summaries | Dynamic allocation |
| 60-89 | In-stroke curve arrays | Dynamic allocation |
| 90-92 | In-stroke axis metadata | Assigned |
| 93-199 | Reserved for future standard fields | Available |
| 200-211 | Dual oarlock per-side | Assigned |
| 212-255 | Reserved for future extensions | Available |

### 9.2 Deprecated Fields

| Field ID | Name | Status | Replacement |
|----------|------|--------|-------------|
| 4 | AverageDriveForceLbs | Deprecated | AverageDriveForceN (ID 6) |
| 5 | PeakDriveForceLbs | Deprecated | PeakDriveForceN (ID 7) |

Producers SHOULD use Newtons for new implementations. Consumers MUST continue to support pounds for backward compatibility.

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.1 | 2026-05-14 | FIT SDK compliance updates: Application ID changed to 16-byte UUID (89e86158-6d47-5c98-9d46-7d29437f27b9); Curve arrays changed from SINT16 to UINT16 for developer field compatibility |
| 1.0 | 2026-05-13 | Initial standard release |

## Appendix A: Terminology

**Stroke Cycle:** Complete rowing motion from catch through drive and recovery back to catch.

**Drive Phase:** Portion of stroke where power is applied (handle moving toward finish).

**Recovery Phase:** Portion of stroke returning to catch position (no power).

**Catch:** Position/angle at start of drive phase (blade entry for OTW).

**Finish:** Position/angle at end of drive phase (blade exit for OTW).

**Slip:** Angular difference between ideal and actual blade entry (early entry).

**Wash:** Angular difference between ideal and actual blade exit (late exit).

**Oar Angle:** Angle of oar relative to boat's longitudinal axis (0° = perpendicular).

**Effective Length:** Horizontal distance from oarlock pin to handle (rigging metric).

**Drive Length:** Actual distance handle travels during drive phase.

**Stroke Distance:** Distance boat travels during one complete stroke cycle.

**OTW:** On-the-water (as opposed to ergometer/indoor).

## Appendix B: Reference Implementation

A reference implementation of this standard is available in the **rowingdata** open-source Python library:

- **Repository:** https://github.com/sanderroosendaal/rowingdata
- **Machine-readable specification:** `rowingdata/data/fit_export_spec.json`
- **Example FIT file:** `testdata/rowingdata_standard_example.fit`

## Appendix C: Acknowledgments

This standard builds upon work by the rowing data community, including contributions from:

- Device manufacturers (Concept2, Nielsen-Kellerman, Quiske, RowPerfect)
- Software platforms (Intervals.icu, OpenRowingMonitor)
- Individual developers and researchers

## Appendix D: Contact and Governance

For questions, clarifications, or proposed amendments to this standard:

- **Issue tracker:** https://github.com/sanderroosendaal/rowingdata/issues
- **Discussion forum:** Rowing Industry Trade Association meetings

Proposed changes SHOULD be discussed with stakeholders before implementation to ensure ecosystem compatibility.

---

**End of Standard**
