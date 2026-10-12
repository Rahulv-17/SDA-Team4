---
name: Stream Analytics Lab
colors:
  surface: '#0e1323'
  surface-dim: '#0e1323'
  surface-bright: '#34394a'
  surface-container-lowest: '#080d1d'
  surface-container-low: '#161b2b'
  surface-container: '#1a1f30'
  surface-container-high: '#25293a'
  surface-container-highest: '#2f3446'
  on-surface: '#dee1f9'
  on-surface-variant: '#bbc9ce'
  inverse-surface: '#dee1f9'
  inverse-on-surface: '#2b3041'
  outline: '#869398'
  outline-variant: '#3c494d'
  surface-tint: '#40d7fd'
  primary: '#b8edff'
  on-primary: '#003642'
  primary-container: '#43d9ff'
  on-primary-container: '#005c6f'
  inverse-primary: '#00677d'
  secondary: '#42deb7'
  on-secondary: '#00382b'
  secondary-container: '#03c19d'
  on-secondary-container: '#004939'
  tertiary: '#ffddde'
  on-tertiary: '#67001b'
  tertiary-container: '#ffb6ba'
  on-tertiary-container: '#ab0033'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#b2ebff'
  primary-fixed-dim: '#40d7fd'
  on-primary-fixed: '#001f27'
  on-primary-fixed-variant: '#004e5e'
  secondary-fixed: '#66fbd3'
  secondary-fixed-dim: '#42deb7'
  on-secondary-fixed: '#002018'
  on-secondary-fixed-variant: '#005140'
  tertiary-fixed: '#ffdadb'
  tertiary-fixed-dim: '#ffb2b7'
  on-tertiary-fixed: '#40000d'
  on-tertiary-fixed-variant: '#92002a'
  background: '#0e1323'
  on-background: '#dee1f9'
  surface-variant: '#2f3446'
typography:
  headline-xl:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
  headline-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-sm:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  metric-display:
    fontFamily: JetBrains Mono
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 32px
  label-code:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
  label-caps:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 16px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1.5rem
  margin: 1.5rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2rem
---

## Brand & Style

This design system is engineered for advanced academic computational workflows, real-time statistical surveillance, and algorithmic verification. Built specifically for high-throughput streaming analytics environments, the interface reconciles the dense informational demands of data science with the serene visual hierarchy of a precision laboratory instrument.

### Design Movement
The visual language operates at the intersection of **Technical Precisionism** and **Restrained Functional Minimalism**. It eschews decorative neon clutter, excessive transparency, and heavy skeuomorphism in favor of:
- High typographic legibility paired with mechanical, tabular data rendering.
- Subtle structural division via crisp, low-luminance borders instead of harsh bounding boxes.
- Purpose-driven luminance contrast: bright accents indicate active distribution shifts, data anomalies, or active stream ingress, while baseline telemetry rests in quiet, muted tonal registers.

### Target Audience & Psychological Response
Designed for graduate researchers, data systems engineers, and computational statisticians. The interface evokes feelings of deep focus, architectural rigor, intellectual discipline, and operational mastery.

## Colors

The palette is anchored in an ultra-deep spectral navy environment, allowing high-frequency visualizations and telemetry signals to stand out with deliberate clarity. Color functions as semantic data signaling rather than aesthetic decoration.

### Core Canvas & Surfaces
- **Canvas Base (`#0B1020`):** Ultra-deep midnight navy serving as the foundational background.
- **Structural Sidebar (`#101729`):** Grounded, low-reflection tone anchoring navigation and stream pipeline controls.
- **Card / Surface Default (`#141D30`):** Elevated structural containers providing legible separation from the base canvas.
- **Card / Surface Hover (`#19243C`):** Interactive lift state for cards and list nodes.
- **Subtle Structural Border (`#1E293B`):** Low-contrast dividers and structural lines.
- **Active Structural Border (`#22324D`):** Container borders and metric panel bounds.

### Typography & Content
- **Text Primary (`#F8FAFC`):** Pure readability for paper titles, statistical readings, and core values.
- **Text Secondary (`#94A3B8`):** Muted slate for metadata, stream parameters, and axis legends.
- **Text Tertiary (`#64748B`):** Diminished tokens, static keys, and timestamps.

### Accents & State Signaling
- **Primary Accent (`#43D9FF`):** Electric Cyan. Reserved for active streams, selected query operators, current hypothesis runs, and primary action buttons.
- **Secondary Accent (`#37D6B0`):** Clean Teal. Denotes converged distribution states, healthy baseline metrics, and steady-state window models.
- **Warning Indicator (`#F59E0B`):** Amber. Denotes soft drift detection, threshold proximity warnings, and memory pressure.
- **Error / Drift Critical (`#F43F5E`):** Coral Red. Flags definitive concept drift events (Kolmogorov-Smirnov / ADWIN triggers), packet losses, and degraded classification boundaries.

## Typography

The typography couples the clean, geometric clarity of **Inter** for operational prose and structure with the computational precision of **JetBrains Mono** for numerical matrices, latency measurements, sliding-window counters, and architectural references.

### Application Rules
- **Numerical Matrices & Metrics:** All floating-point rates, p-values, buffer sizes, and window sizes utilize `metric-display` or `label-code` with tabular numbers (`font-variant-numeric: tabular-nums`) enabled by default to prevent spatial jitter during continuous real-time updates.
- **Section Headers:** Utilize `headline-sm` with tight tracking (`letter-spacing: -0.01em`) to maintain an authoritative, publication-ready posture.
- **Micro-Labels:** Use `label-caps` set in uppercase with slight letter-spacing (`+0.05em`) for card headers, pipeline phases, and table descriptors.

## Layout & Spacing

The layout is built for desktop-first analytical observation with strict containment and stable whitespace ratios.

### Canvas Geometry
- **Target Desktop Viewport:** 1440px wide.
- **Persistent Left Navigation:** Fixed 210px column width spanning full screen height, styled in `#101729` with a persistent right border (`#1E293B`).
- **Main Container:** Centered 1180px maximum content width nestled within the remaining screen real estate, maintaining balanced lateral gutters to eliminate visual claustrophobia.
- **Outer Padding:** 24px (`1.5rem`) global outer padding across the content plane.

### Grid Rhythm & Adapting Views
- **Desktop (1440px+):** Fixed 210px sidebar alongside a 12-column analytical grid with 24px (`1.5rem`) gutters. Metric overview cards span 3 columns each, distribution comparison windows span 6 or 8 columns, and streaming log buffers span 4 or 12 columns.
- **Tablet / Sub-Desktop (1024px - 1439px):** Content container fluidly adapts with 24px side gutters; sidebar collapses to an icon rail (64px).
- **Component Internals:** Spacing inside card modules relies on standard `1rem` (16px) or `1.5rem` (24px) padding, enforcing symmetrical visual rhythm across all dashboard tiles.

## Elevation & Depth

To preserve an authentic scientific demeanor, this system rejects heavy drop shadows and artificial blurred elevations. Depth is achieved via **tonal stratification** and **subtle low-contrast outlines**.

### Stratification Model
1. **Level 0 (Base Canvas):** `#0B1020` — Continuous surface background.
2. **Level 1 (Dock & Sidebar):** `#101729` — Lateral structural plane with a 1px solid border `#1E293B` on the inner edge.
3. **Level 2 (Analytical Cards & Charts):** `#141D30` — Resting content modules enclosed within a 1px solid outline `#22324D`.
4. **Level 3 (Overlays, Tooltips & Dropdowns):** `#1A253D` — Elevated contextual layers bordered by `#334155`, backed by a minimal ambient shadow (`box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.45)`).

### Highlight Accents
Interactive cards apply a restrained top border accent (`1px` or `2px`) in electric cyan (`#43D9FF`) or teal (`#37D6B0`) upon active evaluation states, avoiding full-container glow filters to protect statistical readouts from luminance pollution.

## Shapes

The geometric framework balances human-centered interface ergonomics with crisp mathematical discipline.

### Corner Radii
- **Base Cards & Graph Viewports:** 12px corner rounding (`rounded-xl` equivalent in a scale factor of 2: standard rounded units with 12px bounds for primary containers).
- **Buttons, Inputs & Selectors:** 6px to 8px corner rounding to communicate tactical utility and clickability.
- **Tags, Counters & Status Badges:** 4px micro-rounding, avoiding circular pill shapes to preserve an intentional, data-terminal character.

## Components

### Buttons
- **Primary:** Solid `#43D9FF` background with `#0B1020` typography, semibold weight, 8px radius. Active state shows clean luminance reduction without outer halos.
- **Secondary / Ghost:** Outlined with 1px `#22324D`, filled with transparent or `#141D30`, text in `#F8FAFC`. On hover, border shifts to `#43D9FF` with text color unchanged.
- **Critical / Interruption:** Outlined in `#F43F5E`, transparent background, text `#F43F5E`. Used solely for stream aborts, test set purges, or emergency pipeline stops.

### Cards & Telemetry Containers
- **Container Structure:** Solid `#141D30` surface with a crisp 1px `#22324D` border and 12px border radius.
- **Card Header:** Houses `label-caps` category titles in `#94A3B8` alongside status dots (8px) indicating dynamic stream health.
- **Padding:** 20px on metrics tiles, 24px on complex chart modules.

### Status Indicators & Drift Badges
- **Status Indicator:** 6px circular dot paired with `JetBrains Mono` label text.
  - Green/Teal (`#37D6B0`): Stationary distribution.
  - Amber (`#F59E0B`): Warning zone (e.g., ADWIN margin narrowed).
  - Coral Red (`#F43F5E`): Confirmed drift event.
- **Badge Capsule:** Compact rect with 4px border radius, 4px vertical / 8px horizontal padding, `#0B1020` background, and 1px border matching the status color.

### Input Fields & Controls
- **Parameter Inputs:** Background `#101729`, 1px border `#1E293B`, text `#F8FAFC`, font `JetBrains Mono` (13px). Focused state features a 1px border `#43D9FF` without outer focus rings.
- **Sliding-Window Sliders:** Thin 4px rail in `#1E293B`, active fill in `#43D9FF`, thumb is a sharp 12px square or circular node in `#F8FAFC`.

### Data Tables & Log Streaming Feeds
- **Header:** Sticky `#101729` bar, 1px bottom border `#22324D`, uppercase 11px muted labels with `letter-spacing: 0.05em`.
- **Row:** Height of 36px for dense scanning, alternating faint zebra backgrounds (`#141D30` and `#111827`), hover row highlight `#19243C`.
- **Cells:** JetBrains Mono for sequence IDs, timestamps, and p-values. Right-aligned numerical data.