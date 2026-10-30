---
name: trendlab research report
description: A Tufte-school research handout where every number carries its provenance in the margin.
colors:
  strategy-blue: "#2a78d6"
  benchmark-grey: "#999891"
  examination-carmine: "#b3261e"
  below-ochre: "#b8860b"
  heatmap-sand: "#c99a2e"
  bond-paper: "#fcfcfb"
  ink: "#0b0b0b"
  ink-secondary: "#52514e"
  muted: "#898781"
  rule: "#e1e0d9"
  rule-strong: "#c3c2b7"
  wash: "#f3f3f0"
typography:
  display:
    fontFamily: "ET Book, Iowan Old Style, Palatino, Palatino Linotype, Georgia, serif"
    fontSize: "3.6rem"
    fontWeight: 400
    lineHeight: 1
  headline:
    fontFamily: "ET Book, Iowan Old Style, Palatino, Palatino Linotype, Georgia, serif"
    fontSize: "3.25rem"
    fontWeight: 400
    lineHeight: 1.06
    letterSpacing: "-0.012em"
  headline-section:
    fontFamily: "ET Book, Iowan Old Style, Palatino, Palatino Linotype, Georgia, serif"
    fontSize: "2.1rem"
    fontWeight: 400
    lineHeight: 1.15
    letterSpacing: "-0.006em"
  title:
    fontFamily: "ET Book, Iowan Old Style, Palatino, Palatino Linotype, Georgia, serif"
    fontSize: "1.6rem"
    fontWeight: 400
    lineHeight: 1.22
  standfirst:
    fontFamily: "ET Book, Iowan Old Style, Palatino, Palatino Linotype, Georgia, serif"
    fontSize: "1.45rem"
    fontWeight: 400
    lineHeight: 1.45
  body:
    fontFamily: "ET Book, Iowan Old Style, Palatino, Palatino Linotype, Georgia, serif"
    fontSize: "1.25rem"
    fontWeight: 400
    lineHeight: 1.55
  note:
    fontFamily: "ET Book, Iowan Old Style, Palatino, Palatino Linotype, Georgia, serif"
    fontSize: "0.95rem"
    fontWeight: 400
    lineHeight: 1.45
  label:
    fontFamily: "Helvetica Neue, Helvetica, Arial, Liberation Sans, Nimbus Sans, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 400
    lineHeight: 1.35
    fontFeature: "tnum, lnum"
  label-strong:
    fontFamily: "Helvetica Neue, Helvetica, Arial, Liberation Sans, Nimbus Sans, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 700
    lineHeight: 1.35
  verdict:
    fontFamily: "Helvetica Neue, Helvetica, Arial, Liberation Sans, Nimbus Sans, sans-serif"
    fontSize: "0.7rem"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "0.09em"
  provenance:
    fontFamily: "Helvetica Neue, Helvetica, Arial, Liberation Sans, Nimbus Sans, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 400
    lineHeight: 1.4
  mono:
    fontFamily: "ui-monospace, SF Mono, Menlo, Consolas, Liberation Mono, monospace"
    fontSize: "0.78em"
    fontWeight: 400
    lineHeight: 1.3
rounded:
  hairline: "2px"
  code: "3px"
  swatch: "50%"
spacing:
  step: "1.5rem"
  gutter: "3rem"
  text-column: "40rem"
  margin-column: "17.5rem"
  finding-gap: "3.5rem"
  section-gap: "5.5rem"
components:
  verdict-mark:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    typography: "{typography.verdict}"
    rounded: "0"
    padding: "0.35rem 0.5rem 0.3rem"
  chart-toggle:
    backgroundColor: "transparent"
    textColor: "{colors.ink-secondary}"
    rounded: "{rounded.hairline}"
    padding: "0.4rem 0.55rem"
  chart-toggle-hover:
    textColor: "{colors.ink}"
  chart-toggle-pressed:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.bond-paper}"
  code-inline:
    backgroundColor: "{colors.wash}"
    textColor: "{colors.ink}"
    typography: "{typography.mono}"
    rounded: "{rounded.code}"
    padding: "0.08em 0.32em"
  running-head-link:
    textColor: "{colors.ink-secondary}"
    typography: "{typography.label}"
    padding: "0.2rem 0"
  running-head-link-current:
    textColor: "{colors.ink}"
  margin-note:
    textColor: "{colors.ink-secondary}"
    typography: "{typography.note}"
    width: "{spacing.margin-column}"
---

# Design System: trendlab research report

## Overview

**Creative North Star: "The Annotated Handout"**

A printed research handout on bond paper, set in ET Book, with a wide right margin where every claim's provenance lives. The page is achromatic; colour appears only where data is, and one carmine ink is held back for the single thing under examination: the original +505% claim and the bias each correction removes. The tone is scholarly and plain. It argues against its own headline, so nothing on the page sells.

Density is a reading density, not a dashboard density: a 40rem text column at 1.25rem/1.55, a 17.5rem margin column, generous vertical air between findings (3.5rem) and sections (5.5rem). One master grid governs every page; the index, ten study pages and the methodology page differ only by their content and figures. State is expressed typographically (a boxed verdict word, a strike-through) rather than with hue.

The interactive charts are the signature. Pointing at or arrow-keying through any mark writes its exact values and provenance into the margin readout beside the chart; on narrow screens the readout folds inline beneath it.

**Key Characteristics:**
- Bond paper (#fcfcfb), near-black ink, no surfaces, no shadows, no cards.
- ET Book (line figures) for prose and headings; a Helvetica-class grotesk for labels, nav, tables, chart text and provenance.
- Text column plus margin column on one grid, folding to a single column at 920px.
- Colour is data: strategy blue, buy-and-hold grey, ochre for below benchmark; carmine only for the claim under examination.
- Every headline number carries a sans provenance line naming its study and results key.

## Colors

An achromatic page with colour confined to data marks, and one reserved annotation red.

### Primary
- **Strategy Blue** (strategy-blue): the strategy series in every chart and figure; positive cells in the parameter heatmap; forest-plot whiskers and points; also the focus ring, link-hover underline and selection tint (20% alpha), the one place the data hue crosses into the interface.

### Secondary
- **Benchmark Grey** (benchmark-grey): buy-and-hold and the 60/40 benchmark, as lines and as the dashed reference rule (5 4 dash). Grey because the benchmark is the baseline, not a contender.
- **Below-Benchmark Ochre** (below-ochre): the blend series and anything that trails buy-and-hold in figures; second slot of the matplotlib categorical palette.
- **Heatmap Sand** (heatmap-sand): the negative end of the diverging parameter heatmap, mixed from paper; paired with blue on the positive end.

### Tertiary
- **Examination Carmine** (examination-carmine): the struck-through original claim, the first waterfall bar (the notebook result), and the "pp removed" labels on each subsequent bar. Nothing else.

### Neutral
- **Bond Paper** (bond-paper): page ground, and the halo stroke behind chart value labels.
- **Ink** (ink): body text, headings, the zero line, verdict mark border, pressed toggles.
- **Secondary Ink** (ink-secondary): margin notes, running head, captions, axis ticks, provenance lines.
- **Muted** (muted): list markers only; too light for running text.
- **Rule** (rule): table row hairlines, chart gridlines, running head and colophon borders, study-list separators.
- **Strong Rule** (rule-strong): section-head underlines, pager top rule, chart axes, toggle borders, the mobile margin-note left rule.
- **Wash** (wash): inline code and code blocks; hovered heatmap/forest row band.

### Named Rules
**The One Red Rule.** Carmine marks only the claim under examination and the bias each step removes. No error states, no negative numbers, no emphasis, no decoration in red. The matplotlib palette deliberately carries no red-family hue for this reason.

**The Colour Is Data Rule.** The text field stays achromatic. A hue on the page must be standing in for a series; interface chrome uses ink, secondary ink and rules.

## Typography

**Display Font:** ET Book (with Iowan Old Style, Palatino, Georgia)
**Body Font:** ET Book, roman and bold with line figures; the display italic carries old-style figures
**Label Font:** Helvetica Neue / Helvetica / Arial / Liberation Sans
**Mono Font:** ui-monospace, SF Mono, Menlo, Consolas

**Character:** A Bembo-descended book face for everything read, against a neutral grotesk for everything looked up. The grotesk matches the chart labels so table, axis and margin metadata read as one apparatus.

### Hierarchy
- **Display** (400, 3.6rem, 1): the claim line only: old figure struck in carmine, an arrow, the corrected figure in ink, the benchmark in sans grey. 2.6rem below 920px, 2.2rem below 520px.
- **Headline** (400, 3.25rem, 1.06, -0.012em): page title. 2.5rem below 920px, 2.15rem below 520px.
- **Section Headline** (400, 2.1rem, 1.15): section heads (Findings, Studies, Reproduce), underlined by a strong rule.
- **Title** (400, 1.6rem, 1.22): finding headings, max 34rem.
- **Standfirst** (400, 1.45rem, 1.45): the two-sentence summary under the title.
- **Body** (400, 1.25rem, 1.55): prose, max 40rem, hyphenated; 1.125rem below 920px.
- **Note** (400, 0.95rem, 1.45): margin notes and readouts, in serif, secondary ink.
- **Label** (400/700, 0.8125rem, 1.35, tabular lining figures): running head, chart and table captions (700), tables, figcaptions, readout grids, colophon.
- **Provenance** (400, 0.75rem, 1.4): "Source: e01 · results_key" lines.
- **Verdict** (700, 0.7rem, 0.09em, uppercase): the verdict mark only.

### Named Rules
**The Two Voices Rule.** Serif is read, sans is consulted. Prose, headings and margin explanations are ET Book; anything tabular, numeric-in-a-grid, navigational or provenance is the grotesk. Headings never go bold; hierarchy is size.

**The Tabular Figures Rule.** Every number in a table, readout or chart label uses tabular figures, and the minus sign is a true minus (U+2212).

## Layout

One master grid: text column (40rem, minmax 0) + gutter (3rem) + margin column (17.5rem), centered with 1.5rem side padding. Every row places content in the main column, the side column, or spans both (`full`, used by chart titles, figures and wide tables). Running head and colophon share the same max width.

Vertical rhythm: title row 4.5rem top padding; claim row 2.25rem; sections 5.5rem; findings 3.5rem; charts 1.75rem above / 1.25rem below; figures and tables 1.5rem above / 2rem below; colophon 6rem above.

Responsive: below 1080px the margin narrows to 14rem and the gutter to 2rem. Below 920px the grid collapses to one column, margin notes fold in under their main content with a strong-rule left border and 0.9rem indent, readouts sit inline under the chart with a reserved 7.5rem min height, and the running-head nav drops to its own line. Below 720px raster figures render at 720px inside a horizontal scroller with a scroll hint. Below 520px the side padding drops to 1rem and the pager stacks.

Tables wider than 11 columns use the dense variant (0.75rem, tighter padding); any table may scroll horizontally inside a wrapper with paper-coloured fade masks and inset scroll shadows at its edges.

## Elevation & Depth

Flat. There are no box-shadows anywhere on the page. Depth is conveyed by hairline rules, the wash tint on code and hovered rows, and the paper-coloured halo stroke (4px) behind chart value labels. The single shadow-like device is the scroll cue on overflowing tables: a 0.9rem ink gradient at 14% that appears only while content is clipped.

### Named Rules
**The Paper Rule.** Nothing lifts off the page. If an element needs separating, use a 1px rule or white space, never a shadow, card or tinted panel.

## Shapes

Square and ruled. Corners are effectively square: 2px on toggles and the focus ring, 3px on code, 0 on the verdict mark. Lines are 1px hairlines (rule or strong rule) for structure: under the running head, under section heads, between study-list items and table rows, above the pager and colophon. Circles appear only as data: legend swatches (0.55rem) and chart dots with a paper stroke.

## Components

### Running Head
Quiet, bibliographic. Sans 0.8125rem across the top above a 1px rule: project name in bold ink, "Research report · data through <date>" in secondary ink, nav right-aligned. Nav links have no underline at rest; hover darkens to ink with a strong-rule underline; the current page gets an ink underline (1px border-bottom). Below 920px the nav wraps to a full-width line.

### Claim Line
The page's one display moment. The original figure in carmine with a carmine strike (0.06em thick), a thin drawn SVG arrow in secondary ink, the corrected figure in ink, and the benchmark in sans 1rem secondary ink, all baseline-aligned. Its margin note is the provenance line.

### Verdict Mark
A typographic state mark beside each finding heading in the margin column: one word (Artifact, Holds, Not significant, Noise, Conditional), bold sans 0.7rem uppercase at 0.09em tracking, ink on paper inside a 1px ink box. Never colour-coded: every verdict looks identical so the word carries the meaning.

### Margin Note with Provenance
Serif 0.95rem in secondary ink in the side column, ending in a sans provenance line ("Source: e04 · results_key") linking to the study. Below 920px it folds under the main text with a strong-rule left border.

### Chart Block with Margin Readout (signature)
A bold sans chart title spanning the full row, optional toggle buttons, then a hand-written SVG chart in the main column and a live readout in the side column. Readout: a bold sans head, a one-line serif explanation, and a two-column sans definition grid (labels secondary ink, values ink right-aligned, tabular) with round colour swatches. Pointer or arrow keys select a mark; non-selected marks dim to 38% opacity; heatmap focus is a 2.5px strategy-blue cell outline. Chart text: sans 11-12.5px; values bold ink with paper halo. Gridlines rule, axes strong rule, zero line ink, benchmark reference dashed grey. Bars and lines draw once on first view (bars scaleY over 520ms, lines dash-reveal over 900ms, cubic-bezier(0.16, 1, 0.3, 1)), triggered by IntersectionObserver and disabled under reduced motion. Each chart carries a `details` data fallback with its source.

### Chart Toggles
Small sans buttons (0.75rem) with a 1px strong-rule border and 2px corners, transparent at rest; hover darkens text and border; pressed state is ink fill with paper text. 160ms ease-out colour transitions.

### Figure Row
A matplotlib PNG spanning the full row, linked to the full-size image, with a sans 0.8125rem caption in secondary ink (max 40rem).

### Table Row
Bold sans caption, then a borderless table: sans 0.8125rem tabular lining figures, header in bold secondary ink over a 1px ink rule, body rows on 1px rule hairlines, numeric cells right-aligned, row hover at 3.5% ink. Dense variant at 0.75rem for wide tables.

### Study Index
An unstyled list of studies, each a two-column grid: sans study id (e01) in a 3.25rem gutter, serif title at 1.35rem, then a 1.05rem secondary-ink summary aligned to the title. Rule hairline between entries; hover underlines the title in strategy blue.

### Pager and Colophon
Pager: strong rule on top, previous and next in two columns, each a sans "Previous"/"Next" label over a 1.2rem serif title; stacks below 520px. Colophon: rule on top, 6rem above, sans 0.8125rem secondary ink.

### Links
Inherit ink colour, 1px underline at 32% ink with 0.18em offset; hover turns the underline strategy blue over 160ms. Focus everywhere is a 2px strategy-blue outline at 3px offset.

## Do's and Don'ts

### Do:
- **Do** put every new page on the master grid (40rem text, 3rem gutter, 17.5rem margin) and put explanations and provenance in the margin column, not in the text column.
- **Do** end any headline number with a sans provenance line naming its study and results key.
- **Do** colour strategy series blue (#2a78d6) and benchmark series grey (#999891) in both the site's SVG charts and the matplotlib figures.
- **Do** express a finding's state as a one-word verdict mark in an ink box.
- **Do** write a chart's exact values into its margin readout on pointer and keyboard selection, and give every chart a data fallback.
- **Do** use the grotesk with tabular figures for tables, chart text and readouts; ET Book for everything read.
- **Do** draw charts once on first view and disable motion under prefers-reduced-motion.

### Don't:
- **Don't** use carmine (#b3261e), or any red-family hue, for anything other than the claim under examination and the bias each step removes.
- **Don't** colour-code verdicts or status; the word is the signal.
- **Don't** add shadows, cards, tinted panels, KPI tiles or a dark hero.
- **Don't** bold headings; hierarchy is size in ET Book at weight 400.
- **Don't** use the eight-hue matplotlib categorical palette on the site's interactive charts; it exists for multi-series raster figures only.
- **Don't** use muted (#898781) for running text; it is for list markers.
