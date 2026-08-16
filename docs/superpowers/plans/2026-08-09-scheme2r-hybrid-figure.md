# Scheme2R Hybrid Figure Implementation Plan

> **For agentic workers:** Use this plan task-by-task with a visual QA checkpoint.

**Goal:** Generate an editable hybrid Scheme2R network diagram without changing the algorithm.

**Architecture:** Retain the existing four-column left-to-right layout. Use flat rounded rectangles for computational modules and native PowerPoint stacked cuboids for tensors/features.

**Tech Stack:** Node.js 22+, `@oai/artifact-tool`, PowerPoint-native shapes, rendered PNG visual QA.

## Global Constraints

- Do not modify model code, experiment protocols, metrics, or manuscript text.
- Preserve the frozen task order and tensor shapes exactly.
- Keep the output as an editable `.pptx` plus a rendered `.png` preview.

### Task 1: Update the figure generator — completed

**Files:**
- Modify: `D:/Paper/ppt_build_scheme2r/create_scheme2r_v9.mjs`

**Changes:**
- Add native stacked cuboids for input tensors, task representations, routing tensors, and output tensor.
- Keep DS-TCN, routing, fusion, and prediction-head blocks as flat modules.
- Preserve all existing labels and arrows, changing only visual grouping and depth cues.

### Task 2: Render and inspect — completed

**Files:**
- Create: `D:/Paper/new_paper/Scheme2R_Framework_Diagram_v10.pptx`
- Create: `D:/Paper/ppt_build_scheme2r/rendered_v10/Scheme2R_Framework_Diagram_v10.png`

**Checks:**
- Confirm all text is visible and no object overlaps another object.
- Confirm the output label remains `[B,4,4]`.
- Confirm the gate label remains `[B,H,T_target,T_source]`.

## Verification Result

- Generated `Scheme2R_Framework_Diagram_v10.pptx` and its PNG preview.
- Visual inspection found no clipping or overlap that affects the data flow.
- The algorithm content remains unchanged from v9; only tensor/feature blocks received pseudo-3D depth layers.
