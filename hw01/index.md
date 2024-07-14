# 

<!DOCTYPE html>

<html lang="en">
<head><meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>hw01</title><script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<style type="text/css">
    pre { line-height: 125%; }
td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
.highlight .hll { background-color: var(--jp-cell-editor-active-background) }
.highlight { background: var(--jp-cell-editor-background); color: var(--jp-mirror-editor-variable-color) }
.highlight .c { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment */
.highlight .err { color: var(--jp-mirror-editor-error-color) } /* Error */
.highlight .k { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword */
.highlight .o { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator */
.highlight .p { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation */
.highlight .ch { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Multiline */
.highlight .cp { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Preproc */
.highlight .cpf { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Single */
.highlight .cs { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Special */
.highlight .kc { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Pseudo */
.highlight .kr { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Type */
.highlight .m { color: var(--jp-mirror-editor-number-color) } /* Literal.Number */
.highlight .s { color: var(--jp-mirror-editor-string-color) } /* Literal.String */
.highlight .ow { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator.Word */
.highlight .pm { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation.Marker */
.highlight .w { color: var(--jp-mirror-editor-variable-color) } /* Text.Whitespace */
.highlight .mb { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Bin */
.highlight .mf { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Float */
.highlight .mh { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Hex */
.highlight .mi { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer */
.highlight .mo { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Oct */
.highlight .sa { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Affix */
.highlight .sb { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Backtick */
.highlight .sc { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Char */
.highlight .dl { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Delimiter */
.highlight .sd { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Doc */
.highlight .s2 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Double */
.highlight .se { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Escape */
.highlight .sh { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Heredoc */
.highlight .si { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Interpol */
.highlight .sx { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Other */
.highlight .sr { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Regex */
.highlight .s1 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Single */
.highlight .ss { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Symbol */
.highlight .il { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer.Long */
  </style>
<style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
 * Mozilla scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */
[data-jp-theme-scrollbars='true'] {
  scrollbar-color: rgb(var(--jp-scrollbar-thumb-color))
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar. These selectors
 * will match lower in the tree, and so will override the above */
[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
}

/* tiny scrollbar */

.jp-scrollbar-tiny {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
  scrollbar-width: thin;
}

/* tiny scrollbar */

.jp-scrollbar-tiny::-webkit-scrollbar,
.jp-scrollbar-tiny::-webkit-scrollbar-corner {
  background-color: transparent;
  height: 4px;
  width: 4px;
}

.jp-scrollbar-tiny::-webkit-scrollbar-thumb {
  background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:horizontal {
  border-left: 0 solid transparent;
  border-right: 0 solid transparent;
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:vertical {
  border-top: 0 solid transparent;
  border-bottom: 0 solid transparent;
}

/*
 * Lumino
 */

.lm-ScrollBar[data-orientation='horizontal'] {
  min-height: 16px;
  max-height: 16px;
  min-width: 45px;
  border-top: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] {
  min-width: 16px;
  max-width: 16px;
  min-height: 45px;
  border-left: 1px solid #a0a0a0;
}

.lm-ScrollBar-button {
  background-color: #f0f0f0;
  background-position: center center;
  min-height: 15px;
  max-height: 15px;
  min-width: 15px;
  max-width: 15px;
}

.lm-ScrollBar-button:hover {
  background-color: #dadada;
}

.lm-ScrollBar-button.lm-mod-active {
  background-color: #cdcdcd;
}

.lm-ScrollBar-track {
  background: #f0f0f0;
}

.lm-ScrollBar-thumb {
  background: #cdcdcd;
}

.lm-ScrollBar-thumb:hover {
  background: #bababa;
}

.lm-ScrollBar-thumb.lm-mod-active {
  background: #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal'] .lm-ScrollBar-thumb {
  height: 100%;
  min-width: 15px;
  border-left: 1px solid #a0a0a0;
  border-right: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] .lm-ScrollBar-thumb {
  width: 100%;
  min-height: 15px;
  border-top: 1px solid #a0a0a0;
  border-bottom: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-left);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-right);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-up);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-down);
  background-size: 17px;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-Widget {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
}

.lm-Widget.lm-mod-hidden {
  display: none !important;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.lm-AccordionPanel[data-orientation='horizontal'] > .lm-AccordionPanel-title {
  /* Title is rotated for horizontal accordion panel using CSS */
  display: block;
  transform-origin: top left;
  transform: rotate(-90deg) translate(-100%);
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  display: flex;
  flex-direction: column;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-CommandPalette-search {
  flex: 0 0 auto;
}

.lm-CommandPalette-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  min-height: 0;
  overflow: auto;
  list-style-type: none;
}

.lm-CommandPalette-header {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-CommandPalette-item {
  display: flex;
  flex-direction: row;
}

.lm-CommandPalette-itemIcon {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemContent {
  flex: 1 1 auto;
  overflow: hidden;
}

.lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemLabel {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-close-icon {
  border: 1px solid transparent;
  background-color: transparent;
  position: absolute;
  z-index: 1;
  right: 3%;
  top: 0;
  bottom: 0;
  margin: auto;
  padding: 7px 0;
  display: none;
  vertical-align: middle;
  outline: 0;
  cursor: pointer;
}
.lm-close-icon:after {
  content: 'X';
  display: block;
  width: 15px;
  height: 15px;
  text-align: center;
  color: #000;
  font-weight: normal;
  font-size: 12px;
  cursor: pointer;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-DockPanel {
  z-index: 0;
}

.lm-DockPanel-widget {
  z-index: 0;
}

.lm-DockPanel-tabBar {
  z-index: 1;
}

.lm-DockPanel-handle {
  z-index: 2;
}

.lm-DockPanel-handle.lm-mod-hidden {
  display: none !important;
}

.lm-DockPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}

.lm-DockPanel-handle[data-orientation='horizontal'] {
  cursor: ew-resize;
}

.lm-DockPanel-handle[data-orientation='vertical'] {
  cursor: ns-resize;
}

.lm-DockPanel-handle[data-orientation='horizontal']:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}

.lm-DockPanel-handle[data-orientation='vertical']:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

.lm-DockPanel-overlay {
  z-index: 3;
  box-sizing: border-box;
  pointer-events: none;
}

.lm-DockPanel-overlay.lm-mod-hidden {
  display: none !important;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-Menu {
  z-index: 10000;
  position: absolute;
  white-space: nowrap;
  overflow-x: hidden;
  overflow-y: auto;
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-Menu-content {
  margin: 0;
  padding: 0;
  display: table;
  list-style-type: none;
}

.lm-Menu-item {
  display: table-row;
}

.lm-Menu-item.lm-mod-hidden,
.lm-Menu-item.lm-mod-collapsed {
  display: none !important;
}

.lm-Menu-itemIcon,
.lm-Menu-itemSubmenuIcon {
  display: table-cell;
  text-align: center;
}

.lm-Menu-itemLabel {
  display: table-cell;
  text-align: left;
}

.lm-Menu-itemShortcut {
  display: table-cell;
  text-align: right;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-MenuBar {
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-MenuBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: row;
  list-style-type: none;
}

.lm-MenuBar-item {
  box-sizing: border-box;
}

.lm-MenuBar-itemIcon,
.lm-MenuBar-itemLabel {
  display: inline-block;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-ScrollBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-ScrollBar[data-orientation='horizontal'] {
  flex-direction: row;
}

.lm-ScrollBar[data-orientation='vertical'] {
  flex-direction: column;
}

.lm-ScrollBar-button {
  box-sizing: border-box;
  flex: 0 0 auto;
}

.lm-ScrollBar-track {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  flex: 1 1 auto;
}

.lm-ScrollBar-thumb {
  box-sizing: border-box;
  position: absolute;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-SplitPanel-child {
  z-index: 0;
}

.lm-SplitPanel-handle {
  z-index: 1;
}

.lm-SplitPanel-handle.lm-mod-hidden {
  display: none !important;
}

.lm-SplitPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}

.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle {
  cursor: ew-resize;
}

.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle {
  cursor: ns-resize;
}

.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}

.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-TabBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-TabBar[data-orientation='horizontal'] {
  flex-direction: row;
  align-items: flex-end;
}

.lm-TabBar[data-orientation='vertical'] {
  flex-direction: column;
  align-items: flex-end;
}

.lm-TabBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex: 1 1 auto;
  list-style-type: none;
}

.lm-TabBar[data-orientation='horizontal'] > .lm-TabBar-content {
  flex-direction: row;
}

.lm-TabBar[data-orientation='vertical'] > .lm-TabBar-content {
  flex-direction: column;
}

.lm-TabBar-tab {
  display: flex;
  flex-direction: row;
  box-sizing: border-box;
  overflow: hidden;
  touch-action: none; /* Disable native Drag/Drop */
}

.lm-TabBar-tabIcon,
.lm-TabBar-tabCloseIcon {
  flex: 0 0 auto;
}

.lm-TabBar-tabLabel {
  flex: 1 1 auto;
  overflow: hidden;
  white-space: nowrap;
}

.lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing: border-box;
}

.lm-TabBar-tab.lm-mod-hidden {
  display: none !important;
}

.lm-TabBar-addButton.lm-mod-hidden {
  display: none !important;
}

.lm-TabBar.lm-mod-dragging .lm-TabBar-tab {
  position: relative;
}

.lm-TabBar.lm-mod-dragging[data-orientation='horizontal'] .lm-TabBar-tab {
  left: 0;
  transition: left 150ms ease;
}

.lm-TabBar.lm-mod-dragging[data-orientation='vertical'] .lm-TabBar-tab {
  top: 0;
  transition: top 150ms ease;
}

.lm-TabBar.lm-mod-dragging .lm-TabBar-tab.lm-mod-dragging {
  transition: none;
}

.lm-TabBar-tabLabel .lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing: border-box;
  background: inherit;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-TabPanel-tabBar {
  z-index: 1;
}

.lm-TabPanel-stackedPanel {
  z-index: 0;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapse {
  display: flex;
  flex-direction: column;
  align-items: stretch;
}

.jp-Collapse-header {
  padding: 1px 12px;
  background-color: var(--jp-layout-color1);
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  align-items: center;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  text-transform: uppercase;
  user-select: none;
}

.jp-Collapser-icon {
  height: 16px;
}

.jp-Collapse-header-collapsed .jp-Collapser-icon {
  transform: rotate(-90deg);
  margin: auto 0;
}

.jp-Collapser-title {
  line-height: 25px;
}

.jp-Collapse-contents {
  padding: 0 12px;
  background-color: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensureUiComponents() in @jupyterlab/buildutils */

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

/* Icons urls */

:root {
  --jp-icon-add-above: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzEzN18xOTQ5MikiPgo8cGF0aCBjbGFzcz0ianAtaWNvbjMiIGQ9Ik00Ljc1IDQuOTMwNjZINi42MjVWNi44MDU2NkM2LjYyNSA3LjAxMTkxIDYuNzkzNzUgNy4xODA2NiA3IDcuMTgwNjZDNy4yMDYyNSA3LjE4MDY2IDcuMzc1IDcuMDExOTEgNy4zNzUgNi44MDU2NlY0LjkzMDY2SDkuMjVDOS40NTYyNSA0LjkzMDY2IDkuNjI1IDQuNzYxOTEgOS42MjUgNC41NTU2NkM5LjYyNSA0LjM0OTQxIDkuNDU2MjUgNC4xODA2NiA5LjI1IDQuMTgwNjZINy4zNzVWMi4zMDU2NkM3LjM3NSAyLjA5OTQxIDcuMjA2MjUgMS45MzA2NiA3IDEuOTMwNjZDNi43OTM3NSAxLjkzMDY2IDYuNjI1IDIuMDk5NDEgNi42MjUgMi4zMDU2NlY0LjE4MDY2SDQuNzVDNC41NDM3NSA0LjE4MDY2IDQuMzc1IDQuMzQ5NDEgNC4zNzUgNC41NTU2NkM0LjM3NSA0Ljc2MTkxIDQuNTQzNzUgNC45MzA2NiA0Ljc1IDQuOTMwNjZaIiBmaWxsPSIjNjE2MTYxIiBzdHJva2U9IiM2MTYxNjEiIHN0cm9rZS13aWR0aD0iMC43Ii8+CjwvZz4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTExLjUgOS41VjExLjVMMi41IDExLjVWOS41TDExLjUgOS41Wk0xMiA4QzEyLjU1MjMgOCAxMyA4LjQ0NzcyIDEzIDlWMTJDMTMgMTIuNTUyMyAxMi41NTIzIDEzIDEyIDEzTDIgMTNDMS40NDc3MiAxMyAxIDEyLjU1MjMgMSAxMlY5QzEgOC40NDc3MiAxLjQ0NzcxIDggMiA4TDEyIDhaIiBmaWxsPSIjNjE2MTYxIi8+CjxkZWZzPgo8Y2xpcFBhdGggaWQ9ImNsaXAwXzEzN18xOTQ5MiI+CjxyZWN0IGNsYXNzPSJqcC1pY29uMyIgd2lkdGg9IjYiIGhlaWdodD0iNiIgZmlsbD0id2hpdGUiIHRyYW5zZm9ybT0ibWF0cml4KC0xIDAgMCAxIDEwIDEuNTU1NjYpIi8+CjwvY2xpcFBhdGg+CjwvZGVmcz4KPC9zdmc+Cg==);
  --jp-icon-add-below: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzEzN18xOTQ5OCkiPgo8cGF0aCBjbGFzcz0ianAtaWNvbjMiIGQ9Ik05LjI1IDEwLjA2OTNMNy4zNzUgMTAuMDY5M0w3LjM3NSA4LjE5NDM0QzcuMzc1IDcuOTg4MDkgNy4yMDYyNSA3LjgxOTM0IDcgNy44MTkzNEM2Ljc5Mzc1IDcuODE5MzQgNi42MjUgNy45ODgwOSA2LjYyNSA4LjE5NDM0TDYuNjI1IDEwLjA2OTNMNC43NSAxMC4wNjkzQzQuNTQzNzUgMTAuMDY5MyA0LjM3NSAxMC4yMzgxIDQuMzc1IDEwLjQ0NDNDNC4zNzUgMTAuNjUwNiA0LjU0Mzc1IDEwLjgxOTMgNC43NSAxMC44MTkzTDYuNjI1IDEwLjgxOTNMNi42MjUgMTIuNjk0M0M2LjYyNSAxMi45MDA2IDYuNzkzNzUgMTMuMDY5MyA3IDEzLjA2OTNDNy4yMDYyNSAxMy4wNjkzIDcuMzc1IDEyLjkwMDYgNy4zNzUgMTIuNjk0M0w3LjM3NSAxMC44MTkzTDkuMjUgMTAuODE5M0M5LjQ1NjI1IDEwLjgxOTMgOS42MjUgMTAuNjUwNiA5LjYyNSAxMC40NDQzQzkuNjI1IDEwLjIzODEgOS40NTYyNSAxMC4wNjkzIDkuMjUgMTAuMDY5M1oiIGZpbGw9IiM2MTYxNjEiIHN0cm9rZT0iIzYxNjE2MSIgc3Ryb2tlLXdpZHRoPSIwLjciLz4KPC9nPgo8cGF0aCBjbGFzcz0ianAtaWNvbjMiIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMi41IDUuNUwyLjUgMy41TDExLjUgMy41TDExLjUgNS41TDIuNSA1LjVaTTIgN0MxLjQ0NzcyIDcgMSA2LjU1MjI4IDEgNkwxIDNDMSAyLjQ0NzcyIDEuNDQ3NzIgMiAyIDJMMTIgMkMxMi41NTIzIDIgMTMgMi40NDc3MiAxMyAzTDEzIDZDMTMgNi41NTIyOSAxMi41NTIzIDcgMTIgN0wyIDdaIiBmaWxsPSIjNjE2MTYxIi8+CjxkZWZzPgo8Y2xpcFBhdGggaWQ9ImNsaXAwXzEzN18xOTQ5OCI+CjxyZWN0IGNsYXNzPSJqcC1pY29uMyIgd2lkdGg9IjYiIGhlaWdodD0iNiIgZmlsbD0id2hpdGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMS43NDg0NmUtMDcgMS43NDg0NmUtMDcgLTEgNCAxMy40NDQzKSIvPgo8L2NsaXBQYXRoPgo8L2RlZnM+Cjwvc3ZnPgo=);
  --jp-icon-add: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDEzaC02djZoLTJ2LTZINXYtMmg2VjVoMnY2aDZ2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-bell: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE2IDE2IiB2ZXJzaW9uPSIxLjEiPgogICA8cGF0aCBjbGFzcz0ianAtaWNvbjIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMzMzMzMzIgogICAgICBkPSJtOCAwLjI5Yy0xLjQgMC0yLjcgMC43My0zLjYgMS44LTEuMiAxLjUtMS40IDMuNC0xLjUgNS4yLTAuMTggMi4yLTAuNDQgNC0yLjMgNS4zbDAuMjggMS4zaDVjMC4wMjYgMC42NiAwLjMyIDEuMSAwLjcxIDEuNSAwLjg0IDAuNjEgMiAwLjYxIDIuOCAwIDAuNTItMC40IDAuNi0xIDAuNzEtMS41aDVsMC4yOC0xLjNjLTEuOS0wLjk3LTIuMi0zLjMtMi4zLTUuMy0wLjEzLTEuOC0wLjI2LTMuNy0xLjUtNS4yLTAuODUtMS0yLjItMS44LTMuNi0xLjh6bTAgMS40YzAuODggMCAxLjkgMC41NSAyLjUgMS4zIDAuODggMS4xIDEuMSAyLjcgMS4yIDQuNCAwLjEzIDEuNyAwLjIzIDMuNiAxLjMgNS4yaC0xMGMxLjEtMS42IDEuMi0zLjQgMS4zLTUuMiAwLjEzLTEuNyAwLjMtMy4zIDEuMi00LjQgMC41OS0wLjcyIDEuNi0xLjMgMi41LTEuM3ptLTAuNzQgMTJoMS41Yy0wLjAwMTUgMC4yOCAwLjAxNSAwLjc5LTAuNzQgMC43OS0wLjczIDAuMDAxNi0wLjcyLTAuNTMtMC43NC0wLjc5eiIgLz4KPC9zdmc+Cg==);
  --jp-icon-bug-dot: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiPgogICAgICAgIDxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMTcuMTkgOEgyMFYxMEgxNy45MUMxNy45NiAxMC4zMyAxOCAxMC42NiAxOCAxMVYxMkgyMFYxNEgxOC41SDE4VjE0LjAyNzVDMTUuNzUgMTQuMjc2MiAxNCAxNi4xODM3IDE0IDE4LjVDMTQgMTkuMjA4IDE0LjE2MzUgMTkuODc3OSAxNC40NTQ5IDIwLjQ3MzlDMTMuNzA2MyAyMC44MTE3IDEyLjg3NTcgMjEgMTIgMjFDOS43OCAyMSA3Ljg1IDE5Ljc5IDYuODEgMThINFYxNkg2LjA5QzYuMDQgMTUuNjcgNiAxNS4zNCA2IDE1VjE0SDRWMTJINlYxMUM2IDEwLjY2IDYuMDQgMTAuMzMgNi4wOSAxMEg0VjhINi44MUM3LjI2IDcuMjIgNy44OCA2LjU1IDguNjIgNi4wNEw3IDQuNDFMOC40MSAzTDEwLjU5IDUuMTdDMTEuMDQgNS4wNiAxMS41MSA1IDEyIDVDMTIuNDkgNSAxMi45NiA1LjA2IDEzLjQyIDUuMTdMMTUuNTkgM0wxNyA0LjQxTDE1LjM3IDYuMDRDMTYuMTIgNi41NSAxNi43NCA3LjIyIDE3LjE5IDhaTTEwIDE2SDE0VjE0SDEwVjE2Wk0xMCAxMkgxNFYxMEgxMFYxMloiIGZpbGw9IiM2MTYxNjEiLz4KICAgICAgICA8cGF0aCBkPSJNMjIgMTguNUMyMiAyMC40MzMgMjAuNDMzIDIyIDE4LjUgMjJDMTYuNTY3IDIyIDE1IDIwLjQzMyAxNSAxOC41QzE1IDE2LjU2NyAxNi41NjcgMTUgMTguNSAxNUMyMC40MzMgMTUgMjIgMTYuNTY3IDIyIDE4LjVaIiBmaWxsPSIjNjE2MTYxIi8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-bug: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0yMCA4aC0yLjgxYy0uNDUtLjc4LTEuMDctMS40NS0xLjgyLTEuOTZMMTcgNC40MSAxNS41OSAzbC0yLjE3IDIuMTdDMTIuOTYgNS4wNiAxMi40OSA1IDEyIDVjLS40OSAwLS45Ni4wNi0xLjQxLjE3TDguNDEgMyA3IDQuNDFsMS42MiAxLjYzQzcuODggNi41NSA3LjI2IDcuMjIgNi44MSA4SDR2MmgyLjA5Yy0uMDUuMzMtLjA5LjY2LS4wOSAxdjFINHYyaDJ2MWMwIC4zNC4wNC42Ny4wOSAxSDR2MmgyLjgxYzEuMDQgMS43OSAyLjk3IDMgNS4xOSAzczQuMTUtMS4yMSA1LjE5LTNIMjB2LTJoLTIuMDljLjA1LS4zMy4wOS0uNjYuMDktMXYtMWgydi0yaC0ydi0xYzAtLjM0LS4wNC0uNjctLjA5LTFIMjBWOHptLTYgOGgtNHYtMmg0djJ6bTAtNGgtNHYtMmg0djJ6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-build: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE0LjkgMTcuNDVDMTYuMjUgMTcuNDUgMTcuMzUgMTYuMzUgMTcuMzUgMTVDMTcuMzUgMTMuNjUgMTYuMjUgMTIuNTUgMTQuOSAxMi41NUMxMy41NCAxMi41NSAxMi40NSAxMy42NSAxMi40NSAxNUMxMi40NSAxNi4zNSAxMy41NCAxNy40NSAxNC45IDE3LjQ1Wk0yMC4xIDE1LjY4TDIxLjU4IDE2Ljg0QzIxLjcxIDE2Ljk1IDIxLjc1IDE3LjEzIDIxLjY2IDE3LjI5TDIwLjI2IDE5LjcxQzIwLjE3IDE5Ljg2IDIwIDE5LjkyIDE5LjgzIDE5Ljg2TDE4LjA5IDE5LjE2QzE3LjczIDE5LjQ0IDE3LjMzIDE5LjY3IDE2LjkxIDE5Ljg1TDE2LjY0IDIxLjdDMTYuNjIgMjEuODcgMTYuNDcgMjIgMTYuMyAyMkgxMy41QzEzLjMyIDIyIDEzLjE4IDIxLjg3IDEzLjE1IDIxLjdMMTIuODkgMTkuODVDMTIuNDYgMTkuNjcgMTIuMDcgMTkuNDQgMTEuNzEgMTkuMTZMOS45NjAwMiAxOS44NkM5LjgxMDAyIDE5LjkyIDkuNjIwMDIgMTkuODYgOS41NDAwMiAxOS43MUw4LjE0MDAyIDE3LjI5QzguMDUwMDIgMTcuMTMgOC4wOTAwMiAxNi45NSA4LjIyMDAyIDE2Ljg0TDkuNzAwMDIgMTUuNjhMOS42NTAwMSAxNUw5LjcwMDAyIDE0LjMxTDguMjIwMDIgMTMuMTZDOC4wOTAwMiAxMy4wNSA4LjA1MDAyIDEyLjg2IDguMTQwMDIgMTIuNzFMOS41NDAwMiAxMC4yOUM5LjYyMDAyIDEwLjEzIDkuODEwMDIgMTAuMDcgOS45NjAwMiAxMC4xM0wxMS43MSAxMC44NEMxMi4wNyAxMC41NiAxMi40NiAxMC4zMiAxMi44OSAxMC4xNUwxMy4xNSA4LjI4OTk4QzEzLjE4IDguMTI5OTggMTMuMzIgNy45OTk5OCAxMy41IDcuOTk5OThIMTYuM0MxNi40NyA3Ljk5OTk4IDE2LjYyIDguMTI5OTggMTYuNjQgOC4yODk5OEwxNi45MSAxMC4xNUMxNy4zMyAxMC4zMiAxNy43MyAxMC41NiAxOC4wOSAxMC44NEwxOS44MyAxMC4xM0MyMCAxMC4wNyAyMC4xNyAxMC4xMyAyMC4yNiAxMC4yOUwyMS42NiAxMi43MUMyMS43NSAxMi44NiAyMS43MSAxMy4wNSAyMS41OCAxMy4xNkwyMC4xIDE0LjMxTDIwLjE1IDE1TDIwLjEgMTUuNjhaIi8+CiAgICA8cGF0aCBkPSJNNy4zMjk2NiA3LjQ0NDU0QzguMDgzMSA3LjAwOTU0IDguMzM5MzIgNi4wNTMzMiA3LjkwNDMyIDUuMjk5ODhDNy40NjkzMiA0LjU0NjQzIDYuNTA4MSA0LjI4MTU2IDUuNzU0NjYgNC43MTY1NkM1LjM5MTc2IDQuOTI2MDggNS4xMjY5NSA1LjI3MTE4IDUuMDE4NDkgNS42NzU5NEM0LjkxMDA0IDYuMDgwNzEgNC45NjY4MiA2LjUxMTk4IDUuMTc2MzQgNi44NzQ4OEM1LjYxMTM0IDcuNjI4MzIgNi41NzYyMiA3Ljg3OTU0IDcuMzI5NjYgNy40NDQ1NFpNOS42NTcxOCA0Ljc5NTkzTDEwLjg2NzIgNC45NTE3OUMxMC45NjI4IDQuOTc3NDEgMTEuMDQwMiA1LjA3MTMzIDExLjAzODIgNS4xODc5M0wxMS4wMzg4IDYuOTg4OTNDMTEuMDQ1NSA3LjEwMDU0IDEwLjk2MTYgNy4xOTUxOCAxMC44NTUgNy4yMTA1NEw5LjY2MDAxIDcuMzgwODNMOS4yMzkxNSA4LjEzMTg4TDkuNjY5NjEgOS4yNTc0NUM5LjcwNzI5IDkuMzYyNzEgOS42NjkzNCA5LjQ3Njk5IDkuNTc0MDggOS41MzE5OUw4LjAxNTIzIDEwLjQzMkM3LjkxMTMxIDEwLjQ5MiA3Ljc5MzM3IDEwLjQ2NzcgNy43MjEwNSAxMC4zODI0TDYuOTg3NDggOS40MzE4OEw2LjEwOTMxIDkuNDMwODNMNS4zNDcwNCAxMC4zOTA1QzUuMjg5MDkgMTAuNDcwMiA1LjE3MzgzIDEwLjQ5MDUgNS4wNzE4NyAxMC40MzM5TDMuNTEyNDUgOS41MzI5M0MzLjQxMDQ5IDkuNDc2MzMgMy4zNzY0NyA5LjM1NzQxIDMuNDEwNzUgOS4yNTY3OUwzLjg2MzQ3IDguMTQwOTNMMy42MTc0OSA3Ljc3NDg4TDMuNDIzNDcgNy4zNzg4M0wyLjIzMDc1IDcuMjEyOTdDMi4xMjY0NyA3LjE5MjM1IDIuMDQwNDkgNy4xMDM0MiAyLjA0MjQ1IDYuOTg2ODJMMi4wNDE4NyA1LjE4NTgyQzIuMDQzODMgNS4wNjkyMiAyLjExOTA5IDQuOTc5NTggMi4yMTcwNCA0Ljk2OTIyTDMuNDIwNjUgNC43OTM5M0wzLjg2NzQ5IDQuMDI3ODhMMy40MTEwNSAyLjkxNzMxQzMuMzczMzcgMi44MTIwNCAzLjQxMTMxIDIuNjk3NzYgMy41MTUyMyAyLjYzNzc2TDUuMDc0MDggMS43Mzc3NkM1LjE2OTM0IDEuNjgyNzYgNS4yODcyOSAxLjcwNzA0IDUuMzU5NjEgMS43OTIzMUw2LjExOTE1IDIuNzI3ODhMNi45ODAwMSAyLjczODkzTDcuNzI0OTYgMS43ODkyMkM3Ljc5MTU2IDEuNzA0NTggNy45MTU0OCAxLjY3OTIyIDguMDA4NzkgMS43NDA4Mkw5LjU2ODIxIDIuNjQxODJDOS42NzAxNyAyLjY5ODQyIDkuNzEyODUgMi44MTIzNCA5LjY4NzIzIDIuOTA3OTdMOS4yMTcxOCA0LjAzMzgzTDkuNDYzMTYgNC4zOTk4OEw5LjY1NzE4IDQuNzk1OTNaIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iOS45LDEzLjYgMy42LDcuNCA0LjQsNi42IDkuOSwxMi4yIDE1LjQsNi43IDE2LjEsNy40ICIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNS45TDksOS43bDMuOC0zLjhsMS4yLDEuMmwtNC45LDVsLTQuOS01TDUuMiw1Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNy41TDksMTEuMmwzLjgtMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-left: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik0xMC44LDEyLjhMNy4xLDlsMy44LTMuOGwwLDcuNkgxMC44eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-right: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik03LjIsNS4yTDEwLjksOWwtMy44LDMuOFY1LjJINy4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-up-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iMTUuNCwxMy4zIDkuOSw3LjcgNC40LDEzLjIgMy42LDEyLjUgOS45LDYuMyAxNi4xLDEyLjYgIi8+Cgk8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-up: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik01LjIsMTAuNUw5LDYuOGwzLjgsMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-case-sensitive: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWFjY2VudDIiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTcuNiw4aDAuOWwzLjUsOGgtMS4xTDEwLDE0SDZsLTAuOSwySDRMNy42LDh6IE04LDkuMUw2LjQsMTNoMy4yTDgsOS4xeiIvPgogICAgPHBhdGggZD0iTTE2LjYsOS44Yy0wLjIsMC4xLTAuNCwwLjEtMC43LDAuMWMtMC4yLDAtMC40LTAuMS0wLjYtMC4yYy0wLjEtMC4xLTAuMi0wLjQtMC4yLTAuNyBjLTAuMywwLjMtMC42LDAuNS0wLjksMC43Yy0wLjMsMC4xLTAuNywwLjItMS4xLDAuMmMtMC4zLDAtMC41LDAtMC43LTAuMWMtMC4yLTAuMS0wLjQtMC4yLTAuNi0wLjNjLTAuMi0wLjEtMC4zLTAuMy0wLjQtMC41IGMtMC4xLTAuMi0wLjEtMC40LTAuMS0wLjdjMC0wLjMsMC4xLTAuNiwwLjItMC44YzAuMS0wLjIsMC4zLTAuNCwwLjQtMC41QzEyLDcsMTIuMiw2LjksMTIuNSw2LjhjMC4yLTAuMSwwLjUtMC4xLDAuNy0wLjIgYzAuMy0wLjEsMC41LTAuMSwwLjctMC4xYzAuMiwwLDAuNC0wLjEsMC42LTAuMWMwLjIsMCwwLjMtMC4xLDAuNC0wLjJjMC4xLTAuMSwwLjItMC4yLDAuMi0wLjRjMC0xLTEuMS0xLTEuMy0xIGMtMC40LDAtMS40LDAtMS40LDEuMmgtMC45YzAtMC40LDAuMS0wLjcsMC4yLTFjMC4xLTAuMiwwLjMtMC40LDAuNS0wLjZjMC4yLTAuMiwwLjUtMC4zLDAuOC0wLjNDMTMuMyw0LDEzLjYsNCwxMy45LDQgYzAuMywwLDAuNSwwLDAuOCwwLjFjMC4zLDAsMC41LDAuMSwwLjcsMC4yYzAuMiwwLjEsMC40LDAuMywwLjUsMC41QzE2LDUsMTYsNS4yLDE2LDUuNnYyLjljMCwwLjIsMCwwLjQsMCwwLjUgYzAsMC4xLDAuMSwwLjIsMC4zLDAuMmMwLjEsMCwwLjIsMCwwLjMsMFY5Ljh6IE0xNS4yLDYuOWMtMS4yLDAuNi0zLjEsMC4yLTMuMSwxLjRjMCwxLjQsMy4xLDEsMy4xLTAuNVY2Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik05IDE2LjE3TDQuODMgMTJsLTEuNDIgMS40MUw5IDE5IDIxIDdsLTEuNDEtMS40MXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-circle-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDJDNi40NyAyIDIgNi40NyAyIDEyczQuNDcgMTAgMTAgMTAgMTAtNC40NyAxMC0xMFMxNy41MyAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-circle: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iOSIgY3k9IjkiIHI9IjgiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-clear: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8bWFzayBpZD0iZG9udXRIb2xlIj4KICAgIDxyZWN0IHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0id2hpdGUiIC8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSI4IiBmaWxsPSJibGFjayIvPgogIDwvbWFzaz4KCiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxyZWN0IGhlaWdodD0iMTgiIHdpZHRoPSIyIiB4PSIxMSIgeT0iMyIgdHJhbnNmb3JtPSJyb3RhdGUoMzE1LCAxMiwgMTIpIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgbWFzaz0idXJsKCNkb251dEhvbGUpIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-close: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1ub25lIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIGpwLWljb24zLWhvdmVyIiBmaWxsPSJub25lIj4KICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjExIi8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIGpwLWljb24tYWNjZW50Mi1ob3ZlciIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMTkgNi40MUwxNy41OSA1IDEyIDEwLjU5IDYuNDEgNSA1IDYuNDEgMTAuNTkgMTIgNSAxNy41OSA2LjQxIDE5IDEyIDEzLjQxIDE3LjU5IDE5IDE5IDE3LjU5IDEzLjQxIDEyeiIvPgogIDwvZz4KCiAgPGcgY2xhc3M9ImpwLWljb24tbm9uZSBqcC1pY29uLWJ1c3kiIGZpbGw9Im5vbmUiPgogICAgPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-code-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBzaGFwZS1yZW5kZXJpbmc9Imdlb21ldHJpY1ByZWNpc2lvbiI+CiAgICA8cGF0aCBkPSJNNi41OSwzLjQxTDIsOEw2LjU5LDEyLjZMOCwxMS4xOEw0LjgyLDhMOCw0LjgyTDYuNTksMy40MU0xMi40MSwzLjQxTDExLDQuODJMMTQuMTgsOEwxMSwxMS4xOEwxMi40MSwxMi42TDE3LDhMMTIuNDEsMy40MU0yMS41OSwxMS41OUwxMy41LDE5LjY4TDkuODMsMTZMOC40MiwxNy40MUwxMy41LDIyLjVMMjMsMTNMMjEuNTksMTEuNTlaIiAvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-code: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTExLjQgMTguNkw2LjggMTRMMTEuNCA5LjRMMTAgOEw0IDE0TDEwIDIwTDExLjQgMTguNlpNMTYuNiAxOC42TDIxLjIgMTRMMTYuNiA5LjRMMTggOEwyNCAxNEwxOCAyMEwxNi42IDE4LjZWMTguNloiLz4KCTwvZz4KPC9zdmc+Cg==);
  --jp-icon-collapse-all: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTggMmMxIDAgMTEgMCAxMiAwczIgMSAyIDJjMCAxIDAgMTEgMCAxMnMwIDItMiAyQzIwIDE0IDIwIDQgMjAgNFMxMCA0IDYgNGMwLTIgMS0yIDItMnoiIC8+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTE4IDhjMC0xLTEtMi0yLTJTNSA2IDQgNnMtMiAxLTIgMmMwIDEgMCAxMSAwIDEyczEgMiAyIDJjMSAwIDExIDAgMTIgMHMyLTEgMi0yYzAtMSAwLTExIDAtMTJ6bS0yIDB2MTJINFY4eiIgLz4KICAgICAgICA8cGF0aCBkPSJNNiAxM3YyaDh2LTJ6IiAvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-console: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwMCAyMDAiPgogIDxnIGNsYXNzPSJqcC1jb25zb2xlLWljb24tYmFja2dyb3VuZC1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMjg4RDEiPgogICAgPHBhdGggZD0iTTIwIDE5LjhoMTYwdjE1OS45SDIweiIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtY29uc29sZS1pY29uLWNvbG9yIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIj4KICAgIDxwYXRoIGQ9Ik0xMDUgMTI3LjNoNDB2MTIuOGgtNDB6TTUxLjEgNzdMNzQgOTkuOWwtMjMuMyAyMy4zIDEwLjUgMTAuNSAyMy4zLTIzLjNMOTUgOTkuOSA4NC41IDg5LjQgNjEuNiA2Ni41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-copy: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTExLjksMUgzLjJDMi40LDEsMS43LDEuNywxLjcsMi41djEwLjJoMS41VjIuNWg4LjdWMXogTTE0LjEsMy45aC04Yy0wLjgsMC0xLjUsMC43LTEuNSwxLjV2MTAuMmMwLDAuOCwwLjcsMS41LDEuNSwxLjVoOCBjMC44LDAsMS41LTAuNywxLjUtMS41VjUuNEMxNS41LDQuNiwxNC45LDMuOSwxNC4xLDMuOXogTTE0LjEsMTUuNWgtOFY1LjRoOFYxNS41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-copyright: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGVuYWJsZS1iYWNrZ3JvdW5kPSJuZXcgMCAwIDI0IDI0IiBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCI+CiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0xMS44OCw5LjE0YzEuMjgsMC4wNiwxLjYxLDEuMTUsMS42MywxLjY2aDEuNzljLTAuMDgtMS45OC0xLjQ5LTMuMTktMy40NS0zLjE5QzkuNjQsNy42MSw4LDksOCwxMi4xNCBjMCwxLjk0LDAuOTMsNC4yNCwzLjg0LDQuMjRjMi4yMiwwLDMuNDEtMS42NSwzLjQ0LTIuOTVoLTEuNzljLTAuMDMsMC41OS0wLjQ1LDEuMzgtMS42MywxLjQ0QzEwLjU1LDE0LjgzLDEwLDEzLjgxLDEwLDEyLjE0IEMxMCw5LjI1LDExLjI4LDkuMTYsMTEuODgsOS4xNHogTTEyLDJDNi40OCwyLDIsNi40OCwyLDEyczQuNDgsMTAsMTAsMTBzMTAtNC40OCwxMC0xMFMxNy41MiwyLDEyLDJ6IE0xMiwyMGMtNC40MSwwLTgtMy41OS04LTggczMuNTktOCw4LThzOCwzLjU5LDgsOFMxNi40MSwyMCwxMiwyMHoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-cut: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkuNjQgNy42NGMuMjMtLjUuMzYtMS4wNS4zNi0xLjY0IDAtMi4yMS0xLjc5LTQtNC00UzIgMy43OSAyIDZzMS43OSA0IDQgNGMuNTkgMCAxLjE0LS4xMyAxLjY0LS4zNkwxMCAxMmwtMi4zNiAyLjM2QzcuMTQgMTQuMTMgNi41OSAxNCA2IDE0Yy0yLjIxIDAtNCAxLjc5LTQgNHMxLjc5IDQgNCA0IDQtMS43OSA0LTRjMC0uNTktLjEzLTEuMTQtLjM2LTEuNjRMMTIgMTRsNyA3aDN2LTFMOS42NCA3LjY0ek02IDhjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTAgMTJjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTYtNy41Yy0uMjggMC0uNS0uMjItLjUtLjVzLjIyLS41LjUtLjUuNS4yMi41LjUtLjIyLjUtLjUuNXpNMTkgM2wtNiA2IDIgMiA3LTdWM3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-delete: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2cHgiIGhlaWdodD0iMTZweCI+CiAgICA8cGF0aCBkPSJNMCAwaDI0djI0SDB6IiBmaWxsPSJub25lIiAvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjI2MjYyIiBkPSJNNiAxOWMwIDEuMS45IDIgMiAyaDhjMS4xIDAgMi0uOSAyLTJWN0g2djEyek0xOSA0aC0zLjVsLTEtMWgtNWwtMSAxSDV2MmgxNFY0eiIgLz4KPC9zdmc+Cg==);
  --jp-icon-download: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDloLTRWM0g5djZINWw3IDcgNy03ek01IDE4djJoMTR2LTJINXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-duplicate: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTIuNzk5OTggMC44NzVIOC44OTU4MkM5LjIwMDYxIDAuODc1IDkuNDQ5OTggMS4xMzkxNCA5LjQ0OTk4IDEuNDYxOThDOS40NDk5OCAxLjc4NDgyIDkuMjAwNjEgMi4wNDg5NiA4Ljg5NTgyIDIuMDQ4OTZIMy4zNTQxNUMzLjA0OTM2IDIuMDQ4OTYgMi43OTk5OCAyLjMxMzEgMi43OTk5OCAyLjYzNTk0VjkuNjc5NjlDMi43OTk5OCAxMC4wMDI1IDIuNTUwNjEgMTAuMjY2NyAyLjI0NTgyIDEwLjI2NjdDMS45NDEwMyAxMC4yNjY3IDEuNjkxNjUgMTAuMDAyNSAxLjY5MTY1IDkuNjc5NjlWMi4wNDg5NkMxLjY5MTY1IDEuNDAzMjggMi4xOTA0IDAuODc1IDIuNzk5OTggMC44NzVaTTUuMzY2NjUgMTEuOVY0LjU1SDExLjA4MzNWMTEuOUg1LjM2NjY1Wk00LjE0MTY1IDQuMTQxNjdDNC4xNDE2NSAzLjY5MDYzIDQuNTA3MjggMy4zMjUgNC45NTgzMiAzLjMyNUgxMS40OTE3QzExLjk0MjcgMy4zMjUgMTIuMzA4MyAzLjY5MDYzIDEyLjMwODMgNC4xNDE2N1YxMi4zMDgzQzEyLjMwODMgMTIuNzU5NCAxMS45NDI3IDEzLjEyNSAxMS40OTE3IDEzLjEyNUg0Ljk1ODMyQzQuNTA3MjggMTMuMTI1IDQuMTQxNjUgMTIuNzU5NCA0LjE0MTY1IDEyLjMwODNWNC4xNDE2N1oiIGZpbGw9IiM2MTYxNjEiLz4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBkPSJNOS40MzU3NCA4LjI2NTA3SDguMzY0MzFWOS4zMzY1QzguMzY0MzEgOS40NTQzNSA4LjI2Nzg4IDkuNTUwNzggOC4xNTAwMiA5LjU1MDc4QzguMDMyMTcgOS41NTA3OCA3LjkzNTc0IDkuNDU0MzUgNy45MzU3NCA5LjMzNjVWOC4yNjUwN0g2Ljg2NDMxQzYuNzQ2NDUgOC4yNjUwNyA2LjY1MDAyIDguMTY4NjQgNi42NTAwMiA4LjA1MDc4QzYuNjUwMDIgNy45MzI5MiA2Ljc0NjQ1IDcuODM2NSA2Ljg2NDMxIDcuODM2NUg3LjkzNTc0VjYuNzY1MDdDNy45MzU3NCA2LjY0NzIxIDguMDMyMTcgNi41NTA3OCA4LjE1MDAyIDYuNTUwNzhDOC4yNjc4OCA2LjU1MDc4IDguMzY0MzEgNi42NDcyMSA4LjM2NDMxIDYuNzY1MDdWNy44MzY1SDkuNDM1NzRDOS41NTM2IDcuODM2NSA5LjY1MDAyIDcuOTMyOTIgOS42NTAwMiA4LjA1MDc4QzkuNjUwMDIgOC4xNjg2NCA5LjU1MzYgOC4yNjUwNyA5LjQzNTc0IDguMjY1MDdaIiBmaWxsPSIjNjE2MTYxIiBzdHJva2U9IiM2MTYxNjEiIHN0cm9rZS13aWR0aD0iMC41Ii8+Cjwvc3ZnPgo=);
  --jp-icon-edit: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMgMTcuMjVWMjFoMy43NUwxNy44MSA5Ljk0bC0zLjc1LTMuNzVMMyAxNy4yNXpNMjAuNzEgNy4wNGMuMzktLjM5LjM5LTEuMDIgMC0xLjQxbC0yLjM0LTIuMzRjLS4zOS0uMzktMS4wMi0uMzktMS40MSAwbC0xLjgzIDEuODMgMy43NSAzLjc1IDEuODMtMS44M3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-ellipses: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iNSIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxOSIgY3k9IjEyIiByPSIyIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-error: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjE5IiByPSIyIi8+PHBhdGggZD0iTTEwIDNoNHYxMmgtNHoiLz48L2c+CjxwYXRoIGZpbGw9Im5vbmUiIGQ9Ik0wIDBoMjR2MjRIMHoiLz4KPC9zdmc+Cg==);
  --jp-icon-expand-all: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTggMmMxIDAgMTEgMCAxMiAwczIgMSAyIDJjMCAxIDAgMTEgMCAxMnMwIDItMiAyQzIwIDE0IDIwIDQgMjAgNFMxMCA0IDYgNGMwLTIgMS0yIDItMnoiIC8+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTE4IDhjMC0xLTEtMi0yLTJTNSA2IDQgNnMtMiAxLTIgMmMwIDEgMCAxMSAwIDEyczEgMiAyIDJjMSAwIDExIDAgMTIgMHMyLTEgMi0yYzAtMSAwLTExIDAtMTJ6bS0yIDB2MTJINFY4eiIgLz4KICAgICAgICA8cGF0aCBkPSJNMTEgMTBIOXYzSDZ2MmgzdjNoMnYtM2gzdi0yaC0zeiIgLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-extension: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwLjUgMTFIMTlWN2MwLTEuMS0uOS0yLTItMmgtNFYzLjVDMTMgMi4xMiAxMS44OCAxIDEwLjUgMVM4IDIuMTIgOCAzLjVWNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAydjMuOEgzLjVjMS40OSAwIDIuNyAxLjIxIDIuNyAyLjdzLTEuMjEgMi43LTIuNyAyLjdIMlYyMGMwIDEuMS45IDIgMiAyaDMuOHYtMS41YzAtMS40OSAxLjIxLTIuNyAyLjctMi43IDEuNDkgMCAyLjcgMS4yMSAyLjcgMi43VjIySDE3YzEuMSAwIDItLjkgMi0ydi00aDEuNWMxLjM4IDAgMi41LTEuMTIgMi41LTIuNVMyMS44OCAxMSAyMC41IDExeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-fast-forward: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTQgMThsOC41LTZMNCA2djEyem05LTEydjEybDguNS02TDEzIDZ6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-file-upload: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkgMTZoNnYtNmg0bC03LTctNyA3aDR6bS00IDJoMTR2Mkg1eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-file: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuMyA4LjJsLTUuNS01LjVjLS4zLS4zLS43LS41LTEuMi0uNUgzLjljLS44LjEtMS42LjktMS42IDEuOHYxNC4xYzAgLjkuNyAxLjYgMS42IDEuNmgxNC4yYy45IDAgMS42LS43IDEuNi0xLjZWOS40Yy4xLS41LS4xLS45LS40LTEuMnptLTUuOC0zLjNsMy40IDMuNmgtMy40VjQuOXptMy45IDEyLjdINC43Yy0uMSAwLS4yIDAtLjItLjJWNC43YzAtLjIuMS0uMy4yLS4zaDcuMnY0LjRzMCAuOC4zIDEuMWMuMy4zIDEuMS4zIDEuMS4zaDQuM3Y3LjJzLS4xLjItLjIuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-filter-dot: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTE0LDEyVjE5Ljg4QzE0LjA0LDIwLjE4IDEzLjk0LDIwLjUgMTMuNzEsMjAuNzFDMTMuMzIsMjEuMSAxMi42OSwyMS4xIDEyLjMsMjAuNzFMMTAuMjksMTguN0MxMC4wNiwxOC40NyA5Ljk2LDE4LjE2IDEwLDE3Ljg3VjEySDkuOTdMNC4yMSw0LjYyQzMuODcsNC4xOSAzLjk1LDMuNTYgNC4zOCwzLjIyQzQuNTcsMy4wOCA0Ljc4LDMgNSwzVjNIMTlWM0MxOS4yMiwzIDE5LjQzLDMuMDggMTkuNjIsMy4yMkMyMC4wNSwzLjU2IDIwLjEzLDQuMTkgMTkuNzksNC42MkwxNC4wMywxMkgxNFoiIC8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWRvdCIgZmlsbD0iI0ZGRiI+CiAgICA8Y2lyY2xlIGN4PSIxOCIgY3k9IjE3IiByPSIzIj48L2NpcmNsZT4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-filter-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEwIDE4aDR2LTJoLTR2MnpNMyA2djJoMThWNkgzem0zIDdoMTJ2LTJINnYyeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-filter: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTE0LDEyVjE5Ljg4QzE0LjA0LDIwLjE4IDEzLjk0LDIwLjUgMTMuNzEsMjAuNzFDMTMuMzIsMjEuMSAxMi42OSwyMS4xIDEyLjMsMjAuNzFMMTAuMjksMTguN0MxMC4wNiwxOC40NyA5Ljk2LDE4LjE2IDEwLDE3Ljg3VjEySDkuOTdMNC4yMSw0LjYyQzMuODcsNC4xOSAzLjk1LDMuNTYgNC4zOCwzLjIyQzQuNTcsMy4wOCA0Ljc4LDMgNSwzVjNIMTlWM0MxOS4yMiwzIDE5LjQzLDMuMDggMTkuNjIsMy4yMkMyMC4wNSwzLjU2IDIwLjEzLDQuMTkgMTkuNzksNC42MkwxNC4wMywxMkgxNFoiIC8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-folder-favorite: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAwIDI0IDI0IiB3aWR0aD0iMjRweCIgZmlsbD0iIzAwMDAwMCI+CiAgPHBhdGggZD0iTTAgMGgyNHYyNEgwVjB6IiBmaWxsPSJub25lIi8+PHBhdGggY2xhc3M9ImpwLWljb24zIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxNjE2MSIgZD0iTTIwIDZoLThsLTItMkg0Yy0xLjEgMC0yIC45LTIgMnYxMmMwIDEuMS45IDIgMiAyaDE2YzEuMSAwIDItLjkgMi0yVjhjMC0xLjEtLjktMi0yLTJ6bS0yLjA2IDExTDE1IDE1LjI4IDEyLjA2IDE3bC43OC0zLjMzLTIuNTktMi4yNCAzLjQxLS4yOUwxNSA4bDEuMzQgMy4xNCAzLjQxLjI5LTIuNTkgMi4yNC43OCAzLjMzeiIvPgo8L3N2Zz4K);
  --jp-icon-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY4YzAtMS4xLS45LTItMi0yaC04bC0yLTJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-home: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAwIDI0IDI0IiB3aWR0aD0iMjRweCIgZmlsbD0iIzAwMDAwMCI+CiAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGNsYXNzPSJqcC1pY29uMyBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xMCAyMHYtNmg0djZoNXYtOGgzTDEyIDMgMiAxMmgzdjh6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-html5: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMDAiIGQ9Ik0xMDguNCAwaDIzdjIyLjhoMjEuMlYwaDIzdjY5aC0yM1Y0NmgtMjF2MjNoLTIzLjJNMjA2IDIzaC0yMC4zVjBoNjMuN3YyM0gyMjl2NDZoLTIzbTUzLjUtNjloMjQuMWwxNC44IDI0LjNMMzEzLjIgMGgyNC4xdjY5aC0yM1YzNC44bC0xNi4xIDI0LjgtMTYuMS0yNC44VjY5aC0yMi42bTg5LjItNjloMjN2NDYuMmgzMi42VjY5aC01NS42Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2U0NGQyNiIgZD0iTTEwNy42IDQ3MWwtMzMtMzcwLjRoMzYyLjhsLTMzIDM3MC4yTDI1NS43IDUxMiIvPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNmMTY1MjkiIGQ9Ik0yNTYgNDgwLjVWMTMxaDE0OC4zTDM3NiA0NDciLz4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNlYmViZWIiIGQ9Ik0xNDIgMTc2LjNoMTE0djQ1LjRoLTY0LjJsNC4yIDQ2LjVoNjB2NDUuM0gxNTQuNG0yIDIyLjhIMjAybDMuMiAzNi4zIDUwLjggMTMuNnY0Ny40bC05My4yLTI2Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIiBkPSJNMzY5LjYgMTc2LjNIMjU1Ljh2NDUuNGgxMDkuNm0tNC4xIDQ2LjVIMjU1Ljh2NDUuNGg1NmwtNS4zIDU5LTUwLjcgMTMuNnY0Ny4ybDkzLTI1LjgiLz4KPC9zdmc+Cg==);
  --jp-icon-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1icmFuZDQganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNGRkYiIGQ9Ik0yLjIgMi4yaDE3LjV2MTcuNUgyLjJ6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzNGNTFCNSIgZD0iTTIuMiAyLjJ2MTcuNWgxNy41bC4xLTE3LjVIMi4yem0xMi4xIDIuMmMxLjIgMCAyLjIgMSAyLjIgMi4ycy0xIDIuMi0yLjIgMi4yLTIuMi0xLTIuMi0yLjIgMS0yLjIgMi4yLTIuMnpNNC40IDE3LjZsMy4zLTguOCAzLjMgNi42IDIuMi0zLjIgNC40IDUuNEg0LjR6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-info: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUwLjk3OCA1MC45NzgiPgoJPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KCQk8cGF0aCBkPSJNNDMuNTIsNy40NThDMzguNzExLDIuNjQ4LDMyLjMwNywwLDI1LjQ4OSwwQzE4LjY3LDAsMTIuMjY2LDIuNjQ4LDcuNDU4LDcuNDU4CgkJCWMtOS45NDMsOS45NDEtOS45NDMsMjYuMTE5LDAsMzYuMDYyYzQuODA5LDQuODA5LDExLjIxMiw3LjQ1NiwxOC4wMzEsNy40NThjMCwwLDAuMDAxLDAsMC4wMDIsMAoJCQljNi44MTYsMCwxMy4yMjEtMi42NDgsMTguMDI5LTcuNDU4YzQuODA5LTQuODA5LDcuNDU3LTExLjIxMiw3LjQ1Ny0xOC4wM0M1MC45NzcsMTguNjcsNDguMzI4LDEyLjI2Niw0My41Miw3LjQ1OHoKCQkJIE00Mi4xMDYsNDIuMTA1Yy00LjQzMiw0LjQzMS0xMC4zMzIsNi44NzItMTYuNjE1LDYuODcyaC0wLjAwMmMtNi4yODUtMC4wMDEtMTIuMTg3LTIuNDQxLTE2LjYxNy02Ljg3MgoJCQljLTkuMTYyLTkuMTYzLTkuMTYyLTI0LjA3MSwwLTMzLjIzM0MxMy4zMDMsNC40NCwxOS4yMDQsMiwyNS40ODksMmM2LjI4NCwwLDEyLjE4NiwyLjQ0LDE2LjYxNyw2Ljg3MgoJCQljNC40MzEsNC40MzEsNi44NzEsMTAuMzMyLDYuODcxLDE2LjYxN0M0OC45NzcsMzEuNzcyLDQ2LjUzNiwzNy42NzUsNDIuMTA2LDQyLjEwNXoiLz4KCQk8cGF0aCBkPSJNMjMuNTc4LDMyLjIxOGMtMC4wMjMtMS43MzQsMC4xNDMtMy4wNTksMC40OTYtMy45NzJjMC4zNTMtMC45MTMsMS4xMS0xLjk5NywyLjI3Mi0zLjI1MwoJCQljMC40NjgtMC41MzYsMC45MjMtMS4wNjIsMS4zNjctMS41NzVjMC42MjYtMC43NTMsMS4xMDQtMS40NzgsMS40MzYtMi4xNzVjMC4zMzEtMC43MDcsMC40OTUtMS41NDEsMC40OTUtMi41CgkJCWMwLTEuMDk2LTAuMjYtMi4wODgtMC43NzktMi45NzljLTAuNTY1LTAuODc5LTEuNTAxLTEuMzM2LTIuODA2LTEuMzY5Yy0xLjgwMiwwLjA1Ny0yLjk4NSwwLjY2Ny0zLjU1LDEuODMyCgkJCWMtMC4zMDEsMC41MzUtMC41MDMsMS4xNDEtMC42MDcsMS44MTRjLTAuMTM5LDAuNzA3LTAuMjA3LDEuNDMyLTAuMjA3LDIuMTc0aC0yLjkzN2MtMC4wOTEtMi4yMDgsMC40MDctNC4xMTQsMS40OTMtNS43MTkKCQkJYzEuMDYyLTEuNjQsMi44NTUtMi40ODEsNS4zNzgtMi41MjdjMi4xNiwwLjAyMywzLjg3NCwwLjYwOCw1LjE0MSwxLjc1OGMxLjI3OCwxLjE2LDEuOTI5LDIuNzY0LDEuOTUsNC44MTEKCQkJYzAsMS4xNDItMC4xMzcsMi4xMTEtMC40MSwyLjkxMWMtMC4zMDksMC44NDUtMC43MzEsMS41OTMtMS4yNjgsMi4yNDNjLTAuNDkyLDAuNjUtMS4wNjgsMS4zMTgtMS43MywyLjAwMgoJCQljLTAuNjUsMC42OTctMS4zMTMsMS40NzktMS45ODcsMi4zNDZjLTAuMjM5LDAuMzc3LTAuNDI5LDAuNzc3LTAuNTY1LDEuMTk5Yy0wLjE2LDAuOTU5LTAuMjE3LDEuOTUxLTAuMTcxLDIuOTc5CgkJCUMyNi41ODksMzIuMjE4LDIzLjU3OCwzMi4yMTgsMjMuNTc4LDMyLjIxOHogTTIzLjU3OCwzOC4yMnYtMy40ODRoMy4wNzZ2My40ODRIMjMuNTc4eiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-inspector: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaW5zcGVjdG9yLWljb24tY29sb3IganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY2YzAtMS4xLS45LTItMi0yem0tNSAxNEg0di00aDExdjR6bTAtNUg0VjloMTF2NHptNSA1aC00VjloNHY5eiIvPgo8L3N2Zz4K);
  --jp-icon-json: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtanNvbi1pY29uLWNvbG9yIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI0Y5QTgyNSI+CiAgICA8cGF0aCBkPSJNMjAuMiAxMS44Yy0xLjYgMC0xLjcuNS0xLjcgMSAwIC40LjEuOS4xIDEuMy4xLjUuMS45LjEgMS4zIDAgMS43LTEuNCAyLjMtMy41IDIuM2gtLjl2LTEuOWguNWMxLjEgMCAxLjQgMCAxLjQtLjggMC0uMyAwLS42LS4xLTEgMC0uNC0uMS0uOC0uMS0xLjIgMC0xLjMgMC0xLjggMS4zLTItMS4zLS4yLTEuMy0uNy0xLjMtMiAwLS40LjEtLjguMS0xLjIuMS0uNC4xLS43LjEtMSAwLS44LS40LS43LTEuNC0uOGgtLjVWNC4xaC45YzIuMiAwIDMuNS43IDMuNSAyLjMgMCAuNC0uMS45LS4xIDEuMy0uMS41LS4xLjktLjEgMS4zIDAgLjUuMiAxIDEuNyAxdjEuOHpNMS44IDEwLjFjMS42IDAgMS43LS41IDEuNy0xIDAtLjQtLjEtLjktLjEtMS4zLS4xLS41LS4xLS45LS4xLTEuMyAwLTEuNiAxLjQtMi4zIDMuNS0yLjNoLjl2MS45aC0uNWMtMSAwLTEuNCAwLTEuNC44IDAgLjMgMCAuNi4xIDEgMCAuMi4xLjYuMSAxIDAgMS4zIDAgMS44LTEuMyAyQzYgMTEuMiA2IDExLjcgNiAxM2MwIC40LS4xLjgtLjEgMS4yLS4xLjMtLjEuNy0uMSAxIDAgLjguMy44IDEuNC44aC41djEuOWgtLjljLTIuMSAwLTMuNS0uNi0zLjUtMi4zIDAtLjQuMS0uOS4xLTEuMy4xLS41LjEtLjkuMS0xLjMgMC0uNS0uMi0xLTEuNy0xdi0xLjl6Ii8+CiAgICA8Y2lyY2xlIGN4PSIxMSIgY3k9IjEzLjgiIHI9IjIuMSIvPgogICAgPGNpcmNsZSBjeD0iMTEiIGN5PSI4LjIiIHI9IjIuMSIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-julia: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDMyNSAzMDAiPgogIDxnIGNsYXNzPSJqcC1icmFuZDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjY2IzYzMzIj4KICAgIDxwYXRoIGQ9Ik0gMTUwLjg5ODQzOCAyMjUgQyAxNTAuODk4NDM4IDI2Ni40MjE4NzUgMTE3LjMyMDMxMiAzMDAgNzUuODk4NDM4IDMwMCBDIDM0LjQ3NjU2MiAzMDAgMC44OTg0MzggMjY2LjQyMTg3NSAwLjg5ODQzOCAyMjUgQyAwLjg5ODQzOCAxODMuNTc4MTI1IDM0LjQ3NjU2MiAxNTAgNzUuODk4NDM4IDE1MCBDIDExNy4zMjAzMTIgMTUwIDE1MC44OTg0MzggMTgzLjU3ODEyNSAxNTAuODk4NDM4IDIyNSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzM4OTgyNiI+CiAgICA8cGF0aCBkPSJNIDIzNy41IDc1IEMgMjM3LjUgMTE2LjQyMTg3NSAyMDMuOTIxODc1IDE1MCAxNjIuNSAxNTAgQyAxMjEuMDc4MTI1IDE1MCA4Ny41IDExNi40MjE4NzUgODcuNSA3NSBDIDg3LjUgMzMuNTc4MTI1IDEyMS4wNzgxMjUgMCAxNjIuNSAwIEMgMjAzLjkyMTg3NSAwIDIzNy41IDMzLjU3ODEyNSAyMzcuNSA3NSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzk1NThiMiI+CiAgICA8cGF0aCBkPSJNIDMyNC4xMDE1NjIgMjI1IEMgMzI0LjEwMTU2MiAyNjYuNDIxODc1IDI5MC41MjM0MzggMzAwIDI0OS4xMDE1NjIgMzAwIEMgMjA3LjY3OTY4OCAzMDAgMTc0LjEwMTU2MiAyNjYuNDIxODc1IDE3NC4xMDE1NjIgMjI1IEMgMTc0LjEwMTU2MiAxODMuNTc4MTI1IDIwNy42Nzk2ODggMTUwIDI0OS4xMDE1NjIgMTUwIEMgMjkwLjUyMzQzOCAxNTAgMzI0LjEwMTU2MiAxODMuNTc4MTI1IDMyNC4xMDE1NjIgMjI1Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-jupyter-favicon: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUyIiBoZWlnaHQ9IjE2NSIgdmlld0JveD0iMCAwIDE1MiAxNjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgPGcgY2xhc3M9ImpwLWp1cHl0ZXItaWNvbi1jb2xvciIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA3ODk0NywgMTEwLjU4MjkyNykiIGQ9Ik03NS45NDIyODQyLDI5LjU4MDQ1NjEgQzQzLjMwMjM5NDcsMjkuNTgwNDU2MSAxNC43OTY3ODMyLDE3LjY1MzQ2MzQgMCwwIEM1LjUxMDgzMjExLDE1Ljg0MDY4MjkgMTUuNzgxNTM4OSwyOS41NjY3NzMyIDI5LjM5MDQ5NDcsMzkuMjc4NDE3MSBDNDIuOTk5Nyw0OC45ODk4NTM3IDU5LjI3MzcsNTQuMjA2NzgwNSA3NS45NjA1Nzg5LDU0LjIwNjc4MDUgQzkyLjY0NzQ1NzksNTQuMjA2NzgwNSAxMDguOTIxNDU4LDQ4Ljk4OTg1MzcgMTIyLjUzMDY2MywzOS4yNzg0MTcxIEMxMzYuMTM5NDUzLDI5LjU2Njc3MzIgMTQ2LjQxMDI4NCwxNS44NDA2ODI5IDE1MS45MjExNTgsMCBDMTM3LjA4Nzg2OCwxNy42NTM0NjM0IDEwOC41ODI1ODksMjkuNTgwNDU2MSA3NS45NDIyODQyLDI5LjU4MDQ1NjEgTDc1Ljk0MjI4NDIsMjkuNTgwNDU2MSBaIiAvPgogICAgPHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMzczNjgsIDAuNzA0ODc4KSIgZD0iTTc1Ljk3ODQ1NzksMjQuNjI2NDA3MyBDMTA4LjYxODc2MywyNC42MjY0MDczIDEzNy4xMjQ0NTgsMzYuNTUzNDQxNSAxNTEuOTIxMTU4LDU0LjIwNjc4MDUgQzE0Ni40MTAyODQsMzguMzY2MjIyIDEzNi4xMzk0NTMsMjQuNjQwMTMxNyAxMjIuNTMwNjYzLDE0LjkyODQ4NzggQzEwOC45MjE0NTgsNS4yMTY4NDM5IDkyLjY0NzQ1NzksMCA3NS45NjA1Nzg5LDAgQzU5LjI3MzcsMCA0Mi45OTk3LDUuMjE2ODQzOSAyOS4zOTA0OTQ3LDE0LjkyODQ4NzggQzE1Ljc4MTUzODksMjQuNjQwMTMxNyA1LjUxMDgzMjExLDM4LjM2NjIyMiAwLDU0LjIwNjc4MDUgQzE0LjgzMzA4MTYsMzYuNTg5OTI5MyA0My4zMzg1Njg0LDI0LjYyNjQwNzMgNzUuOTc4NDU3OSwyNC42MjY0MDczIEw3NS45Nzg0NTc5LDI0LjYyNjQwNzMgWiIgLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-jupyter: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzkiIGhlaWdodD0iNTEiIHZpZXdCb3g9IjAgMCAzOSA1MSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtMTYzOCAtMjI4MSkiPgogICAgIDxnIGNsYXNzPSJqcC1qdXB5dGVyLWljb24tY29sb3IiIGZpbGw9IiNGMzc3MjYiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5Ljc0IDIzMTEuOTgpIiBkPSJNIDE4LjI2NDYgNy4xMzQxMUMgMTAuNDE0NSA3LjEzNDExIDMuNTU4NzIgNC4yNTc2IDAgMEMgMS4zMjUzOSAzLjgyMDQgMy43OTU1NiA3LjEzMDgxIDcuMDY4NiA5LjQ3MzAzQyAxMC4zNDE3IDExLjgxNTIgMTQuMjU1NyAxMy4wNzM0IDE4LjI2OSAxMy4wNzM0QyAyMi4yODIzIDEzLjA3MzQgMjYuMTk2MyAxMS44MTUyIDI5LjQ2OTQgOS40NzMwM0MgMzIuNzQyNCA3LjEzMDgxIDM1LjIxMjYgMy44MjA0IDM2LjUzOCAwQyAzMi45NzA1IDQuMjU3NiAyNi4xMTQ4IDcuMTM0MTEgMTguMjY0NiA3LjEzNDExWiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5LjczIDIyODUuNDgpIiBkPSJNIDE4LjI3MzMgNS45MzkzMUMgMjYuMTIzNSA1LjkzOTMxIDMyLjk3OTMgOC44MTU4MyAzNi41MzggMTMuMDczNEMgMzUuMjEyNiA5LjI1MzAzIDMyLjc0MjQgNS45NDI2MiAyOS40Njk0IDMuNjAwNEMgMjYuMTk2MyAxLjI1ODE4IDIyLjI4MjMgMCAxOC4yNjkgMEMgMTQuMjU1NyAwIDEwLjM0MTcgMS4yNTgxOCA3LjA2ODYgMy42MDA0QyAzLjc5NTU2IDUuOTQyNjIgMS4zMjUzOSA5LjI1MzAzIDAgMTMuMDczNEMgMy41Njc0NSA4LjgyNDYzIDEwLjQyMzIgNS45MzkzMSAxOC4yNzMzIDUuOTM5MzFaIi8+CiAgICA8L2c+CiAgICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjY5LjMgMjI4MS4zMSkiIGQ9Ik0gNS44OTM1MyAyLjg0NEMgNS45MTg4OSAzLjQzMTY1IDUuNzcwODUgNC4wMTM2NyA1LjQ2ODE1IDQuNTE2NDVDIDUuMTY1NDUgNS4wMTkyMiA0LjcyMTY4IDUuNDIwMTUgNC4xOTI5OSA1LjY2ODUxQyAzLjY2NDMgNS45MTY4OCAzLjA3NDQ0IDYuMDAxNTEgMi40OTgwNSA1LjkxMTcxQyAxLjkyMTY2IDUuODIxOSAxLjM4NDYzIDUuNTYxNyAwLjk1NDg5OCA1LjE2NDAxQyAwLjUyNTE3IDQuNzY2MzMgMC4yMjIwNTYgNC4yNDkwMyAwLjA4MzkwMzcgMy42Nzc1N0MgLTAuMDU0MjQ4MyAzLjEwNjExIC0wLjAyMTIzIDIuNTA2MTcgMC4xNzg3ODEgMS45NTM2NEMgMC4zNzg3OTMgMS40MDExIDAuNzM2ODA5IDAuOTIwODE3IDEuMjA3NTQgMC41NzM1MzhDIDEuNjc4MjYgMC4yMjYyNTkgMi4yNDA1NSAwLjAyNzU5MTkgMi44MjMyNiAwLjAwMjY3MjI5QyAzLjYwMzg5IC0wLjAzMDcxMTUgNC4zNjU3MyAwLjI0OTc4OSA0Ljk0MTQyIDAuNzgyNTUxQyA1LjUxNzExIDEuMzE1MzEgNS44NTk1NiAyLjA1Njc2IDUuODkzNTMgMi44NDRaIi8+CiAgICAgIDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE2MzkuOCAyMzIzLjgxKSIgZD0iTSA3LjQyNzg5IDMuNTgzMzhDIDcuNDYwMDggNC4zMjQzIDcuMjczNTUgNS4wNTgxOSA2Ljg5MTkzIDUuNjkyMTNDIDYuNTEwMzEgNi4zMjYwNyA1Ljk1MDc1IDYuODMxNTYgNS4yODQxMSA3LjE0NDZDIDQuNjE3NDcgNy40NTc2MyAzLjg3MzcxIDcuNTY0MTQgMy4xNDcwMiA3LjQ1MDYzQyAyLjQyMDMyIDcuMzM3MTIgMS43NDMzNiA3LjAwODcgMS4yMDE4NCA2LjUwNjk1QyAwLjY2MDMyOCA2LjAwNTIgMC4yNzg2MSA1LjM1MjY4IDAuMTA1MDE3IDQuNjMyMDJDIC0wLjA2ODU3NTcgMy45MTEzNSAtMC4wMjYyMzYxIDMuMTU0OTQgMC4yMjY2NzUgMi40NTg1NkMgMC40Nzk1ODcgMS43NjIxNyAwLjkzMTY5NyAxLjE1NzEzIDEuNTI1NzYgMC43MjAwMzNDIDIuMTE5ODMgMC4yODI5MzUgMi44MjkxNCAwLjAzMzQzOTUgMy41NjM4OSAwLjAwMzEzMzQ0QyA0LjU0NjY3IC0wLjAzNzQwMzMgNS41MDUyOSAwLjMxNjcwNiA2LjIyOTYxIDAuOTg3ODM1QyA2Ljk1MzkzIDEuNjU4OTYgNy4zODQ4NCAyLjU5MjM1IDcuNDI3ODkgMy41ODMzOEwgNy40Mjc4OSAzLjU4MzM4WiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM4LjM2IDIyODYuMDYpIiBkPSJNIDIuMjc0NzEgNC4zOTYyOUMgMS44NDM2MyA0LjQxNTA4IDEuNDE2NzEgNC4zMDQ0NSAxLjA0Nzk5IDQuMDc4NDNDIDAuNjc5MjY4IDMuODUyNCAwLjM4NTMyOCAzLjUyMTE0IDAuMjAzMzcxIDMuMTI2NTZDIDAuMDIxNDEzNiAyLjczMTk4IC0wLjA0MDM3OTggMi4yOTE4MyAwLjAyNTgxMTYgMS44NjE4MUMgMC4wOTIwMDMxIDEuNDMxOCAwLjI4MzIwNCAxLjAzMTI2IDAuNTc1MjEzIDAuNzEwODgzQyAwLjg2NzIyMiAwLjM5MDUxIDEuMjQ2OTEgMC4xNjQ3MDggMS42NjYyMiAwLjA2MjA1OTJDIDIuMDg1NTMgLTAuMDQwNTg5NyAyLjUyNTYxIC0wLjAxNTQ3MTQgMi45MzA3NiAwLjEzNDIzNUMgMy4zMzU5MSAwLjI4Mzk0MSAzLjY4NzkyIDAuNTUxNTA1IDMuOTQyMjIgMC45MDMwNkMgNC4xOTY1MiAxLjI1NDYyIDQuMzQxNjkgMS42NzQzNiA0LjM1OTM1IDIuMTA5MTZDIDQuMzgyOTkgMi42OTEwNyA0LjE3Njc4IDMuMjU4NjkgMy43ODU5NyAzLjY4NzQ2QyAzLjM5NTE2IDQuMTE2MjQgMi44NTE2NiA0LjM3MTE2IDIuMjc0NzEgNC4zOTYyOUwgMi4yNzQ3MSA0LjM5NjI5WiIvPgogICAgPC9nPgogIDwvZz4+Cjwvc3ZnPgo=);
  --jp-icon-jupyterlab-wordmark: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIHZpZXdCb3g9IjAgMCAxODYwLjggNDc1Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0RTRFNEUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ4MC4xMzY0MDEsIDY0LjI3MTQ5MykiPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDU4Ljg3NTU2NikiPgogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA4NzYwMywgMC4xNDAyOTQpIj4KICAgICAgICA8cGF0aCBkPSJNLTQyNi45LDE2OS44YzAsNDguNy0zLjcsNjQuNy0xMy42LDc2LjRjLTEwLjgsMTAtMjUsMTUuNS0zOS43LDE1LjVsMy43LDI5IGMyMi44LDAuMyw0NC44LTcuOSw2MS45LTIzLjFjMTcuOC0xOC41LDI0LTQ0LjEsMjQtODMuM1YwSC00Mjd2MTcwLjFMLTQyNi45LDE2OS44TC00MjYuOSwxNjkuOHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTU1LjA0NTI5NiwgNTYuODM3MTA0KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTYyNDUzLCAxLjc5OTg0MikiPgogICAgICAgIDxwYXRoIGQ9Ik0tMzEyLDE0OGMwLDIxLDAsMzkuNSwxLjcsNTUuNGgtMzEuOGwtMi4xLTMzLjNoLTAuOGMtNi43LDExLjYtMTYuNCwyMS4zLTI4LDI3LjkgYy0xMS42LDYuNi0yNC44LDEwLTM4LjIsOS44Yy0zMS40LDAtNjktMTcuNy02OS04OVYwaDM2LjR2MTEyLjdjMCwzOC43LDExLjYsNjQuNyw0NC42LDY0LjdjMTAuMy0wLjIsMjAuNC0zLjUsMjguOS05LjQgYzguNS01LjksMTUuMS0xNC4zLDE4LjktMjMuOWMyLjItNi4xLDMuMy0xMi41LDMuMy0xOC45VjAuMmgzNi40VjE0OEgtMzEyTC0zMTIsMTQ4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzOTAuMDEzMzIyLCA1My40Nzk2MzgpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS43MDY0NTgsIDAuMjMxNDI1KSI+CiAgICAgICAgPHBhdGggZD0iTS00NzguNiw3MS40YzAtMjYtMC44LTQ3LTEuNy02Ni43aDMyLjdsMS43LDM0LjhoMC44YzcuMS0xMi41LDE3LjUtMjIuOCwzMC4xLTI5LjcgYzEyLjUtNywyNi43LTEwLjMsNDEtOS44YzQ4LjMsMCw4NC43LDQxLjcsODQuNywxMDMuM2MwLDczLjEtNDMuNywxMDkuMi05MSwxMDkuMmMtMTIuMSwwLjUtMjQuMi0yLjItMzUtNy44IGMtMTAuOC01LjYtMTkuOS0xMy45LTI2LjYtMjQuMmgtMC44VjI5MWgtMzZ2LTIyMEwtNDc4LjYsNzEuNEwtNDc4LjYsNzEuNHogTS00NDIuNiwxMjUuNmMwLjEsNS4xLDAuNiwxMC4xLDEuNywxNS4xIGMzLDEyLjMsOS45LDIzLjMsMTkuOCwzMS4xYzkuOSw3LjgsMjIuMSwxMi4xLDM0LjcsMTIuMWMzOC41LDAsNjAuNy0zMS45LDYwLjctNzguNWMwLTQwLjctMjEuMS03NS42LTU5LjUtNzUuNiBjLTEyLjksMC40LTI1LjMsNS4xLTM1LjMsMTMuNGMtOS45LDguMy0xNi45LDE5LjctMTkuNiwzMi40Yy0xLjUsNC45LTIuMywxMC0yLjUsMTUuMVYxMjUuNkwtNDQyLjYsMTI1LjZMLTQ0Mi42LDEyNS42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2MDYuNzQwNzI2LCA1Ni44MzcxMDQpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC43NTEyMjYsIDEuOTg5Mjk5KSI+CiAgICAgICAgPHBhdGggZD0iTS00NDAuOCwwbDQzLjcsMTIwLjFjNC41LDEzLjQsOS41LDI5LjQsMTIuOCw0MS43aDAuOGMzLjctMTIuMiw3LjktMjcuNywxMi44LTQyLjQgbDM5LjctMTE5LjJoMzguNUwtMzQ2LjksMTQ1Yy0yNiw2OS43LTQzLjcsMTA1LjQtNjguNiwxMjcuMmMtMTIuNSwxMS43LTI3LjksMjAtNDQuNiwyMy45bC05LjEtMzEuMSBjMTEuNy0zLjksMjIuNS0xMC4xLDMxLjgtMTguMWMxMy4yLTExLjEsMjMuNy0yNS4yLDMwLjYtNDEuMmMxLjUtMi44LDIuNS01LjcsMi45LTguOGMtMC4zLTMuMy0xLjItNi42LTIuNS05LjdMLTQ4MC4yLDAuMSBoMzkuN0wtNDQwLjgsMEwtNDQwLjgsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODIyLjc0ODEwNCwgMC4wMDAwMDApIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS40NjQwNTAsIDAuMzc4OTE0KSI+CiAgICAgICAgPHBhdGggZD0iTS00MTMuNywwdjU4LjNoNTJ2MjguMmgtNTJWMTk2YzAsMjUsNywzOS41LDI3LjMsMzkuNWM3LjEsMC4xLDE0LjItMC43LDIxLjEtMi41IGwxLjcsMjcuN2MtMTAuMywzLjctMjEuMyw1LjQtMzIuMiw1Yy03LjMsMC40LTE0LjYtMC43LTIxLjMtMy40Yy02LjgtMi43LTEyLjktNi44LTE3LjktMTIuMWMtMTAuMy0xMC45LTE0LjEtMjktMTQuMS01Mi45IFY4Ni41aC0zMVY1OC4zaDMxVjkuNkwtNDEzLjcsMEwtNDEzLjcsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOTc0LjQzMzI4NiwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuOTkwMDM0LCAwLjYxMDMzOSkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDQ1LjgsMTEzYzAuOCw1MCwzMi4yLDcwLjYsNjguNiw3MC42YzE5LDAuNiwzNy45LTMsNTUuMy0xMC41bDYuMiwyNi40IGMtMjAuOSw4LjktNDMuNSwxMy4xLTY2LjIsMTIuNmMtNjEuNSwwLTk4LjMtNDEuMi05OC4zLTEwMi41Qy00ODAuMiw0OC4yLTQ0NC43LDAtMzg2LjUsMGM2NS4yLDAsODIuNyw1OC4zLDgyLjcsOTUuNyBjLTAuMSw1LjgtMC41LDExLjUtMS4yLDE3LjJoLTE0MC42SC00NDUuOEwtNDQ1LjgsMTEzeiBNLTMzOS4yLDg2LjZjMC40LTIzLjUtOS41LTYwLjEtNTAuNC02MC4xIGMtMzYuOCwwLTUyLjgsMzQuNC01NS43LDYwLjFILTMzOS4yTC0zMzkuMiw4Ni42TC0zMzkuMiw4Ni42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjAxLjk2MTA1OCwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuMTc5NjQwLCAwLjcwNTA2OCkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDc4LjYsNjhjMC0yMy45LTAuNC00NC41LTEuNy02My40aDMxLjhsMS4yLDM5LjloMS43YzkuMS0yNy4zLDMxLTQ0LjUsNTUuMy00NC41IGMzLjUtMC4xLDcsMC40LDEwLjMsMS4ydjM0LjhjLTQuMS0wLjktOC4yLTEuMy0xMi40LTEuMmMtMjUuNiwwLTQzLjcsMTkuNy00OC43LDQ3LjRjLTEsNS43LTEuNiwxMS41LTEuNywxNy4ydjEwOC4zaC0zNlY2OCBMLTQ3OC42LDY4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCBkPSJNMTM1Mi4zLDMyNi4yaDM3VjI4aC0zN1YzMjYuMnogTTE2MDQuOCwzMjYuMmMtMi41LTEzLjktMy40LTMxLjEtMy40LTQ4Ljd2LTc2IGMwLTQwLjctMTUuMS04My4xLTc3LjMtODMuMWMtMjUuNiwwLTUwLDcuMS02Ni44LDE4LjFsOC40LDI0LjRjMTQuMy05LjIsMzQtMTUuMSw1My0xNS4xYzQxLjYsMCw0Ni4yLDMwLjIsNDYuMiw0N3Y0LjIgYy03OC42LTAuNC0xMjIuMywyNi41LTEyMi4zLDc1LjZjMCwyOS40LDIxLDU4LjQsNjIuMiw1OC40YzI5LDAsNTAuOS0xNC4zLDYyLjItMzAuMmgxLjNsMi45LDI1LjZIMTYwNC44eiBNMTU2NS43LDI1Ny43IGMwLDMuOC0wLjgsOC0yLjEsMTEuOGMtNS45LDE3LjItMjIuNywzNC00OS4yLDM0Yy0xOC45LDAtMzQuOS0xMS4zLTM0LjktMzUuM2MwLTM5LjUsNDUuOC00Ni42LDg2LjItNDUuOFYyNTcuN3ogTTE2OTguNSwzMjYuMiBsMS43LTMzLjZoMS4zYzE1LjEsMjYuOSwzOC43LDM4LjIsNjguMSwzOC4yYzQ1LjQsMCw5MS4yLTM2LjEsOTEuMi0xMDguOGMwLjQtNjEuNy0zNS4zLTEwMy43LTg1LjctMTAzLjcgYy0zMi44LDAtNTYuMywxNC43LTY5LjMsMzcuNGgtMC44VjI4aC0zNi42djI0NS43YzAsMTguMS0wLjgsMzguNi0xLjcsNTIuNUgxNjk4LjV6IE0xNzA0LjgsMjA4LjJjMC01LjksMS4zLTEwLjksMi4xLTE1LjEgYzcuNi0yOC4xLDMxLjEtNDUuNCw1Ni4zLTQ1LjRjMzkuNSwwLDYwLjUsMzQuOSw2MC41LDc1LjZjMCw0Ni42LTIzLjEsNzguMS02MS44LDc4LjFjLTI2LjksMC00OC4zLTE3LjYtNTUuNS00My4zIGMtMC44LTQuMi0xLjctOC44LTEuNy0xMy40VjIwOC4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzYxNjE2MSIgZD0iTTE1IDlIOXY2aDZWOXptLTIgNGgtMnYtMmgydjJ6bTgtMlY5aC0yVjdjMC0xLjEtLjktMi0yLTJoLTJWM2gtMnYyaC0yVjNIOXYySDdjLTEuMSAwLTIgLjktMiAydjJIM3YyaDJ2MkgzdjJoMnYyYzAgMS4xLjkgMiAyIDJoMnYyaDJ2LTJoMnYyaDJ2LTJoMmMxLjEgMCAyLS45IDItMnYtMmgydi0yaC0ydi0yaDJ6bS00IDZIN1Y3aDEwdjEweiIvPgo8L3N2Zz4K);
  --jp-icon-keyboard: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMTdjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0tOSAzaDJ2MmgtMlY4em0wIDNoMnYyaC0ydi0yek04IDhoMnYySDhWOHptMCAzaDJ2Mkg4di0yem0tMSAySDV2LTJoMnYyem0wLTNINVY4aDJ2MnptOSA3SDh2LTJoOHYyem0wLTRoLTJ2LTJoMnYyem0wLTNoLTJWOGgydjJ6bTMgM2gtMnYtMmgydjJ6bTAtM2gtMlY4aDJ2MnoiLz4KPC9zdmc+Cg==);
  --jp-icon-launch: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMzIgMzIiIHdpZHRoPSIzMiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0yNiwyOEg2YTIuMDAyNywyLjAwMjcsMCwwLDEtMi0yVjZBMi4wMDI3LDIuMDAyNywwLDAsMSw2LDRIMTZWNkg2VjI2SDI2VjE2aDJWMjZBMi4wMDI3LDIuMDAyNywwLDAsMSwyNiwyOFoiLz4KICAgIDxwb2x5Z29uIHBvaW50cz0iMjAgMiAyMCA0IDI2LjU4NiA0IDE4IDEyLjU4NiAxOS40MTQgMTQgMjggNS40MTQgMjggMTIgMzAgMTIgMzAgMiAyMCAyIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-launcher: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkgMTlINVY1aDdWM0g1YTIgMiAwIDAwLTIgMnYxNGEyIDIgMCAwMDIgMmgxNGMxLjEgMCAyLS45IDItMnYtN2gtMnY3ek0xNCAzdjJoMy41OWwtOS44MyA5LjgzIDEuNDEgMS40MUwxOSA2LjQxVjEwaDJWM2gtN3oiLz4KPC9zdmc+Cg==);
  --jp-icon-line-form: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGZpbGw9IndoaXRlIiBkPSJNNS44OCA0LjEyTDEzLjc2IDEybC03Ljg4IDcuODhMOCAyMmwxMC0xMEw4IDJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-link: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMuOSAxMmMwLTEuNzEgMS4zOS0zLjEgMy4xLTMuMWg0VjdIN2MtMi43NiAwLTUgMi4yNC01IDVzMi4yNCA1IDUgNWg0di0xLjlIN2MtMS43MSAwLTMuMS0xLjM5LTMuMS0zLjF6TTggMTNoOHYtMkg4djJ6bTktNmgtNHYxLjloNGMxLjcxIDAgMy4xIDEuMzkgMy4xIDMuMXMtMS4zOSAzLjEtMy4xIDMuMWgtNFYxN2g0YzIuNzYgMCA1LTIuMjQgNS01cy0yLjI0LTUtNS01eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xOSA1djE0SDVWNWgxNG0xLjEtMkgzLjljLS41IDAtLjkuNC0uOS45djE2LjJjMCAuNC40LjkuOS45aDE2LjJjLjQgMCAuOS0uNS45LS45VjMuOWMwLS41LS41LS45LS45LS45ek0xMSA3aDZ2MmgtNlY3em0wIDRoNnYyaC02di0yem0wIDRoNnYyaC02ek03IDdoMnYySDd6bTAgNGgydjJIN3ptMCA0aDJ2Mkg3eiIvPgo8L3N2Zz4K);
  --jp-icon-markdown: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjN0IxRkEyIiBkPSJNNSAxNC45aDEybC02LjEgNnptOS40LTYuOGMwLTEuMy0uMS0yLjktLjEtNC41LS40IDEuNC0uOSAyLjktMS4zIDQuM2wtMS4zIDQuM2gtMkw4LjUgNy45Yy0uNC0xLjMtLjctMi45LTEtNC4zLS4xIDEuNi0uMSAzLjItLjIgNC42TDcgMTIuNEg0LjhsLjctMTFoMy4zTDEwIDVjLjQgMS4yLjcgMi43IDEgMy45LjMtMS4yLjctMi42IDEtMy45bDEuMi0zLjdoMy4zbC42IDExaC0yLjRsLS4zLTQuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-move-down: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBkPSJNMTIuNDcxIDcuNTI4OTlDMTIuNzYzMiA3LjIzNjg0IDEyLjc2MzIgNi43NjMxNiAxMi40NzEgNi40NzEwMVY2LjQ3MTAxQzEyLjE3OSA2LjE3OTA1IDExLjcwNTcgNi4xNzg4NCAxMS40MTM1IDYuNDcwNTRMNy43NSAxMC4xMjc1VjEuNzVDNy43NSAxLjMzNTc5IDcuNDE0MjEgMSA3IDFWMUM2LjU4NTc5IDEgNi4yNSAxLjMzNTc5IDYuMjUgMS43NVYxMC4xMjc1TDIuNTk3MjYgNi40NjgyMkMyLjMwMzM4IDYuMTczODEgMS44MjY0MSA2LjE3MzU5IDEuNTMyMjYgNi40Njc3NFY2LjQ2Nzc0QzEuMjM4MyA2Ljc2MTcgMS4yMzgzIDcuMjM4MyAxLjUzMjI2IDcuNTMyMjZMNi4yOTI4OSAxMi4yOTI5QzYuNjgzNDIgMTIuNjgzNCA3LjMxNjU4IDEyLjY4MzQgNy43MDcxMSAxMi4yOTI5TDEyLjQ3MSA3LjUyODk5WiIgZmlsbD0iIzYxNjE2MSIvPgo8L3N2Zz4K);
  --jp-icon-move-up: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBkPSJNMS41Mjg5OSA2LjQ3MTAxQzEuMjM2ODQgNi43NjMxNiAxLjIzNjg0IDcuMjM2ODQgMS41Mjg5OSA3LjUyODk5VjcuNTI4OTlDMS44MjA5NSA3LjgyMDk1IDIuMjk0MjYgNy44MjExNiAyLjU4NjQ5IDcuNTI5NDZMNi4yNSAzLjg3MjVWMTIuMjVDNi4yNSAxMi42NjQyIDYuNTg1NzkgMTMgNyAxM1YxM0M3LjQxNDIxIDEzIDcuNzUgMTIuNjY0MiA3Ljc1IDEyLjI1VjMuODcyNUwxMS40MDI3IDcuNTMxNzhDMTEuNjk2NiA3LjgyNjE5IDEyLjE3MzYgNy44MjY0MSAxMi40Njc3IDcuNTMyMjZWNy41MzIyNkMxMi43NjE3IDcuMjM4MyAxMi43NjE3IDYuNzYxNyAxMi40Njc3IDYuNDY3NzRMNy43MDcxMSAxLjcwNzExQzcuMzE2NTggMS4zMTY1OCA2LjY4MzQyIDEuMzE2NTggNi4yOTI4OSAxLjcwNzExTDEuNTI4OTkgNi40NzEwMVoiIGZpbGw9IiM2MTYxNjEiLz4KPC9zdmc+Cg==);
  --jp-icon-new-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwIDZoLThsLTItMkg0Yy0xLjExIDAtMS45OS44OS0xLjk5IDJMMiAxOGMwIDEuMTEuODkgMiAyIDJoMTZjMS4xMSAwIDItLjg5IDItMlY4YzAtMS4xMS0uODktMi0yLTJ6bS0xIDhoLTN2M2gtMnYtM2gtM3YtMmgzVjloMnYzaDN2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-not-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI1IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMTkgMTcuMTg0NCAyLjk2OTY4IDE0LjMwMzIgMS44NjA5NCAxMS40NDA5WiIvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24yIiBzdHJva2U9IiMzMzMzMzMiIHN0cm9rZS13aWR0aD0iMiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOS4zMTU5MiA5LjMyMDMxKSIgZD0iTTcuMzY4NDIgMEwwIDcuMzY0NzkiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDkuMzE1OTIgMTYuNjgzNikgc2NhbGUoMSAtMSkiIGQ9Ik03LjM2ODQyIDBMMCA3LjM2NDc5Ii8+Cjwvc3ZnPgo=);
  --jp-icon-notebook: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtbm90ZWJvb2staWNvbi1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNFRjZDMDAiPgogICAgPHBhdGggZD0iTTE4LjcgMy4zdjE1LjRIMy4zVjMuM2gxNS40bTEuNS0xLjVIMS44djE4LjNoMTguM2wuMS0xOC4zeiIvPgogICAgPHBhdGggZD0iTTE2LjUgMTYuNWwtNS40LTQuMy01LjYgNC4zdi0xMWgxMXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-numbering: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTQgMTlINlYxOS41SDVWMjAuNUg2VjIxSDRWMjJIN1YxOEg0VjE5Wk01IDEwSDZWNkg0VjdINVYxMFpNNCAxM0g1LjhMNCAxNS4xVjE2SDdWMTVINS4yTDcgMTIuOVYxMkg0VjEzWk05IDdWOUgyM1Y3SDlaTTkgMjFIMjNWMTlIOVYyMVpNOSAxNUgyM1YxM0g5VjE1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-offline-bolt: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDIuMDJjLTUuNTEgMC05Ljk4IDQuNDctOS45OCA5Ljk4czQuNDcgOS45OCA5Ljk4IDkuOTggOS45OC00LjQ3IDkuOTgtOS45OFMxNy41MSAyLjAyIDEyIDIuMDJ6TTExLjQ4IDIwdi02LjI2SDhMMTMgNHY2LjI2aDMuMzVMMTEuNDggMjB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-palette: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE4IDEzVjIwSDRWNkg5LjAyQzkuMDcgNS4yOSA5LjI0IDQuNjIgOS41IDRINEMyLjkgNCAyIDQuOSAyIDZWMjBDMiAyMS4xIDIuOSAyMiA0IDIySDE4QzE5LjEgMjIgMjAgMjEuMSAyMCAyMFYxNUwxOCAxM1pNMTkuMyA4Ljg5QzE5Ljc0IDguMTkgMjAgNy4zOCAyMCA2LjVDMjAgNC4wMSAxNy45OSAyIDE1LjUgMkMxMy4wMSAyIDExIDQuMDEgMTEgNi41QzExIDguOTkgMTMuMDEgMTEgMTUuNDkgMTFDMTYuMzcgMTEgMTcuMTkgMTAuNzQgMTcuODggMTAuM0wyMSAxMy40MkwyMi40MiAxMkwxOS4zIDguODlaTTE1LjUgOUMxNC4xMiA5IDEzIDcuODggMTMgNi41QzEzIDUuMTIgMTQuMTIgNCAxNS41IDRDMTYuODggNCAxOCA1LjEyIDE4IDYuNUMxOCA3Ljg4IDE2Ljg4IDkgMTUuNSA5WiIvPgogICAgPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik00IDZIOS4wMTg5NEM5LjAwNjM5IDYuMTY1MDIgOSA2LjMzMTc2IDkgNi41QzkgOC44MTU3NyAxMC4yMTEgMTAuODQ4NyAxMi4wMzQzIDEySDlWMTRIMTZWMTIuOTgxMUMxNi41NzAzIDEyLjkzNzcgMTcuMTIgMTIuODIwNyAxNy42Mzk2IDEyLjYzOTZMMTggMTNWMjBINFY2Wk04IDhINlYxMEg4VjhaTTYgMTJIOFYxNEg2VjEyWk04IDE2SDZWMThIOFYxNlpNOSAxNkgxNlYxOEg5VjE2WiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-paste: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE5IDJoLTQuMThDMTQuNC44NCAxMy4zIDAgMTIgMGMtMS4zIDAtMi40Ljg0LTIuODIgMkg1Yy0xLjEgMC0yIC45LTIgMnYxNmMwIDEuMS45IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjRjMC0xLjEtLjktMi0yLTJ6bS03IDBjLjU1IDAgMSAuNDUgMSAxcy0uNDUgMS0xIDEtMS0uNDUtMS0xIC40NS0xIDEtMXptNyAxOEg1VjRoMnYzaDEwVjRoMnYxNnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-pdf: url(data:image/svg+xml;base64,PHN2ZwogICB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMiAyMiIgd2lkdGg9IjE2Ij4KICAgIDxwYXRoIHRyYW5zZm9ybT0icm90YXRlKDQ1KSIgY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI0ZGMkEyQSIKICAgICAgIGQ9Im0gMjIuMzQ0MzY5LC0zLjAxNjM2NDIgaCA1LjYzODYwNCB2IDEuNTc5MjQzMyBoIC0zLjU0OTIyNyB2IDEuNTA4NjkyOTkgaCAzLjMzNzU3NiBWIDEuNjUwODE1NCBoIC0zLjMzNzU3NiB2IDMuNDM1MjYxMyBoIC0yLjA4OTM3NyB6IG0gLTcuMTM2NDQ0LDEuNTc5MjQzMyB2IDQuOTQzOTU0MyBoIDAuNzQ4OTIgcSAxLjI4MDc2MSwwIDEuOTUzNzAzLC0wLjYzNDk1MzUgMC42NzgzNjksLTAuNjM0OTUzNSAwLjY3ODM2OSwtMS44NDUxNjQxIDAsLTEuMjA0NzgzNTUgLTAuNjcyOTQyLC0xLjgzNDMxMDExIC0wLjY3Mjk0MiwtMC42Mjk1MjY1OSAtMS45NTkxMywtMC42Mjk1MjY1OSB6IG0gLTIuMDg5Mzc3LC0xLjU3OTI0MzMgaCAyLjIwMzM0MyBxIDEuODQ1MTY0LDAgMi43NDYwMzksMC4yNjU5MjA3IDAuOTA2MzAxLDAuMjYwNDkzNyAxLjU1MjEwOCwwLjg5MDAyMDMgMC41Njk4MywwLjU0ODEyMjMgMC44NDY2MDUsMS4yNjQ0ODAwNiAwLjI3Njc3NCwwLjcxNjM1NzgxIDAuMjc2Nzc0LDEuNjIyNjU4OTQgMCwwLjkxNzE1NTEgLTAuMjc2Nzc0LDEuNjM4OTM5OSAtMC4yNzY3NzUsMC43MTYzNTc4IC0wLjg0NjYwNSwxLjI2NDQ4IC0wLjY1MTIzNCwwLjYyOTUyNjYgLTEuNTYyOTYyLDAuODk1NDQ3MyAtMC45MTE3MjgsMC4yNjA0OTM3IC0yLjczNTE4NSwwLjI2MDQ5MzcgaCAtMi4yMDMzNDMgeiBtIC04LjE0NTg1NjUsMCBoIDMuNDY3ODIzIHEgMS41NDY2ODE2LDAgMi4zNzE1Nzg1LDAuNjg5MjIzIDAuODMwMzI0LDAuNjgzNzk2MSAwLjgzMDMyNCwxLjk1MzcwMzE0IDAsMS4yNzUzMzM5NyAtMC44MzAzMjQsMS45NjQ1NTcwNiBRIDkuOTg3MTk2MSwyLjI3NDkxNSA4LjQ0MDUxNDUsMi4yNzQ5MTUgSCA3LjA2MjA2ODQgViA1LjA4NjA3NjcgSCA0Ljk3MjY5MTUgWiBtIDIuMDg5Mzc2OSwxLjUxNDExOTkgdiAyLjI2MzAzOTQzIGggMS4xNTU5NDEgcSAwLjYwNzgxODgsMCAwLjkzODg2MjksLTAuMjkzMDU1NDcgMC4zMzEwNDQxLC0wLjI5ODQ4MjQxIDAuMzMxMDQ0MSwtMC44NDExNzc3MiAwLC0wLjU0MjY5NTMxIC0wLjMzMTA0NDEsLTAuODM1NzUwNzQgLTAuMzMxMDQ0MSwtMC4yOTMwNTU1IC0wLjkzODg2MjksLTAuMjkzMDU1NSB6IgovPgo8L3N2Zz4K);
  --jp-icon-python: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iLTEwIC0xMCAxMzEuMTYxMzYxNjk0MzM1OTQgMTMyLjM4ODk5OTkzODk2NDg0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMzA2OTk4IiBkPSJNIDU0LjkxODc4NSw5LjE5Mjc0MjFlLTQgQyA1MC4zMzUxMzIsMC4wMjIyMTcyNyA0NS45NTc4NDYsMC40MTMxMzY5NyA0Mi4xMDYyODUsMS4wOTQ2NjkzIDMwLjc2MDA2OSwzLjA5OTE3MzEgMjguNzAwMDM2LDcuMjk0NzcxNCAyOC43MDAwMzUsMTUuMDMyMTY5IHYgMTAuMjE4NzUgaCAyNi44MTI1IHYgMy40MDYyNSBoIC0yNi44MTI1IC0xMC4wNjI1IGMgLTcuNzkyNDU5LDAgLTE0LjYxNTc1ODgsNC42ODM3MTcgLTE2Ljc0OTk5OTgsMTMuNTkzNzUgLTIuNDYxODE5OTgsMTAuMjEyOTY2IC0yLjU3MTAxNTA4LDE2LjU4NjAyMyAwLDI3LjI1IDEuOTA1OTI4Myw3LjkzNzg1MiA2LjQ1NzU0MzIsMTMuNTkzNzQ4IDE0LjI0OTk5OTgsMTMuNTkzNzUgaCA5LjIxODc1IHYgLTEyLjI1IGMgMCwtOC44NDk5MDIgNy42NTcxNDQsLTE2LjY1NjI0OCAxNi43NSwtMTYuNjU2MjUgaCAyNi43ODEyNSBjIDcuNDU0OTUxLDAgMTMuNDA2MjUzLC02LjEzODE2NCAxMy40MDYyNSwtMTMuNjI1IHYgLTI1LjUzMTI1IGMgMCwtNy4yNjYzMzg2IC02LjEyOTk4LC0xMi43MjQ3NzcxIC0xMy40MDYyNSwtMTMuOTM3NDk5NyBDIDY0LjI4MTU0OCwwLjMyNzk0Mzk3IDU5LjUwMjQzOCwtMC4wMjAzNzkwMyA1NC45MTg3ODUsOS4xOTI3NDIxZS00IFogbSAtMTQuNSw4LjIxODc1MDEyNTc5IGMgMi43Njk1NDcsMCA1LjAzMTI1LDIuMjk4NjQ1NiA1LjAzMTI1LDUuMTI0OTk5NiAtMmUtNiwyLjgxNjMzNiAtMi4yNjE3MDMsNS4wOTM3NSAtNS4wMzEyNSw1LjA5Mzc1IC0yLjc3OTQ3NiwtMWUtNiAtNS4wMzEyNSwtMi4yNzc0MTUgLTUuMDMxMjUsLTUuMDkzNzUgLTEwZS03LC0yLjgyNjM1MyAyLjI1MTc3NCwtNS4xMjQ5OTk2IDUuMDMxMjUsLTUuMTI0OTk5NiB6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2ZmZDQzYiIgZD0ibSA4NS42Mzc1MzUsMjguNjU3MTY5IHYgMTEuOTA2MjUgYyAwLDkuMjMwNzU1IC03LjgyNTg5NSwxNi45OTk5OTkgLTE2Ljc1LDE3IGggLTI2Ljc4MTI1IGMgLTcuMzM1ODMzLDAgLTEzLjQwNjI0OSw2LjI3ODQ4MyAtMTMuNDA2MjUsMTMuNjI1IHYgMjUuNTMxMjQ3IGMgMCw3LjI2NjM0NCA2LjMxODU4OCwxMS41NDAzMjQgMTMuNDA2MjUsMTMuNjI1MDA0IDguNDg3MzMxLDIuNDk1NjEgMTYuNjI2MjM3LDIuOTQ2NjMgMjYuNzgxMjUsMCA2Ljc1MDE1NSwtMS45NTQzOSAxMy40MDYyNTMsLTUuODg3NjEgMTMuNDA2MjUsLTEzLjYyNTAwNCBWIDg2LjUwMDkxOSBoIC0yNi43ODEyNSB2IC0zLjQwNjI1IGggMjYuNzgxMjUgMTMuNDA2MjU0IGMgNy43OTI0NjEsMCAxMC42OTYyNTEsLTUuNDM1NDA4IDEzLjQwNjI0MSwtMTMuNTkzNzUgMi43OTkzMywtOC4zOTg4ODYgMi42ODAyMiwtMTYuNDc1Nzc2IDAsLTI3LjI1IC0xLjkyNTc4LC03Ljc1NzQ0MSAtNS42MDM4NywtMTMuNTkzNzUgLTEzLjQwNjI0MSwtMTMuNTkzNzUgeiBtIC0xNS4wNjI1LDY0LjY1NjI1IGMgMi43Nzk0NzgsM2UtNiA1LjAzMTI1LDIuMjc3NDE3IDUuMDMxMjUsNS4wOTM3NDcgLTJlLTYsMi44MjYzNTQgLTIuMjUxNzc1LDUuMTI1MDA0IC01LjAzMTI1LDUuMTI1MDA0IC0yLjc2OTU1LDAgLTUuMDMxMjUsLTIuMjk4NjUgLTUuMDMxMjUsLTUuMTI1MDA0IDJlLTYsLTIuODE2MzMgMi4yNjE2OTcsLTUuMDkzNzQ3IDUuMDMxMjUsLTUuMDkzNzQ3IHoiLz4KPC9zdmc+Cg==);
  --jp-icon-r-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjE5NkYzIiBkPSJNNC40IDIuNWMxLjItLjEgMi45LS4zIDQuOS0uMyAyLjUgMCA0LjEuNCA1LjIgMS4zIDEgLjcgMS41IDEuOSAxLjUgMy41IDAgMi0xLjQgMy41LTIuOSA0LjEgMS4yLjQgMS43IDEuNiAyLjIgMyAuNiAxLjkgMSAzLjkgMS4zIDQuNmgtMy44Yy0uMy0uNC0uOC0xLjctMS4yLTMuN3MtMS4yLTIuNi0yLjYtMi42aC0uOXY2LjRINC40VjIuNXptMy43IDYuOWgxLjRjMS45IDAgMi45LS45IDIuOS0yLjNzLTEtMi4zLTIuOC0yLjNjLS43IDAtMS4zIDAtMS42LjJ2NC41aC4xdi0uMXoiLz4KPC9zdmc+Cg==);
  --jp-icon-react: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMTUwIDE1MCA1NDEuOSAyOTUuMyI+CiAgPGcgY2xhc3M9ImpwLWljb24tYnJhbmQyIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxREFGQiI+CiAgICA8cGF0aCBkPSJNNjY2LjMgMjk2LjVjMC0zMi41LTQwLjctNjMuMy0xMDMuMS04Mi40IDE0LjQtNjMuNiA4LTExNC4yLTIwLjItMTMwLjQtNi41LTMuOC0xNC4xLTUuNi0yMi40LTUuNnYyMi4zYzQuNiAwIDguMy45IDExLjQgMi42IDEzLjYgNy44IDE5LjUgMzcuNSAxNC45IDc1LjctMS4xIDkuNC0yLjkgMTkuMy01LjEgMjkuNC0xOS42LTQuOC00MS04LjUtNjMuNS0xMC45LTEzLjUtMTguNS0yNy41LTM1LjMtNDEuNi01MCAzMi42LTMwLjMgNjMuMi00Ni45IDg0LTQ2LjlWNzhjLTI3LjUgMC02My41IDE5LjYtOTkuOSA1My42LTM2LjQtMzMuOC03Mi40LTUzLjItOTkuOS01My4ydjIyLjNjMjAuNyAwIDUxLjQgMTYuNSA4NCA0Ni42LTE0IDE0LjctMjggMzEuNC00MS4zIDQ5LjktMjIuNiAyLjQtNDQgNi4xLTYzLjYgMTEtMi4zLTEwLTQtMTkuNy01LjItMjktNC43LTM4LjIgMS4xLTY3LjkgMTQuNi03NS44IDMtMS44IDYuOS0yLjYgMTEuNS0yLjZWNzguNWMtOC40IDAtMTYgMS44LTIyLjYgNS42LTI4LjEgMTYuMi0zNC40IDY2LjctMTkuOSAxMzAuMS02Mi4yIDE5LjItMTAyLjcgNDkuOS0xMDIuNyA4Mi4zIDAgMzIuNSA0MC43IDYzLjMgMTAzLjEgODIuNC0xNC40IDYzLjYtOCAxMTQuMiAyMC4yIDEzMC40IDYuNSAzLjggMTQuMSA1LjYgMjIuNSA1LjYgMjcuNSAwIDYzLjUtMTkuNiA5OS45LTUzLjYgMzYuNCAzMy44IDcyLjQgNTMuMiA5OS45IDUzLjIgOC40IDAgMTYtMS44IDIyLjYtNS42IDI4LjEtMTYuMiAzNC40LTY2LjcgMTkuOS0xMzAuMSA2Mi0xOS4xIDEwMi41LTQ5LjkgMTAyLjUtODIuM3ptLTEzMC4yLTY2LjdjLTMuNyAxMi45LTguMyAyNi4yLTEzLjUgMzkuNS00LjEtOC04LjQtMTYtMTMuMS0yNC00LjYtOC05LjUtMTUuOC0xNC40LTIzLjQgMTQuMiAyLjEgMjcuOSA0LjcgNDEgNy45em0tNDUuOCAxMDYuNWMtNy44IDEzLjUtMTUuOCAyNi4zLTI0LjEgMzguMi0xNC45IDEuMy0zMCAyLTQ1LjIgMi0xNS4xIDAtMzAuMi0uNy00NS0xLjktOC4zLTExLjktMTYuNC0yNC42LTI0LjItMzgtNy42LTEzLjEtMTQuNS0yNi40LTIwLjgtMzkuOCA2LjItMTMuNCAxMy4yLTI2LjggMjAuNy0zOS45IDcuOC0xMy41IDE1LjgtMjYuMyAyNC4xLTM4LjIgMTQuOS0xLjMgMzAtMiA0NS4yLTIgMTUuMSAwIDMwLjIuNyA0NSAxLjkgOC4zIDExLjkgMTYuNCAyNC42IDI0LjIgMzggNy42IDEzLjEgMTQuNSAyNi40IDIwLjggMzkuOC02LjMgMTMuNC0xMy4yIDI2LjgtMjAuNyAzOS45em0zMi4zLTEzYzUuNCAxMy40IDEwIDI2LjggMTMuOCAzOS44LTEzLjEgMy4yLTI2LjkgNS45LTQxLjIgOCA0LjktNy43IDkuOC0xNS42IDE0LjQtMjMuNyA0LjYtOCA4LjktMTYuMSAxMy0yNC4xek00MjEuMiA0MzBjLTkuMy05LjYtMTguNi0yMC4zLTI3LjgtMzIgOSAuNCAxOC4yLjcgMjcuNS43IDkuNCAwIDE4LjctLjIgMjcuOC0uNy05IDExLjctMTguMyAyMi40LTI3LjUgMzJ6bS03NC40LTU4LjljLTE0LjItMi4xLTI3LjktNC43LTQxLTcuOSAzLjctMTIuOSA4LjMtMjYuMiAxMy41LTM5LjUgNC4xIDggOC40IDE2IDEzLjEgMjQgNC43IDggOS41IDE1LjggMTQuNCAyMy40ek00MjAuNyAxNjNjOS4zIDkuNiAxOC42IDIwLjMgMjcuOCAzMi05LS40LTE4LjItLjctMjcuNS0uNy05LjQgMC0xOC43LjItMjcuOC43IDktMTEuNyAxOC4zLTIyLjQgMjcuNS0zMnptLTc0IDU4LjljLTQuOSA3LjctOS44IDE1LjYtMTQuNCAyMy43LTQuNiA4LTguOSAxNi0xMyAyNC01LjQtMTMuNC0xMC0yNi44LTEzLjgtMzkuOCAxMy4xLTMuMSAyNi45LTUuOCA0MS4yLTcuOXptLTkwLjUgMTI1LjJjLTM1LjQtMTUuMS01OC4zLTM0LjktNTguMy01MC42IDAtMTUuNyAyMi45LTM1LjYgNTguMy01MC42IDguNi0zLjcgMTgtNyAyNy43LTEwLjEgNS43IDE5LjYgMTMuMiA0MCAyMi41IDYwLjktOS4yIDIwLjgtMTYuNiA0MS4xLTIyLjIgNjAuNi05LjktMy4xLTE5LjMtNi41LTI4LTEwLjJ6TTMxMCA0OTBjLTEzLjYtNy44LTE5LjUtMzcuNS0xNC45LTc1LjcgMS4xLTkuNCAyLjktMTkuMyA1LjEtMjkuNCAxOS42IDQuOCA0MSA4LjUgNjMuNSAxMC45IDEzLjUgMTguNSAyNy41IDM1LjMgNDEuNiA1MC0zMi42IDMwLjMtNjMuMiA0Ni45LTg0IDQ2LjktNC41LS4xLTguMy0xLTExLjMtMi43em0yMzcuMi03Ni4yYzQuNyAzOC4yLTEuMSA2Ny45LTE0LjYgNzUuOC0zIDEuOC02LjkgMi42LTExLjUgMi42LTIwLjcgMC01MS40LTE2LjUtODQtNDYuNiAxNC0xNC43IDI4LTMxLjQgNDEuMy00OS45IDIyLjYtMi40IDQ0LTYuMSA2My42LTExIDIuMyAxMC4xIDQuMSAxOS44IDUuMiAyOS4xem0zOC41LTY2LjdjLTguNiAzLjctMTggNy0yNy43IDEwLjEtNS43LTE5LjYtMTMuMi00MC0yMi41LTYwLjkgOS4yLTIwLjggMTYuNi00MS4xIDIyLjItNjAuNiA5LjkgMy4xIDE5LjMgNi41IDI4LjEgMTAuMiAzNS40IDE1LjEgNTguMyAzNC45IDU4LjMgNTAuNi0uMSAxNS43LTIzIDM1LjYtNTguNCA1MC42ek0zMjAuOCA3OC40eiIvPgogICAgPGNpcmNsZSBjeD0iNDIwLjkiIGN5PSIyOTYuNSIgcj0iNDUuNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-redo: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCBkPSJNMCAwaDI0djI0SDB6IiBmaWxsPSJub25lIi8+PHBhdGggZD0iTTE4LjQgMTAuNkMxNi41NSA4Ljk5IDE0LjE1IDggMTEuNSA4Yy00LjY1IDAtOC41OCAzLjAzLTkuOTYgNy4yMkwzLjkgMTZjMS4wNS0zLjE5IDQuMDUtNS41IDcuNi01LjUgMS45NSAwIDMuNzMuNzIgNS4xMiAxLjg4TDEzIDE2aDlWN2wtMy42IDMuNnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-refresh: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTkgMTMuNWMtMi40OSAwLTQuNS0yLjAxLTQuNS00LjVTNi41MSA0LjUgOSA0LjVjMS4yNCAwIDIuMzYuNTIgMy4xNyAxLjMzTDEwIDhoNVYzbC0xLjc2IDEuNzZDMTIuMTUgMy42OCAxMC42NiAzIDkgMyA1LjY5IDMgMy4wMSA1LjY5IDMuMDEgOVM1LjY5IDE1IDkgMTVjMi45NyAwIDUuNDMtMi4xNiA1LjktNWgtMS41MmMtLjQ2IDItMi4yNCAzLjUtNC4zOCAzLjV6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-regex: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiBmaWxsPSIjRkZGIj4KICAgIDxjaXJjbGUgY2xhc3M9InN0MiIgY3g9IjUuNSIgY3k9IjE0LjUiIHI9IjEuNSIvPgogICAgPHJlY3QgeD0iMTIiIHk9IjQiIGNsYXNzPSJzdDIiIHdpZHRoPSIxIiBoZWlnaHQ9IjgiLz4KICAgIDxyZWN0IHg9IjguNSIgeT0iNy41IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjg2NiAtMC41IDAuNSAwLjg2NiAtMi4zMjU1IDcuMzIxOSkiIGNsYXNzPSJzdDIiIHdpZHRoPSI4IiBoZWlnaHQ9IjEiLz4KICAgIDxyZWN0IHg9IjEyIiB5PSI0IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjUgLTAuODY2IDAuODY2IDAuNSAtMC42Nzc5IDE0LjgyNTIpIiBjbGFzcz0ic3QyIiB3aWR0aD0iMSIgaGVpZ2h0PSI4Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-run: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTggNXYxNGwxMS03eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-running: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMjU2IDhDMTE5IDggOCAxMTkgOCAyNTZzMTExIDI0OCAyNDggMjQ4IDI0OC0xMTEgMjQ4LTI0OFMzOTMgOCAyNTYgOHptOTYgMzI4YzAgOC44LTcuMiAxNi0xNiAxNkgxNzZjLTguOCAwLTE2LTcuMi0xNi0xNlYxNzZjMC04LjggNy4yLTE2IDE2LTE2aDE2MGM4LjggMCAxNiA3LjIgMTYgMTZ2MTYweiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-save: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE3IDNINWMtMS4xMSAwLTIgLjktMiAydjE0YzAgMS4xLjg5IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjdsLTQtNHptLTUgMTZjLTEuNjYgMC0zLTEuMzQtMy0zczEuMzQtMyAzLTMgMyAxLjM0IDMgMy0xLjM0IDMtMyAzem0zLTEwSDVWNWgxMHY0eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-search: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjEsMTAuOWgtMC43bC0wLjItMC4yYzAuOC0wLjksMS4zLTIuMiwxLjMtMy41YzAtMy0yLjQtNS40LTUuNC01LjRTMS44LDQuMiwxLjgsNy4xczIuNCw1LjQsNS40LDUuNCBjMS4zLDAsMi41LTAuNSwzLjUtMS4zbDAuMiwwLjJ2MC43bDQuMSw0LjFsMS4yLTEuMkwxMi4xLDEwLjl6IE03LjEsMTAuOWMtMi4xLDAtMy43LTEuNy0zLjctMy43czEuNy0zLjcsMy43LTMuN3MzLjcsMS43LDMuNywzLjcgUzkuMiwxMC45LDcuMSwxMC45eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-settings: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuNDMgMTIuOThjLjA0LS4zMi4wNy0uNjQuMDctLjk4cy0uMDMtLjY2LS4wNy0uOThsMi4xMS0xLjY1Yy4xOS0uMTUuMjQtLjQyLjEyLS42NGwtMi0zLjQ2Yy0uMTItLjIyLS4zOS0uMy0uNjEtLjIybC0yLjQ5IDFjLS41Mi0uNC0xLjA4LS43My0xLjY5LS45OGwtLjM4LTIuNjVBLjQ4OC40ODggMCAwMDE0IDJoLTRjLS4yNSAwLS40Ni4xOC0uNDkuNDJsLS4zOCAyLjY1Yy0uNjEuMjUtMS4xNy41OS0xLjY5Ljk4bC0yLjQ5LTFjLS4yMy0uMDktLjQ5IDAtLjYxLjIybC0yIDMuNDZjLS4xMy4yMi0uMDcuNDkuMTIuNjRsMi4xMSAxLjY1Yy0uMDQuMzItLjA3LjY1LS4wNy45OHMuMDMuNjYuMDcuOThsLTIuMTEgMS42NWMtLjE5LjE1LS4yNC40Mi0uMTIuNjRsMiAzLjQ2Yy4xMi4yMi4zOS4zLjYxLjIybDIuNDktMWMuNTIuNCAxLjA4LjczIDEuNjkuOThsLjM4IDIuNjVjLjAzLjI0LjI0LjQyLjQ5LjQyaDRjLjI1IDAgLjQ2LS4xOC40OS0uNDJsLjM4LTIuNjVjLjYxLS4yNSAxLjE3LS41OSAxLjY5LS45OGwyLjQ5IDFjLjIzLjA5LjQ5IDAgLjYxLS4yMmwyLTMuNDZjLjEyLS4yMi4wNy0uNDktLjEyLS42NGwtMi4xMS0xLjY1ek0xMiAxNS41Yy0xLjkzIDAtMy41LTEuNTctMy41LTMuNXMxLjU3LTMuNSAzLjUtMy41IDMuNSAxLjU3IDMuNSAzLjUtMS41NyAzLjUtMy41IDMuNXoiLz4KPC9zdmc+Cg==);
  --jp-icon-share: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTSAxOCAyIEMgMTYuMzU0OTkgMiAxNSAzLjM1NDk5MDQgMTUgNSBDIDE1IDUuMTkwOTUyOSAxNS4wMjE3OTEgNS4zNzcxMjI0IDE1LjA1NjY0MSA1LjU1ODU5MzggTCA3LjkyMTg3NSA5LjcyMDcwMzEgQyA3LjM5ODUzOTkgOS4yNzc4NTM5IDYuNzMyMDc3MSA5IDYgOSBDIDQuMzU0OTkwNCA5IDMgMTAuMzU0OTkgMyAxMiBDIDMgMTMuNjQ1MDEgNC4zNTQ5OTA0IDE1IDYgMTUgQyA2LjczMjA3NzEgMTUgNy4zOTg1Mzk5IDE0LjcyMjE0NiA3LjkyMTg3NSAxNC4yNzkyOTcgTCAxNS4wNTY2NDEgMTguNDM5NDUzIEMgMTUuMDIxNTU1IDE4LjYyMTUxNCAxNSAxOC44MDgzODYgMTUgMTkgQyAxNSAyMC42NDUwMSAxNi4zNTQ5OSAyMiAxOCAyMiBDIDE5LjY0NTAxIDIyIDIxIDIwLjY0NTAxIDIxIDE5IEMgMjEgMTcuMzU0OTkgMTkuNjQ1MDEgMTYgMTggMTYgQyAxNy4yNjc0OCAxNiAxNi42MDE1OTMgMTYuMjc5MzI4IDE2LjA3ODEyNSAxNi43MjI2NTYgTCA4Ljk0MzM1OTQgMTIuNTU4NTk0IEMgOC45NzgyMDk1IDEyLjM3NzEyMiA5IDEyLjE5MDk1MyA5IDEyIEMgOSAxMS44MDkwNDcgOC45NzgyMDk1IDExLjYyMjg3OCA4Ljk0MzM1OTQgMTEuNDQxNDA2IEwgMTYuMDc4MTI1IDcuMjc5Mjk2OSBDIDE2LjYwMTQ2IDcuNzIyMTQ2MSAxNy4yNjc5MjMgOCAxOCA4IEMgMTkuNjQ1MDEgOCAyMSA2LjY0NTAwOTYgMjEgNSBDIDIxIDMuMzU0OTkwNCAxOS42NDUwMSAyIDE4IDIgeiBNIDE4IDQgQyAxOC41NjQxMjkgNCAxOSA0LjQzNTg3MDYgMTkgNSBDIDE5IDUuNTY0MTI5NCAxOC41NjQxMjkgNiAxOCA2IEMgMTcuNDM1ODcxIDYgMTcgNS41NjQxMjk0IDE3IDUgQyAxNyA0LjQzNTg3MDYgMTcuNDM1ODcxIDQgMTggNCB6IE0gNiAxMSBDIDYuNTY0MTI5NCAxMSA3IDExLjQzNTg3MSA3IDEyIEMgNyAxMi41NjQxMjkgNi41NjQxMjk0IDEzIDYgMTMgQyA1LjQzNTg3MDYgMTMgNSAxMi41NjQxMjkgNSAxMiBDIDUgMTEuNDM1ODcxIDUuNDM1ODcwNiAxMSA2IDExIHogTSAxOCAxOCBDIDE4LjU2NDEyOSAxOCAxOSAxOC40MzU4NzEgMTkgMTkgQyAxOSAxOS41NjQxMjkgMTguNTY0MTI5IDIwIDE4IDIwIEMgMTcuNDM1ODcxIDIwIDE3IDE5LjU2NDEyOSAxNyAxOSBDIDE3IDE4LjQzNTg3MSAxNy40MzU4NzEgMTggMTggMTggeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-spreadsheet: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNENBRjUwIiBkPSJNMi4yIDIuMnYxNy42aDE3LjZWMi4ySDIuMnptMTUuNCA3LjdoLTUuNVY0LjRoNS41djUuNXpNOS45IDQuNHY1LjVINC40VjQuNGg1LjV6bS01LjUgNy43aDUuNXY1LjVINC40di01LjV6bTcuNyA1LjV2LTUuNWg1LjV2NS41aC01LjV6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-stop: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik02IDZoMTJ2MTJINnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tab: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIxIDNIM2MtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxOGMxLjEgMCAyLS45IDItMlY1YzAtMS4xLS45LTItMi0yem0wIDE2SDNWNWgxMHY0aDh2MTB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-table-rows: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMSw4SDNWNGgxOFY4eiBNMjEsMTBIM3Y0aDE4VjEweiBNMjEsMTZIM3Y0aDE4VjE2eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-tag: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgiIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCA0MyAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTI4LjgzMzIgMTIuMzM0TDMyLjk5OTggMTYuNTAwN0wzNy4xNjY1IDEyLjMzNEgyOC44MzMyWiIvPgoJCTxwYXRoIGQ9Ik0xNi4yMDk1IDIxLjYxMDRDMTUuNjg3MyAyMi4xMjk5IDE0Ljg0NDMgMjIuMTI5OSAxNC4zMjQ4IDIxLjYxMDRMNi45ODI5IDE0LjcyNDVDNi41NzI0IDE0LjMzOTQgNi4wODMxMyAxMy42MDk4IDYuMDQ3ODYgMTMuMDQ4MkM1Ljk1MzQ3IDExLjUyODggNi4wMjAwMiA4LjYxOTQ0IDYuMDY2MjEgNy4wNzY5NUM2LjA4MjgxIDYuNTE0NzcgNi41NTU0OCA2LjA0MzQ3IDcuMTE4MDQgNi4wMzA1NUM5LjA4ODYzIDUuOTg0NzMgMTMuMjYzOCA1LjkzNTc5IDEzLjY1MTggNi4zMjQyNUwyMS43MzY5IDEzLjYzOUMyMi4yNTYgMTQuMTU4NSAyMS43ODUxIDE1LjQ3MjQgMjEuMjYyIDE1Ljk5NDZMMTYuMjA5NSAyMS42MTA0Wk05Ljc3NTg1IDguMjY1QzkuMzM1NTEgNy44MjU2NiA4LjYyMzUxIDcuODI1NjYgOC4xODI4IDguMjY1QzcuNzQzNDYgOC43MDU3MSA3Ljc0MzQ2IDkuNDE3MzMgOC4xODI4IDkuODU2NjdDOC42MjM4MiAxMC4yOTY0IDkuMzM1ODIgMTAuMjk2NCA5Ljc3NTg1IDkuODU2NjdDMTAuMjE1NiA5LjQxNzMzIDEwLjIxNTYgOC43MDUzMyA5Ljc3NTg1IDguMjY1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-terminal: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiA+CiAgICA8cmVjdCBjbGFzcz0ianAtdGVybWluYWwtaWNvbi1iYWNrZ3JvdW5kLWNvbG9yIGpwLWljb24tc2VsZWN0YWJsZSIgd2lkdGg9IjIwIiBoZWlnaHQ9IjIwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyIDIpIiBmaWxsPSIjMzMzMzMzIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtdGVybWluYWwtaWNvbi1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUtaW52ZXJzZSIgZD0iTTUuMDU2NjQgOC43NjE3MkM1LjA1NjY0IDguNTk3NjYgNS4wMzEyNSA4LjQ1MzEyIDQuOTgwNDcgOC4zMjgxMkM0LjkzMzU5IDguMTk5MjIgNC44NTU0NyA4LjA4MjAzIDQuNzQ2MDkgNy45NzY1NkM0LjY0MDYyIDcuODcxMDkgNC41IDcuNzc1MzkgNC4zMjQyMiA3LjY4OTQ1QzQuMTUyMzQgNy41OTk2MSAzLjk0MzM2IDcuNTExNzIgMy42OTcyNyA3LjQyNTc4QzMuMzAyNzMgNy4yODUxNiAyLjk0MzM2IDcuMTM2NzIgMi42MTkxNCA2Ljk4MDQ3QzIuMjk0OTIgNi44MjQyMiAyLjAxNzU4IDYuNjQyNTggMS43ODcxMSA2LjQzNTU1QzEuNTYwNTUgNi4yMjg1MiAxLjM4NDc3IDUuOTg4MjggMS4yNTk3NyA1LjcxNDg0QzEuMTM0NzcgNS40Mzc1IDEuMDcyMjcgNS4xMDkzOCAxLjA3MjI3IDQuNzMwNDdDMS4wNzIyNyA0LjM5ODQ0IDEuMTI4OTEgNC4wOTU3IDEuMjQyMTkgMy44MjIyN0MxLjM1NTQ3IDMuNTQ0OTIgMS41MTU2MiAzLjMwNDY5IDEuNzIyNjYgMy4xMDE1NkMxLjkyOTY5IDIuODk4NDQgMi4xNzk2OSAyLjczNDM3IDIuNDcyNjYgMi42MDkzOEMyLjc2NTYyIDIuNDg0MzggMy4wOTE4IDIuNDA0MyAzLjQ1MTE3IDIuMzY5MTRWMS4xMDkzOEg0LjM4ODY3VjIuMzgwODZDNC43NDAyMyAyLjQyNzczIDUuMDU2NjQgMi41MjM0NCA1LjMzNzg5IDIuNjY3OTdDNS42MTkxNCAyLjgxMjUgNS44NTc0MiAzLjAwMTk1IDYuMDUyNzMgMy4yMzYzM0M2LjI1MTk1IDMuNDY2OCA2LjQwNDMgMy43NDAyMyA2LjUwOTc3IDQuMDU2NjRDNi42MTkxNCA0LjM2OTE0IDYuNjczODMgNC43MjA3IDYuNjczODMgNS4xMTEzM0g1LjA0NDkyQzUuMDQ0OTIgNC42Mzg2NyA0LjkzNzUgNC4yODEyNSA0LjcyMjY2IDQuMDM5MDZDNC41MDc4MSAzLjc5Mjk3IDQuMjE2OCAzLjY2OTkyIDMuODQ5NjEgMy42Njk5MkMzLjY1MDM5IDMuNjY5OTIgMy40NzY1NiAzLjY5NzI3IDMuMzI4MTIgMy43NTE5NUMzLjE4MzU5IDMuODAyNzMgMy4wNjQ0NSAzLjg3Njk1IDIuOTcwNyAzLjk3NDYxQzIuODc2OTUgNC4wNjgzNiAyLjgwNjY0IDQuMTc5NjkgMi43NTk3NyA0LjMwODU5QzIuNzE2OCA0LjQzNzUgMi42OTUzMSA0LjU3ODEyIDIuNjk1MzEgNC43MzA0N0MyLjY5NTMxIDQuODgyODEgMi43MTY4IDUuMDE5NTMgMi43NTk3NyA1LjE0MDYyQzIuODA2NjQgNS4yNTc4MSAyLjg4MjgxIDUuMzY3MTkgMi45ODgyOCA1LjQ2ODc1QzMuMDk3NjYgNS41NzAzMSAzLjI0MDIzIDUuNjY3OTcgMy40MTYwMiA1Ljc2MTcyQzMuNTkxOCA1Ljg1MTU2IDMuODEwNTUgNS45NDMzNiA0LjA3MjI3IDYuMDM3MTFDNC40NjY4IDYuMTg1NTUgNC44MjQyMiA2LjMzOTg0IDUuMTQ0NTMgNi41QzUuNDY0ODQgNi42NTYyNSA1LjczODI4IDYuODM5ODQgNS45NjQ4NCA3LjA1MDc4QzYuMTk1MzEgNy4yNTc4MSA2LjM3MTA5IDcuNSA2LjQ5MjE5IDcuNzc3MzRDNi42MTcxOSA4LjA1MDc4IDYuNjc5NjkgOC4zNzUgNi42Nzk2OSA4Ljc1QzYuNjc5NjkgOS4wOTM3NSA2LjYyMzA1IDkuNDA0MyA2LjUwOTc3IDkuNjgxNjRDNi4zOTY0OCA5Ljk1NTA4IDYuMjM0MzggMTAuMTkxNCA2LjAyMzQ0IDEwLjM5MDZDNS44MTI1IDEwLjU4OTggNS41NTg1OSAxMC43NSA1LjI2MTcyIDEwLjg3MTFDNC45NjQ4NCAxMC45ODgzIDQuNjMyODEgMTEuMDY0NSA0LjI2NTYyIDExLjA5OTZWMTIuMjQ4SDMuMzMzOThWMTEuMDk5NkMzLjAwMTk1IDExLjA2ODQgMi42Nzk2OSAxMC45OTYxIDIuMzY3MTkgMTAuODgyOEMyLjA1NDY5IDEwLjc2NTYgMS43NzczNCAxMC41OTc3IDEuNTM1MTYgMTAuMzc4OUMxLjI5Njg4IDEwLjE2MDIgMS4xMDU0NyA5Ljg4NDc3IDAuOTYwOTM4IDkuNTUyNzNDMC44MTY0MDYgOS4yMTY4IDAuNzQ0MTQxIDguODE0NDUgMC43NDQxNDEgOC4zNDU3SDIuMzc4OTFDMi4zNzg5MSA4LjYyNjk1IDIuNDE5OTIgOC44NjMyOCAyLjUwMTk1IDkuMDU0NjlDMi41ODM5OCA5LjI0MjE5IDIuNjg5NDUgOS4zOTI1OCAyLjgxODM2IDkuNTA1ODZDMi45NTExNyA5LjYxNTIzIDMuMTAxNTYgOS42OTMzNiAzLjI2OTUzIDkuNzQwMjNDMy40Mzc1IDkuNzg3MTEgMy42MDkzOCA5LjgxMDU1IDMuNzg1MTYgOS44MTA1NUM0LjIwMzEyIDkuODEwNTUgNC41MTk1MyA5LjcxMjg5IDQuNzM0MzggOS41MTc1OEM0Ljk0OTIyIDkuMzIyMjcgNS4wNTY2NCA5LjA3MDMxIDUuMDU2NjQgOC43NjE3MlpNMTMuNDE4IDEyLjI3MTVIOC4wNzQyMlYxMUgxMy40MThWMTIuMjcxNVoiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMuOTUyNjQgNikiIGZpbGw9IndoaXRlIi8+Cjwvc3ZnPgo=);
  --jp-icon-text-editor: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtdGV4dC1lZGl0b3ItaWNvbi1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xNSAxNUgzdjJoMTJ2LTJ6bTAtOEgzdjJoMTJWN3pNMyAxM2gxOHYtMkgzdjJ6bTAgOGgxOHYtMkgzdjJ6TTMgM3YyaDE4VjNIM3oiLz4KPC9zdmc+Cg==);
  --jp-icon-toc: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik03LDVIMjFWN0g3VjVNNywxM1YxMUgyMVYxM0g3TTQsNC41QTEuNSwxLjUgMCAwLDEgNS41LDZBMS41LDEuNSAwIDAsMSA0LDcuNUExLjUsMS41IDAgMCwxIDIuNSw2QTEuNSwxLjUgMCAwLDEgNCw0LjVNNCwxMC41QTEuNSwxLjUgMCAwLDEgNS41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMy41QTEuNSwxLjUgMCAwLDEgMi41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMC41TTcsMTlWMTdIMjFWMTlIN000LDE2LjVBMS41LDEuNSAwIDAsMSA1LjUsMThBMS41LDEuNSAwIDAsMSA0LDE5LjVBMS41LDEuNSAwIDAsMSAyLjUsMThBMS41LDEuNSAwIDAsMSA0LDE2LjVaIiAvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tree-view: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMiAxMVYzaC03djNIOVYzSDJ2OGg3VjhoMnYxMGg0djNoN3YtOGgtN3YzaC0yVjhoMnYzeiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMiAxNy4xODQ0IDIuOTY5NjggMTQuMzAzMiAxLjg2MDk0IDExLjQ0MDlaIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiMzMzMzMzMiIHN0cm9rZT0iIzMzMzMzMyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOCA5Ljg2NzE5KSIgZD0iTTIuODYwMTUgNC44NjUzNUwwLjcyNjU0OSAyLjk5OTU5TDAgMy42MzA0NUwyLjg2MDE1IDYuMTMxNTdMOCAwLjYzMDg3Mkw3LjI3ODU3IDBMMi44NjAxNSA0Ljg2NTM1WiIvPgo8L3N2Zz4K);
  --jp-icon-undo: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjUgOGMtMi42NSAwLTUuMDUuOTktNi45IDIuNkwyIDd2OWg5bC0zLjYyLTMuNjJjMS4zOS0xLjE2IDMuMTYtMS44OCA1LjEyLTEuODggMy41NCAwIDYuNTUgMi4zMSA3LjYgNS41bDIuMzctLjc4QzIxLjA4IDExLjAzIDE3LjE1IDggMTIuNSA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-user: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE2IDdhNCA0IDAgMTEtOCAwIDQgNCAwIDAxOCAwek0xMiAxNGE3IDcgMCAwMC03IDdoMTRhNyA3IDAgMDAtNy03eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-users: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZlcnNpb249IjEuMSIgdmlld0JveD0iMCAwIDM2IDI0IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogPGcgY2xhc3M9ImpwLWljb24zIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjczMjcgMCAwIDEuNzMyNyAtMy42MjgyIC4wOTk1NzcpIiBmaWxsPSIjNjE2MTYxIj4KICA8cGF0aCB0cmFuc2Zvcm09Im1hdHJpeCgxLjUsMCwwLDEuNSwwLC02KSIgZD0ibTEyLjE4NiA3LjUwOThjLTEuMDUzNSAwLTEuOTc1NyAwLjU2NjUtMi40Nzg1IDEuNDEwMiAwLjc1MDYxIDAuMzEyNzcgMS4zOTc0IDAuODI2NDggMS44NzMgMS40NzI3aDMuNDg2M2MwLTEuNTkyLTEuMjg4OS0yLjg4MjgtMi44ODA5LTIuODgyOHoiLz4KICA8cGF0aCBkPSJtMjAuNDY1IDIuMzg5NWEyLjE4ODUgMi4xODg1IDAgMCAxLTIuMTg4NCAyLjE4ODUgMi4xODg1IDIuMTg4NSAwIDAgMS0yLjE4ODUtMi4xODg1IDIuMTg4NSAyLjE4ODUgMCAwIDEgMi4xODg1LTIuMTg4NSAyLjE4ODUgMi4xODg1IDAgMCAxIDIuMTg4NCAyLjE4ODV6Ii8+CiAgPHBhdGggdHJhbnNmb3JtPSJtYXRyaXgoMS41LDAsMCwxLjUsMCwtNikiIGQ9Im0zLjU4OTggOC40MjE5Yy0xLjExMjYgMC0yLjAxMzcgMC45MDExMS0yLjAxMzcgMi4wMTM3aDIuODE0NWMwLjI2Nzk3LTAuMzczMDkgMC41OTA3LTAuNzA0MzUgMC45NTg5OC0wLjk3ODUyLTAuMzQ0MzMtMC42MTY4OC0xLjAwMzEtMS4wMzUyLTEuNzU5OC0xLjAzNTJ6Ii8+CiAgPHBhdGggZD0ibTYuOTE1NCA0LjYyM2ExLjUyOTQgMS41Mjk0IDAgMCAxLTEuNTI5NCAxLjUyOTQgMS41Mjk0IDEuNTI5NCAwIDAgMS0xLjUyOTQtMS41Mjk0IDEuNTI5NCAxLjUyOTQgMCAwIDEgMS41Mjk0LTEuNTI5NCAxLjUyOTQgMS41Mjk0IDAgMCAxIDEuNTI5NCAxLjUyOTR6Ii8+CiAgPHBhdGggZD0ibTYuMTM1IDEzLjUzNWMwLTMuMjM5MiAyLjYyNTktNS44NjUgNS44NjUtNS44NjUgMy4yMzkyIDAgNS44NjUgMi42MjU5IDUuODY1IDUuODY1eiIvPgogIDxjaXJjbGUgY3g9IjEyIiBjeT0iMy43Njg1IiByPSIyLjk2ODUiLz4KIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-vega: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbjEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjEyMTIxIj4KICAgIDxwYXRoIGQ9Ik0xMC42IDUuNGwyLjItMy4ySDIuMnY3LjNsNC02LjZ6Ii8+CiAgICA8cGF0aCBkPSJNMTUuOCAyLjJsLTQuNCA2LjZMNyA2LjNsLTQuOCA4djUuNWgxNy42VjIuMmgtNHptLTcgMTUuNEg1LjV2LTQuNGgzLjN2NC40em00LjQgMEg5LjhWOS44aDMuNHY3Ljh6bTQuNCAwaC0zLjRWNi41aDMuNHYxMS4xeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-word: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KIDxnIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzQxNDE0MSI+CiAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiA8L2c+CiA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSguNDMgLjA0MDEpIiBmaWxsPSIjZmZmIj4KICA8cGF0aCBkPSJtNC4xNCA4Ljc2cTAuMDY4Mi0xLjg5IDIuNDItMS44OSAxLjE2IDAgMS42OCAwLjQyIDAuNTY3IDAuNDEgMC41NjcgMS4xNnYzLjQ3cTAgMC40NjIgMC41MTQgMC40NjIgMC4xMDMgMCAwLjItMC4wMjMxdjAuNzE0cS0wLjM5OSAwLjEwMy0wLjY1MSAwLjEwMy0wLjQ1MiAwLTAuNjkzLTAuMjItMC4yMzEtMC4yLTAuMjg0LTAuNjYyLTAuOTU2IDAuODcyLTIgMC44NzItMC45MDMgMC0xLjQ3LTAuNDcyLTAuNTI1LTAuNDcyLTAuNTI1LTEuMjYgMC0wLjI2MiAwLjA0NTItMC40NzIgMC4wNTY3LTAuMjIgMC4xMTYtMC4zNzggMC4wNjgyLTAuMTY4IDAuMjMxLTAuMzA0IDAuMTU4LTAuMTQ3IDAuMjYyLTAuMjQyIDAuMTE2LTAuMDkxNCAwLjM2OC0wLjE2OCAwLjI2Mi0wLjA5MTQgMC4zOTktMC4xMjYgMC4xMzYtMC4wNDUyIDAuNDcyLTAuMTAzIDAuMzM2LTAuMDU3OCAwLjUwNC0wLjA3OTggMC4xNTgtMC4wMjMxIDAuNTY3LTAuMDc5OCAwLjU1Ni0wLjA2ODIgMC43NzctMC4yMjEgMC4yMi0wLjE1MiAwLjIyLTAuNDQxdi0wLjI1MnEwLTAuNDMtMC4zNTctMC42NjItMC4zMzYtMC4yMzEtMC45NzYtMC4yMzEtMC42NjIgMC0wLjk5OCAwLjI2Mi0wLjMzNiAwLjI1Mi0wLjM5OSAwLjc5OHptMS44OSAzLjY4cTAuNzg4IDAgMS4yNi0wLjQxIDAuNTA0LTAuNDIgMC41MDQtMC45MDN2LTEuMDVxLTAuMjg0IDAuMTM2LTAuODYxIDAuMjMxLTAuNTY3IDAuMDkxNC0wLjk4NyAwLjE1OC0wLjQyIDAuMDY4Mi0wLjc2NiAwLjMyNi0wLjMzNiAwLjI1Mi0wLjMzNiAwLjcwNHQwLjMwNCAwLjcwNCAwLjg2MSAwLjI1MnoiIHN0cm9rZS13aWR0aD0iMS4wNSIvPgogIDxwYXRoIGQ9Im0xMCA0LjU2aDAuOTQ1djMuMTVxMC42NTEtMC45NzYgMS44OS0wLjk3NiAxLjE2IDAgMS44OSAwLjg0IDAuNjgyIDAuODQgMC42ODIgMi4zMSAwIDEuNDctMC43MDQgMi40Mi0wLjcwNCAwLjg4Mi0xLjg5IDAuODgyLTEuMjYgMC0xLjg5LTEuMDJ2MC43NjZoLTAuODV6bTIuNjIgMy4wNHEtMC43NDYgMC0xLjE2IDAuNjQtMC40NTIgMC42My0wLjQ1MiAxLjY4IDAgMS4wNSAwLjQ1MiAxLjY4dDEuMTYgMC42M3EwLjc3NyAwIDEuMjYtMC42MyAwLjQ5NC0wLjY0IDAuNDk0LTEuNjggMC0xLjA1LTAuNDcyLTEuNjgtMC40NjItMC42NC0xLjI2LTAuNjR6IiBzdHJva2Utd2lkdGg9IjEuMDUiLz4KICA8cGF0aCBkPSJtMi43MyAxNS44IDEzLjYgMC4wMDgxYzAuMDA2OSAwIDAtMi42IDAtMi42IDAtMC4wMDc4LTEuMTUgMC0xLjE1IDAtMC4wMDY5IDAtMC4wMDgzIDEuNS0wLjAwODMgMS41LTJlLTMgLTAuMDAxNC0xMS4zLTAuMDAxNC0xMS4zLTAuMDAxNGwtMC4wMDU5Mi0xLjVjMC0wLjAwNzgtMS4xNyAwLjAwMTMtMS4xNyAwLjAwMTN6IiBzdHJva2Utd2lkdGg9Ii45NzUiLz4KIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-yaml: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1jb250cmFzdDIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjRDgxQjYwIj4KICAgIDxwYXRoIGQ9Ik03LjIgMTguNnYtNS40TDMgNS42aDMuM2wxLjQgMy4xYy4zLjkuNiAxLjYgMSAyLjUuMy0uOC42LTEuNiAxLTIuNWwxLjQtMy4xaDMuNGwtNC40IDcuNnY1LjVsLTIuOS0uMXoiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxNi41IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxMSIgcj0iMi4xIi8+CiAgPC9nPgo8L3N2Zz4K);
}

/* Icon CSS class declarations */

.jp-AddAboveIcon {
  background-image: var(--jp-icon-add-above);
}

.jp-AddBelowIcon {
  background-image: var(--jp-icon-add-below);
}

.jp-AddIcon {
  background-image: var(--jp-icon-add);
}

.jp-BellIcon {
  background-image: var(--jp-icon-bell);
}

.jp-BugDotIcon {
  background-image: var(--jp-icon-bug-dot);
}

.jp-BugIcon {
  background-image: var(--jp-icon-bug);
}

.jp-BuildIcon {
  background-image: var(--jp-icon-build);
}

.jp-CaretDownEmptyIcon {
  background-image: var(--jp-icon-caret-down-empty);
}

.jp-CaretDownEmptyThinIcon {
  background-image: var(--jp-icon-caret-down-empty-thin);
}

.jp-CaretDownIcon {
  background-image: var(--jp-icon-caret-down);
}

.jp-CaretLeftIcon {
  background-image: var(--jp-icon-caret-left);
}

.jp-CaretRightIcon {
  background-image: var(--jp-icon-caret-right);
}

.jp-CaretUpEmptyThinIcon {
  background-image: var(--jp-icon-caret-up-empty-thin);
}

.jp-CaretUpIcon {
  background-image: var(--jp-icon-caret-up);
}

.jp-CaseSensitiveIcon {
  background-image: var(--jp-icon-case-sensitive);
}

.jp-CheckIcon {
  background-image: var(--jp-icon-check);
}

.jp-CircleEmptyIcon {
  background-image: var(--jp-icon-circle-empty);
}

.jp-CircleIcon {
  background-image: var(--jp-icon-circle);
}

.jp-ClearIcon {
  background-image: var(--jp-icon-clear);
}

.jp-CloseIcon {
  background-image: var(--jp-icon-close);
}

.jp-CodeCheckIcon {
  background-image: var(--jp-icon-code-check);
}

.jp-CodeIcon {
  background-image: var(--jp-icon-code);
}

.jp-CollapseAllIcon {
  background-image: var(--jp-icon-collapse-all);
}

.jp-ConsoleIcon {
  background-image: var(--jp-icon-console);
}

.jp-CopyIcon {
  background-image: var(--jp-icon-copy);
}

.jp-CopyrightIcon {
  background-image: var(--jp-icon-copyright);
}

.jp-CutIcon {
  background-image: var(--jp-icon-cut);
}

.jp-DeleteIcon {
  background-image: var(--jp-icon-delete);
}

.jp-DownloadIcon {
  background-image: var(--jp-icon-download);
}

.jp-DuplicateIcon {
  background-image: var(--jp-icon-duplicate);
}

.jp-EditIcon {
  background-image: var(--jp-icon-edit);
}

.jp-EllipsesIcon {
  background-image: var(--jp-icon-ellipses);
}

.jp-ErrorIcon {
  background-image: var(--jp-icon-error);
}

.jp-ExpandAllIcon {
  background-image: var(--jp-icon-expand-all);
}

.jp-ExtensionIcon {
  background-image: var(--jp-icon-extension);
}

.jp-FastForwardIcon {
  background-image: var(--jp-icon-fast-forward);
}

.jp-FileIcon {
  background-image: var(--jp-icon-file);
}

.jp-FileUploadIcon {
  background-image: var(--jp-icon-file-upload);
}

.jp-FilterDotIcon {
  background-image: var(--jp-icon-filter-dot);
}

.jp-FilterIcon {
  background-image: var(--jp-icon-filter);
}

.jp-FilterListIcon {
  background-image: var(--jp-icon-filter-list);
}

.jp-FolderFavoriteIcon {
  background-image: var(--jp-icon-folder-favorite);
}

.jp-FolderIcon {
  background-image: var(--jp-icon-folder);
}

.jp-HomeIcon {
  background-image: var(--jp-icon-home);
}

.jp-Html5Icon {
  background-image: var(--jp-icon-html5);
}

.jp-ImageIcon {
  background-image: var(--jp-icon-image);
}

.jp-InfoIcon {
  background-image: var(--jp-icon-info);
}

.jp-InspectorIcon {
  background-image: var(--jp-icon-inspector);
}

.jp-JsonIcon {
  background-image: var(--jp-icon-json);
}

.jp-JuliaIcon {
  background-image: var(--jp-icon-julia);
}

.jp-JupyterFaviconIcon {
  background-image: var(--jp-icon-jupyter-favicon);
}

.jp-JupyterIcon {
  background-image: var(--jp-icon-jupyter);
}

.jp-JupyterlabWordmarkIcon {
  background-image: var(--jp-icon-jupyterlab-wordmark);
}

.jp-KernelIcon {
  background-image: var(--jp-icon-kernel);
}

.jp-KeyboardIcon {
  background-image: var(--jp-icon-keyboard);
}

.jp-LaunchIcon {
  background-image: var(--jp-icon-launch);
}

.jp-LauncherIcon {
  background-image: var(--jp-icon-launcher);
}

.jp-LineFormIcon {
  background-image: var(--jp-icon-line-form);
}

.jp-LinkIcon {
  background-image: var(--jp-icon-link);
}

.jp-ListIcon {
  background-image: var(--jp-icon-list);
}

.jp-MarkdownIcon {
  background-image: var(--jp-icon-markdown);
}

.jp-MoveDownIcon {
  background-image: var(--jp-icon-move-down);
}

.jp-MoveUpIcon {
  background-image: var(--jp-icon-move-up);
}

.jp-NewFolderIcon {
  background-image: var(--jp-icon-new-folder);
}

.jp-NotTrustedIcon {
  background-image: var(--jp-icon-not-trusted);
}

.jp-NotebookIcon {
  background-image: var(--jp-icon-notebook);
}

.jp-NumberingIcon {
  background-image: var(--jp-icon-numbering);
}

.jp-OfflineBoltIcon {
  background-image: var(--jp-icon-offline-bolt);
}

.jp-PaletteIcon {
  background-image: var(--jp-icon-palette);
}

.jp-PasteIcon {
  background-image: var(--jp-icon-paste);
}

.jp-PdfIcon {
  background-image: var(--jp-icon-pdf);
}

.jp-PythonIcon {
  background-image: var(--jp-icon-python);
}

.jp-RKernelIcon {
  background-image: var(--jp-icon-r-kernel);
}

.jp-ReactIcon {
  background-image: var(--jp-icon-react);
}

.jp-RedoIcon {
  background-image: var(--jp-icon-redo);
}

.jp-RefreshIcon {
  background-image: var(--jp-icon-refresh);
}

.jp-RegexIcon {
  background-image: var(--jp-icon-regex);
}

.jp-RunIcon {
  background-image: var(--jp-icon-run);
}

.jp-RunningIcon {
  background-image: var(--jp-icon-running);
}

.jp-SaveIcon {
  background-image: var(--jp-icon-save);
}

.jp-SearchIcon {
  background-image: var(--jp-icon-search);
}

.jp-SettingsIcon {
  background-image: var(--jp-icon-settings);
}

.jp-ShareIcon {
  background-image: var(--jp-icon-share);
}

.jp-SpreadsheetIcon {
  background-image: var(--jp-icon-spreadsheet);
}

.jp-StopIcon {
  background-image: var(--jp-icon-stop);
}

.jp-TabIcon {
  background-image: var(--jp-icon-tab);
}

.jp-TableRowsIcon {
  background-image: var(--jp-icon-table-rows);
}

.jp-TagIcon {
  background-image: var(--jp-icon-tag);
}

.jp-TerminalIcon {
  background-image: var(--jp-icon-terminal);
}

.jp-TextEditorIcon {
  background-image: var(--jp-icon-text-editor);
}

.jp-TocIcon {
  background-image: var(--jp-icon-toc);
}

.jp-TreeViewIcon {
  background-image: var(--jp-icon-tree-view);
}

.jp-TrustedIcon {
  background-image: var(--jp-icon-trusted);
}

.jp-UndoIcon {
  background-image: var(--jp-icon-undo);
}

.jp-UserIcon {
  background-image: var(--jp-icon-user);
}

.jp-UsersIcon {
  background-image: var(--jp-icon-users);
}

.jp-VegaIcon {
  background-image: var(--jp-icon-vega);
}

.jp-WordIcon {
  background-image: var(--jp-icon-word);
}

.jp-YamlIcon {
  background-image: var(--jp-icon-yaml);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

.jp-Icon,
.jp-MaterialIcon {
  background-position: center;
  background-repeat: no-repeat;
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-cover {
  background-position: center;
  background-repeat: no-repeat;
  background-size: cover;
}

/**
 * (DEPRECATED) Support for specific CSS icon sizes
 */

.jp-Icon-16 {
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-18 {
  background-size: 18px;
  min-width: 18px;
  min-height: 18px;
}

.jp-Icon-20 {
  background-size: 20px;
  min-width: 20px;
  min-height: 20px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.lm-TabBar .lm-TabBar-addButton {
  align-items: center;
  display: flex;
  padding: 4px;
  padding-bottom: 5px;
  margin-right: 1px;
  background-color: var(--jp-layout-color2);
}

.lm-TabBar .lm-TabBar-addButton:hover {
  background-color: var(--jp-layout-color1);
}

.lm-DockPanel-tabBar .lm-TabBar-tab {
  width: var(--jp-private-horizontal-tab-width);
}

.lm-DockPanel-tabBar .lm-TabBar-content {
  flex: unset;
}

.lm-DockPanel-tabBar[data-orientation='horizontal'] {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for icons as inline SVG HTMLElements
 */

/* recolor the primary elements of an icon */
.jp-icon0[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon1[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon2[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon3[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-accent0[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-accent1[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-accent2[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-accent3[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-accent4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-accent0[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-accent1[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-accent2[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-accent3[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-accent4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-none[fill] {
  fill: none;
}

.jp-icon-none[stroke] {
  stroke: none;
}

/* brand icon colors. Same for light and dark */
.jp-icon-brand0[fill] {
  fill: var(--jp-brand-color0);
}

.jp-icon-brand1[fill] {
  fill: var(--jp-brand-color1);
}

.jp-icon-brand2[fill] {
  fill: var(--jp-brand-color2);
}

.jp-icon-brand3[fill] {
  fill: var(--jp-brand-color3);
}

.jp-icon-brand4[fill] {
  fill: var(--jp-brand-color4);
}

.jp-icon-brand0[stroke] {
  stroke: var(--jp-brand-color0);
}

.jp-icon-brand1[stroke] {
  stroke: var(--jp-brand-color1);
}

.jp-icon-brand2[stroke] {
  stroke: var(--jp-brand-color2);
}

.jp-icon-brand3[stroke] {
  stroke: var(--jp-brand-color3);
}

.jp-icon-brand4[stroke] {
  stroke: var(--jp-brand-color4);
}

/* warn icon colors. Same for light and dark */
.jp-icon-warn0[fill] {
  fill: var(--jp-warn-color0);
}

.jp-icon-warn1[fill] {
  fill: var(--jp-warn-color1);
}

.jp-icon-warn2[fill] {
  fill: var(--jp-warn-color2);
}

.jp-icon-warn3[fill] {
  fill: var(--jp-warn-color3);
}

.jp-icon-warn0[stroke] {
  stroke: var(--jp-warn-color0);
}

.jp-icon-warn1[stroke] {
  stroke: var(--jp-warn-color1);
}

.jp-icon-warn2[stroke] {
  stroke: var(--jp-warn-color2);
}

.jp-icon-warn3[stroke] {
  stroke: var(--jp-warn-color3);
}

/* icon colors that contrast well with each other and most backgrounds */
.jp-icon-contrast0[fill] {
  fill: var(--jp-icon-contrast-color0);
}

.jp-icon-contrast1[fill] {
  fill: var(--jp-icon-contrast-color1);
}

.jp-icon-contrast2[fill] {
  fill: var(--jp-icon-contrast-color2);
}

.jp-icon-contrast3[fill] {
  fill: var(--jp-icon-contrast-color3);
}

.jp-icon-contrast0[stroke] {
  stroke: var(--jp-icon-contrast-color0);
}

.jp-icon-contrast1[stroke] {
  stroke: var(--jp-icon-contrast-color1);
}

.jp-icon-contrast2[stroke] {
  stroke: var(--jp-icon-contrast-color2);
}

.jp-icon-contrast3[stroke] {
  stroke: var(--jp-icon-contrast-color3);
}

.jp-icon-dot[fill] {
  fill: var(--jp-warn-color0);
}

.jp-jupyter-icon-color[fill] {
  fill: var(--jp-jupyter-icon-color, var(--jp-warn-color0));
}

.jp-notebook-icon-color[fill] {
  fill: var(--jp-notebook-icon-color, var(--jp-warn-color0));
}

.jp-json-icon-color[fill] {
  fill: var(--jp-json-icon-color, var(--jp-warn-color1));
}

.jp-console-icon-color[fill] {
  fill: var(--jp-console-icon-color, white);
}

.jp-console-icon-background-color[fill] {
  fill: var(--jp-console-icon-background-color, var(--jp-brand-color1));
}

.jp-terminal-icon-color[fill] {
  fill: var(--jp-terminal-icon-color, var(--jp-layout-color2));
}

.jp-terminal-icon-background-color[fill] {
  fill: var(
    --jp-terminal-icon-background-color,
    var(--jp-inverse-layout-color2)
  );
}

.jp-text-editor-icon-color[fill] {
  fill: var(--jp-text-editor-icon-color, var(--jp-inverse-layout-color3));
}

.jp-inspector-icon-color[fill] {
  fill: var(--jp-inspector-icon-color, var(--jp-inverse-layout-color3));
}

/* CSS for icons in selected filebrowser listing items */
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

.jp-DirListing-item.jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* stylelint-disable selector-max-class, selector-max-compound-selectors */

/**
* TODO: come up with non css-hack solution for showing the busy icon on top
*  of the close icon
* CSS for complex behavior of close icon of tabs in the main area tabbar
*/
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}

.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

/* stylelint-enable selector-max-class, selector-max-compound-selectors */

/* CSS for icons in status bar */
#jp-main-statusbar .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

#jp-main-statusbar .jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* special handling for splash icon CSS. While the theme CSS reloads during
   splash, the splash icon can loose theming. To prevent that, we set a
   default for its color variable */
:root {
  --jp-warn-color0: var(--md-orange-700);
}

/* not sure what to do with this one, used in filebrowser listing */
.jp-DragIcon {
  margin-right: 4px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for alt colors for icons as inline SVG HTMLElements
 */

/* alt recolor the primary elements of an icon */
.jp-icon-alt .jp-icon0[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-alt .jp-icon1[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-alt .jp-icon2[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-alt .jp-icon3[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-alt .jp-icon4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-alt .jp-icon0[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-alt .jp-icon1[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-alt .jp-icon2[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-alt .jp-icon3[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-alt .jp-icon4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* alt recolor the accent elements of an icon */
.jp-icon-alt .jp-icon-accent0[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-alt .jp-icon-accent1[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-alt .jp-icon-accent2[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-alt .jp-icon-accent3[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-alt .jp-icon-accent4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-alt .jp-icon-accent0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-alt .jp-icon-accent1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-alt .jp-icon-accent2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-alt .jp-icon-accent3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-alt .jp-icon-accent4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-icon-hoverShow:not(:hover) .jp-icon-hoverShow-content {
  display: none !important;
}

/**
 * Support for hover colors for icons as inline SVG HTMLElements
 */

/**
 * regular colors
 */

/* recolor the primary elements of an icon */
.jp-icon-hover :hover .jp-icon0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-hover :hover .jp-icon1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-hover :hover .jp-icon2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-hover :hover .jp-icon3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-hover :hover .jp-icon4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-hover :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-hover :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-hover :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-hover :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-hover :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-hover :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-hover :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-hover :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-hover :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-hover :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-hover :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-hover :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-hover :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-hover :hover .jp-icon-none-hover[fill] {
  fill: none;
}

.jp-icon-hover :hover .jp-icon-none-hover[stroke] {
  stroke: none;
}

/**
 * inverse colors
 */

/* inverse recolor the primary elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* inverse recolor the accent elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-IFrame {
  width: 100%;
  height: 100%;
}

.jp-IFrame > iframe {
  border: none;
}

/*
When drag events occur, `lm-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-IFrame {
  position: relative;
}

body.lm-mod-override-cursor .jp-IFrame::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-HoverBox {
  position: fixed;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FormGroup-content fieldset {
  border: none;
  padding: 0;
  min-width: 0;
  width: 100%;
}

/* stylelint-disable selector-max-type */

.jp-FormGroup-content fieldset .jp-inputFieldWrapper input,
.jp-FormGroup-content fieldset .jp-inputFieldWrapper select,
.jp-FormGroup-content fieldset .jp-inputFieldWrapper textarea {
  font-size: var(--jp-content-font-size2);
  border-color: var(--jp-input-border-color);
  border-style: solid;
  border-radius: var(--jp-border-radius);
  border-width: 1px;
  padding: 6px 8px;
  background: none;
  color: var(--jp-ui-font-color0);
  height: inherit;
}

.jp-FormGroup-content fieldset input[type='checkbox'] {
  position: relative;
  top: 2px;
  margin-left: 0;
}

.jp-FormGroup-content button.jp-mod-styled {
  cursor: pointer;
}

.jp-FormGroup-content .checkbox label {
  cursor: pointer;
  font-size: var(--jp-content-font-size1);
}

.jp-FormGroup-content .jp-root > fieldset > legend {
  display: none;
}

.jp-FormGroup-content .jp-root > fieldset > p {
  display: none;
}

/** copy of `input.jp-mod-styled:focus` style */
.jp-FormGroup-content fieldset input:focus,
.jp-FormGroup-content fieldset select:focus {
  -moz-outline-radius: unset;
  outline: var(--jp-border-width) solid var(--md-blue-500);
  outline-offset: -1px;
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-FormGroup-content fieldset input:hover:not(:focus),
.jp-FormGroup-content fieldset select:hover:not(:focus) {
  background-color: var(--jp-border-color2);
}

/* stylelint-enable selector-max-type */

.jp-FormGroup-content .checkbox .field-description {
  /* Disable default description field for checkbox:
   because other widgets do not have description fields,
   we add descriptions to each widget on the field level.
  */
  display: none;
}

.jp-FormGroup-content #root__description {
  display: none;
}

.jp-FormGroup-content .jp-modifiedIndicator {
  width: 5px;
  background-color: var(--jp-brand-color2);
  margin-top: 0;
  margin-left: calc(var(--jp-private-settingeditor-modifier-indent) * -1);
  flex-shrink: 0;
}

.jp-FormGroup-content .jp-modifiedIndicator.jp-errorIndicator {
  background-color: var(--jp-error-color0);
  margin-right: 0.5em;
}

/* RJSF ARRAY style */

.jp-arrayFieldWrapper legend {
  font-size: var(--jp-content-font-size2);
  color: var(--jp-ui-font-color0);
  flex-basis: 100%;
  padding: 4px 0;
  font-weight: var(--jp-content-heading-font-weight);
  border-bottom: 1px solid var(--jp-border-color2);
}

.jp-arrayFieldWrapper .field-description {
  padding: 4px 0;
  white-space: pre-wrap;
}

.jp-arrayFieldWrapper .array-item {
  width: 100%;
  border: 1px solid var(--jp-border-color2);
  border-radius: 4px;
  margin: 4px;
}

.jp-ArrayOperations {
  display: flex;
  margin-left: 8px;
}

.jp-ArrayOperationsButton {
  margin: 2px;
}

.jp-ArrayOperationsButton .jp-icon3[fill] {
  fill: var(--jp-ui-font-color0);
}

button.jp-ArrayOperationsButton.jp-mod-styled:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

/* RJSF form validation error */

.jp-FormGroup-content .validationErrors {
  color: var(--jp-error-color0);
}

/* Hide panel level error as duplicated the field level error */
.jp-FormGroup-content .panel.errors {
  display: none;
}

/* RJSF normal content (settings-editor) */

.jp-FormGroup-contentNormal {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

.jp-FormGroup-contentNormal .jp-FormGroup-contentItem {
  margin-left: 7px;
  color: var(--jp-ui-font-color0);
}

.jp-FormGroup-contentNormal .jp-FormGroup-description {
  flex-basis: 100%;
  padding: 4px 7px;
}

.jp-FormGroup-contentNormal .jp-FormGroup-default {
  flex-basis: 100%;
  padding: 4px 7px;
}

.jp-FormGroup-contentNormal .jp-FormGroup-fieldLabel {
  font-size: var(--jp-content-font-size1);
  font-weight: normal;
  min-width: 120px;
}

.jp-FormGroup-contentNormal fieldset:not(:first-child) {
  margin-left: 7px;
}

.jp-FormGroup-contentNormal .field-array-of-string .array-item {
  /* Display `jp-ArrayOperations` buttons side-by-side with content except
    for small screens where flex-wrap will place them one below the other.
  */
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

.jp-FormGroup-contentNormal .jp-objectFieldWrapper .form-group {
  padding: 2px 8px 2px var(--jp-private-settingeditor-modifier-indent);
  margin-top: 2px;
}

/* RJSF compact content (metadata-form) */

.jp-FormGroup-content.jp-FormGroup-contentCompact {
  width: 100%;
}

.jp-FormGroup-contentCompact .form-group {
  display: flex;
  padding: 0.5em 0.2em 0.5em 0;
}

.jp-FormGroup-contentCompact
  .jp-FormGroup-compactTitle
  .jp-FormGroup-description {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color2);
}

.jp-FormGroup-contentCompact .jp-FormGroup-fieldLabel {
  padding-bottom: 0.3em;
}

.jp-FormGroup-contentCompact .jp-inputFieldWrapper .form-control {
  width: 100%;
  box-sizing: border-box;
}

.jp-FormGroup-contentCompact .jp-arrayFieldWrapper .jp-FormGroup-compactTitle {
  padding-bottom: 7px;
}

.jp-FormGroup-contentCompact
  .jp-objectFieldWrapper
  .jp-objectFieldWrapper
  .form-group {
  padding: 2px 8px 2px var(--jp-private-settingeditor-modifier-indent);
  margin-top: 2px;
}

.jp-FormGroup-contentCompact ul.error-detail {
  margin-block-start: 0.5em;
  margin-block-end: 0.5em;
  padding-inline-start: 1em;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-SidePanel {
  display: flex;
  flex-direction: column;
  min-width: var(--jp-sidebar-min-width);
  overflow-y: auto;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  font-size: var(--jp-ui-font-size1);
}

.jp-SidePanel-header {
  flex: 0 0 auto;
  display: flex;
  border-bottom: var(--jp-border-width) solid var(--jp-border-color2);
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin: 0;
  padding: 2px;
  text-transform: uppercase;
}

.jp-SidePanel-toolbar {
  flex: 0 0 auto;
}

.jp-SidePanel-content {
  flex: 1 1 auto;
}

.jp-SidePanel-toolbar,
.jp-AccordionPanel-toolbar {
  height: var(--jp-private-toolbar-height);
}

.jp-SidePanel-toolbar.jp-Toolbar-micro {
  display: none;
}

.lm-AccordionPanel .jp-AccordionPanel-title {
  box-sizing: border-box;
  line-height: 25px;
  margin: 0;
  display: flex;
  align-items: center;
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  font-size: var(--jp-ui-font-size0);
}

.jp-AccordionPanel-title {
  cursor: pointer;
  user-select: none;
  -moz-user-select: none;
  -webkit-user-select: none;
  text-transform: uppercase;
}

.lm-AccordionPanel[data-orientation='horizontal'] > .jp-AccordionPanel-title {
  /* Title is rotated for horizontal accordion panel using CSS */
  display: block;
  transform-origin: top left;
  transform: rotate(-90deg) translate(-100%);
}

.jp-AccordionPanel-title .lm-AccordionPanel-titleLabel {
  user-select: none;
  text-overflow: ellipsis;
  white-space: nowrap;
  overflow: hidden;
}

.jp-AccordionPanel-title .lm-AccordionPanel-titleCollapser {
  transform: rotate(-90deg);
  margin: auto 0;
  height: 16px;
}

.jp-AccordionPanel-title.lm-mod-expanded .lm-AccordionPanel-titleCollapser {
  transform: rotate(0deg);
}

.lm-AccordionPanel .jp-AccordionPanel-toolbar {
  background: none;
  box-shadow: none;
  border: none;
  margin-left: auto;
}

.lm-AccordionPanel .lm-SplitPanel-handle:hover {
  background: var(--jp-layout-color3);
}

.jp-text-truncated {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Spinner {
  position: absolute;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 10;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-layout-color0);
  outline: none;
}

.jp-SpinnerContent {
  font-size: 10px;
  margin: 50px auto;
  text-indent: -9999em;
  width: 3em;
  height: 3em;
  border-radius: 50%;
  background: var(--jp-brand-color3);
  background: linear-gradient(
    to right,
    #f37626 10%,
    rgba(255, 255, 255, 0) 42%
  );
  position: relative;
  animation: load3 1s infinite linear, fadeIn 1s;
}

.jp-SpinnerContent::before {
  width: 50%;
  height: 50%;
  background: #f37626;
  border-radius: 100% 0 0;
  position: absolute;
  top: 0;
  left: 0;
  content: '';
}

.jp-SpinnerContent::after {
  background: var(--jp-layout-color0);
  width: 75%;
  height: 75%;
  border-radius: 50%;
  content: '';
  margin: auto;
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
}

@keyframes fadeIn {
  0% {
    opacity: 0;
  }

  100% {
    opacity: 1;
  }
}

@keyframes load3 {
  0% {
    transform: rotate(0deg);
  }

  100% {
    transform: rotate(360deg);
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

button.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: none;
  box-sizing: border-box;
  text-align: center;
  line-height: 32px;
  height: 32px;
  padding: 0 12px;
  letter-spacing: 0.8px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input.jp-mod-styled {
  background: var(--jp-input-background);
  height: 28px;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color1);
  padding-left: 7px;
  padding-right: 7px;
  font-size: var(--jp-ui-font-size2);
  color: var(--jp-ui-font-color0);
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input[type='checkbox'].jp-mod-styled {
  appearance: checkbox;
  -webkit-appearance: checkbox;
  -moz-appearance: checkbox;
  height: auto;
}

input.jp-mod-styled:focus {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-select-wrapper {
  display: flex;
  position: relative;
  flex-direction: column;
  padding: 1px;
  background-color: var(--jp-layout-color1);
  box-sizing: border-box;
  margin-bottom: 12px;
}

.jp-select-wrapper:not(.multiple) {
  height: 28px;
}

.jp-select-wrapper.jp-mod-focused select.jp-mod-styled {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-input-active-background);
}

select.jp-mod-styled:hover {
  cursor: pointer;
  color: var(--jp-ui-font-color0);
  background-color: var(--jp-input-hover-background);
  box-shadow: inset 0 0 1px rgba(0, 0, 0, 0.5);
}

select.jp-mod-styled {
  flex: 1 1 auto;
  width: 100%;
  font-size: var(--jp-ui-font-size2);
  background: var(--jp-input-background);
  color: var(--jp-ui-font-color0);
  padding: 0 25px 0 8px;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

select.jp-mod-styled:not([multiple]) {
  height: 32px;
}

select.jp-mod-styled[multiple] {
  max-height: 200px;
  overflow-y: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-switch {
  display: flex;
  align-items: center;
  padding-left: 4px;
  padding-right: 4px;
  font-size: var(--jp-ui-font-size1);
  background-color: transparent;
  color: var(--jp-ui-font-color1);
  border: none;
  height: 20px;
}

.jp-switch:hover {
  background-color: var(--jp-layout-color2);
}

.jp-switch-label {
  margin-right: 5px;
  font-family: var(--jp-ui-font-family);
}

.jp-switch-track {
  cursor: pointer;
  background-color: var(--jp-switch-color, var(--jp-border-color1));
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 34px;
  height: 16px;
  width: 35px;
  position: relative;
}

.jp-switch-track::before {
  content: '';
  position: absolute;
  height: 10px;
  width: 10px;
  margin: 3px;
  left: 0;
  background-color: var(--jp-ui-inverse-font-color1);
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 50%;
}

.jp-switch[aria-checked='true'] .jp-switch-track {
  background-color: var(--jp-switch-true-position-color, var(--jp-warn-color0));
}

.jp-switch[aria-checked='true'] .jp-switch-track::before {
  /* track width (35) - margins (3 + 3) - thumb width (10) */
  left: 19px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toolbar-height: calc(
    28px + var(--jp-border-width)
  ); /* leave 28px for content */
}

.jp-Toolbar {
  color: var(--jp-ui-font-color1);
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: 2px;
  z-index: 8;
  overflow-x: hidden;
}

/* Toolbar items */

.jp-Toolbar > .jp-Toolbar-item.jp-Toolbar-spacer {
  flex-grow: 1;
  flex-shrink: 1;
}

.jp-Toolbar-item.jp-Toolbar-kernelStatus {
  display: inline-block;
  width: 32px;
  background-repeat: no-repeat;
  background-position: center;
  background-size: 16px;
}

.jp-Toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  display: flex;
  padding-left: 1px;
  padding-right: 1px;
  font-size: var(--jp-ui-font-size1);
  line-height: var(--jp-private-toolbar-height);
  height: 100%;
}

/* Toolbar buttons */

/* This is the div we use to wrap the react component into a Widget */
div.jp-ToolbarButton {
  color: transparent;
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0;
  margin: 0;
}

button.jp-ToolbarButtonComponent {
  background: var(--jp-layout-color1);
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0 6px;
  margin: 0;
  height: 24px;
  border-radius: var(--jp-border-radius);
  display: flex;
  align-items: center;
  text-align: center;
  font-size: 14px;
  min-width: unset;
  min-height: unset;
}

button.jp-ToolbarButtonComponent:disabled {
  opacity: 0.4;
}

button.jp-ToolbarButtonComponent > span {
  padding: 0;
  flex: 0 0 auto;
}

button.jp-ToolbarButtonComponent .jp-ToolbarButtonComponent-label {
  font-size: var(--jp-ui-font-size1);
  line-height: 100%;
  padding-left: 2px;
  color: var(--jp-ui-font-color1);
  font-family: var(--jp-ui-font-family);
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar.jp-Toolbar-micro {
  padding: 0;
  min-height: 0;
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar {
  border: none;
  box-shadow: none;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-WindowedPanel-outer {
  position: relative;
  overflow-y: auto;
}

.jp-WindowedPanel-inner {
  position: relative;
}

.jp-WindowedPanel-window {
  position: absolute;
  left: 0;
  right: 0;
  overflow: visible;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* Sibling imports */

body {
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
}

/* Disable native link decoration styles everywhere outside of dialog boxes */
a {
  text-decoration: unset;
  color: unset;
}

a:hover {
  text-decoration: unset;
  color: unset;
}

/* Accessibility for links inside dialog box text */
.jp-Dialog-content a {
  text-decoration: revert;
  color: var(--jp-content-link-color);
}

.jp-Dialog-content a:hover {
  text-decoration: revert;
}

/* Styles for ui-components */
.jp-Button {
  color: var(--jp-ui-font-color2);
  border-radius: var(--jp-border-radius);
  padding: 0 12px;
  font-size: var(--jp-ui-font-size1);

  /* Copy from blueprint 3 */
  display: inline-flex;
  flex-direction: row;
  border: none;
  cursor: pointer;
  align-items: center;
  justify-content: center;
  text-align: left;
  vertical-align: middle;
  min-height: 30px;
  min-width: 30px;
}

.jp-Button:disabled {
  cursor: not-allowed;
}

.jp-Button:empty {
  padding: 0 !important;
}

.jp-Button.jp-mod-small {
  min-height: 24px;
  min-width: 24px;
  font-size: 12px;
  padding: 0 7px;
}

/* Use our own theme for hover styles */
.jp-Button.jp-mod-minimal:hover {
  background-color: var(--jp-layout-color2);
}

.jp-Button.jp-mod-minimal {
  background: none;
}

.jp-InputGroup {
  display: block;
  position: relative;
}

.jp-InputGroup input {
  box-sizing: border-box;
  border: none;
  border-radius: 0;
  background-color: transparent;
  color: var(--jp-ui-font-color0);
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
  padding-bottom: 0;
  padding-top: 0;
  padding-left: 10px;
  padding-right: 28px;
  position: relative;
  width: 100%;
  -webkit-appearance: none;
  -moz-appearance: none;
  appearance: none;
  font-size: 14px;
  font-weight: 400;
  height: 30px;
  line-height: 30px;
  outline: none;
  vertical-align: middle;
}

.jp-InputGroup input:focus {
  box-shadow: inset 0 0 0 var(--jp-border-width)
      var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-InputGroup input:disabled {
  cursor: not-allowed;
  resize: block;
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color2);
}

.jp-InputGroup input:disabled ~ span {
  cursor: not-allowed;
  color: var(--jp-ui-font-color2);
}

.jp-InputGroup input::placeholder,
input::placeholder {
  color: var(--jp-ui-font-color2);
}

.jp-InputGroupAction {
  position: absolute;
  bottom: 1px;
  right: 0;
  padding: 6px;
}

.jp-HTMLSelect.jp-DefaultStyle select {
  background-color: initial;
  border: none;
  border-radius: 0;
  box-shadow: none;
  color: var(--jp-ui-font-color0);
  display: block;
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  height: 24px;
  line-height: 14px;
  padding: 0 25px 0 10px;
  text-align: left;
  -moz-appearance: none;
  -webkit-appearance: none;
}

.jp-HTMLSelect.jp-DefaultStyle select:disabled {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color2);
  cursor: not-allowed;
  resize: block;
}

.jp-HTMLSelect.jp-DefaultStyle select:disabled ~ span {
  cursor: not-allowed;
}

/* Use our own theme for hover and option styles */
/* stylelint-disable-next-line selector-max-type */
.jp-HTMLSelect.jp-DefaultStyle select:hover,
.jp-HTMLSelect.jp-DefaultStyle select > option {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color0);
}

select {
  box-sizing: border-box;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-StatusBar-Widget {
  display: flex;
  align-items: center;
  background: var(--jp-layout-color2);
  min-height: var(--jp-statusbar-height);
  justify-content: space-between;
  padding: 0 10px;
}

.jp-StatusBar-Left {
  display: flex;
  align-items: center;
  flex-direction: row;
}

.jp-StatusBar-Middle {
  display: flex;
  align-items: center;
}

.jp-StatusBar-Right {
  display: flex;
  align-items: center;
  flex-direction: row-reverse;
}

.jp-StatusBar-Item {
  max-height: var(--jp-statusbar-height);
  margin: 0 2px;
  height: var(--jp-statusbar-height);
  white-space: nowrap;
  text-overflow: ellipsis;
  color: var(--jp-ui-font-color1);
  padding: 0 6px;
}

.jp-mod-highlighted:hover {
  background-color: var(--jp-layout-color3);
}

.jp-mod-clicked {
  background-color: var(--jp-brand-color1);
}

.jp-mod-clicked:hover {
  background-color: var(--jp-brand-color0);
}

.jp-mod-clicked .jp-StatusBar-TextItem {
  color: var(--jp-ui-inverse-font-color1);
}

.jp-StatusBar-HoverItem {
  box-shadow: '0px 4px 4px rgba(0, 0, 0, 0.25)';
}

.jp-StatusBar-TextItem {
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  line-height: 24px;
  color: var(--jp-ui-font-color1);
}

.jp-StatusBar-GroupItem {
  display: flex;
  align-items: center;
  flex-direction: row;
}

.jp-Statusbar-ProgressCircle svg {
  display: block;
  margin: 0 auto;
  width: 16px;
  height: 24px;
  align-self: normal;
}

.jp-Statusbar-ProgressCircle path {
  fill: var(--jp-inverse-layout-color3);
}

.jp-Statusbar-ProgressBar-progress-bar {
  height: 10px;
  width: 100px;
  border: solid 0.25px var(--jp-brand-color2);
  border-radius: 3px;
  overflow: hidden;
  align-self: center;
}

.jp-Statusbar-ProgressBar-progress-bar > div {
  background-color: var(--jp-brand-color2);
  background-image: linear-gradient(
    -45deg,
    rgba(255, 255, 255, 0.2) 25%,
    transparent 25%,
    transparent 50%,
    rgba(255, 255, 255, 0.2) 50%,
    rgba(255, 255, 255, 0.2) 75%,
    transparent 75%,
    transparent
  );
  background-size: 40px 40px;
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 14px;
  color: #fff;
  text-align: center;
  animation: jp-Statusbar-ExecutionTime-progress-bar 2s linear infinite;
}

.jp-Statusbar-ProgressBar-progress-bar p {
  color: var(--jp-ui-font-color1);
  font-family: var(--jp-ui-font-family);
  font-size: var(--jp-ui-font-size1);
  line-height: 10px;
  width: 100px;
}

@keyframes jp-Statusbar-ExecutionTime-progress-bar {
  0% {
    background-position: 0 0;
  }

  100% {
    background-position: 40px 40px;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-commandpalette-search-height: 28px;
}

/*-----------------------------------------------------------------------------
| Overall styles
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  padding-bottom: 0;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);

  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Modal variant
|----------------------------------------------------------------------------*/

.jp-ModalCommandPalette {
  position: absolute;
  z-index: 10000;
  top: 38px;
  left: 30%;
  margin: 0;
  padding: 4px;
  width: 40%;
  box-shadow: var(--jp-elevation-z4);
  border-radius: 4px;
  background: var(--jp-layout-color0);
}

.jp-ModalCommandPalette .lm-CommandPalette {
  max-height: 40vh;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-close-icon::after {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-header {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-item {
  margin-left: 4px;
  margin-right: 4px;
}

.jp-ModalCommandPalette
  .lm-CommandPalette
  .lm-CommandPalette-item.lm-mod-disabled {
  display: none;
}

/*-----------------------------------------------------------------------------
| Search
|----------------------------------------------------------------------------*/

.lm-CommandPalette-search {
  padding: 4px;
  background-color: var(--jp-layout-color1);
  z-index: 2;
}

.lm-CommandPalette-wrapper {
  overflow: overlay;
  padding: 0 9px;
  background-color: var(--jp-input-active-background);
  height: 30px;
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.lm-CommandPalette.lm-mod-focused .lm-CommandPalette-wrapper {
  box-shadow: inset 0 0 0 1px var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-SearchIconGroup {
  color: white;
  background-color: var(--jp-brand-color1);
  position: absolute;
  top: 4px;
  right: 4px;
  padding: 5px 5px 1px;
}

.jp-SearchIconGroup svg {
  height: 20px;
  width: 20px;
}

.jp-SearchIconGroup .jp-icon3[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-input {
  background: transparent;
  width: calc(100% - 18px);
  float: left;
  border: none;
  outline: none;
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  line-height: var(--jp-private-commandpalette-search-height);
}

.lm-CommandPalette-input::-webkit-input-placeholder,
.lm-CommandPalette-input::-moz-placeholder,
.lm-CommandPalette-input:-ms-input-placeholder {
  color: var(--jp-ui-font-color2);
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Results
|----------------------------------------------------------------------------*/

.lm-CommandPalette-header:first-child {
  margin-top: 0;
}

.lm-CommandPalette-header {
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin-top: 8px;
  padding: 8px 0 8px 12px;
  text-transform: uppercase;
}

.lm-CommandPalette-header.lm-mod-active {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-header > mark {
  background-color: transparent;
  font-weight: bold;
  color: var(--jp-ui-font-color1);
}

.lm-CommandPalette-item {
  padding: 4px 12px 4px 4px;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  font-weight: 400;
  display: flex;
}

.lm-CommandPalette-item.lm-mod-disabled {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item.lm-mod-active {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item.lm-mod-active .lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-inverse-font-color0);
}

.lm-CommandPalette-item.lm-mod-active .jp-icon-selectable[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-item.lm-mod-active:hover:not(.lm-mod-disabled) {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item:hover:not(.lm-mod-active):not(.lm-mod-disabled) {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-itemContent {
  overflow: hidden;
}

.lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.lm-CommandPalette-item.lm-mod-disabled mark {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item .lm-CommandPalette-itemIcon {
  margin: 0 4px 0 0;
  position: relative;
  width: 16px;
  top: 2px;
  flex: 0 0 auto;
}

.lm-CommandPalette-item.lm-mod-disabled .lm-CommandPalette-itemIcon {
  opacity: 0.6;
}

.lm-CommandPalette-item .lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemCaption {
  display: none;
}

.lm-CommandPalette-content {
  background-color: var(--jp-layout-color1);
}

.lm-CommandPalette-content:empty::after {
  content: 'No results';
  margin: auto;
  margin-top: 20px;
  width: 100px;
  display: block;
  font-size: var(--jp-ui-font-size2);
  font-family: var(--jp-ui-font-family);
  font-weight: lighter;
}

.lm-CommandPalette-emptyMessage {
  text-align: center;
  margin-top: 24px;
  line-height: 1.32;
  padding: 0 8px;
  color: var(--jp-content-font-color3);
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Dialog {
  position: absolute;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  top: 0;
  left: 0;
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-dialog-background);
}

.jp-Dialog-content {
  display: flex;
  flex-direction: column;
  margin-left: auto;
  margin-right: auto;
  background: var(--jp-layout-color1);
  padding: 24px 24px 12px;
  min-width: 300px;
  min-height: 150px;
  max-width: 1000px;
  max-height: 500px;
  box-sizing: border-box;
  box-shadow: var(--jp-elevation-z20);
  word-wrap: break-word;
  border-radius: var(--jp-border-radius);

  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color1);
  resize: both;
}

.jp-Dialog-content.jp-Dialog-content-small {
  max-width: 500px;
}

.jp-Dialog-button {
  overflow: visible;
}

button.jp-Dialog-button:focus {
  outline: 1px solid var(--jp-brand-color1);
  outline-offset: 4px;
  -moz-outline-radius: 0;
}

button.jp-Dialog-button:focus::-moz-focus-inner {
  border: 0;
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-accept:focus,
button.jp-Dialog-button.jp-mod-styled.jp-mod-warn:focus,
button.jp-Dialog-button.jp-mod-styled.jp-mod-reject:focus {
  outline-offset: 4px;
  -moz-outline-radius: 0;
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-accept:focus {
  outline: 1px solid var(--jp-accept-color-normal, var(--jp-brand-color1));
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-warn:focus {
  outline: 1px solid var(--jp-warn-color-normal, var(--jp-error-color1));
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-reject:focus {
  outline: 1px solid var(--jp-reject-color-normal, var(--md-grey-600));
}

button.jp-Dialog-close-button {
  padding: 0;
  height: 100%;
  min-width: unset;
  min-height: unset;
}

.jp-Dialog-header {
  display: flex;
  justify-content: space-between;
  flex: 0 0 auto;
  padding-bottom: 12px;
  font-size: var(--jp-ui-font-size3);
  font-weight: 400;
  color: var(--jp-ui-font-color1);
}

.jp-Dialog-body {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  font-size: var(--jp-ui-font-size1);
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

.jp-Dialog-footer {
  display: flex;
  flex-direction: row;
  justify-content: flex-end;
  align-items: center;
  flex: 0 0 auto;
  margin-left: -12px;
  margin-right: -12px;
  padding: 12px;
}

.jp-Dialog-checkbox {
  padding-right: 5px;
}

.jp-Dialog-checkbox > input:focus-visible {
  outline: 1px solid var(--jp-input-active-border-color);
  outline-offset: 1px;
}

.jp-Dialog-spacer {
  flex: 1 1 auto;
}

.jp-Dialog-title {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.jp-Dialog-body > .jp-select-wrapper {
  width: 100%;
}

.jp-Dialog-body > button {
  padding: 0 16px;
}

.jp-Dialog-body > label {
  line-height: 1.4;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-button.jp-mod-styled:not(:last-child) {
  margin-right: 12px;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-Input-Boolean-Dialog {
  flex-direction: row-reverse;
  align-items: end;
  width: 100%;
}

.jp-Input-Boolean-Dialog > label {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MainAreaWidget > :focus {
  outline: none;
}

.jp-MainAreaWidget .jp-MainAreaWidget-error {
  padding: 6px;
}

.jp-MainAreaWidget .jp-MainAreaWidget-error > pre {
  width: auto;
  padding: 10px;
  background: var(--jp-error-color3);
  border: var(--jp-border-width) solid var(--jp-error-color1);
  border-radius: var(--jp-border-radius);
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  white-space: pre-wrap;
  word-wrap: break-word;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/**
 * google-material-color v1.2.6
 * https://github.com/danlevan/google-material-color
 */
:root {
  --md-red-50: #ffebee;
  --md-red-100: #ffcdd2;
  --md-red-200: #ef9a9a;
  --md-red-300: #e57373;
  --md-red-400: #ef5350;
  --md-red-500: #f44336;
  --md-red-600: #e53935;
  --md-red-700: #d32f2f;
  --md-red-800: #c62828;
  --md-red-900: #b71c1c;
  --md-red-A100: #ff8a80;
  --md-red-A200: #ff5252;
  --md-red-A400: #ff1744;
  --md-red-A700: #d50000;
  --md-pink-50: #fce4ec;
  --md-pink-100: #f8bbd0;
  --md-pink-200: #f48fb1;
  --md-pink-300: #f06292;
  --md-pink-400: #ec407a;
  --md-pink-500: #e91e63;
  --md-pink-600: #d81b60;
  --md-pink-700: #c2185b;
  --md-pink-800: #ad1457;
  --md-pink-900: #880e4f;
  --md-pink-A100: #ff80ab;
  --md-pink-A200: #ff4081;
  --md-pink-A400: #f50057;
  --md-pink-A700: #c51162;
  --md-purple-50: #f3e5f5;
  --md-purple-100: #e1bee7;
  --md-purple-200: #ce93d8;
  --md-purple-300: #ba68c8;
  --md-purple-400: #ab47bc;
  --md-purple-500: #9c27b0;
  --md-purple-600: #8e24aa;
  --md-purple-700: #7b1fa2;
  --md-purple-800: #6a1b9a;
  --md-purple-900: #4a148c;
  --md-purple-A100: #ea80fc;
  --md-purple-A200: #e040fb;
  --md-purple-A400: #d500f9;
  --md-purple-A700: #a0f;
  --md-deep-purple-50: #ede7f6;
  --md-deep-purple-100: #d1c4e9;
  --md-deep-purple-200: #b39ddb;
  --md-deep-purple-300: #9575cd;
  --md-deep-purple-400: #7e57c2;
  --md-deep-purple-500: #673ab7;
  --md-deep-purple-600: #5e35b1;
  --md-deep-purple-700: #512da8;
  --md-deep-purple-800: #4527a0;
  --md-deep-purple-900: #311b92;
  --md-deep-purple-A100: #b388ff;
  --md-deep-purple-A200: #7c4dff;
  --md-deep-purple-A400: #651fff;
  --md-deep-purple-A700: #6200ea;
  --md-indigo-50: #e8eaf6;
  --md-indigo-100: #c5cae9;
  --md-indigo-200: #9fa8da;
  --md-indigo-300: #7986cb;
  --md-indigo-400: #5c6bc0;
  --md-indigo-500: #3f51b5;
  --md-indigo-600: #3949ab;
  --md-indigo-700: #303f9f;
  --md-indigo-800: #283593;
  --md-indigo-900: #1a237e;
  --md-indigo-A100: #8c9eff;
  --md-indigo-A200: #536dfe;
  --md-indigo-A400: #3d5afe;
  --md-indigo-A700: #304ffe;
  --md-blue-50: #e3f2fd;
  --md-blue-100: #bbdefb;
  --md-blue-200: #90caf9;
  --md-blue-300: #64b5f6;
  --md-blue-400: #42a5f5;
  --md-blue-500: #2196f3;
  --md-blue-600: #1e88e5;
  --md-blue-700: #1976d2;
  --md-blue-800: #1565c0;
  --md-blue-900: #0d47a1;
  --md-blue-A100: #82b1ff;
  --md-blue-A200: #448aff;
  --md-blue-A400: #2979ff;
  --md-blue-A700: #2962ff;
  --md-light-blue-50: #e1f5fe;
  --md-light-blue-100: #b3e5fc;
  --md-light-blue-200: #81d4fa;
  --md-light-blue-300: #4fc3f7;
  --md-light-blue-400: #29b6f6;
  --md-light-blue-500: #03a9f4;
  --md-light-blue-600: #039be5;
  --md-light-blue-700: #0288d1;
  --md-light-blue-800: #0277bd;
  --md-light-blue-900: #01579b;
  --md-light-blue-A100: #80d8ff;
  --md-light-blue-A200: #40c4ff;
  --md-light-blue-A400: #00b0ff;
  --md-light-blue-A700: #0091ea;
  --md-cyan-50: #e0f7fa;
  --md-cyan-100: #b2ebf2;
  --md-cyan-200: #80deea;
  --md-cyan-300: #4dd0e1;
  --md-cyan-400: #26c6da;
  --md-cyan-500: #00bcd4;
  --md-cyan-600: #00acc1;
  --md-cyan-700: #0097a7;
  --md-cyan-800: #00838f;
  --md-cyan-900: #006064;
  --md-cyan-A100: #84ffff;
  --md-cyan-A200: #18ffff;
  --md-cyan-A400: #00e5ff;
  --md-cyan-A700: #00b8d4;
  --md-teal-50: #e0f2f1;
  --md-teal-100: #b2dfdb;
  --md-teal-200: #80cbc4;
  --md-teal-300: #4db6ac;
  --md-teal-400: #26a69a;
  --md-teal-500: #009688;
  --md-teal-600: #00897b;
  --md-teal-700: #00796b;
  --md-teal-800: #00695c;
  --md-teal-900: #004d40;
  --md-teal-A100: #a7ffeb;
  --md-teal-A200: #64ffda;
  --md-teal-A400: #1de9b6;
  --md-teal-A700: #00bfa5;
  --md-green-50: #e8f5e9;
  --md-green-100: #c8e6c9;
  --md-green-200: #a5d6a7;
  --md-green-300: #81c784;
  --md-green-400: #66bb6a;
  --md-green-500: #4caf50;
  --md-green-600: #43a047;
  --md-green-700: #388e3c;
  --md-green-800: #2e7d32;
  --md-green-900: #1b5e20;
  --md-green-A100: #b9f6ca;
  --md-green-A200: #69f0ae;
  --md-green-A400: #00e676;
  --md-green-A700: #00c853;
  --md-light-green-50: #f1f8e9;
  --md-light-green-100: #dcedc8;
  --md-light-green-200: #c5e1a5;
  --md-light-green-300: #aed581;
  --md-light-green-400: #9ccc65;
  --md-light-green-500: #8bc34a;
  --md-light-green-600: #7cb342;
  --md-light-green-700: #689f38;
  --md-light-green-800: #558b2f;
  --md-light-green-900: #33691e;
  --md-light-green-A100: #ccff90;
  --md-light-green-A200: #b2ff59;
  --md-light-green-A400: #76ff03;
  --md-light-green-A700: #64dd17;
  --md-lime-50: #f9fbe7;
  --md-lime-100: #f0f4c3;
  --md-lime-200: #e6ee9c;
  --md-lime-300: #dce775;
  --md-lime-400: #d4e157;
  --md-lime-500: #cddc39;
  --md-lime-600: #c0ca33;
  --md-lime-700: #afb42b;
  --md-lime-800: #9e9d24;
  --md-lime-900: #827717;
  --md-lime-A100: #f4ff81;
  --md-lime-A200: #eeff41;
  --md-lime-A400: #c6ff00;
  --md-lime-A700: #aeea00;
  --md-yellow-50: #fffde7;
  --md-yellow-100: #fff9c4;
  --md-yellow-200: #fff59d;
  --md-yellow-300: #fff176;
  --md-yellow-400: #ffee58;
  --md-yellow-500: #ffeb3b;
  --md-yellow-600: #fdd835;
  --md-yellow-700: #fbc02d;
  --md-yellow-800: #f9a825;
  --md-yellow-900: #f57f17;
  --md-yellow-A100: #ffff8d;
  --md-yellow-A200: #ff0;
  --md-yellow-A400: #ffea00;
  --md-yellow-A700: #ffd600;
  --md-amber-50: #fff8e1;
  --md-amber-100: #ffecb3;
  --md-amber-200: #ffe082;
  --md-amber-300: #ffd54f;
  --md-amber-400: #ffca28;
  --md-amber-500: #ffc107;
  --md-amber-600: #ffb300;
  --md-amber-700: #ffa000;
  --md-amber-800: #ff8f00;
  --md-amber-900: #ff6f00;
  --md-amber-A100: #ffe57f;
  --md-amber-A200: #ffd740;
  --md-amber-A400: #ffc400;
  --md-amber-A700: #ffab00;
  --md-orange-50: #fff3e0;
  --md-orange-100: #ffe0b2;
  --md-orange-200: #ffcc80;
  --md-orange-300: #ffb74d;
  --md-orange-400: #ffa726;
  --md-orange-500: #ff9800;
  --md-orange-600: #fb8c00;
  --md-orange-700: #f57c00;
  --md-orange-800: #ef6c00;
  --md-orange-900: #e65100;
  --md-orange-A100: #ffd180;
  --md-orange-A200: #ffab40;
  --md-orange-A400: #ff9100;
  --md-orange-A700: #ff6d00;
  --md-deep-orange-50: #fbe9e7;
  --md-deep-orange-100: #ffccbc;
  --md-deep-orange-200: #ffab91;
  --md-deep-orange-300: #ff8a65;
  --md-deep-orange-400: #ff7043;
  --md-deep-orange-500: #ff5722;
  --md-deep-orange-600: #f4511e;
  --md-deep-orange-700: #e64a19;
  --md-deep-orange-800: #d84315;
  --md-deep-orange-900: #bf360c;
  --md-deep-orange-A100: #ff9e80;
  --md-deep-orange-A200: #ff6e40;
  --md-deep-orange-A400: #ff3d00;
  --md-deep-orange-A700: #dd2c00;
  --md-brown-50: #efebe9;
  --md-brown-100: #d7ccc8;
  --md-brown-200: #bcaaa4;
  --md-brown-300: #a1887f;
  --md-brown-400: #8d6e63;
  --md-brown-500: #795548;
  --md-brown-600: #6d4c41;
  --md-brown-700: #5d4037;
  --md-brown-800: #4e342e;
  --md-brown-900: #3e2723;
  --md-grey-50: #fafafa;
  --md-grey-100: #f5f5f5;
  --md-grey-200: #eee;
  --md-grey-300: #e0e0e0;
  --md-grey-400: #bdbdbd;
  --md-grey-500: #9e9e9e;
  --md-grey-600: #757575;
  --md-grey-700: #616161;
  --md-grey-800: #424242;
  --md-grey-900: #212121;
  --md-blue-grey-50: #eceff1;
  --md-blue-grey-100: #cfd8dc;
  --md-blue-grey-200: #b0bec5;
  --md-blue-grey-300: #90a4ae;
  --md-blue-grey-400: #78909c;
  --md-blue-grey-500: #607d8b;
  --md-blue-grey-600: #546e7a;
  --md-blue-grey-700: #455a64;
  --md-blue-grey-800: #37474f;
  --md-blue-grey-900: #263238;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| RenderedText
|----------------------------------------------------------------------------*/

:root {
  /* This is the padding value to fill the gaps between lines containing spans with background color. */
  --jp-private-code-span-padding: calc(
    (var(--jp-code-line-height) - 1) * var(--jp-code-font-size) / 2
  );
}

.jp-RenderedText {
  text-align: left;
  padding-left: var(--jp-code-padding);
  line-height: var(--jp-code-line-height);
  font-family: var(--jp-code-font-family);
}

.jp-RenderedText pre,
.jp-RenderedJavaScript pre,
.jp-RenderedHTMLCommon pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
  border: none;
  margin: 0;
  padding: 0;
}

.jp-RenderedText pre a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedText pre a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedText pre a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* console foregrounds and backgrounds */
.jp-RenderedText pre .ansi-black-fg {
  color: #3e424d;
}

.jp-RenderedText pre .ansi-red-fg {
  color: #e75c58;
}

.jp-RenderedText pre .ansi-green-fg {
  color: #00a250;
}

.jp-RenderedText pre .ansi-yellow-fg {
  color: #ddb62b;
}

.jp-RenderedText pre .ansi-blue-fg {
  color: #208ffb;
}

.jp-RenderedText pre .ansi-magenta-fg {
  color: #d160c4;
}

.jp-RenderedText pre .ansi-cyan-fg {
  color: #60c6c8;
}

.jp-RenderedText pre .ansi-white-fg {
  color: #c5c1b4;
}

.jp-RenderedText pre .ansi-black-bg {
  background-color: #3e424d;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-red-bg {
  background-color: #e75c58;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-green-bg {
  background-color: #00a250;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-yellow-bg {
  background-color: #ddb62b;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-blue-bg {
  background-color: #208ffb;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-magenta-bg {
  background-color: #d160c4;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-cyan-bg {
  background-color: #60c6c8;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-white-bg {
  background-color: #c5c1b4;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-black-intense-fg {
  color: #282c36;
}

.jp-RenderedText pre .ansi-red-intense-fg {
  color: #b22b31;
}

.jp-RenderedText pre .ansi-green-intense-fg {
  color: #007427;
}

.jp-RenderedText pre .ansi-yellow-intense-fg {
  color: #b27d12;
}

.jp-RenderedText pre .ansi-blue-intense-fg {
  color: #0065ca;
}

.jp-RenderedText pre .ansi-magenta-intense-fg {
  color: #a03196;
}

.jp-RenderedText pre .ansi-cyan-intense-fg {
  color: #258f8f;
}

.jp-RenderedText pre .ansi-white-intense-fg {
  color: #a1a6b2;
}

.jp-RenderedText pre .ansi-black-intense-bg {
  background-color: #282c36;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-red-intense-bg {
  background-color: #b22b31;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-green-intense-bg {
  background-color: #007427;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-yellow-intense-bg {
  background-color: #b27d12;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-blue-intense-bg {
  background-color: #0065ca;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-magenta-intense-bg {
  background-color: #a03196;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-cyan-intense-bg {
  background-color: #258f8f;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-white-intense-bg {
  background-color: #a1a6b2;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-default-inverse-fg {
  color: var(--jp-ui-inverse-font-color0);
}

.jp-RenderedText pre .ansi-default-inverse-bg {
  background-color: var(--jp-inverse-layout-color0);
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-bold {
  font-weight: bold;
}

.jp-RenderedText pre .ansi-underline {
  text-decoration: underline;
}

.jp-RenderedText[data-mime-type='application/vnd.jupyter.stderr'] {
  background: var(--jp-rendermime-error-background);
  padding-top: var(--jp-code-padding);
}

/*-----------------------------------------------------------------------------
| RenderedLatex
|----------------------------------------------------------------------------*/

.jp-RenderedLatex {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
}

/* Left-justify outputs.*/
.jp-OutputArea-output.jp-RenderedLatex {
  padding: var(--jp-code-padding);
  text-align: left;
}

/*-----------------------------------------------------------------------------
| RenderedHTML
|----------------------------------------------------------------------------*/

.jp-RenderedHTMLCommon {
  color: var(--jp-content-font-color1);
  font-family: var(--jp-content-font-family);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);

  /* Give a bit more R padding on Markdown text to keep line lengths reasonable */
  padding-right: 20px;
}

.jp-RenderedHTMLCommon em {
  font-style: italic;
}

.jp-RenderedHTMLCommon strong {
  font-weight: bold;
}

.jp-RenderedHTMLCommon u {
  text-decoration: underline;
}

.jp-RenderedHTMLCommon a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* Headings */

.jp-RenderedHTMLCommon h1,
.jp-RenderedHTMLCommon h2,
.jp-RenderedHTMLCommon h3,
.jp-RenderedHTMLCommon h4,
.jp-RenderedHTMLCommon h5,
.jp-RenderedHTMLCommon h6 {
  line-height: var(--jp-content-heading-line-height);
  font-weight: var(--jp-content-heading-font-weight);
  font-style: normal;
  margin: var(--jp-content-heading-margin-top) 0
    var(--jp-content-heading-margin-bottom) 0;
}

.jp-RenderedHTMLCommon h1:first-child,
.jp-RenderedHTMLCommon h2:first-child,
.jp-RenderedHTMLCommon h3:first-child,
.jp-RenderedHTMLCommon h4:first-child,
.jp-RenderedHTMLCommon h5:first-child,
.jp-RenderedHTMLCommon h6:first-child {
  margin-top: calc(0.5 * var(--jp-content-heading-margin-top));
}

.jp-RenderedHTMLCommon h1:last-child,
.jp-RenderedHTMLCommon h2:last-child,
.jp-RenderedHTMLCommon h3:last-child,
.jp-RenderedHTMLCommon h4:last-child,
.jp-RenderedHTMLCommon h5:last-child,
.jp-RenderedHTMLCommon h6:last-child {
  margin-bottom: calc(0.5 * var(--jp-content-heading-margin-bottom));
}

.jp-RenderedHTMLCommon h1 {
  font-size: var(--jp-content-font-size5);
}

.jp-RenderedHTMLCommon h2 {
  font-size: var(--jp-content-font-size4);
}

.jp-RenderedHTMLCommon h3 {
  font-size: var(--jp-content-font-size3);
}

.jp-RenderedHTMLCommon h4 {
  font-size: var(--jp-content-font-size2);
}

.jp-RenderedHTMLCommon h5 {
  font-size: var(--jp-content-font-size1);
}

.jp-RenderedHTMLCommon h6 {
  font-size: var(--jp-content-font-size0);
}

/* Lists */

/* stylelint-disable selector-max-type, selector-max-compound-selectors */

.jp-RenderedHTMLCommon ul:not(.list-inline),
.jp-RenderedHTMLCommon ol:not(.list-inline) {
  padding-left: 2em;
}

.jp-RenderedHTMLCommon ul {
  list-style: disc;
}

.jp-RenderedHTMLCommon ul ul {
  list-style: square;
}

.jp-RenderedHTMLCommon ul ul ul {
  list-style: circle;
}

.jp-RenderedHTMLCommon ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol ol {
  list-style: upper-alpha;
}

.jp-RenderedHTMLCommon ol ol ol {
  list-style: lower-alpha;
}

.jp-RenderedHTMLCommon ol ol ol ol {
  list-style: lower-roman;
}

.jp-RenderedHTMLCommon ol ol ol ol ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol,
.jp-RenderedHTMLCommon ul {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon ul ul,
.jp-RenderedHTMLCommon ul ol,
.jp-RenderedHTMLCommon ol ul,
.jp-RenderedHTMLCommon ol ol {
  margin-bottom: 0;
}

/* stylelint-enable selector-max-type, selector-max-compound-selectors */

.jp-RenderedHTMLCommon hr {
  color: var(--jp-border-color2);
  background-color: var(--jp-border-color1);
  margin-top: 1em;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon > pre {
  margin: 1.5em 2em;
}

.jp-RenderedHTMLCommon pre,
.jp-RenderedHTMLCommon code {
  border: 0;
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  line-height: var(--jp-code-line-height);
  padding: 0;
  white-space: pre-wrap;
}

.jp-RenderedHTMLCommon :not(pre) > code {
  background-color: var(--jp-layout-color2);
  padding: 1px 5px;
}

/* Tables */

.jp-RenderedHTMLCommon table {
  border-collapse: collapse;
  border-spacing: 0;
  border: none;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  table-layout: fixed;
  margin-left: auto;
  margin-bottom: 1em;
  margin-right: auto;
}

.jp-RenderedHTMLCommon thead {
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  vertical-align: bottom;
}

.jp-RenderedHTMLCommon td,
.jp-RenderedHTMLCommon th,
.jp-RenderedHTMLCommon tr {
  vertical-align: middle;
  padding: 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}

.jp-RenderedMarkdown.jp-RenderedHTMLCommon td,
.jp-RenderedMarkdown.jp-RenderedHTMLCommon th {
  max-width: none;
}

:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon td,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon th,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon tr {
  text-align: right;
}

.jp-RenderedHTMLCommon th {
  font-weight: bold;
}

.jp-RenderedHTMLCommon tbody tr:nth-child(odd) {
  background: var(--jp-layout-color0);
}

.jp-RenderedHTMLCommon tbody tr:nth-child(even) {
  background: var(--jp-rendermime-table-row-background);
}

.jp-RenderedHTMLCommon tbody tr:hover {
  background: var(--jp-rendermime-table-row-hover-background);
}

.jp-RenderedHTMLCommon p {
  text-align: left;
  margin: 0;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon img {
  -moz-force-broken-image-icon: 1;
}

/* Restrict to direct children as other images could be nested in other content. */
.jp-RenderedHTMLCommon > img {
  display: block;
  margin-left: 0;
  margin-right: 0;
  margin-bottom: 1em;
}

/* Change color behind transparent images if they need it... */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-light-background {
  background-color: var(--jp-inverse-layout-color1);
}

[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-dark-background {
  background-color: var(--jp-inverse-layout-color1);
}

.jp-RenderedHTMLCommon img,
.jp-RenderedImage img,
.jp-RenderedHTMLCommon svg,
.jp-RenderedSVG svg {
  max-width: 100%;
  height: auto;
}

.jp-RenderedHTMLCommon img.jp-mod-unconfined,
.jp-RenderedImage img.jp-mod-unconfined,
.jp-RenderedHTMLCommon svg.jp-mod-unconfined,
.jp-RenderedSVG svg.jp-mod-unconfined {
  max-width: none;
}

.jp-RenderedHTMLCommon .alert {
  padding: var(--jp-notebook-padding);
  border: var(--jp-border-width) solid transparent;
  border-radius: var(--jp-border-radius);
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon .alert-info {
  color: var(--jp-info-color0);
  background-color: var(--jp-info-color3);
  border-color: var(--jp-info-color2);
}

.jp-RenderedHTMLCommon .alert-info hr {
  border-color: var(--jp-info-color3);
}

.jp-RenderedHTMLCommon .alert-info > p:last-child,
.jp-RenderedHTMLCommon .alert-info > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-warning {
  color: var(--jp-warn-color0);
  background-color: var(--jp-warn-color3);
  border-color: var(--jp-warn-color2);
}

.jp-RenderedHTMLCommon .alert-warning hr {
  border-color: var(--jp-warn-color3);
}

.jp-RenderedHTMLCommon .alert-warning > p:last-child,
.jp-RenderedHTMLCommon .alert-warning > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-success {
  color: var(--jp-success-color0);
  background-color: var(--jp-success-color3);
  border-color: var(--jp-success-color2);
}

.jp-RenderedHTMLCommon .alert-success hr {
  border-color: var(--jp-success-color3);
}

.jp-RenderedHTMLCommon .alert-success > p:last-child,
.jp-RenderedHTMLCommon .alert-success > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-danger {
  color: var(--jp-error-color0);
  background-color: var(--jp-error-color3);
  border-color: var(--jp-error-color2);
}

.jp-RenderedHTMLCommon .alert-danger hr {
  border-color: var(--jp-error-color3);
}

.jp-RenderedHTMLCommon .alert-danger > p:last-child,
.jp-RenderedHTMLCommon .alert-danger > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon blockquote {
  margin: 1em 2em;
  padding: 0 1em;
  border-left: 5px solid var(--jp-border-color2);
}

a.jp-InternalAnchorLink {
  visibility: hidden;
  margin-left: 8px;
  color: var(--md-blue-800);
}

h1:hover .jp-InternalAnchorLink,
h2:hover .jp-InternalAnchorLink,
h3:hover .jp-InternalAnchorLink,
h4:hover .jp-InternalAnchorLink,
h5:hover .jp-InternalAnchorLink,
h6:hover .jp-InternalAnchorLink {
  visibility: visible;
}

.jp-RenderedHTMLCommon kbd {
  background-color: var(--jp-rendermime-table-row-background);
  border: 1px solid var(--jp-border-color0);
  border-bottom-color: var(--jp-border-color2);
  border-radius: 3px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
  display: inline-block;
  font-size: var(--jp-ui-font-size0);
  line-height: 1em;
  padding: 0.2em 0.5em;
}

/* Most direct children of .jp-RenderedHTMLCommon have a margin-bottom of 1.0.
 * At the bottom of cells this is a bit too much as there is also spacing
 * between cells. Going all the way to 0 gets too tight between markdown and
 * code cells.
 */
.jp-RenderedHTMLCommon > *:last-child {
  margin-bottom: 0.5em;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-cursor-backdrop {
  position: fixed;
  width: 200px;
  height: 200px;
  margin-top: -100px;
  margin-left: -100px;
  will-change: transform;
  z-index: 100;
}

.lm-mod-drag-image {
  will-change: transform;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-lineFormSearch {
  padding: 4px 12px;
  background-color: var(--jp-layout-color2);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
  font-size: var(--jp-ui-font-size1);
}

.jp-lineFormCaption {
  font-size: var(--jp-ui-font-size0);
  line-height: var(--jp-ui-font-size1);
  margin-top: 4px;
  color: var(--jp-ui-font-color0);
}

.jp-baseLineForm {
  border: none;
  border-radius: 0;
  position: absolute;
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
  outline: none;
}

.jp-lineFormButtonContainer {
  top: 4px;
  right: 8px;
  height: 24px;
  padding: 0 12px;
  width: 12px;
}

.jp-lineFormButtonIcon {
  top: 0;
  right: 0;
  background-color: var(--jp-brand-color1);
  height: 100%;
  width: 100%;
  box-sizing: border-box;
  padding: 4px 6px;
}

.jp-lineFormButton {
  top: 0;
  right: 0;
  background-color: transparent;
  height: 100%;
  width: 100%;
  box-sizing: border-box;
}

.jp-lineFormWrapper {
  overflow: hidden;
  padding: 0 8px;
  border: 1px solid var(--jp-border-color0);
  background-color: var(--jp-input-active-background);
  height: 22px;
}

.jp-lineFormWrapperFocusWithin {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-lineFormInput {
  background: transparent;
  width: 200px;
  height: 100%;
  border: none;
  outline: none;
  color: var(--jp-ui-font-color0);
  line-height: 28px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-JSONEditor {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.jp-JSONEditor-host {
  flex: 1 1 auto;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0;
  background: var(--jp-layout-color0);
  min-height: 50px;
  padding: 1px;
}

.jp-JSONEditor.jp-mod-error .jp-JSONEditor-host {
  border-color: red;
  outline-color: red;
}

.jp-JSONEditor-header {
  display: flex;
  flex: 1 0 auto;
  padding: 0 0 0 12px;
}

.jp-JSONEditor-header label {
  flex: 0 0 auto;
}

.jp-JSONEditor-commitButton {
  height: 16px;
  width: 16px;
  background-size: 18px;
  background-repeat: no-repeat;
  background-position: center;
}

.jp-JSONEditor-host.jp-mod-focused {
  background-color: var(--jp-input-active-background);
  border: 1px solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

.jp-Editor.jp-mod-dropTarget {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/
.jp-DocumentSearch-input {
  border: none;
  outline: none;
  color: var(--jp-ui-font-color0);
  font-size: var(--jp-ui-font-size1);
  background-color: var(--jp-layout-color0);
  font-family: var(--jp-ui-font-family);
  padding: 2px 1px;
  resize: none;
}

.jp-DocumentSearch-overlay {
  position: absolute;
  background-color: var(--jp-toolbar-background);
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  border-left: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  top: 0;
  right: 0;
  z-index: 7;
  min-width: 405px;
  padding: 2px;
  font-size: var(--jp-ui-font-size1);

  --jp-private-document-search-button-height: 20px;
}

.jp-DocumentSearch-overlay button {
  background-color: var(--jp-toolbar-background);
  outline: 0;
}

.jp-DocumentSearch-overlay button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-overlay button:active {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-overlay-row {
  display: flex;
  align-items: center;
  margin-bottom: 2px;
}

.jp-DocumentSearch-button-content {
  display: inline-block;
  cursor: pointer;
  box-sizing: border-box;
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-button-content svg {
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-input-wrapper {
  border: var(--jp-border-width) solid var(--jp-border-color0);
  display: flex;
  background-color: var(--jp-layout-color0);
  margin: 2px;
}

.jp-DocumentSearch-input-wrapper:focus-within {
  border-color: var(--jp-cell-editor-active-border-color);
}

.jp-DocumentSearch-toggle-wrapper,
.jp-DocumentSearch-button-wrapper {
  all: initial;
  overflow: hidden;
  display: inline-block;
  border: none;
  box-sizing: border-box;
}

.jp-DocumentSearch-toggle-wrapper {
  width: 14px;
  height: 14px;
}

.jp-DocumentSearch-button-wrapper {
  width: var(--jp-private-document-search-button-height);
  height: var(--jp-private-document-search-button-height);
}

.jp-DocumentSearch-toggle-wrapper:focus,
.jp-DocumentSearch-button-wrapper:focus {
  outline: var(--jp-border-width) solid
    var(--jp-cell-editor-active-border-color);
  outline-offset: -1px;
}

.jp-DocumentSearch-toggle-wrapper,
.jp-DocumentSearch-button-wrapper,
.jp-DocumentSearch-button-content:focus {
  outline: none;
}

.jp-DocumentSearch-toggle-placeholder {
  width: 5px;
}

.jp-DocumentSearch-input-button::before {
  display: block;
  padding-top: 100%;
}

.jp-DocumentSearch-input-button-off {
  opacity: var(--jp-search-toggle-off-opacity);
}

.jp-DocumentSearch-input-button-off:hover {
  opacity: var(--jp-search-toggle-hover-opacity);
}

.jp-DocumentSearch-input-button-on {
  opacity: var(--jp-search-toggle-on-opacity);
}

.jp-DocumentSearch-index-counter {
  padding-left: 10px;
  padding-right: 10px;
  user-select: none;
  min-width: 35px;
  display: inline-block;
}

.jp-DocumentSearch-up-down-wrapper {
  display: inline-block;
  padding-right: 2px;
  margin-left: auto;
  white-space: nowrap;
}

.jp-DocumentSearch-spacer {
  margin-left: auto;
}

.jp-DocumentSearch-up-down-wrapper button {
  outline: 0;
  border: none;
  width: var(--jp-private-document-search-button-height);
  height: var(--jp-private-document-search-button-height);
  vertical-align: middle;
  margin: 1px 5px 2px;
}

.jp-DocumentSearch-up-down-button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-up-down-button:active {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-filter-button {
  border-radius: var(--jp-border-radius);
}

.jp-DocumentSearch-filter-button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-filter-button-enabled {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-filter-button-enabled:hover {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-search-options {
  padding: 0 8px;
  margin-left: 3px;
  width: 100%;
  display: grid;
  justify-content: start;
  grid-template-columns: 1fr 1fr;
  align-items: center;
  justify-items: stretch;
}

.jp-DocumentSearch-search-filter-disabled {
  color: var(--jp-ui-font-color2);
}

.jp-DocumentSearch-search-filter {
  display: flex;
  align-items: center;
  user-select: none;
}

.jp-DocumentSearch-regex-error {
  color: var(--jp-error-color0);
}

.jp-DocumentSearch-replace-button-wrapper {
  overflow: hidden;
  display: inline-block;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color0);
  margin: auto 2px;
  padding: 1px 4px;
  height: calc(var(--jp-private-document-search-button-height) + 2px);
}

.jp-DocumentSearch-replace-button-wrapper:focus {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
}

.jp-DocumentSearch-replace-button {
  display: inline-block;
  text-align: center;
  cursor: pointer;
  box-sizing: border-box;
  color: var(--jp-ui-font-color1);

  /* height - 2 * (padding of wrapper) */
  line-height: calc(var(--jp-private-document-search-button-height) - 2px);
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-replace-button:focus {
  outline: none;
}

.jp-DocumentSearch-replace-wrapper-class {
  margin-left: 14px;
  display: flex;
}

.jp-DocumentSearch-replace-toggle {
  border: none;
  background-color: var(--jp-toolbar-background);
  border-radius: var(--jp-border-radius);
}

.jp-DocumentSearch-replace-toggle:hover {
  background-color: var(--jp-layout-color2);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.cm-editor {
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  border: 0;
  border-radius: 0;
  height: auto;

  /* Changed to auto to autogrow */
}

.cm-editor pre {
  padding: 0 var(--jp-code-padding);
}

.jp-CodeMirrorEditor[data-type='inline'] .cm-dialog {
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

.jp-CodeMirrorEditor {
  cursor: text;
}

/* When zoomed out 67% and 33% on a screen of 1440 width x 900 height */
@media screen and (min-width: 2138px) and (max-width: 4319px) {
  .jp-CodeMirrorEditor[data-type='inline'] .cm-cursor {
    border-left: var(--jp-code-cursor-width1) solid
      var(--jp-editor-cursor-color);
  }
}

/* When zoomed out less than 33% */
@media screen and (min-width: 4320px) {
  .jp-CodeMirrorEditor[data-type='inline'] .cm-cursor {
    border-left: var(--jp-code-cursor-width2) solid
      var(--jp-editor-cursor-color);
  }
}

.cm-editor.jp-mod-readOnly .cm-cursor {
  display: none;
}

.jp-CollaboratorCursor {
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: none;
  border-bottom: 3px solid;
  background-clip: content-box;
  margin-left: -5px;
  margin-right: -5px;
}

.cm-searching,
.cm-searching span {
  /* `.cm-searching span`: we need to override syntax highlighting */
  background-color: var(--jp-search-unselected-match-background-color);
  color: var(--jp-search-unselected-match-color);
}

.cm-searching::selection,
.cm-searching span::selection {
  background-color: var(--jp-search-unselected-match-background-color);
  color: var(--jp-search-unselected-match-color);
}

.jp-current-match > .cm-searching,
.jp-current-match > .cm-searching span,
.cm-searching > .jp-current-match,
.cm-searching > .jp-current-match span {
  background-color: var(--jp-search-selected-match-background-color);
  color: var(--jp-search-selected-match-color);
}

.jp-current-match > .cm-searching::selection,
.cm-searching > .jp-current-match::selection,
.jp-current-match > .cm-searching span::selection {
  background-color: var(--jp-search-selected-match-background-color);
  color: var(--jp-search-selected-match-color);
}

.cm-trailingspace {
  background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAsElEQVQIHQGlAFr/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA7+r3zKmT0/+pk9P/7+r3zAAAAAAAAAAABAAAAAAAAAAA6OPzM+/q9wAAAAAA6OPzMwAAAAAAAAAAAgAAAAAAAAAAGR8NiRQaCgAZIA0AGR8NiQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQyoYJ/SY80UAAAAASUVORK5CYII=);
  background-position: center left;
  background-repeat: repeat-x;
}

.jp-CollaboratorCursor-hover {
  position: absolute;
  z-index: 1;
  transform: translateX(-50%);
  color: white;
  border-radius: 3px;
  padding-left: 4px;
  padding-right: 4px;
  padding-top: 1px;
  padding-bottom: 1px;
  text-align: center;
  font-size: var(--jp-ui-font-size1);
  white-space: nowrap;
}

.jp-CodeMirror-ruler {
  border-left: 1px dashed var(--jp-border-color2);
}

/* Styles for shared cursors (remote cursor locations and selected ranges) */
.jp-CodeMirrorEditor .cm-ySelectionCaret {
  position: relative;
  border-left: 1px solid black;
  margin-left: -1px;
  margin-right: -1px;
  box-sizing: border-box;
}

.jp-CodeMirrorEditor .cm-ySelectionCaret > .cm-ySelectionInfo {
  white-space: nowrap;
  position: absolute;
  top: -1.15em;
  padding-bottom: 0.05em;
  left: -1px;
  font-size: 0.95em;
  font-family: var(--jp-ui-font-family);
  font-weight: bold;
  line-height: normal;
  user-select: none;
  color: white;
  padding-left: 2px;
  padding-right: 2px;
  z-index: 101;
  transition: opacity 0.3s ease-in-out;
}

.jp-CodeMirrorEditor .cm-ySelectionInfo {
  transition-delay: 0.7s;
  opacity: 0;
}

.jp-CodeMirrorEditor .cm-ySelectionCaret:hover > .cm-ySelectionInfo {
  opacity: 1;
  transition-delay: 0s;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MimeDocument {
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-filebrowser-button-height: 28px;
  --jp-private-filebrowser-button-width: 48px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FileBrowser .jp-SidePanel-content {
  display: flex;
  flex-direction: column;
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  flex-wrap: wrap;
  row-gap: 12px;
  border-bottom: none;
  height: auto;
  margin: 8px 12px 0;
  box-shadow: none;
  padding: 0;
  justify-content: flex-start;
}

.jp-FileBrowser-Panel {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
}

.jp-BreadCrumbs {
  flex: 0 0 auto;
  margin: 8px 12px;
}

.jp-BreadCrumbs-item {
  margin: 0 2px;
  padding: 0 2px;
  border-radius: var(--jp-border-radius);
  cursor: pointer;
}

.jp-BreadCrumbs-item:hover {
  background-color: var(--jp-layout-color2);
}

.jp-BreadCrumbs-item:first-child {
  margin-left: 0;
}

.jp-BreadCrumbs-item.jp-mod-dropTarget {
  background-color: var(--jp-brand-color2);
  opacity: 0.7;
}

/*-----------------------------------------------------------------------------
| Buttons
|----------------------------------------------------------------------------*/

.jp-FileBrowser-toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  padding-left: 0;
  padding-right: 2px;
  align-items: center;
  height: unset;
}

.jp-FileBrowser-toolbar > .jp-Toolbar-item .jp-ToolbarButtonComponent {
  width: 40px;
}

/*-----------------------------------------------------------------------------
| Other styles
|----------------------------------------------------------------------------*/

.jp-FileDialog.jp-mod-conflict input {
  color: var(--jp-error-color1);
}

.jp-FileDialog .jp-new-name-title {
  margin-top: 12px;
}

.jp-LastModified-hidden {
  display: none;
}

.jp-FileSize-hidden {
  display: none;
}

.jp-FileBrowser .lm-AccordionPanel > h3:first-child {
  display: none;
}

/*-----------------------------------------------------------------------------
| DirListing
|----------------------------------------------------------------------------*/

.jp-DirListing {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  outline: 0;
}

.jp-DirListing-header {
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  align-items: center;
  overflow: hidden;
  border-top: var(--jp-border-width) solid var(--jp-border-color2);
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
}

.jp-DirListing-headerItem {
  padding: 4px 12px 2px;
  font-weight: 500;
}

.jp-DirListing-headerItem:hover {
  background: var(--jp-layout-color2);
}

.jp-DirListing-headerItem.jp-id-name {
  flex: 1 0 84px;
}

.jp-DirListing-headerItem.jp-id-modified {
  flex: 0 0 112px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-DirListing-headerItem.jp-id-filesize {
  flex: 0 0 75px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-id-narrow {
  display: none;
  flex: 0 0 5px;
  padding: 4px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
  color: var(--jp-border-color2);
}

.jp-DirListing-narrow .jp-id-narrow {
  display: block;
}

.jp-DirListing-narrow .jp-id-modified,
.jp-DirListing-narrow .jp-DirListing-itemModified {
  display: none;
}

.jp-DirListing-headerItem.jp-mod-selected {
  font-weight: 600;
}

/* increase specificity to override bundled default */
.jp-DirListing-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

.jp-DirListing-content mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.jp-DirListing-content .jp-DirListing-item.jp-mod-selected mark {
  color: var(--jp-ui-inverse-font-color0);
}

/* Style the directory listing content when a user drops a file to upload */
.jp-DirListing.jp-mod-native-drop .jp-DirListing-content {
  outline: 5px dashed rgba(128, 128, 128, 0.5);
  outline-offset: -10px;
  cursor: copy;
}

.jp-DirListing-item {
  display: flex;
  flex-direction: row;
  align-items: center;
  padding: 4px 12px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-DirListing-checkboxWrapper {
  /* Increases hit area of checkbox. */
  padding: 4px;
}

.jp-DirListing-header
  .jp-DirListing-checkboxWrapper
  + .jp-DirListing-headerItem {
  padding-left: 4px;
}

.jp-DirListing-content .jp-DirListing-checkboxWrapper {
  position: relative;
  left: -4px;
  margin: -4px 0 -4px -8px;
}

.jp-DirListing-checkboxWrapper.jp-mod-visible {
  visibility: visible;
}

/* For devices that support hovering, hide checkboxes until hovered, selected...
*/
@media (hover: hover) {
  .jp-DirListing-checkboxWrapper {
    visibility: hidden;
  }

  .jp-DirListing-item:hover .jp-DirListing-checkboxWrapper,
  .jp-DirListing-item.jp-mod-selected .jp-DirListing-checkboxWrapper {
    visibility: visible;
  }
}

.jp-DirListing-item[data-is-dot] {
  opacity: 75%;
}

.jp-DirListing-item.jp-mod-selected {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.jp-DirListing-item.jp-mod-dropTarget {
  background: var(--jp-brand-color3);
}

.jp-DirListing-item:hover:not(.jp-mod-selected) {
  background: var(--jp-layout-color2);
}

.jp-DirListing-itemIcon {
  flex: 0 0 20px;
  margin-right: 4px;
}

.jp-DirListing-itemText {
  flex: 1 0 64px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  user-select: none;
}

.jp-DirListing-itemText:focus {
  outline-width: 2px;
  outline-color: var(--jp-inverse-layout-color1);
  outline-style: solid;
  outline-offset: 1px;
}

.jp-DirListing-item.jp-mod-selected .jp-DirListing-itemText:focus {
  outline-color: var(--jp-layout-color1);
}

.jp-DirListing-itemModified {
  flex: 0 0 125px;
  text-align: right;
}

.jp-DirListing-itemFileSize {
  flex: 0 0 90px;
  text-align: right;
}

.jp-DirListing-editor {
  flex: 1 0 64px;
  outline: none;
  border: none;
  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color1);
}

.jp-DirListing-item.jp-mod-running .jp-DirListing-itemIcon::before {
  color: var(--jp-success-color1);
  content: '\25CF';
  font-size: 8px;
  position: absolute;
  left: -8px;
}

.jp-DirListing-item.jp-mod-running.jp-mod-selected
  .jp-DirListing-itemIcon::before {
  color: var(--jp-ui-inverse-font-color1);
}

.jp-DirListing-item.lm-mod-drag-image,
.jp-DirListing-item.jp-mod-selected.lm-mod-drag-image {
  font-size: var(--jp-ui-font-size1);
  padding-left: 4px;
  margin-left: 4px;
  width: 160px;
  background-color: var(--jp-ui-inverse-font-color2);
  box-shadow: var(--jp-elevation-z2);
  border-radius: 0;
  color: var(--jp-ui-font-color1);
  transform: translateX(-40%) translateY(-58%);
}

.jp-Document {
  min-width: 120px;
  min-height: 120px;
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Main OutputArea
| OutputArea has a list of Outputs
|----------------------------------------------------------------------------*/

.jp-OutputArea {
  overflow-y: auto;
}

.jp-OutputArea-child {
  display: table;
  table-layout: fixed;
  width: 100%;
  overflow: hidden;
}

.jp-OutputPrompt {
  width: var(--jp-cell-prompt-width);
  color: var(--jp-cell-outprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);

  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;

  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-OutputArea-prompt {
  display: table-cell;
  vertical-align: top;
}

.jp-OutputArea-output {
  display: table-cell;
  width: 100%;
  height: auto;
  overflow: auto;
  user-select: text;
  -moz-user-select: text;
  -webkit-user-select: text;
  -ms-user-select: text;
}

.jp-OutputArea .jp-RenderedText {
  padding-left: 1ch;
}

/**
 * Prompt overlay.
 */

.jp-OutputArea-promptOverlay {
  position: absolute;
  top: 0;
  width: var(--jp-cell-prompt-width);
  height: 100%;
  opacity: 0.5;
}

.jp-OutputArea-promptOverlay:hover {
  background: var(--jp-layout-color2);
  box-shadow: inset 0 0 1px var(--jp-inverse-layout-color0);
  cursor: zoom-out;
}

.jp-mod-outputsScrolled .jp-OutputArea-promptOverlay:hover {
  cursor: zoom-in;
}

/**
 * Isolated output.
 */
.jp-OutputArea-output.jp-mod-isolated {
  width: 100%;
  display: block;
}

/*
When drag events occur, `lm-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated {
  position: relative;
}

body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/* pre */

.jp-OutputArea-output pre {
  border: none;
  margin: 0;
  padding: 0;
  overflow-x: auto;
  overflow-y: auto;
  word-break: break-all;
  word-wrap: break-word;
  white-space: pre-wrap;
}

/* tables */

.jp-OutputArea-output.jp-RenderedHTMLCommon table {
  margin-left: 0;
  margin-right: 0;
}

/* description lists */

.jp-OutputArea-output dl,
.jp-OutputArea-output dt,
.jp-OutputArea-output dd {
  display: block;
}

.jp-OutputArea-output dl {
  width: 100%;
  overflow: hidden;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dt {
  font-weight: bold;
  float: left;
  width: 20%;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dd {
  float: left;
  width: 80%;
  padding: 0;
  margin: 0;
}

.jp-TrimmedOutputs pre {
  background: var(--jp-layout-color3);
  font-size: calc(var(--jp-code-font-size) * 1.4);
  text-align: center;
  text-transform: uppercase;
}

/* Hide the gutter in case of
 *  - nested output areas (e.g. in the case of output widgets)
 *  - mirrored output areas
 */
.jp-OutputArea .jp-OutputArea .jp-OutputArea-prompt {
  display: none;
}

/* Hide empty lines in the output area, for instance due to cleared widgets */
.jp-OutputArea-prompt:empty {
  padding: 0;
  border: 0;
}

/*-----------------------------------------------------------------------------
| executeResult is added to any Output-result for the display of the object
| returned by a cell
|----------------------------------------------------------------------------*/

.jp-OutputArea-output.jp-OutputArea-executeResult {
  margin-left: 0;
  width: 100%;
}

/* Text output with the Out[] prompt needs a top padding to match the
 * alignment of the Out[] prompt itself.
 */
.jp-OutputArea-executeResult .jp-RenderedText.jp-OutputArea-output {
  padding-top: var(--jp-code-padding);
  border-top: var(--jp-border-width) solid transparent;
}

/*-----------------------------------------------------------------------------
| The Stdin output
|----------------------------------------------------------------------------*/

.jp-Stdin-prompt {
  color: var(--jp-content-font-color0);
  padding-right: var(--jp-code-padding);
  vertical-align: baseline;
  flex: 0 0 auto;
}

.jp-Stdin-input {
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  color: inherit;
  background-color: inherit;
  width: 42%;
  min-width: 200px;

  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;

  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0 0.25em;
  margin: 0 0.25em;
  flex: 0 0 70%;
}

.jp-Stdin-input::placeholder {
  opacity: 0;
}

.jp-Stdin-input:focus {
  box-shadow: none;
}

.jp-Stdin-input:focus::placeholder {
  opacity: 1;
}

/*-----------------------------------------------------------------------------
| Output Area View
|----------------------------------------------------------------------------*/

.jp-LinkedOutputView .jp-OutputArea {
  height: 100%;
  display: block;
}

.jp-LinkedOutputView .jp-OutputArea-output:only-child {
  height: 100%;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

@media print {
  .jp-OutputArea-child {
    break-inside: avoid-page;
  }
}

/*-----------------------------------------------------------------------------
| Mobile
|----------------------------------------------------------------------------*/
@media only screen and (max-width: 760px) {
  .jp-OutputPrompt {
    display: table-row;
    text-align: left;
  }

  .jp-OutputArea-child .jp-OutputArea-output {
    display: table-row;
    margin-left: var(--jp-notebook-padding);
  }
}

/* Trimmed outputs warning */
.jp-TrimmedOutputs > a {
  margin: 10px;
  text-decoration: none;
  cursor: pointer;
}

.jp-TrimmedOutputs > a:hover {
  text-decoration: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Table of Contents
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toc-active-width: 4px;
}

.jp-TableOfContents {
  display: flex;
  flex-direction: column;
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  height: 100%;
}

.jp-TableOfContents-placeholder {
  text-align: center;
}

.jp-TableOfContents-placeholderContent {
  color: var(--jp-content-font-color2);
  padding: 8px;
}

.jp-TableOfContents-placeholderContent > h3 {
  margin-bottom: var(--jp-content-heading-margin-bottom);
}

.jp-TableOfContents .jp-SidePanel-content {
  overflow-y: auto;
}

.jp-TableOfContents-tree {
  margin: 4px;
}

.jp-TableOfContents ol {
  list-style-type: none;
}

/* stylelint-disable-next-line selector-max-type */
.jp-TableOfContents li > ol {
  /* Align left border with triangle icon center */
  padding-left: 11px;
}

.jp-TableOfContents-content {
  /* left margin for the active heading indicator */
  margin: 0 0 0 var(--jp-private-toc-active-width);
  padding: 0;
  background-color: var(--jp-layout-color1);
}

.jp-tocItem {
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-tocItem-heading {
  display: flex;
  cursor: pointer;
}

.jp-tocItem-heading:hover {
  background-color: var(--jp-layout-color2);
}

.jp-tocItem-content {
  display: block;
  padding: 4px 0;
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow-x: hidden;
}

.jp-tocItem-collapser {
  height: 20px;
  margin: 2px 2px 0;
  padding: 0;
  background: none;
  border: none;
  cursor: pointer;
}

.jp-tocItem-collapser:hover {
  background-color: var(--jp-layout-color3);
}

/* Active heading indicator */

.jp-tocItem-heading::before {
  content: ' ';
  background: transparent;
  width: var(--jp-private-toc-active-width);
  height: 24px;
  position: absolute;
  left: 0;
  border-radius: var(--jp-border-radius);
}

.jp-tocItem-heading.jp-tocItem-active::before {
  background-color: var(--jp-brand-color1);
}

.jp-tocItem-heading:hover.jp-tocItem-active::before {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapser {
  flex: 0 0 var(--jp-cell-collapser-width);
  padding: 0;
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
  border-radius: var(--jp-border-radius);
  opacity: 1;
}

.jp-Collapser-child {
  display: block;
  width: 100%;
  box-sizing: border-box;

  /* height: 100% doesn't work because the height of its parent is computed from content */
  position: absolute;
  top: 0;
  bottom: 0;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

/*
Hiding collapsers in print mode.

Note: input and output wrappers have "display: block" propery in print mode.
*/

@media print {
  .jp-Collapser {
    display: none;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Header/Footer
|----------------------------------------------------------------------------*/

/* Hidden by zero height by default */
.jp-CellHeader,
.jp-CellFooter {
  height: 0;
  width: 100%;
  padding: 0;
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Input
|----------------------------------------------------------------------------*/

/* All input areas */
.jp-InputArea {
  display: table;
  table-layout: fixed;
  width: 100%;
  overflow: hidden;
}

.jp-InputArea-editor {
  display: table-cell;
  overflow: hidden;
  vertical-align: top;

  /* This is the non-active, default styling */
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0;
  background: var(--jp-cell-editor-background);
}

.jp-InputPrompt {
  display: table-cell;
  vertical-align: top;
  width: var(--jp-cell-prompt-width);
  color: var(--jp-cell-inprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  opacity: var(--jp-cell-prompt-opacity);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;

  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;

  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

/*-----------------------------------------------------------------------------
| Mobile
|----------------------------------------------------------------------------*/
@media only screen and (max-width: 760px) {
  .jp-InputArea-editor {
    display: table-row;
    margin-left: var(--jp-notebook-padding);
  }

  .jp-InputPrompt {
    display: table-row;
    text-align: left;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Placeholder {
  display: table;
  table-layout: fixed;
  width: 100%;
}

.jp-Placeholder-prompt {
  display: table-cell;
  box-sizing: border-box;
}

.jp-Placeholder-content {
  display: table-cell;
  padding: 4px 6px;
  border: 1px solid transparent;
  border-radius: 0;
  background: none;
  box-sizing: border-box;
  cursor: pointer;
}

.jp-Placeholder-contentContainer {
  display: flex;
}

.jp-Placeholder-content:hover,
.jp-InputPlaceholder > .jp-Placeholder-content:hover {
  border-color: var(--jp-layout-color3);
}

.jp-Placeholder-content .jp-MoreHorizIcon {
  width: 32px;
  height: 16px;
  border: 1px solid transparent;
  border-radius: var(--jp-border-radius);
}

.jp-Placeholder-content .jp-MoreHorizIcon:hover {
  border: 1px solid var(--jp-border-color1);
  box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.25);
  background-color: var(--jp-layout-color0);
}

.jp-PlaceholderText {
  white-space: nowrap;
  overflow-x: hidden;
  color: var(--jp-inverse-layout-color3);
  font-family: var(--jp-code-font-family);
}

.jp-InputPlaceholder > .jp-Placeholder-content {
  border-color: var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-cell-scrolling-output-offset: 5px;
}

/*-----------------------------------------------------------------------------
| Cell
|----------------------------------------------------------------------------*/

.jp-Cell {
  padding: var(--jp-cell-padding);
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Common input/output
|----------------------------------------------------------------------------*/

.jp-Cell-inputWrapper,
.jp-Cell-outputWrapper {
  display: flex;
  flex-direction: row;
  padding: 0;
  margin: 0;

  /* Added to reveal the box-shadow on the input and output collapsers. */
  overflow: visible;
}

/* Only input/output areas inside cells */
.jp-Cell-inputArea,
.jp-Cell-outputArea {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Collapser
|----------------------------------------------------------------------------*/

/* Make the output collapser disappear when there is not output, but do so
 * in a manner that leaves it in the layout and preserves its width.
 */
.jp-Cell.jp-mod-noOutputs .jp-Cell-outputCollapser {
  border: none !important;
  background: transparent !important;
}

.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputCollapser {
  min-height: var(--jp-cell-collapser-min-height);
}

/*-----------------------------------------------------------------------------
| Output
|----------------------------------------------------------------------------*/

/* Put a space between input and output when there IS output */
.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputWrapper {
  margin-top: 5px;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea {
  overflow-y: auto;
  max-height: 24em;
  margin-left: var(--jp-private-cell-scrolling-output-offset);
  resize: vertical;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea[style*='height'] {
  max-height: unset;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea::after {
  content: ' ';
  box-shadow: inset 0 0 6px 2px rgb(0 0 0 / 30%);
  width: 100%;
  height: 100%;
  position: sticky;
  bottom: 0;
  top: 0;
  margin-top: -50%;
  float: left;
  display: block;
  pointer-events: none;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-child {
  padding-top: 6px;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
  width: calc(
    var(--jp-cell-prompt-width) - var(--jp-private-cell-scrolling-output-offset)
  );
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-promptOverlay {
  left: calc(-1 * var(--jp-private-cell-scrolling-output-offset));
}

/*-----------------------------------------------------------------------------
| CodeCell
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| MarkdownCell
|----------------------------------------------------------------------------*/

.jp-MarkdownOutput {
  display: table-cell;
  width: 100%;
  margin-top: 0;
  margin-bottom: 0;
  padding-left: var(--jp-code-padding);
}

.jp-MarkdownOutput.jp-RenderedHTMLCommon {
  overflow: auto;
}

/* collapseHeadingButton (show always if hiddenCellsButton is _not_ shown) */
.jp-collapseHeadingButton {
  display: flex;
  min-height: var(--jp-cell-collapser-min-height);
  font-size: var(--jp-code-font-size);
  position: absolute;
  background-color: transparent;
  background-size: 25px;
  background-repeat: no-repeat;
  background-position-x: center;
  background-position-y: top;
  background-image: var(--jp-icon-caret-down);
  right: 0;
  top: 0;
  bottom: 0;
}

.jp-collapseHeadingButton.jp-mod-collapsed {
  background-image: var(--jp-icon-caret-right);
}

/*
 set the container font size to match that of content
 so that the nested collapse buttons have the right size
*/
.jp-MarkdownCell .jp-InputPrompt {
  font-size: var(--jp-content-font-size1);
}

/*
  Align collapseHeadingButton with cell top header
  The font sizes are identical to the ones in packages/rendermime/style/base.css
*/
.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='1'] {
  font-size: var(--jp-content-font-size5);
  background-position-y: calc(0.3 * var(--jp-content-font-size5));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='2'] {
  font-size: var(--jp-content-font-size4);
  background-position-y: calc(0.3 * var(--jp-content-font-size4));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='3'] {
  font-size: var(--jp-content-font-size3);
  background-position-y: calc(0.3 * var(--jp-content-font-size3));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='4'] {
  font-size: var(--jp-content-font-size2);
  background-position-y: calc(0.3 * var(--jp-content-font-size2));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='5'] {
  font-size: var(--jp-content-font-size1);
  background-position-y: top;
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='6'] {
  font-size: var(--jp-content-font-size0);
  background-position-y: top;
}

/* collapseHeadingButton (show only on (hover,active) if hiddenCellsButton is shown) */
.jp-Notebook.jp-mod-showHiddenCellsButton .jp-collapseHeadingButton {
  display: none;
}

.jp-Notebook.jp-mod-showHiddenCellsButton
  :is(.jp-MarkdownCell:hover, .jp-mod-active)
  .jp-collapseHeadingButton {
  display: flex;
}

/* showHiddenCellsButton (only show if jp-mod-showHiddenCellsButton is set, which
is a consequence of the showHiddenCellsButton option in Notebook Settings)*/
.jp-Notebook.jp-mod-showHiddenCellsButton .jp-showHiddenCellsButton {
  margin-left: calc(var(--jp-cell-prompt-width) + 2 * var(--jp-code-padding));
  margin-top: var(--jp-code-padding);
  border: 1px solid var(--jp-border-color2);
  background-color: var(--jp-border-color3) !important;
  color: var(--jp-content-font-color0) !important;
  display: flex;
}

.jp-Notebook.jp-mod-showHiddenCellsButton .jp-showHiddenCellsButton:hover {
  background-color: var(--jp-border-color2) !important;
}

.jp-showHiddenCellsButton {
  display: none;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

/*
Using block instead of flex to allow the use of the break-inside CSS property for
cell outputs.
*/

@media print {
  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: block;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-notebook-toolbar-padding: 2px 5px 2px 2px;
}

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-NotebookPanel-toolbar {
  padding: var(--jp-notebook-toolbar-padding);

  /* disable paint containment from lumino 2.0 default strict CSS containment */
  contain: style size !important;
}

.jp-Toolbar-item.jp-Notebook-toolbarCellType .jp-select-wrapper.jp-mod-focused {
  border: none;
  box-shadow: none;
}

.jp-Notebook-toolbarCellTypeDropdown select {
  height: 24px;
  font-size: var(--jp-ui-font-size1);
  line-height: 14px;
  border-radius: 0;
  display: block;
}

.jp-Notebook-toolbarCellTypeDropdown span {
  top: 5px !important;
}

.jp-Toolbar-responsive-popup {
  position: absolute;
  height: fit-content;
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  justify-content: flex-end;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: var(--jp-notebook-toolbar-padding);
  z-index: 1;
  right: 0;
  top: 0;
}

.jp-Toolbar > .jp-Toolbar-responsive-opener {
  margin-left: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-Notebook-ExecutionIndicator {
  position: relative;
  display: inline-block;
  height: 100%;
  z-index: 9997;
}

.jp-Notebook-ExecutionIndicator-tooltip {
  visibility: hidden;
  height: auto;
  width: max-content;
  width: -moz-max-content;
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color1);
  text-align: justify;
  border-radius: 6px;
  padding: 0 5px;
  position: fixed;
  display: table;
}

.jp-Notebook-ExecutionIndicator-tooltip.up {
  transform: translateX(-50%) translateY(-100%) translateY(-32px);
}

.jp-Notebook-ExecutionIndicator-tooltip.down {
  transform: translateX(calc(-100% + 16px)) translateY(5px);
}

.jp-Notebook-ExecutionIndicator-tooltip.hidden {
  display: none;
}

.jp-Notebook-ExecutionIndicator:hover .jp-Notebook-ExecutionIndicator-tooltip {
  visibility: visible;
}

.jp-Notebook-ExecutionIndicator span {
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  color: var(--jp-ui-font-color1);
  line-height: 24px;
  display: block;
}

.jp-Notebook-ExecutionIndicator-progress-bar {
  display: flex;
  justify-content: center;
  height: 100%;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*
 * Execution indicator
 */
.jp-tocItem-content::after {
  content: '';

  /* Must be identical to form a circle */
  width: 12px;
  height: 12px;
  background: none;
  border: none;
  position: absolute;
  right: 0;
}

.jp-tocItem-content[data-running='0']::after {
  border-radius: 50%;
  border: var(--jp-border-width) solid var(--jp-inverse-layout-color3);
  background: none;
}

.jp-tocItem-content[data-running='1']::after {
  border-radius: 50%;
  border: var(--jp-border-width) solid var(--jp-inverse-layout-color3);
  background-color: var(--jp-inverse-layout-color3);
}

.jp-tocItem-content[data-running='0'],
.jp-tocItem-content[data-running='1'] {
  margin-right: 12px;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-Notebook-footer {
  height: 27px;
  margin-left: calc(
    var(--jp-cell-prompt-width) + var(--jp-cell-collapser-width) +
      var(--jp-cell-padding)
  );
  width: calc(
    100% -
      (
        var(--jp-cell-prompt-width) + var(--jp-cell-collapser-width) +
          var(--jp-cell-padding) + var(--jp-cell-padding)
      )
  );
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  color: var(--jp-ui-font-color3);
  margin-top: 6px;
  background: none;
  cursor: pointer;
}

.jp-Notebook-footer:focus {
  border-color: var(--jp-cell-editor-active-border-color);
}

/* For devices that support hovering, hide footer until hover */
@media (hover: hover) {
  .jp-Notebook-footer {
    opacity: 0;
  }

  .jp-Notebook-footer:focus,
  .jp-Notebook-footer:hover {
    opacity: 1;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Imports
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-side-by-side-output-size: 1fr;
  --jp-side-by-side-resized-cell: var(--jp-side-by-side-output-size);
  --jp-private-notebook-dragImage-width: 304px;
  --jp-private-notebook-dragImage-height: 36px;
  --jp-private-notebook-selected-color: var(--md-blue-400);
  --jp-private-notebook-active-color: var(--md-green-400);
}

/*-----------------------------------------------------------------------------
| Notebook
|----------------------------------------------------------------------------*/

/* stylelint-disable selector-max-class */

.jp-NotebookPanel {
  display: block;
  height: 100%;
}

.jp-NotebookPanel.jp-Document {
  min-width: 240px;
  min-height: 120px;
}

.jp-Notebook {
  padding: var(--jp-notebook-padding);
  outline: none;
  overflow: auto;
  background: var(--jp-layout-color0);
}

.jp-Notebook.jp-mod-scrollPastEnd::after {
  display: block;
  content: '';
  min-height: var(--jp-notebook-scroll-padding);
}

.jp-MainAreaWidget-ContainStrict .jp-Notebook * {
  contain: strict;
}

.jp-Notebook .jp-Cell {
  overflow: visible;
}

.jp-Notebook .jp-Cell .jp-InputPrompt {
  cursor: move;
}

/*-----------------------------------------------------------------------------
| Notebook state related styling
|
| The notebook and cells each have states, here are the possibilities:
|
| - Notebook
|   - Command
|   - Edit
| - Cell
|   - None
|   - Active (only one can be active)
|   - Selected (the cells actions are applied to)
|   - Multiselected (when multiple selected, the cursor)
|   - No outputs
|----------------------------------------------------------------------------*/

/* Command or edit modes */

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-InputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-OutputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

/* cell is active */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser {
  background: var(--jp-brand-color1);
}

/* cell is dirty */
.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt {
  color: var(--jp-warn-color1);
}

.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt::before {
  color: var(--jp-warn-color1);
  content: '•';
}

.jp-Notebook .jp-Cell.jp-mod-active.jp-mod-dirty .jp-Collapser {
  background: var(--jp-warn-color1);
}

/* collapser is hovered */
.jp-Notebook .jp-Cell .jp-Collapser:hover {
  box-shadow: var(--jp-elevation-z2);
  background: var(--jp-brand-color1);
  opacity: var(--jp-cell-collapser-not-active-hover-opacity);
}

/* cell is active and collapser is hovered */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser:hover {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/* Command mode */

.jp-Notebook.jp-mod-commandMode .jp-Cell.jp-mod-selected {
  background: var(--jp-notebook-multiselected-color);
}

.jp-Notebook.jp-mod-commandMode
  .jp-Cell.jp-mod-active.jp-mod-selected:not(.jp-mod-multiSelected) {
  background: transparent;
}

/* Edit mode */

.jp-Notebook.jp-mod-editMode .jp-Cell.jp-mod-active .jp-InputArea-editor {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-cell-editor-active-background);
}

/*-----------------------------------------------------------------------------
| Notebook drag and drop
|----------------------------------------------------------------------------*/

.jp-Notebook-cell.jp-mod-dropSource {
  opacity: 0.5;
}

.jp-Notebook-cell.jp-mod-dropTarget,
.jp-Notebook.jp-mod-commandMode
  .jp-Notebook-cell.jp-mod-active.jp-mod-selected.jp-mod-dropTarget {
  border-top-color: var(--jp-private-notebook-selected-color);
  border-top-style: solid;
  border-top-width: 2px;
}

.jp-dragImage {
  display: block;
  flex-direction: row;
  width: var(--jp-private-notebook-dragImage-width);
  height: var(--jp-private-notebook-dragImage-height);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
  overflow: visible;
}

.jp-dragImage-singlePrompt {
  box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.12);
}

.jp-dragImage .jp-dragImage-content {
  flex: 1 1 auto;
  z-index: 2;
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  line-height: var(--jp-code-line-height);
  padding: var(--jp-code-padding);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background-color);
  color: var(--jp-content-font-color3);
  text-align: left;
  margin: 4px 4px 4px 0;
}

.jp-dragImage .jp-dragImage-prompt {
  flex: 0 0 auto;
  min-width: 36px;
  color: var(--jp-cell-inprompt-font-color);
  padding: var(--jp-code-padding);
  padding-left: 12px;
  font-family: var(--jp-cell-prompt-font-family);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: 1.9;
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
}

.jp-dragImage-multipleBack {
  z-index: -1;
  position: absolute;
  height: 32px;
  width: 300px;
  top: 8px;
  left: 8px;
  background: var(--jp-layout-color2);
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.12);
}

/*-----------------------------------------------------------------------------
| Cell toolbar
|----------------------------------------------------------------------------*/

.jp-NotebookTools {
  display: block;
  min-width: var(--jp-sidebar-min-width);
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);

  /* This is needed so that all font sizing of children done in ems is
    * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  overflow: auto;
}

.jp-ActiveCellTool {
  padding: 12px 0;
  display: flex;
}

.jp-ActiveCellTool-Content {
  flex: 1 1 auto;
}

.jp-ActiveCellTool .jp-ActiveCellTool-CellContent {
  background: var(--jp-cell-editor-background);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0;
  min-height: 29px;
}

.jp-ActiveCellTool .jp-InputPrompt {
  min-width: calc(var(--jp-cell-prompt-width) * 0.75);
}

.jp-ActiveCellTool-CellContent > pre {
  padding: 5px 4px;
  margin: 0;
  white-space: normal;
}

.jp-MetadataEditorTool {
  flex-direction: column;
  padding: 12px 0;
}

.jp-RankedPanel > :not(:first-child) {
  margin-top: 12px;
}

.jp-KeySelector select.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: var(--jp-border-width) solid var(--jp-border-color1);
}

.jp-KeySelector label,
.jp-MetadataEditorTool label,
.jp-NumberSetter label {
  line-height: 1.4;
}

.jp-NotebookTools .jp-select-wrapper {
  margin-top: 4px;
  margin-bottom: 0;
}

.jp-NumberSetter input {
  width: 100%;
  margin-top: 4px;
}

.jp-NotebookTools .jp-Collapse {
  margin-top: 16px;
}

/*-----------------------------------------------------------------------------
| Presentation Mode (.jp-mod-presentationMode)
|----------------------------------------------------------------------------*/

.jp-mod-presentationMode .jp-Notebook {
  --jp-content-font-size1: var(--jp-content-presentation-font-size1);
  --jp-code-font-size: var(--jp-code-presentation-font-size);
}

.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-InputPrompt,
.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-OutputPrompt {
  flex: 0 0 110px;
}

/*-----------------------------------------------------------------------------
| Side-by-side Mode (.jp-mod-sideBySide)
|----------------------------------------------------------------------------*/
.jp-mod-sideBySide.jp-Notebook .jp-Notebook-cell {
  margin-top: 3em;
  margin-bottom: 3em;
  margin-left: 5%;
  margin-right: 5%;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell {
  display: grid;
  grid-template-columns: minmax(0, 1fr) min-content minmax(
      0,
      var(--jp-side-by-side-output-size)
    );
  grid-template-rows: auto minmax(0, 1fr) auto;
  grid-template-areas:
    'header header header'
    'input handle output'
    'footer footer footer';
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell.jp-mod-resizedCell {
  grid-template-columns: minmax(0, 1fr) min-content minmax(
      0,
      var(--jp-side-by-side-resized-cell)
    );
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellHeader {
  grid-area: header;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-Cell-inputWrapper {
  grid-area: input;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-Cell-outputWrapper {
  /* overwrite the default margin (no vertical separation needed in side by side move */
  margin-top: 0;
  grid-area: output;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellFooter {
  grid-area: footer;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellResizeHandle {
  grid-area: handle;
  user-select: none;
  display: block;
  height: 100%;
  cursor: ew-resize;
  padding: 0 var(--jp-cell-padding);
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellResizeHandle::after {
  content: '';
  display: block;
  background: var(--jp-border-color2);
  height: 100%;
  width: 5px;
}

.jp-mod-sideBySide.jp-Notebook
  .jp-CodeCell.jp-mod-resizedCell
  .jp-CellResizeHandle::after {
  background: var(--jp-border-color0);
}

.jp-CellResizeHandle {
  display: none;
}

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Cell-Placeholder {
  padding-left: 55px;
}

.jp-Cell-Placeholder-wrapper {
  background: #fff;
  border: 1px solid;
  border-color: #e5e6e9 #dfe0e4 #d0d1d5;
  border-radius: 4px;
  -webkit-border-radius: 4px;
  margin: 10px 15px;
}

.jp-Cell-Placeholder-wrapper-inner {
  padding: 15px;
  position: relative;
}

.jp-Cell-Placeholder-wrapper-body {
  background-repeat: repeat;
  background-size: 50% auto;
}

.jp-Cell-Placeholder-wrapper-body div {
  background: #f6f7f8;
  background-image: -webkit-linear-gradient(
    left,
    #f6f7f8 0%,
    #edeef1 20%,
    #f6f7f8 40%,
    #f6f7f8 100%
  );
  background-repeat: no-repeat;
  background-size: 800px 104px;
  height: 104px;
  position: absolute;
  right: 15px;
  left: 15px;
  top: 15px;
}

div.jp-Cell-Placeholder-h1 {
  top: 20px;
  height: 20px;
  left: 15px;
  width: 150px;
}

div.jp-Cell-Placeholder-h2 {
  left: 15px;
  top: 50px;
  height: 10px;
  width: 100px;
}

div.jp-Cell-Placeholder-content-1,
div.jp-Cell-Placeholder-content-2,
div.jp-Cell-Placeholder-content-3 {
  left: 15px;
  right: 15px;
  height: 10px;
}

div.jp-Cell-Placeholder-content-1 {
  top: 100px;
}

div.jp-Cell-Placeholder-content-2 {
  top: 120px;
}

div.jp-Cell-Placeholder-content-3 {
  top: 140px;
}

</style>
<style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
The following CSS variables define the main, public API for styling JupyterLab.
These variables should be used by all plugins wherever possible. In other
words, plugins should not define custom colors, sizes, etc unless absolutely
necessary. This enables users to change the visual theme of JupyterLab
by changing these variables.

Many variables appear in an ordered sequence (0,1,2,3). These sequences
are designed to work well together, so for example, `--jp-border-color1` should
be used with `--jp-layout-color1`. The numbers have the following meanings:

* 0: super-primary, reserved for special emphasis
* 1: primary, most important under normal situations
* 2: secondary, next most important under normal situations
* 3: tertiary, next most important under normal situations

Throughout JupyterLab, we are mostly following principles from Google's
Material Design when selecting colors. We are not, however, following
all of MD as it is not optimized for dense, information rich UIs.
*/

:root {
  /* Elevation
   *
   * We style box-shadows using Material Design's idea of elevation. These particular numbers are taken from here:
   *
   * https://github.com/material-components/material-components-web
   * https://material-components-web.appspot.com/elevation.html
   */

  --jp-shadow-base-lightness: 0;
  --jp-shadow-umbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.2
  );
  --jp-shadow-penumbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.14
  );
  --jp-shadow-ambient-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.12
  );
  --jp-elevation-z0: none;
  --jp-elevation-z1: 0 2px 1px -1px var(--jp-shadow-umbra-color),
    0 1px 1px 0 var(--jp-shadow-penumbra-color),
    0 1px 3px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z2: 0 3px 1px -2px var(--jp-shadow-umbra-color),
    0 2px 2px 0 var(--jp-shadow-penumbra-color),
    0 1px 5px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z4: 0 2px 4px -1px var(--jp-shadow-umbra-color),
    0 4px 5px 0 var(--jp-shadow-penumbra-color),
    0 1px 10px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z6: 0 3px 5px -1px var(--jp-shadow-umbra-color),
    0 6px 10px 0 var(--jp-shadow-penumbra-color),
    0 1px 18px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z8: 0 5px 5px -3px var(--jp-shadow-umbra-color),
    0 8px 10px 1px var(--jp-shadow-penumbra-color),
    0 3px 14px 2px var(--jp-shadow-ambient-color);
  --jp-elevation-z12: 0 7px 8px -4px var(--jp-shadow-umbra-color),
    0 12px 17px 2px var(--jp-shadow-penumbra-color),
    0 5px 22px 4px var(--jp-shadow-ambient-color);
  --jp-elevation-z16: 0 8px 10px -5px var(--jp-shadow-umbra-color),
    0 16px 24px 2px var(--jp-shadow-penumbra-color),
    0 6px 30px 5px var(--jp-shadow-ambient-color);
  --jp-elevation-z20: 0 10px 13px -6px var(--jp-shadow-umbra-color),
    0 20px 31px 3px var(--jp-shadow-penumbra-color),
    0 8px 38px 7px var(--jp-shadow-ambient-color);
  --jp-elevation-z24: 0 11px 15px -7px var(--jp-shadow-umbra-color),
    0 24px 38px 3px var(--jp-shadow-penumbra-color),
    0 9px 46px 8px var(--jp-shadow-ambient-color);

  /* Borders
   *
   * The following variables, specify the visual styling of borders in JupyterLab.
   */

  --jp-border-width: 1px;
  --jp-border-color0: var(--md-grey-400);
  --jp-border-color1: var(--md-grey-400);
  --jp-border-color2: var(--md-grey-300);
  --jp-border-color3: var(--md-grey-200);
  --jp-inverse-border-color: var(--md-grey-600);
  --jp-border-radius: 2px;

  /* UI Fonts
   *
   * The UI font CSS variables are used for the typography all of the JupyterLab
   * user interface elements that are not directly user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-ui-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-ui-font-scale-factor: 1.2;
  --jp-ui-font-size0: 0.83333em;
  --jp-ui-font-size1: 13px; /* Base font size */
  --jp-ui-font-size2: 1.2em;
  --jp-ui-font-size3: 1.44em;
  --jp-ui-font-family: system-ui, -apple-system, blinkmacsystemfont, 'Segoe UI',
    helvetica, arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji',
    'Segoe UI Symbol';

  /*
   * Use these font colors against the corresponding main layout colors.
   * In a light theme, these go from dark to light.
   */

  /* Defaults use Material Design specification */
  --jp-ui-font-color0: rgba(0, 0, 0, 1);
  --jp-ui-font-color1: rgba(0, 0, 0, 0.87);
  --jp-ui-font-color2: rgba(0, 0, 0, 0.54);
  --jp-ui-font-color3: rgba(0, 0, 0, 0.38);

  /*
   * Use these against the brand/accent/warn/error colors.
   * These will typically go from light to darker, in both a dark and light theme.
   */

  --jp-ui-inverse-font-color0: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color1: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color2: rgba(255, 255, 255, 0.7);
  --jp-ui-inverse-font-color3: rgba(255, 255, 255, 0.5);

  /* Content Fonts
   *
   * Content font variables are used for typography of user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-content-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-content-line-height: 1.6;
  --jp-content-font-scale-factor: 1.2;
  --jp-content-font-size0: 0.83333em;
  --jp-content-font-size1: 14px; /* Base font size */
  --jp-content-font-size2: 1.2em;
  --jp-content-font-size3: 1.44em;
  --jp-content-font-size4: 1.728em;
  --jp-content-font-size5: 2.0736em;

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-content-presentation-font-size1: 17px;
  --jp-content-heading-line-height: 1;
  --jp-content-heading-margin-top: 1.2em;
  --jp-content-heading-margin-bottom: 0.8em;
  --jp-content-heading-font-weight: 500;

  /* Defaults use Material Design specification */
  --jp-content-font-color0: rgba(0, 0, 0, 1);
  --jp-content-font-color1: rgba(0, 0, 0, 0.87);
  --jp-content-font-color2: rgba(0, 0, 0, 0.54);
  --jp-content-font-color3: rgba(0, 0, 0, 0.38);
  --jp-content-link-color: var(--md-blue-900);
  --jp-content-font-family: system-ui, -apple-system, blinkmacsystemfont,
    'Segoe UI', helvetica, arial, sans-serif, 'Apple Color Emoji',
    'Segoe UI Emoji', 'Segoe UI Symbol';

  /*
   * Code Fonts
   *
   * Code font variables are used for typography of code and other monospaces content.
   */

  --jp-code-font-size: 13px;
  --jp-code-line-height: 1.3077; /* 17px for 13px base */
  --jp-code-padding: 5px; /* 5px for 13px base, codemirror highlighting needs integer px value */
  --jp-code-font-family-default: menlo, consolas, 'DejaVu Sans Mono', monospace;
  --jp-code-font-family: var(--jp-code-font-family-default);

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-code-presentation-font-size: 16px;

  /* may need to tweak cursor width if you change font size */
  --jp-code-cursor-width0: 1.4px;
  --jp-code-cursor-width1: 2px;
  --jp-code-cursor-width2: 4px;

  /* Layout
   *
   * The following are the main layout colors use in JupyterLab. In a light
   * theme these would go from light to dark.
   */

  --jp-layout-color0: white;
  --jp-layout-color1: white;
  --jp-layout-color2: var(--md-grey-200);
  --jp-layout-color3: var(--md-grey-400);
  --jp-layout-color4: var(--md-grey-600);

  /* Inverse Layout
   *
   * The following are the inverse layout colors use in JupyterLab. In a light
   * theme these would go from dark to light.
   */

  --jp-inverse-layout-color0: #111;
  --jp-inverse-layout-color1: var(--md-grey-900);
  --jp-inverse-layout-color2: var(--md-grey-800);
  --jp-inverse-layout-color3: var(--md-grey-700);
  --jp-inverse-layout-color4: var(--md-grey-600);

  /* Brand/accent */

  --jp-brand-color0: var(--md-blue-900);
  --jp-brand-color1: var(--md-blue-700);
  --jp-brand-color2: var(--md-blue-300);
  --jp-brand-color3: var(--md-blue-100);
  --jp-brand-color4: var(--md-blue-50);
  --jp-accent-color0: var(--md-green-900);
  --jp-accent-color1: var(--md-green-700);
  --jp-accent-color2: var(--md-green-300);
  --jp-accent-color3: var(--md-green-100);

  /* State colors (warn, error, success, info) */

  --jp-warn-color0: var(--md-orange-900);
  --jp-warn-color1: var(--md-orange-700);
  --jp-warn-color2: var(--md-orange-300);
  --jp-warn-color3: var(--md-orange-100);
  --jp-error-color0: var(--md-red-900);
  --jp-error-color1: var(--md-red-700);
  --jp-error-color2: var(--md-red-300);
  --jp-error-color3: var(--md-red-100);
  --jp-success-color0: var(--md-green-900);
  --jp-success-color1: var(--md-green-700);
  --jp-success-color2: var(--md-green-300);
  --jp-success-color3: var(--md-green-100);
  --jp-info-color0: var(--md-cyan-900);
  --jp-info-color1: var(--md-cyan-700);
  --jp-info-color2: var(--md-cyan-300);
  --jp-info-color3: var(--md-cyan-100);

  /* Cell specific styles */

  --jp-cell-padding: 5px;
  --jp-cell-collapser-width: 8px;
  --jp-cell-collapser-min-height: 20px;
  --jp-cell-collapser-not-active-hover-opacity: 0.6;
  --jp-cell-editor-background: var(--md-grey-100);
  --jp-cell-editor-border-color: var(--md-grey-300);
  --jp-cell-editor-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-cell-editor-active-background: var(--jp-layout-color0);
  --jp-cell-editor-active-border-color: var(--jp-brand-color1);
  --jp-cell-prompt-width: 64px;
  --jp-cell-prompt-font-family: var(--jp-code-font-family-default);
  --jp-cell-prompt-letter-spacing: 0;
  --jp-cell-prompt-opacity: 1;
  --jp-cell-prompt-not-active-opacity: 0.5;
  --jp-cell-prompt-not-active-font-color: var(--md-grey-700);

  /* A custom blend of MD grey and blue 600
   * See https://meyerweb.com/eric/tools/color-blend/#546E7A:1E88E5:5:hex */
  --jp-cell-inprompt-font-color: #307fc1;

  /* A custom blend of MD grey and orange 600
   * https://meyerweb.com/eric/tools/color-blend/#546E7A:F4511E:5:hex */
  --jp-cell-outprompt-font-color: #bf5b3d;

  /* Notebook specific styles */

  --jp-notebook-padding: 10px;
  --jp-notebook-select-background: var(--jp-layout-color1);
  --jp-notebook-multiselected-color: var(--md-blue-50);

  /* The scroll padding is calculated to fill enough space at the bottom of the
  notebook to show one single-line cell (with appropriate padding) at the top
  when the notebook is scrolled all the way to the bottom. We also subtract one
  pixel so that no scrollbar appears if we have just one single-line cell in the
  notebook. This padding is to enable a 'scroll past end' feature in a notebook.
  */
  --jp-notebook-scroll-padding: calc(
    100% - var(--jp-code-font-size) * var(--jp-code-line-height) -
      var(--jp-code-padding) - var(--jp-cell-padding) - 1px
  );

  /* Rendermime styles */

  --jp-rendermime-error-background: #fdd;
  --jp-rendermime-table-row-background: var(--md-grey-100);
  --jp-rendermime-table-row-hover-background: var(--md-light-blue-50);

  /* Dialog specific styles */

  --jp-dialog-background: rgba(0, 0, 0, 0.25);

  /* Console specific styles */

  --jp-console-padding: 10px;

  /* Toolbar specific styles */

  --jp-toolbar-border-color: var(--jp-border-color1);
  --jp-toolbar-micro-height: 8px;
  --jp-toolbar-background: var(--jp-layout-color1);
  --jp-toolbar-box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.24);
  --jp-toolbar-header-margin: 4px 4px 0 4px;
  --jp-toolbar-active-background: var(--md-grey-300);

  /* Statusbar specific styles */

  --jp-statusbar-height: 24px;

  /* Input field styles */

  --jp-input-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-input-active-background: var(--jp-layout-color1);
  --jp-input-hover-background: var(--jp-layout-color1);
  --jp-input-background: var(--md-grey-100);
  --jp-input-border-color: var(--jp-inverse-border-color);
  --jp-input-active-border-color: var(--jp-brand-color1);
  --jp-input-active-box-shadow-color: rgba(19, 124, 189, 0.3);

  /* General editor styles */

  --jp-editor-selected-background: #d9d9d9;
  --jp-editor-selected-focused-background: #d7d4f0;
  --jp-editor-cursor-color: var(--jp-ui-font-color0);

  /* Code mirror specific styles */

  --jp-mirror-editor-keyword-color: #008000;
  --jp-mirror-editor-atom-color: #88f;
  --jp-mirror-editor-number-color: #080;
  --jp-mirror-editor-def-color: #00f;
  --jp-mirror-editor-variable-color: var(--md-grey-900);
  --jp-mirror-editor-variable-2-color: rgb(0, 54, 109);
  --jp-mirror-editor-variable-3-color: #085;
  --jp-mirror-editor-punctuation-color: #05a;
  --jp-mirror-editor-property-color: #05a;
  --jp-mirror-editor-operator-color: #a2f;
  --jp-mirror-editor-comment-color: #408080;
  --jp-mirror-editor-string-color: #ba2121;
  --jp-mirror-editor-string-2-color: #708;
  --jp-mirror-editor-meta-color: #a2f;
  --jp-mirror-editor-qualifier-color: #555;
  --jp-mirror-editor-builtin-color: #008000;
  --jp-mirror-editor-bracket-color: #997;
  --jp-mirror-editor-tag-color: #170;
  --jp-mirror-editor-attribute-color: #00c;
  --jp-mirror-editor-header-color: blue;
  --jp-mirror-editor-quote-color: #090;
  --jp-mirror-editor-link-color: #00c;
  --jp-mirror-editor-error-color: #f00;
  --jp-mirror-editor-hr-color: #999;

  /*
    RTC user specific colors.
    These colors are used for the cursor, username in the editor,
    and the icon of the user.
  */

  --jp-collaborator-color1: #ffad8e;
  --jp-collaborator-color2: #dac83d;
  --jp-collaborator-color3: #72dd76;
  --jp-collaborator-color4: #00e4d0;
  --jp-collaborator-color5: #45d4ff;
  --jp-collaborator-color6: #e2b1ff;
  --jp-collaborator-color7: #ff9de6;

  /* Vega extension styles */

  --jp-vega-background: white;

  /* Sidebar-related styles */

  --jp-sidebar-min-width: 250px;

  /* Search-related styles */

  --jp-search-toggle-off-opacity: 0.5;
  --jp-search-toggle-hover-opacity: 0.8;
  --jp-search-toggle-on-opacity: 1;
  --jp-search-selected-match-background-color: rgb(245, 200, 0);
  --jp-search-selected-match-color: black;
  --jp-search-unselected-match-background-color: var(
    --jp-inverse-layout-color0
  );
  --jp-search-unselected-match-color: var(--jp-ui-inverse-font-color0);

  /* Icon colors that work well with light or dark backgrounds */
  --jp-icon-contrast-color0: var(--md-purple-600);
  --jp-icon-contrast-color1: var(--md-green-600);
  --jp-icon-contrast-color2: var(--md-pink-600);
  --jp-icon-contrast-color3: var(--md-blue-600);

  /* Button colors */
  --jp-accept-color-normal: var(--md-blue-700);
  --jp-accept-color-hover: var(--md-blue-800);
  --jp-accept-color-active: var(--md-blue-900);
  --jp-warn-color-normal: var(--md-red-700);
  --jp-warn-color-hover: var(--md-red-800);
  --jp-warn-color-active: var(--md-red-900);
  --jp-reject-color-normal: var(--md-grey-600);
  --jp-reject-color-hover: var(--md-grey-700);
  --jp-reject-color-active: var(--md-grey-800);

  /* File or activity icons and switch semantic variables */
  --jp-jupyter-icon-color: #f37626;
  --jp-notebook-icon-color: #f37626;
  --jp-json-icon-color: var(--md-orange-700);
  --jp-console-icon-background-color: var(--md-blue-700);
  --jp-console-icon-color: white;
  --jp-terminal-icon-background-color: var(--md-grey-800);
  --jp-terminal-icon-color: var(--md-grey-200);
  --jp-text-editor-icon-color: var(--md-grey-700);
  --jp-inspector-icon-color: var(--md-grey-700);
  --jp-switch-color: var(--md-grey-400);
  --jp-switch-true-position-color: var(--md-orange-900);
}
</style>
<style type="text/css">
/* Force rendering true colors when outputing to pdf */
* {
  -webkit-print-color-adjust: exact;
}

/* Misc */
a.anchor-link {
  display: none;
}

/* Input area styling */
.jp-InputArea {
  overflow: hidden;
}

.jp-InputArea-editor {
  overflow: hidden;
}

.cm-editor.cm-s-jupyter .highlight pre {
/* weird, but --jp-code-padding defined to be 5px but 4px horizontal padding is hardcoded for pre.cm-line */
  padding: var(--jp-code-padding) 4px;
  margin: 0;

  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
  color: inherit;

}

.jp-OutputArea-output pre {
  line-height: inherit;
  font-family: inherit;
}

.jp-RenderedText pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
}

/* Hiding the collapser by default */
.jp-Collapser {
  display: none;
}

@page {
    margin: 0.5in; /* Margin for each printed piece of paper */
}

@media print {
  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: block;
  }
}
</style>
<!-- Load mathjax -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_CHTML-full,Safe"> </script>
<!-- MathJax configuration -->
<script type="text/x-mathjax-config">
    init_mathjax = function() {
        if (window.MathJax) {
        // MathJax loaded
            MathJax.Hub.Config({
                TeX: {
                    equationNumbers: {
                    autoNumber: "AMS",
                    useLabelIds: true
                    }
                },
                tex2jax: {
                    inlineMath: [ ['$','$'], ["\\(","\\)"] ],
                    displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
                    processEscapes: true,
                    processEnvironments: true
                },
                displayAlign: 'center',
                CommonHTML: {
                    linebreaks: {
                    automatic: true
                    }
                }
            });

            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
    }
    init_mathjax();
    </script>
<!-- End of mathjax configuration --><script type="module">
  document.addEventListener("DOMContentLoaded", async () => {
    const diagrams = document.querySelectorAll(".jp-Mermaid > pre.mermaid");
    // do not load mermaidjs if not needed
    if (!diagrams.length) {
      return;
    }
    const mermaid = (await import("https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.7.0/mermaid.esm.min.mjs")).default;
    const parser = new DOMParser();

    mermaid.initialize({
      maxTextSize: 100000,
      maxEdges: 100000,
      startOnLoad: false,
      fontFamily: window
        .getComputedStyle(document.body)
        .getPropertyValue("--jp-ui-font-family"),
      theme: document.querySelector("body[data-jp-theme-light='true']")
        ? "default"
        : "dark",
    });

    let _nextMermaidId = 0;

    function makeMermaidImage(svg) {
      const img = document.createElement("img");
      const doc = parser.parseFromString(svg, "image/svg+xml");
      const svgEl = doc.querySelector("svg");
      const { maxWidth } = svgEl?.style || {};
      const firstTitle = doc.querySelector("title");
      const firstDesc = doc.querySelector("desc");

      img.setAttribute("src", `data:image/svg+xml,${encodeURIComponent(svg)}`);
      if (maxWidth) {
        img.width = parseInt(maxWidth);
      }
      if (firstTitle) {
        img.setAttribute("alt", firstTitle.textContent);
      }
      if (firstDesc) {
        const caption = document.createElement("figcaption");
        caption.className = "sr-only";
        caption.textContent = firstDesc.textContent;
        return [img, caption];
      }
      return [img];
    }

    async function makeMermaidError(text) {
      let errorMessage = "";
      try {
        await mermaid.parse(text);
      } catch (err) {
        errorMessage = `${err}`;
      }

      const result = document.createElement("details");
      result.className = 'jp-RenderedMermaid-Details';
      const summary = document.createElement("summary");
      summary.className = 'jp-RenderedMermaid-Summary';
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.innerText = text;
      pre.appendChild(code);
      summary.appendChild(pre);
      result.appendChild(summary);

      const warning = document.createElement("pre");
      warning.innerText = errorMessage;
      result.appendChild(warning);
      return [result];
    }

    async function renderOneMarmaid(src) {
      const id = `jp-mermaid-${_nextMermaidId++}`;
      const parent = src.parentNode;
      let raw = src.textContent.trim();
      const el = document.createElement("div");
      el.style.visibility = "hidden";
      document.body.appendChild(el);
      let results = null;
      let output = null;
      try {
        let { svg } = await mermaid.render(id, raw, el);
        svg = cleanMermaidSvg(svg);
        results = makeMermaidImage(svg);
        output = document.createElement("figure");
        results.map(output.appendChild, output);
      } catch (err) {
        parent.classList.add("jp-mod-warning");
        results = await makeMermaidError(raw);
        output = results[0];
      } finally {
        el.remove();
      }
      parent.classList.add("jp-RenderedMermaid");
      parent.appendChild(output);
    }


    /**
     * Post-process to ensure mermaid diagrams contain only valid SVG and XHTML.
     */
    function cleanMermaidSvg(svg) {
      return svg.replace(RE_VOID_ELEMENT, replaceVoidElement);
    }


    /**
     * A regular expression for all void elements, which may include attributes and
     * a slash.
     *
     * @see https://developer.mozilla.org/en-US/docs/Glossary/Void_element
     *
     * Of these, only `<br>` is generated by Mermaid in place of `\n`,
     * but _any_ "malformed" tag will break the SVG rendering entirely.
     */
    const RE_VOID_ELEMENT =
      /<\s*(area|base|br|col|embed|hr|img|input|link|meta|param|source|track|wbr)\s*([^>]*?)\s*>/gi;

    /**
     * Ensure a void element is closed with a slash, preserving any attributes.
     */
    function replaceVoidElement(match, tag, rest) {
      rest = rest.trim();
      if (!rest.endsWith('/')) {
        rest = `${rest} /`;
      }
      return `<${tag} ${rest}>`;
    }

    void Promise.all([...diagrams].map(renderOneMarmaid));
  });
</script>
<style>
  .jp-Mermaid:not(.jp-RenderedMermaid) {
    display: none;
  }

  .jp-RenderedMermaid {
    overflow: auto;
    display: flex;
  }

  .jp-RenderedMermaid.jp-mod-warning {
    width: auto;
    padding: 0.5em;
    margin-top: 0.5em;
    border: var(--jp-border-width) solid var(--jp-warn-color2);
    border-radius: var(--jp-border-radius);
    color: var(--jp-ui-font-color1);
    font-size: var(--jp-ui-font-size1);
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  .jp-RenderedMermaid figure {
    margin: 0;
    overflow: auto;
    max-width: 100%;
  }

  .jp-RenderedMermaid img {
    max-width: 100%;
  }

  .jp-RenderedMermaid-Details > pre {
    margin-top: 1em;
  }

  .jp-RenderedMermaid-Summary {
    color: var(--jp-warn-color2);
  }

  .jp-RenderedMermaid:not(.jp-mod-warning) pre {
    display: none;
  }

  .jp-RenderedMermaid-Summary > pre {
    display: inline-block;
    white-space: normal;
  }
</style>
<!-- End of mermaid configuration --></head>
<body class="jp-Notebook" data-jp-theme-light="true" data-jp-theme-name="JupyterLab Light">
<main><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Initialize Otter</span>
<span class="kn">import</span> <span class="nn">otter</span>
<span class="n">grader</span> <span class="o">=</span> <span class="n">otter</span><span class="o">.</span><span class="n">Notebook</span><span class="p">(</span><span class="s2">"hw01.ipynb"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h1 id="HW-1:-Math-Review-and-Plotting-%E9%87%8D%E7%82%B9%E5%9C%A8coding%EF%BC%8C%E7%90%86%E8%AE%BA%E4%BE%9B%E5%8F%82%E8%80%83">HW 1: Math Review and Plotting 重点在coding，理论供参考<a class="anchor-link" href="#HW-1:-Math-Review-and-Plotting-%E9%87%8D%E7%82%B9%E5%9C%A8coding%EF%BC%8C%E7%90%86%E8%AE%BA%E4%BE%9B%E5%8F%82%E8%80%83">¶</a></h1><h2 id="Due-Date:-Thursday-Jan-27,-11:59-PM">Due Date: Thursday Jan 27, 11:59 PM<a class="anchor-link" href="#Due-Date:-Thursday-Jan-27,-11:59-PM">¶</a></h2><h2 id="Collaboration-Policy">Collaboration Policy<a class="anchor-link" href="#Collaboration-Policy">¶</a></h2><p>Data science is a collaborative activity. While you may talk with others about
the homework, we ask that you <strong>write your solutions individually</strong>. If you do
discuss the assignments with others please <strong>include their names</strong> at the top
of your notebook.</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><strong>Collaborators</strong>: <em>list collaborators here</em></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h2 id="This-Assignment">This Assignment<a class="anchor-link" href="#This-Assignment">¶</a></h2><p>The purpose of this assignment is for you to combine Python, math, and the ideas in Data 8 to draw some interesting conclusions. The methods and results will help build the foundation of Data 100.</p>
<h2 id="Score-Breakdown">Score Breakdown<a class="anchor-link" href="#Score-Breakdown">¶</a></h2><table>
<thead>
<tr>
<th>Question</th>
<th>Points</th>
</tr>
</thead>
<tbody>
<tr>
<td>1a</td>
<td>1</td>
</tr>
<tr>
<td>1b</td>
<td>2</td>
</tr>
<tr>
<td>2a</td>
<td>1</td>
</tr>
<tr>
<td>2b</td>
<td>1</td>
</tr>
<tr>
<td>2c</td>
<td>2</td>
</tr>
<tr>
<td>2d</td>
<td>2</td>
</tr>
<tr>
<td>2e</td>
<td>1</td>
</tr>
<tr>
<td>3a</td>
<td>2</td>
</tr>
<tr>
<td>3b</td>
<td>2</td>
</tr>
<tr>
<td>3c</td>
<td>1</td>
</tr>
<tr>
<td>3d</td>
<td>2</td>
</tr>
<tr>
<td>3e</td>
<td>2</td>
</tr>
<tr>
<td>4a</td>
<td>1</td>
</tr>
<tr>
<td>4b</td>
<td>1</td>
</tr>
<tr>
<td>4c</td>
<td>1</td>
</tr>
<tr>
<td>4d</td>
<td>1</td>
</tr>
<tr>
<td>5a</td>
<td>1</td>
</tr>
<tr>
<td>5b</td>
<td>1</td>
</tr>
<tr>
<td>5d</td>
<td>3</td>
</tr>
<tr>
<td>6a</td>
<td>2</td>
</tr>
<tr>
<td>6b(i)</td>
<td>2</td>
</tr>
<tr>
<td>6b(ii)</td>
<td>2</td>
</tr>
<tr>
<td>6c</td>
<td>2</td>
</tr>
<tr>
<td>Total</td>
<td>36</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h2 id="Before-You-Start">Before You Start<a class="anchor-link" href="#Before-You-Start">¶</a></h2><p>For each question in the assignment, please write down your answer in the answer cell(s) right below the question.</p>
<p>We understand that it is helpful to have extra cells breaking down the process towards reaching your final answer. If you happen to create new cells <em>below</em> your answer to run code, <strong>NEVER</strong> add cells between a question cell and the answer cell below it. It will cause errors when we run the autograder, and it will sometimes cause a failure to generate the PDF file.</p>
<p><strong>Important note: The local autograder tests will not be comprehensive. You can pass the automated tests in your notebook but still fail tests in the autograder.</strong> Please be sure to check your results carefully.</p>
<h3 id="Initialize-your-environment">Initialize your environment<a class="anchor-link" href="#Initialize-your-environment">¶</a></h3><p>This cell should run without error if you're using the course Jupyter Hub or you have <a href="http://www.ds100.org/sp20/setup">set up your personal computer correctly</a>.</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">matplotlib</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="n">plt</span><span class="o">.</span><span class="n">style</span><span class="o">.</span><span class="n">use</span><span class="p">(</span><span class="s1">'fivethirtyeight'</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Preliminary:-Jupyter-Shortcuts">Preliminary: Jupyter Shortcuts<a class="anchor-link" href="#Preliminary:-Jupyter-Shortcuts">¶</a></h3><p>Here are some useful Jupyter notebook keyboard shortcuts.  To learn more keyboard shortcuts, go to <strong>Help -&gt; Keyboard Shortcuts</strong> in the menu above.</p>
<p>Here are a few we like:</p>
<ol>
<li><code>ctrl</code>+<code>return</code> : <em>Evaluate the current cell</em></li>
<li><code>shift</code>+<code>return</code>: <em>Evaluate the current cell and move to the next</em></li>
<li><code>esc</code> : <em>command mode</em> (may need to press before using any of the commands below)</li>
<li><code>a</code> : <em>create a cell above</em></li>
<li><code>b</code> : <em>create a cell below</em></li>
<li><code>dd</code> : <em>delete a cell</em></li>
<li><code>m</code> : <em>convert a cell to markdown</em></li>
<li><code>y</code> : <em>convert a cell to code</em></li>
</ol>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Preliminary:-NumPy">Preliminary: NumPy<a class="anchor-link" href="#Preliminary:-NumPy">¶</a></h3><p>You should be able to understand the code in the following cells. If not, review the following:</p>
<ul>
<li><a href="https://www.inferentialthinking.com/chapters/05/1/Arrays">The Data 8 Textbook Chapter on NumPy</a></li>
<li><a href="http://ds100.org/fa17/assets/notebooks/numpy/Numpy_Review.html">DS100 NumPy Review</a></li>
<li><a href="http://cs231n.github.io/python-numpy-tutorial/#numpy">Condensed NumPy Review</a></li>
<li><a href="https://numpy.org/doc/stable/user/quickstart.html">The Official NumPy Tutorial</a></li>
</ul>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><strong>Jupyter pro-tip</strong>: Pull up the docs for any function in Jupyter by running a cell with
the function name and a <code>?</code> at the end:</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span>np.arange<span class="o">?</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre><span class="ansi-red-intense-fg ansi-bold">Docstring:</span>
arange([start,] stop[, step,], dtype=None, *, device=None, like=None)

Return evenly spaced values within a given interval.

``arange`` can be called with a varying number of positional arguments:

* ``arange(stop)``: Values are generated within the half-open interval
  ``[0, stop)`` (in other words, the interval including `start` but
  excluding `stop`).
* ``arange(start, stop)``: Values are generated within the half-open
  interval ``[start, stop)``.
* ``arange(start, stop, step)`` Values are generated within the half-open
  interval ``[start, stop)``, with spacing between values given by
  ``step``.

For integer arguments the function is roughly equivalent to the Python
built-in :py:class:`range`, but returns an ndarray rather than a ``range``
instance.

When using a non-integer step, such as 0.1, it is often better to use
`numpy.linspace`.

See the Warning sections below for more information.

Parameters
----------
start : integer or real, optional
    Start of interval.  The interval includes this value.  The default
    start value is 0.
stop : integer or real
    End of interval.  The interval does not include this value, except
    in some cases where `step` is not an integer and floating point
    round-off affects the length of `out`.
step : integer or real, optional
    Spacing between values.  For any output `out`, this is the distance
    between two adjacent values, ``out[i+1] - out[i]``.  The default
    step size is 1.  If `step` is specified as a position argument,
    `start` must also be given.
dtype : dtype, optional
    The type of the output array.  If `dtype` is not given, infer the data
    type from the other input arguments.
device : str, optional
    The device on which to place the created array. Default: None.
    For Array-API interoperability only, so must be ``"cpu"`` if passed.

    .. versionadded:: 2.0.0
like : array_like, optional
    Reference object to allow the creation of arrays which are not
    NumPy arrays. If an array-like passed in as ``like`` supports
    the ``__array_function__`` protocol, the result will be defined
    by it. In this case, it ensures the creation of an array object
    compatible with that passed in via this argument.

    .. versionadded:: 1.20.0

Returns
-------
arange : ndarray
    Array of evenly spaced values.

    For floating point arguments, the length of the result is
    ``ceil((stop - start)/step)``.  Because of floating point overflow,
    this rule may result in the last element of `out` being greater
    than `stop`.

Warnings
--------
The length of the output might not be numerically stable.

Another stability issue is due to the internal implementation of
`numpy.arange`.
The actual step value used to populate the array is
``dtype(start + step) - dtype(start)`` and not `step`. Precision loss
can occur here, due to casting or due to using floating points when
`start` is much larger than `step`. This can lead to unexpected
behaviour. For example::

  &gt;&gt;&gt; np.arange(0, 5, 0.5, dtype=int)
  array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  &gt;&gt;&gt; np.arange(-3, 3, 0.5, dtype=int)
  array([-3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8])

In such cases, the use of `numpy.linspace` should be preferred.

The built-in :py:class:`range` generates :std:doc:`Python built-in integers
that have arbitrary size &lt;python:c-api/long&gt;`, while `numpy.arange`
produces `numpy.int32` or `numpy.int64` numbers. This may result in
incorrect results for large integer values::

  &gt;&gt;&gt; power = 40
  &gt;&gt;&gt; modulo = 10000
  &gt;&gt;&gt; x1 = [(n ** power) % modulo for n in range(8)]
  &gt;&gt;&gt; x2 = [(n ** power) % modulo for n in np.arange(8)]
  &gt;&gt;&gt; print(x1)
  [0, 1, 7776, 8801, 6176, 625, 6576, 4001]  # correct
  &gt;&gt;&gt; print(x2)
  [0, 1, 7776, 7185, 0, 5969, 4816, 3361]  # incorrect

See Also
--------
numpy.linspace : Evenly spaced numbers with careful handling of endpoints.
numpy.ogrid: Arrays of evenly spaced numbers in N-dimensions.
numpy.mgrid: Grid-shaped arrays of evenly spaced numbers in N-dimensions.
:ref:`how-to-partition`

Examples
--------
&gt;&gt;&gt; np.arange(3)
array([0, 1, 2])
&gt;&gt;&gt; np.arange(3.0)
array([ 0.,  1.,  2.])
&gt;&gt;&gt; np.arange(3,7)
array([3, 4, 5, 6])
&gt;&gt;&gt; np.arange(3,7,2)
array([3, 5])
<span class="ansi-red-intense-fg ansi-bold">Type:</span>      builtin_function_or_method</pre>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>You can close the window at the bottom by pressing <code>esc</code> several times or clicking on the x at the right hand side.</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><strong>Another Jupyter pro-tip</strong>: Pull up the docs for any function in Jupyter by typing the function
name, then <code>&lt;Shift&gt;&lt;Tab&gt;</code> on your keyboard. This is super convenient when you forget the order
of the arguments to a function. You can press <code>&lt;Tab&gt;</code> multiple times to expand the docs and reveal additional information.</p>
<p>Try it on the function below:</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">np</span><span class="o">.</span><span class="n">linspace</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Preliminary:-LaTeX">Preliminary: LaTeX<a class="anchor-link" href="#Preliminary:-LaTeX">¶</a></h3><p>You should use LaTeX to format math in your answers. If you aren't familiar with LaTeX, not to worry. It's not hard to use in a Jupyter notebook. Just place your math in between dollar signs within Markdown cells:</p>
<p><code>$ f(x) = 2x $</code> becomes $ f(x) = 2x $.</p>
<p>If you have a longer equation, use double dollar signs to place it on a line by itself:</p>
<p><code>$$ \sum_{i=0}^n i^2 $$</code> becomes:</p>
<p>$$ \sum_{i=0}^n i^2$$</p>
<p>You can align multiple lines using the <code>&amp;</code> anchor, <code>\\</code> newline, in an <code>align</code> block as follows:</p>
<pre><code>\begin{align}
f(x) &amp;= (x - 1)^2 \\
&amp;= x^2 - 2x + 1
\end{align}
</code></pre>
<p>becomes</p>
<p>\begin{align}
f(x) &amp;= (x - 1)^2 \\
&amp;= x^2 - 2x + 1
\end{align}</p>
<ul>
<li><a href="latex_tips.pdf">This PDF</a> has some handy LaTeX.</li>
<li><a href="https://www.sharelatex.com/learn/Mathematical_expressions">For more about basic LaTeX formatting, you can read this article.</a></li>
</ul>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Preliminary:-Sums">Preliminary: Sums<a class="anchor-link" href="#Preliminary:-Sums">¶</a></h3><p>Here's a recap of some basic algebra written in sigma notation. The facts are all just applications of the ordinary associative and distributive properties of addition and multiplication, written compactly and without the possibly ambiguous "...". But if you are ever unsure of whether you're working correctly with a sum, you can always try writing $\sum_{i=1}^n a_i$ as $a_1 + a_2 + \cdots + a_n$ and see if that helps.</p>
<p>You can use any reasonable notation for the index over which you are summing, just as in Python you can use any reasonable name in <code>for name in list</code>. Thus $\sum_{i=1}^n a_i = \sum_{k=1}^n a_k$.</p>
<ul>
<li>$\sum_{i=1}^n (a_i + b_i) = \sum_{i=1}^n a_i + \sum_{i=1}^n b_i$</li>
<li>$\sum_{i=1}^n d = nd$</li>
<li>$\sum_{i=1}^n (ca_i + d) = c\sum_{i=1}^n a_i + nd$</li>
</ul>
<p>These properties may be useful in the Least Squares Predictor question. To see the LaTeX we used, double-click this cell. Evaluate the cell to exit.</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span> 
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- BEGIN QUESTION -->
<h2 id="Question-1:-Calculus">Question 1: Calculus<a class="anchor-link" href="#Question-1:-Calculus">¶</a></h2><p>In this question we will review some fundamental properties of the sigmoid function, which will be discussed when we talk more about logistic regression in the latter half of the class. The sigmoid function is defined to be
$$\sigma(x) = 
\frac{1}{1+e^{-x}}$$</p>
<!--
BEGIN QUESTION
name: q1a
manual: true
-->
<h3 id="Question-1a">Question 1a<a class="anchor-link" href="#Question-1a">¶</a></h3><p>Show that $\sigma(-x) = 1 - \sigma(x)$.</p>
<p><strong>Note, again: In this class, you must always put your answer in the cell that immediately follows the question. DO NOT create any cells between this one and the one that says</strong> <em>Type your answer here, replacing this text.</em></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><em>Type your answer here, replacing this text.</em></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- END QUESTION -->
<!-- BEGIN QUESTION -->
<h3 id="Question-1b">Question 1b<a class="anchor-link" href="#Question-1b">¶</a></h3><p>Show that the derivative of the sigmoid function can be written as:</p>
<p>$$\frac{d}{dx}\sigma(x) = \sigma(x)(1 - \sigma(x))$$</p>
<p><a href="latex_tips.pdf">This PDF</a> has some handy LaTeX.</p>
<!--
BEGIN QUESTION
name: q1b
manual: true
-->
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><em>Type your answer here, replacing this text.</em></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- END QUESTION -->
<h2 id="Question-2:-Probabilities-and-Proportions">Question 2: Probabilities and Proportions<a class="anchor-link" href="#Question-2:-Probabilities-and-Proportions">¶</a></h2>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- BEGIN QUESTION -->
<p>Much of data analysis involves interpreting proportions – lots and lots of related proportions. So let's recall the basics. It might help to start by reviewing <a href="https://www.inferentialthinking.com/chapters/09/5/Finding_Probabilities.html">the main rules</a> from Data 8, with particular attention to what's being multiplied in the multiplication rule.</p>
<!--
    BEGIN QUESTION
    name: q2a
    manual: true
-->
<h3 id="Question-2a">Question 2a<a class="anchor-link" href="#Question-2a">¶</a></h3><p>The Pew Research Foundation publishes the results of numerous surveys, one of which is about the <a href="https://www.pewresearch.org/fact-tank/2019/03/22/public-confidence-in-scientists-has-remained-stable-for-decades/">trust that Americans have</a> in groups such as the military, scientists, and elected officials to act in the public interest. A table in the article summarizes the results.</p>
<p>Pick one of the options (i) and (ii) to answer the question below; if you pick (i), fill in the blank with the percent. Then, explain your choice.</p>
<p>The percent of surveyed U.S. adults who had a great deal of confidence in both scientists and religious leaders</p>
<p>(i) is equal to ______________________.</p>
<p>(ii) cannot be found with the information in the article.</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><em>Type your answer here, replacing this text.</em></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- END QUESTION -->
<h3 id="Question-2b">Question 2b<a class="anchor-link" href="#Question-2b">¶</a></h3><p>In a famous (or infamous) survey, members of the Harvard medical school were asked to consider a scenario in which "a test to detect a disease whose prevalence is 1/1,000 has a false positive rate of 5 percent". The terminology, the specific question asked in the survey, and the answer, are discussed in detail in a Stat 88 textbook <a href="http://stat88.org/textbook/notebooks/Chapter_02/04_Use_and_Interpretation.html#Harvard-Medical-School-Survey">section</a> that you are strongly encouraged to read. As Stat 88 is a Data 8 connector course, the section is another look at the same ideas as in the corresponding <a href="https://www.inferentialthinking.com/chapters/18/2/Making_Decisions.html">Data 8 textbook section</a>.</p>
<p>The corresponding tree diagram is copied below for your reference.</p>
<img alt="No description has been provided for this image" src="tree_disease_harvard.png"/>
<p>The survey did not provide the true positive rate. The respondents and Stat 88 were allowed to assume that the true positive rate is 1, but we will not do so here. <strong>Let the true positive rate be some unknown proportion $p$.</strong></p>
<p>Suppose a person is picked at random from the population. Let $N$ be the event that the person doesn't have the disease and let $T_N$ be the event that the person's test result is negative.</p>
<p>Fill in Blanks 1 and 2 with options chosen from (1)-(9).</p>
<p>The proportion $P(N \mid T_N)$ is the number of people who $\underline{~~~~~~1~~~~~~}$ relative to the total number of people who $\underline{~~~~~~2~~~~~~}$.</p>
<p>(1) are in the population</p>
<p>(2) have the disease</p>
<p>(3) don't have the disease</p>
<p>(4) test positive</p>
<p>(5) test negative</p>
<p>(6) have the disease and test positive</p>
<p>(7) have the disease and test negative</p>
<p>(8) don't have the disease and test positive</p>
<p>(9) don't have the disease and test negative</p>
<p>Assign the variable <code>q4bi</code> to your answer to the first blank and <code>q4bii</code> to your answer to the second blank.</p>
<!--
    BEGIN QUESTION
    name: q2b
    points: 1
-->
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">q4bi</span> <span class="o">=</span> <span class="o">...</span>
<span class="n">q4bii</span> <span class="o">=</span> <span class="o">...</span>
<span class="n">q4bi</span><span class="p">,</span> <span class="n">q4bii</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">grader</span><span class="o">.</span><span class="n">check</span><span class="p">(</span><span class="s2">"q2b"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Question-2c">Question 2c<a class="anchor-link" href="#Question-2c">¶</a></h3><p>(This is a continuation of the previous part.) Define a function <code>no_disease_given_negative</code> that takes $p$ as its argument and returns $P(N \mid T_N)$.</p>
<!--
    BEGIN QUESTION
    name: q4c
    points: 2
-->
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">no_disease_given_negative</span><span class="p">(</span><span class="n">p</span><span class="p">):</span>
    <span class="o">...</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">grader</span><span class="o">.</span><span class="n">check</span><span class="p">(</span><span class="s2">"q4c"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- BEGIN QUESTION -->
<h3 id="Question-2d">Question 2d<a class="anchor-link" href="#Question-2d">¶</a></h3><p>(This part is a continuation of the previous two.) Pick all of the options (i)-(iv) that are true for all values of $p$. Explain by algebraic or probailistic reasoning; you are welcome to use your function <code>no_disease_given_negative</code> to try a few cases numerically. Your explanation should include the reasons why you <em>didn't</em> choose some options.</p>
<p>$P(N \mid T_N)$ is</p>
<p>(i) equal to $0.95$.</p>
<p>(ii) equal to $0.999 \times 0.95$.</p>
<p>(iii) greater than $0.999 \times 0.95$.</p>
<p>(iv) greater than $0.95$.</p>
<!--
BEGIN QUESTION
name: q2d
manual: true
-->
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><em>Type your answer here, replacing this text.</em></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- END QUESTION -->
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Use this cell for experimenting if you wish, but your answer should be written in the cell above.</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- BEGIN QUESTION -->
<h3 id="Question-2e">Question 2e<a class="anchor-link" href="#Question-2e">¶</a></h3><p>Suzuki is one of most commonly owned makes of cars in our county (Alameda). A car heading from Berkeley to San Francisco is pulled over on the freeway for speeding. Suppose I tell you that the car is either a Suzuki or a Lamborghini, and you have to guess which of the two is more likely.</p>
<p>What would you guess, and why? Make some reasonable assumptions and explain them (data scientists often have to do this), justify your answer, and say how it's connected to the previous parts.</p>
<!--
    BEGIN QUESTION
    name: q2e
    manual: true
-->
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><em>Type your answer here, replacing this text.</em></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- END QUESTION -->
<h2 id="Question-3:-Distributions">Question 3: Distributions<a class="anchor-link" href="#Question-3:-Distributions">¶</a></h2>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>Visualizing distributions, both categorical and numerical, helps us understand variability. In Data 8 you visualized numerical distributions by drawing <a href="https://www.inferentialthinking.com/chapters/07/2/Visualizing_Numerical_Distributions.html#A-Histogram">histograms</a>, which look like bar charts but represent proportions by the <em>areas</em> of the bars instead of the heights or lengths. In this exercise you will use the <code>hist</code> function in <code>matplotlib</code> instead of the corresponding <code>Table</code> method to draw histograms.</p>
<p>To start off, suppose we want to plot the probability distribution of the number of spots on a single roll of a die. That should be a flat histogram since the chance of each of the values 1 through 6 is 1/6. Here is a first attempt at drawing the histogram.</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">faces</span> <span class="o">=</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">7</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">faces</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[ ]:</div>
<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain" tabindex="0">
<pre>(array([1., 0., 1., 0., 1., 0., 1., 0., 1., 1.]),
 array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. ]),
 &lt;BarContainer object of 10 artists&gt;)</pre>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,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atQoxcXF6bLLLtPgwYNVXV2t/Px8VVRUKDExUTk5OQoI4K46lt5//31dd911CgoK0okTJ5Sfn6+rr77aelq9yujRo1VTU6PFixe32efxePi7xcCpU6d0zz33aPPmzYqOjlZSUpJCQkL06aefas+ePfrNb36jcePGmczN0Vcn9XQbN27UsGHD5PF49MQTTyg9Pd16Sr1aXFycduzYoYSEhFbb9+7dq5kzZyotLU3Tp09X3759jWbYO33rW9/SJ598IpfL1Wr7yZMnNWvWLBUWFio/P18/+MEPjGaIxsZGLV68WKNHj9awYcP0u9/9znpKvVZYWJgefPBB62ngP5555hlt3rxZt99+u9asWdPmuyxPnjxpNDNuDNvK5MmT23y/JuzMmDGjTYxJ0vjx4zVhwgRVV1fr/fffN5hZ7xYQENAmxiQpKChIN954oyS1+xVq8K3HH39cH3zwgZ566im/+vJkwNK///1vrVmzRlFRUfrVr3512j8bQUF256k4Q4Zu6YILLpAk/rLxI83Nzd7b38TExBjPpvd67733tHbtWj300EO6/PLLrafT6zU0NGjLli2qrKzUwIEDFRcXp6uuusp6Wr1SYWGhqqurNX/+fDU1NelPf/qTysrKFBYWpsmTJ7d7H1VfI8jQ7VRUVOjNN9/URRddpFGjRllPp9dqaGjQ2rVrderUKX3xxRd66623VFpaqvnz52vSpEnW0+uVvvrqK+9blampqdbTgb6+afpdd93ValtcXJxeeOEFRUdHG82qd3rvvfckff0/8vHx8Tp48KB3X0BAgJYsWaJVq1YZzY4gQzfT2NioO++8U1999ZUefvhhzpAZamho0Jo1a7z/3adPHy1dulS//OUvDWfVuz366KMqKyvTm2++yZ8NPzB//nyNGzdOMTExGjBggA4ePKinn35ar732mmbMmKG9e/dq4MCB1tPsNT7//HNJ0tNPP63vfve7Kiws1MiRI1VSUqK7775bTz31lKKjo7Vo0SKT+XENGbqN5uZmLVmyRHv37tXChQtb3YQYXS8kJETV1dU6duyY/v73v+vxxx9XVlaWbrzxxjN+3y185+2339bGjRt177338paxn3jggQc0adIkXXjhherfv79iY2P17LPP6uabb1ZFRYVefvll6yn2Ks3NzZIkl8ulLVu2KC4uTiEhIRo/frw2b96sgIAAPfXUU2bzI8jQLTQ3N+uuu+7S66+/rptuuklPPPGE9ZTwHwEBAbrkkku0aNEiPfnkk9q/f7/Wrl1rPa1e5eTJk1q8eLFGjRqle+65x3o6aEdKSook6a9//avxTHqX0NBQSdL3vvc9XXzxxa32xcTEKCoqSocOHVJ1dbXB7HjLEt1Ay5mxV199VXPmzFFmZib3uPJT1113nSSpuLjYeCa9y/Hjx1VWViZJuvDCC097zNSpUyVJr7zyivfTsLAxaNAgSdKXX35pPJPeZcSIEZK+vhXJ6bRsr6+v77I5fRNBBr/2zRhLTk7Ws88+y7UxfqyyslLSfz8Fi67Rt29fLViw4LT79u7dq7KyMt1www0aPHgwt/bxAy3fNMJadK0JEyZIkkpLS9vsa2xsVHl5uQYMGGD2bUEEGfxWy9uUr776qmbNmqXnnnuOGPMDH3zwgTwej/r3799q+5dffqkVK1ZI+u/ZGHSNfv36aePGjafdt3jxYpWVlSktLY079Xeh0tJSXXrppW3+nJSWlurhhx+WJM2ZM8dgZr1XdHS0EhMTVVhYqKysLN16663efU888YRqamp00003md2LjCD7hqysLO3bt0+SvDcc/e1vf+t9+2XcuHGtFhC+tWbNGmVnZyskJESXXXaZfv3rX7c5Zvr06YqNjTWYXe+Vm5urTZs26dprr5XH49HAgQN15MgR7dq1S8eOHdO4ceO0ZMkS62kCpn7/+99r06ZNGj9+vCIjI9W/f38dPHhQ+fn5amxsVFpamuLj462n2eusXbtW06ZN07Jly5SXl6cRI0aopKRERUVFioyMVEZGhtncCLJv2Ldvn7Kzs1tt279/v/bv3+/9b4Ks63zyySeSvr4+5vHHHz/tMR6PhyDrYtdff70qKyv19ttv6+2339aJEycUGhqqUaNGafbs2frxj39serdrwB9MmDBBpaWlKikp0b59+/Tll19q0KBBmjp1qm6//XYlJiZaT7FXio6O1u7du/Xoo4+qoKBAhYWFcrvduuOOO7R8+fIzXoPZFfhycQAAAGN8VA0AAMAYQQYAAGCMIAMAADBGkAEAABgjyAAAAIwRZAAAAMYIMgAAAGMEGQAAgDGCDAAAwBhBBgAAYIwgAwAAMEaQAQAAGPt/YNl0cWrz868AAAAASUVORK5CYII="/>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>This default plot is not helpful. We have to choose some arguments to get a visualization that we can interpret.</p>
<p>Note that the second printed line shows the left ends of the default bins, as well as the right end of the last bin. The first line shows the counts in the bins. If you don't want the printed lines you can add a semi-colon at the end of the call to <code>plt.hist</code>, but we'll keep the lines for now.</p>
<p>Let's redraw the histogram with bins of unit length centered at the possible values. By the end of the exercise you'll see a reason for centering. Notice that the argument for specifying bins is the same as the one for the <code>Table</code> method <code>hist</code>.</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">unit_bins</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">6.6</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">faces</span><span class="p">,</span> <span class="n">bins</span> <span class="o">=</span> <span class="n">unit_bins</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[ ]:</div>
<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain" tabindex="0">
<pre>(array([1., 1., 1., 1., 1., 1.]),
 array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]),
 &lt;BarContainer object of 6 artists&gt;)</pre>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,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"/>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>We need to see the edges of the bars! Let's specify the edge color <code>ec</code> to be white. <a href="https://matplotlib.org/3.1.0/gallery/color/named_colors.html">Here</a> are all the colors you could use, but do try to drag yourself away from the poetic names.</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">faces</span><span class="p">,</span> <span class="n">bins</span> <span class="o">=</span> <span class="n">unit_bins</span><span class="p">,</span> <span class="n">ec</span><span class="o">=</span><span class="s1">'white'</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[ ]:</div>
<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain" tabindex="0">
<pre>(array([1., 1., 1., 1., 1., 1.]),
 array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]),
 &lt;BarContainer object of 6 artists&gt;)</pre>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,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"/>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>That's much better, but look at the vertical axis. It is not drawn to the <a href="https://www.inferentialthinking.com/chapters/07/2/Visualizing_Numerical_Distributions.html#The-Histogram:-General-Principles-and-Calculation">density scale</a> defined in Data 8. We want a histogram of a probability distribution, so the total area should be 1. We just have to ask for that.</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">faces</span><span class="p">,</span> <span class="n">bins</span> <span class="o">=</span> <span class="n">unit_bins</span><span class="p">,</span> <span class="n">ec</span><span class="o">=</span><span class="s1">'white'</span><span class="p">,</span> <span class="n">density</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[ ]:</div>
<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain" tabindex="0">
<pre>(array([0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667]),
 array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]),
 &lt;BarContainer object of 6 artists&gt;)</pre>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,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eromT54sf39/jR8/XgkJCTpx4kSPjo+JiZGfn5/8/f17cwkAAADGcrd6QElJiaKiouTh4aHIyEh5eXlpx44dio+P17Fjx7R8+fJu52hpaVFsbKyKioo0depUzZ8/X2VlZcrOztbevXu1e/duDRs27JLH//a3v1VRUZE8PDzU2tpq9RIAAACMZmkHrrm5WQkJCXI6nSooKNDTTz+tn//859q3b59uvPFGpaSk6OjRo93Ok5OTo6KiIkVHR+tPf/qTVq9ereeff17r1q3TkSNHlJqaesljP/30U61atUoPP/ywhg8fbmX5AAAA/YKlAFdSUqKKigpFR0crJCTE1e7r66vExEQ1NjYqNze323mys7MlSU888YQcDoerPT4+XsHBwdq+fbvOnTvX4bjW1lb98Ic/lL+/v37yk59YWToAAEC/YSnA7du3T5IUFhbWoS88PFyStH///i7naGho0MGDBzVmzBgFBga263M4HJo9e7bOnj2r999/v8OxzzzzjPbv369NmzbJ09PTytIBAAD6DUvPwJWVlUmSRo8e3aHP399fXl5eKi8v73KOiooKtbS0aNSoUZ32t7WXlZVp+vTp7c69Zs0aLV26VN/4xjesLLudhoaGXh/bnVan5UcK0S2ecbQX9bQfNbUX9bQX9bRba2vLFcsSHh4ePR5rKXGcPn1akuTj49Npv7e3t2tMd3P4+vp22t8298XztLS0aNmyZfL399dPf/pTK0vuoKqqShcuXLisOS5lQMCYKzLv1YzvqNiLetqPmtqLetqLetqvualZlVWVts/r5uZ2yc2tzhixZbRhwwa9++672rlzpwYPHnxZcwUEBNi0qo5q2IGz3UWPSMIG1NN+1NRe1NNe1NN+7gPcdd2IEX29DGsBrrPdsYudOXNGfn5+PZqjrq6u0/7/3OX75JNPlJaWpgcffFC33367leV2ysr2pFWOJt6LbD8+fexFPe1HTe1FPe1FPe3mcDivaJboKUuJo+3Zt7Zn4S5WXV2t+vr6brf/goOD5XQ6L/msXFt727k++ugjnT9/Xlu3bpWfn1+7vyorK3X+/HnXn2tra61cDgAAgJEs7cCFhoYqIyNDxcXFioqKatdXVFTkGtMVT09PTZkyRe+++66OHj3a7puora2t2rNnj4YMGaJbb71VkhQYGKjFixd3Old+fr7OnTun2NhYSdKgQYOs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/76vl2QsX19f/fd//3dfL6Pf+NWvfqXnnntODz74oNLT0+Xm5tauv7m5uY9WZgb+d/Yqcccdd3T43Vn0zvz58zuEN0maPn26ZsyYodraWv3jH//og5WZy+l0dghvkuTu7q677rpLkrr9iT50bu3atfroo4+0adOmDv+BBPrKuXPnlJ6eruDgYD355JOd/rPp7s4eU1eoDmCjAQMGSBL/obRJS0uL6zVBEyZM6OPVmOeDDz7QunXr9JOf/ETjx4/v6+UYr7GxUS+++KKOHz8ub29vTZ48WV//+tf7ellGKi4uVm1treLi4nThwgX98Y9/VFlZmXx9fXXHHXd0+z5ZEOAA21RWVurNN9/Uddddp4kTJ/b1cozU2NiodevWqbW1VZ999pn27t2r0tJSxcXFadasWX29PKOcP3/edes0ISGhr5fTL1RXV+vhhx9u1zZ58mT95je/0ciRI/toVWb64IMPJH3xP7uhoaH65JNPXH1Op1MPPfSQUlNT+2h1ZiDAATZoamrS0qVLdf78ea1evZoduF5qbGxUenq6688Oh0PLly/Xz372sz5clZl+8YtfqKysTG+++Sb/PNogLi5O06ZN04QJEzRkyBB98skn2rx5s1566SXNnz9fBw4ckLe3d18v0xg1NTWSpM2bN+uWW25RcXGxxo4dq8OHD+uRRx7Rpk2bNHLkSD3wwAN9vNIvL56BAy5TS0uLHnroIR04cEBLlixp9yJrWOPl5aXa2lqdOnVKf//737V27VplZ2frrrvuuuRvIaOjv/zlL9q4caMee+wxbj3bJCkpSbNmzdLw4cM1ePBghYSE6JlnntG9996ryspK/fa3v+3rJRqlpaVFkjRw4EC9+OKLmjx5sry8vDR9+nQ999xzcjqd2rRpUx+v8suNAAdchpaWFj388MPavn277rnnHj311FN9vaR+wel06vrrr9cDDzygp59+Wm+//bbWrVvX18syQnNzs5YtW6aJEyfq0Ucf7evl9Hvx8fGSpHfeeaePV2IWHx8fSdLXvvY1ffWrX23XN2HCBAUHB6uiokK1tbV9sDozcAsV6KW2nbff/e53io6OVmZmJu8puwJmz54tSdq3b18fr8QM9fX1KisrkyQNHz680zFz586VJL3wwguub/mid4YOHSpJ+vzzz/t4JWYZM2aMpC9ezdKZtvaGhob/b2syDQEO6IWLw1tkZKSeeeYZnjO6Qo4fPy7p/77hi64NGjRIixcv7rTvwIEDKisr05133qlhw4bxaiEbtP36CrW0ZsaMGZKk0tLSDn1NTU0qLy/XkCFDOv1VJnyBAAdY1Hbb9He/+50WLlyoZ599lvB2mT766CMFBgZq8ODB7do///xzPf7445L+b9cIXfP09NTGjRs77Vu2bJnKysqUmJjILzFYUFpaqhtuuKHDP5+lpaVavXq1JCk6OroPVmaukSNHKiwsTMXFxcrOztZ3v/tdV99TTz2luro63XPPPbwLrgtU5iqRnZ2tt956S5JcL5l9/vnnXbelpk2b1u5fIFxaenq6cnNz5eXlpRtvvFG//OUvO4yJiIhQSEhIH6zOTPn5+dqyZYu+8Y1vKDAwUN7e3qqqqtLu3bt16tQpTZs2TQ899FBfLxNXqZdffllbtmzR9OnTNWLECA0ePFiffPKJCgsL1dTUpMTERIWGhvb1Mo2zbt06ffOb39SPfvQjFRQUaMyYMTp8+LBKSko0YsQIpaSk9PUSv9QIcFeJt956S7m5ue3a3n77bb399tuuPxPgeubo0aOSvnjWaO3atZ2OCQwMJMBZ8K1vfUvHjx/XX/7yF/3lL3/R2bNn5ePjo4kTJyoqKkr3338//yeOPjNjxgyVlpbq8OHDeuutt/T5559r6NChmjt3rh588EGFhYX19RKNNHLkSO3Zs0e/+MUvVFRUpOLiYvn7++v73/++Vq5ceclnOPEFfsweAADAMHxlDgAAwDAEOAAAAMMQ4AAAAAxDgAMAADAMAQ4AAMAwBDgAAADDEOAAAAAMQ4ADAAAwDAEOAADAMAQ4AAAAwxDgAAAADEOAAwAAMMz/A9F3IeQ4jU/pAAAAAElFTkSuQmCC"/>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>That's the probability histogram of the number of spots on one roll of a die. The proportion is $1/6$ in each of the bins.</p>
<p><strong>Note</strong>: You may notice that running the above cells also displayed the return value of the last function call of each cell. This was intentional on our part to show you how <code>plt.hist()</code> (<a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html">documentation</a>) returned different values per plot.</p>
<p><strong>Note 2</strong>: Going forward, you can use a semicolon <code>;</code> on the last line to suppress additional display, as below.</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">faces</span><span class="p">,</span> <span class="n">bins</span> <span class="o">=</span> <span class="n">unit_bins</span><span class="p">,</span> <span class="n">ec</span><span class="o">=</span><span class="s1">'white'</span><span class="p">,</span> <span class="n">density</span><span class="o">=</span><span class="kc">True</span><span class="p">);</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,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eromT54sf39/jR8/XgkJCTpx4kSPjo+JiZGfn5/8/f17cwkAAADGcrd6QElJiaKiouTh4aHIyEh5eXlpx44dio+P17Fjx7R8+fJu52hpaVFsbKyKioo0depUzZ8/X2VlZcrOztbevXu1e/duDRs27JLH//a3v1VRUZE8PDzU2tpq9RIAAACMZmkHrrm5WQkJCXI6nSooKNDTTz+tn//859q3b59uvPFGpaSk6OjRo93Ok5OTo6KiIkVHR+tPf/qTVq9ereeff17r1q3TkSNHlJqaesljP/30U61atUoPP/ywhg8fbmX5AAAA/YKlAFdSUqKKigpFR0crJCTE1e7r66vExEQ1NjYqNze323mys7MlSU888YQcDoerPT4+XsHBwdq+fbvOnTvX4bjW1lb98Ic/lL+/v37yk59YWToAAEC/YSnA7du3T5IUFhbWoS88PFyStH///i7naGho0MGDBzVmzBgFBga263M4HJo9e7bOnj2r999/v8OxzzzzjPbv369NmzbJ09PTytIBAAD6DUvPwJWVlUmSRo8e3aHP399fXl5eKi8v73KOiooKtbS0aNSoUZ32t7WXlZVp+vTp7c69Zs0aLV26VN/4xjesLLudhoaGXh/bnVan5UcK0S2ecbQX9bQfNbUX9bQX9bRba2vLFcsSHh4ePR5rKXGcPn1akuTj49Npv7e3t2tMd3P4+vp22t8298XztLS0aNmyZfL399dPf/pTK0vuoKqqShcuXLisOS5lQMCYKzLv1YzvqNiLetqPmtqLetqLetqvualZlVWVts/r5uZ2yc2tzhixZbRhwwa9++672rlzpwYPHnxZcwUEBNi0qo5q2IGz3UWPSMIG1NN+1NRe1NNe1NN+7gPcdd2IEX29DGsBrrPdsYudOXNGfn5+PZqjrq6u0/7/3OX75JNPlJaWpgcffFC33367leV2ysr2pFWOJt6LbD8+fexFPe1HTe1FPe1FPe3mcDivaJboKUuJo+3Zt7Zn4S5WXV2t+vr6brf/goOD5XQ6L/msXFt727k++ugjnT9/Xlu3bpWfn1+7vyorK3X+/HnXn2tra61cDgAAgJEs7cCFhoYqIyNDxcXFioqKatdXVFTkGtMVT09PTZkyRe+++66OHj3a7puora2t2rNnj4YMGaJbb71VkhQYGKjFixd3Old+fr7OnTun2NhYSdKgQYOs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/76vl2QsX19f/fd//3dfL6Pf+NWvfqXnnntODz74oNLT0+Xm5tauv7m5uY9WZgb+d/Yqcccdd3T43Vn0zvz58zuEN0maPn26ZsyYodraWv3jH//og5WZy+l0dghvkuTu7q677rpLkrr9iT50bu3atfroo4+0adOmDv+BBPrKuXPnlJ6eruDgYD355JOd/rPp7s4eU1eoDmCjAQMGSBL/obRJS0uL6zVBEyZM6OPVmOeDDz7QunXr9JOf/ETjx4/v6+UYr7GxUS+++KKOHz8ub29vTZ48WV//+tf7ellGKi4uVm1treLi4nThwgX98Y9/VFlZmXx9fXXHHXd0+z5ZEOAA21RWVurNN9/Uddddp4kTJ/b1cozU2NiodevWqbW1VZ999pn27t2r0tJSxcXFadasWX29PKOcP3/edes0ISGhr5fTL1RXV+vhhx9u1zZ58mT95je/0ciRI/toVWb64IMPJH3xP7uhoaH65JNPXH1Op1MPPfSQUlNT+2h1ZiDAATZoamrS0qVLdf78ea1evZoduF5qbGxUenq6688Oh0PLly/Xz372sz5clZl+8YtfqKysTG+++Sb/PNogLi5O06ZN04QJEzRkyBB98skn2rx5s1566SXNnz9fBw4ckLe3d18v0xg1NTWSpM2bN+uWW25RcXGxxo4dq8OHD+uRRx7Rpk2bNHLkSD3wwAN9vNIvL56BAy5TS0uLHnroIR04cEBLlixp9yJrWOPl5aXa2lqdOnVKf//737V27VplZ2frrrvuuuRvIaOjv/zlL9q4caMee+wxbj3bJCkpSbNmzdLw4cM1ePBghYSE6JlnntG9996ryspK/fa3v+3rJRqlpaVFkjRw4EC9+OKLmjx5sry8vDR9+nQ999xzcjqd2rRpUx+v8suNAAdchpaWFj388MPavn277rnnHj311FN9vaR+wel06vrrr9cDDzygp59+Wm+//bbWrVvX18syQnNzs5YtW6aJEyfq0Ucf7evl9Hvx8fGSpHfeeaePV2IWHx8fSdLXvvY1ffWrX23XN2HCBAUHB6uiokK1tbV9sDozcAsV6KW2nbff/e53io6OVmZmJu8puwJmz54tSdq3b18fr8QM9fX1KisrkyQNHz680zFz586VJL3wwguub/mid4YOHSpJ+vzzz/t4JWYZM2aMpC9ezdKZtvaGhob/b2syDQEO6IWLw1tkZKSeeeYZnjO6Qo4fPy7p/77hi64NGjRIixcv7rTvwIEDKisr05133qlhw4bxaiEbtP36CrW0ZsaMGZKk0tLSDn1NTU0qLy/XkCFDOv1VJnyBAAdY1Hbb9He/+50WLlyoZ599lvB2mT766CMFBgZq8ODB7do///xzPf7445L+b9cIXfP09NTGjRs77Vu2bJnKysqUmJjILzFYUFpaqhtuuKHDP5+lpaVavXq1JCk6OroPVmaukSNHKiwsTMXFxcrOztZ3v/tdV99TTz2luro63XPPPbwLrgtU5iqRnZ2tt956S5JcL5l9/vnnXbelpk2b1u5fIFxaenq6cnNz5eXlpRtvvFG//OUvO4yJiIhQSEhIH6zOTPn5+dqyZYu+8Y1vKDAwUN7e3qqqqtLu3bt16tQpTZs2TQ899FBfLxNXqZdffllbtmzR9OnTNWLECA0ePFiffPKJCgsL1dTUpMTERIWGhvb1Mo2zbt06ffOb39SPfvQjFRQUaMyYMTp8+LBKSko0YsQIpaSk9PUSv9QIcFeJt956S7m5ue3a3n77bb399tuuPxPgeubo0aOSvnjWaO3atZ2OCQwMJMBZ8K1vfUvHjx/XX/7yF/3lL3/R2bNn5ePjo4kTJyoqKkr3338//yeOPjNjxgyVlpbq8OHDeuutt/T5559r6NChmjt3rh588EGFhYX19RKNNHLkSO3Zs0e/+MUvVFRUpOLiYvn7++v73/++Vq5ceclnOPEFfsweAADAMHxlDgAAwDAEOAAAAMMQ4AAAAAxDgAMAADAMAQ4AAMAwBDgAAADDEOAAAAAMQ4ADAAAwDAEOAADAMAQ4AAAAwxDgAAAADEOAAwAAMMz/A9F3IeQ4jU/pAAAAAElFTkSuQmCC"/>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- BEGIN QUESTION -->
<h3 id="Question-3a">Question 3a<a class="anchor-link" href="#Question-3a">¶</a></h3><p>Define a function <code>integer_distribution</code> that takes an array of integers and draws the histogram of the distribution using unit bins centered at the integers and white edges for the bars. The histogram should be drawn to the density scale. The left-most bar should be centered at the smallest integer in the array, and the right-most bar at the largest.</p>
<p>Your function does not have to check that the input is an array consisting only of integers. The display does not need to include the printed proportions and bins.</p>
<p>If you have trouble defining the function, go back and carefully read all the lines of code that resulted in the probability histogram of the number of spots on one roll of a die. Pay special attention to the bins.</p>
<!--
    BEGIN QUESTION
    name: q3a
    manual: true
-->
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">integer_distribution</span><span class="p">(</span><span class="n">x</span><span class="p">):</span>
    <span class="n">bins</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="nb">min</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> <span class="o">-</span> <span class="mf">0.5</span><span class="p">,</span> <span class="nb">max</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> <span class="o">+</span> <span class="mf">1.5</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">bins</span><span class="o">=</span><span class="n">bins</span><span class="p">,</span> <span class="n">ec</span><span class="o">=</span><span class="s1">'white'</span><span class="p">,</span><span class="n">density</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="n">integer_distribution</span><span class="p">(</span><span class="n">faces</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,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eromT54sf39/jR8/XgkJCTpx4kSPjo+JiZGfn5/8/f17cwkAAADGcrd6QElJiaKiouTh4aHIyEh5eXlpx44dio+P17Fjx7R8+fJu52hpaVFsbKyKioo0depUzZ8/X2VlZcrOztbevXu1e/duDRs27JLH//a3v1VRUZE8PDzU2tpq9RIAAACMZmkHrrm5WQkJCXI6nSooKNDTTz+tn//859q3b59uvPFGpaSk6OjRo93Ok5OTo6KiIkVHR+tPf/qTVq9ereeff17r1q3TkSNHlJqaesljP/30U61atUoPP/ywhg8fbmX5AAAA/YKlAFdSUqKKigpFR0crJCTE1e7r66vExEQ1NjYqNze323mys7MlSU888YQcDoerPT4+XsHBwdq+fbvOnTvX4bjW1lb98Ic/lL+/v37yk59YWToAAEC/YSnA7du3T5IUFhbWoS88PFyStH///i7naGho0MGDBzVmzBgFBga263M4HJo9e7bOnj2r999/v8OxzzzzjPbv369NmzbJ09PTytIBAAD6DUvPwJWVlUmSRo8e3aHP399fXl5eKi8v73KOiooKtbS0aNSoUZ32t7WXlZVp+vTp7c69Zs0aLV26VN/4xjesLLudhoaGXh/bnVan5UcK0S2ecbQX9bQfNbUX9bQX9bRba2vLFcsSHh4ePR5rKXGcPn1akuTj49Npv7e3t2tMd3P4+vp22t8298XztLS0aNmyZfL399dPf/pTK0vuoKqqShcuXLisOS5lQMCYKzLv1YzvqNiLetqPmtqLetqLetqvualZlVWVts/r5uZ2yc2tzhixZbRhwwa9++672rlzpwYPHnxZcwUEBNi0qo5q2IGz3UWPSMIG1NN+1NRe1NNe1NN+7gPcdd2IEX29DGsBrrPdsYudOXNGfn5+PZqjrq6u0/7/3OX75JNPlJaWpgcffFC33367leV2ysr2pFWOJt6LbD8+fexFPe1HTe1FPe1FPe3mcDivaJboKUuJo+3Zt7Zn4S5WXV2t+vr6brf/goOD5XQ6L/msXFt727k++ugjnT9/Xlu3bpWfn1+7vyorK3X+/HnXn2tra61cDgAAgJEs7cCFhoYqIyNDxcXFioqKatdXVFTkGtMVT09PTZkyRe+++66OHj3a7puora2t2rNnj4YMGaJbb71VkhQYGKjFixd3Old+fr7OnTun2NhYSdKgQYOs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/76vl2QsX19f/fd//3dfL6Pf+NWvfqXnnntODz74oNLT0+Xm5tauv7m5uY9WZgb+d/Yqcccdd3T43Vn0zvz58zuEN0maPn26ZsyYodraWv3jH//og5WZy+l0dghvkuTu7q677rpLkrr9iT50bu3atfroo4+0adOmDv+BBPrKuXPnlJ6eruDgYD355JOd/rPp7s4eU1eoDmCjAQMGSBL/obRJS0uL6zVBEyZM6OPVmOeDDz7QunXr9JOf/ETjx4/v6+UYr7GxUS+++KKOHz8ub29vTZ48WV//+tf7ellGKi4uVm1treLi4nThwgX98Y9/VFlZmXx9fXXHHXd0+z5ZEOAA21RWVurNN9/Uddddp4kTJ/b1cozU2NiodevWqbW1VZ999pn27t2r0tJSxcXFadasWX29PKOcP3/edes0ISGhr5fTL1RXV+vhhx9u1zZ58mT95je/0ciRI/toVWb64IMPJH3xP7uhoaH65JNPXH1Op1MPPfSQUlNT+2h1ZiDAATZoamrS0qVLdf78ea1evZoduF5qbGxUenq6688Oh0PLly/Xz372sz5clZl+8YtfqKysTG+++Sb/PNogLi5O06ZN04QJEzRkyBB98skn2rx5s1566SXNnz9fBw4ckLe3d18v0xg1NTWSpM2bN+uWW25RcXGxxo4dq8OHD+uRRx7Rpk2bNHLkSD3wwAN9vNIvL56BAy5TS0uLHnroIR04cEBLlixp9yJrWOPl5aXa2lqdOnVKf//737V27VplZ2frrrvuuuRvIaOjv/zlL9q4caMee+wxbj3bJCkpSbNmzdLw4cM1ePBghYSE6JlnntG9996ryspK/fa3v+3rJRqlpaVFkjRw4EC9+OKLmjx5sry8vDR9+nQ999xzcjqd2rRpUx+v8suNAAdchpaWFj388MPavn277rnnHj311FN9vaR+wel06vrrr9cDDzygp59+Wm+//bbWrVvX18syQnNzs5YtW6aJEyfq0Ucf7evl9Hvx8fGSpHfeeaePV2IWHx8fSdLXvvY1ffWrX23XN2HCBAUHB6uiokK1tbV9sDozcAsV6KW2nbff/e53io6OVmZmJu8puwJmz54tSdq3b18fr8QM9fX1KisrkyQNHz680zFz586VJL3wwguub/mid4YOHSpJ+vzzz/t4JWYZM2aMpC9ezdKZtvaGhob/b2syDQEO6IWLw1tkZKSeeeYZnjO6Qo4fPy7p/77hi64NGjRIixcv7rTvwIEDKisr05133qlhw4bxaiEbtP36CrW0ZsaMGZKk0tLSDn1NTU0qLy/XkCFDOv1VJnyBAAdY1Hbb9He/+50WLlyoZ599lvB2mT766CMFBgZq8ODB7do///xzPf7445L+b9cIXfP09NTGjRs77Vu2bJnKysqUmJjILzFYUFpaqhtuuKHDP5+lpaVavXq1JCk6OroPVmaukSNHKiwsTMXFxcrOztZ3v/tdV99TTz2luro63XPPPbwLrgtU5iqRnZ2tt956S5JcL5l9/vnnXbelpk2b1u5fIFxaenq6cnNz5eXlpRtvvFG//OUvO4yJiIhQSEhIH6zOTPn5+dqyZYu+8Y1vKDAwUN7e3qqqqtLu3bt16tQpTZs2TQ899FBfLxNXqZdffllbtmzR9OnTNWLECA0ePFiffPKJCgsL1dTUpMTERIWGhvb1Mo2zbt06ffOb39SPfvQjFRQUaMyYMTp8+LBKSko0YsQIpaSk9PUSv9QIcFeJt956S7m5ue3a3n77bb399tuuPxPgeubo0aOSvnjWaO3atZ2OCQwMJMBZ8K1vfUvHjx/XX/7yF/3lL3/R2bNn5ePjo4kTJyoqKkr3338//yeOPjNjxgyVlpbq8OHDeuutt/T5559r6NChmjt3rh588EGFhYX19RKNNHLkSO3Zs0e/+MUvVFRUpOLiYvn7++v73/++Vq5ceclnOPEFfsweAADAMHxlDgAAwDAEOAAAAMMQ4AAAAAxDgAMAADAMAQ4AAMAwBDgAAADDEOAAAAAMQ4ADAAAwDAEOAADAMAQ4AAAAwxDgAAAADEOAAwAAMMz/A9F3IeQ4jU/pAAAAAElFTkSuQmCC"/>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- END QUESTION -->
<h3 id="Question-3b">Question 3b<a class="anchor-link" href="#Question-3b">¶</a></h3><p>(Note: You can complete this part with just prerequisite knowledge for Data 100. That being said, Lecture 2 provides additional historical context and definitions for probability sample, sampling bias, and chance error).</p>
<p>One way to use probability samples is to quantify sampling bias and chance error. Put briefly, if we assume that a sample distribution was selected at random from a known population, then we can quantify how likely that sample is to have arisen due to random chance (<strong>chance error</strong>). If the difference in sample and population distributions is too great, then we suspect that the given sample has <strong>bias</strong> in how it was selected from the population.</p>
<p>Let's see this process in a <em>post</em>-analysis of <em>pre</em>-election polling of the 1936 U.S. Presidential Election. Through the U.S. electoral college process (we'll ignore it in this question, but read more <a href="https://en.wikipedia.org/wiki/United_States_Electoral_College">here</a>), Franklin D. Roosevelt won the election by an overwhelming margin. The popular vote results were approximately 61% Roosevelt (Democrat, incumbent), 37% Alf Landon (Republican), and 2% other candidates. For this problem, this is our <strong>population distribution</strong>.</p>
<p>You can use <code>np.random.multinomial</code> to simulate drawing at random with replacement from a categorical distribution. The arguments are the sample size <code>n</code> and an array <code>pvals</code> of the proportions in all the categories. The function simulates <code>n</code> independent random draws from the distribution and returns the observed counts in all the categories. Read the documentation to see how this is described formally; we will use the formal terminology and notation in future assignments after we have discussed them in class.</p>
<p>You will see that the function also takes a third argument <code>size</code>, which for our purposes will be an integer that specifies the number of times to run the entire simulation. All the runs are independent of each other.</p>
<p>Write one line of code that uses <code>np.random.multinomial</code> to run 10 independent simulations of drawing 100 times at random with replacement from a population in which 61% of the people vote for Roosevelt, 37% for Landon, and 2% for other candidatdes. The output should be an array containing the counts in the <strong>Roosevelt</strong> category in the 10 simulations. It will help to recall how to slice <code>NumPy</code> arrays. Assign your answer to the variable <code>sample</code>.</p>
<!--
    BEGIN QUESTION
    name: q3b
    points: 2
-->
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">sample</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">multinomial</span><span class="p">(</span><span class="mi">100</span><span class="p">,</span> <span class="p">[</span><span class="mf">0.61</span><span class="p">,</span> <span class="mf">0.37</span><span class="p">,</span> <span class="mf">0.02</span><span class="p">],</span> <span class="n">size</span><span class="o">=</span><span class="mi">10</span><span class="p">)[:,</span> <span class="mi">0</span><span class="p">]</span>
<span class="n">sample</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[ ]:</div>
<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain" tabindex="0">
<pre>array([54, 65, 58, 53, 66, 58, 61, 56, 60, 57], dtype=int32)</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">grader</span><span class="o">.</span><span class="n">check</span><span class="p">(</span><span class="s2">"q3b"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[ ]:</div>
<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html" tabindex="0">
<p><strong><pre style="display: inline;">q3b</pre></strong> passed! 🙌</p>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- BEGIN QUESTION -->
<h3 id="Question-3c">Question 3c<a class="anchor-link" href="#Question-3c">¶</a></h3><p>Replace the "..." in the code cell below with a Python expression so that the output of the cell is an empirical histogram of 500,000 simulated counts of voters for Roosevelt in 100 draws made at random with replacement from the voting population.</p>
<p>After you have drawn the histogram, you might want to take a moment to recall the conclusion reached by the <em>Literary Digest</em>, a magazine that---while having successfully predicted the outcome of many previous presidential elections---failed to correctly predict the winner of the 1936 presidential election. In their survey of 10 million individuals, they predicted the popular vote as just 43% for Roosevelt and 57% for Landon. Based on our simulation, there was most definitely sampling bias in the <em>Digest</em>'s sampling process.</p>
<!--
    BEGIN QUESTION
    name: q3c
    manual: true
-->
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">simulated_counts</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">multinomial</span><span class="p">(</span><span class="mi">100</span><span class="p">,</span> <span class="p">[</span><span class="mf">0.61</span><span class="p">,</span> <span class="mf">0.37</span><span class="p">,</span> <span class="mf">0.02</span><span class="p">],</span> <span class="n">size</span><span class="o">=</span><span class="mi">500000</span><span class="p">)[:,</span> <span class="mi">0</span><span class="p">]</span>
<span class="n">integer_distribution</span><span class="p">(</span><span class="n">simulated_counts</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnAAAAGwCAYAAAApE1iKAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/TGe4hAAAACXBIWXMAAA9hAAAPYQGoP6dpAABD10lEQVR4nO3df1xUZd7/8feM/AwQvFNnYxNQo1xbLeU2S0pXsbXWVlt+lEHm8qit221dNuo2tshNcdfc3dQsdTcfYVHCmuxyh1mWjia3trcraWK1u+YAipncWiuICQPOfP/oy9xOjDDHQjjwej4ePrRzfeaa63Q5w9tzrnOO5eTJk24BAADANKxdPQAAAAAYQ4ADAAAwGQIcAACAyRDgAAAATIYABwAAYDIEOAAAAJMhwAEAAJgMAQ4AAMBkCHAAAAAmQ4ADAAAwGQKciTU2NqqyslKNjY1dPRRcAObP/JhDc2P+zK23zx8BzuTOnj3b1UPA18D8mR9zaG7Mn7n15vkjwAEAAJgMAQ4AAMBkCHAAAAAmQ4ADAAAwGQIcAACAyRDgAAAATIYABwAAYDIEOAAAAJMhwAEAAJgMAQ4AAMBkCHAAAAAmQ4ADAAAwGQIcAACAyRDgAAAATIYABwAAYDIBXT0AAOhpTilY9c1uv2r7BloUoaZOHhGAnoYABwDfsPpmt65+pcqv2g/vHqyIwE4eEIAeh1OoAAAAJkOAAwAAMJkLCnB79uxRWlqaYmJiFB0drcmTJ6ukpMRQH01NTVq8eLFGjx4tm82mYcOGKSsrS8ePH/dZf+bMGT333HMaP368YmNjFRMTo8TERP3+979XXV3dhewGAACAKRleA1dWVqaUlBSFhIQoOTlZ4eHhKi0tVWZmpo4cOaI5c+Z02IfL5VJ6errsdrvGjBmjadOmyeFwqKCgQNu3b9eWLVvUv39/T31zc7N++MMfqry8XCNGjFB6erok6b//+7+1cOFC/fnPf5bdbtcll1xidHcAAABMx1CAa2lpUVZWlqxWqzZu3KiRI0dKkubOnaukpCTl5eVp+vTpiomJabefwsJC2e12paamavXq1bJYLJKk/Px8ZWdna+HChVq2bJmn/vXXX1d5ebluu+02vfLKK159paen64033tBrr72mu+66y8juAAAAmJKhU6hlZWWqqqpSamqqJ7xJUmRkpLKzs+V0OlVUVNRhPwUFBZKkefPmecKbJGVmZiouLk7r16/XmTNnPNurq6slSTfffHObvqZMmSJJOnHihJFdAYBuITjAqk+ag/z+dUrBXT1kAN2AoSNwO3bskCRNmjSpTVtSUpIkaefOne320djYqPLycsXHx7c5UmexWDRx4kStWbNGe/fu1bhx4yRJ3/nOdyRJmzdv1qxZs7xe89Zbb8liseimm24ysisA0C2cbnbrmrX+3XJE4rYjAL5kKMA5HA5J0tChQ9u02Ww2hYeHq7Kyst0+qqqq5HK5NGTIEJ/trdsdDocnwE2ZMkVTp07V66+/rptuukk33nijpC/XwB0+fFjPPPOMrr32Wr/2obGx0a86M3A6nV6/w1yYP/M73xy6rUa+Wv274a+n2u3qUd9jXYnPoLn1xPkLCQnxu9ZQgKuvr5ck9e3b12d7RESEp6ajPiIjI322t/Z9bj8Wi0Uvv/yyFixYoGeeeUb79+/3tN1111363ve+5/c+HD16VGfPnvW73gxqa2u7egj4Gpg/8/vqHAZGx/v9Wrex/KaW5hbVHK0x9iK0i8+gufWU+evTp895D275YoonMXzxxRe699579d577+mFF17wBLZ33nlHOTk52rJli7Zs2aLY2NgO+4qOju7k0V48TqdTtbW1stlsCgoK6urhwCDmz/zON4cnDByBO2cZsF8CAgP0rUGDjL0IPvEZNLfePn+GApyvo2PnOnXqlKKiovzq43z3bvN1lG/JkiV68803VVhYqB/84Aee7cnJyQoODlZGRoaefvppLV++vMN9MHJ40iyCgoJ65H71Fsyf+X11Di3NRq4PM5bgLBYrf1++YXwGza23zp+hANe69s3hcLRZc1ZbW6uGhgaNHj263T7i4uJktVrPu1audfu56+w2b94sST4vVGjdVlFR4d9OAMAF8PWAerc1QIHR8TphDfAKbS6DoQwAjDIU4BITE7VkyRJt3bpVKSkpXm12u91T057Q0FAlJCRo9+7dOnz4sNeVqG63W9u2bVNYWJhGjRrl2d7c3CxJ+uyzzxQREeHV32effSZJCg7m0noAncfIA+r3Zfi/jgUALoSh+8BNmDBBcXFxKi4u9jriVVdXpyVLligoKEgzZszwbD927JgOHDjQ5nRp661AFixYIPc5K3jXrFmj6upqpaWlKTQ01LN97NixkqSnnnpKLpfLs/3s2bNatGiR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9S394Ac/UEREhD744ANt3bpV4eHheuuttzR8+HBJvXMODT8LFRdXc3OzZs+erREjRmjIkCF69dVX29Q888wz2r9/v37xi1/oySef9Gx/8skntWzZMq1cuVLZ2dkXcdSQvrxR9S9/+csO6woLC2W325WamqrVq1d7ng2cn5+v7OxsLVy4UMuWLevk0eKr/vCHP+jFF1/Ufffdp8WLF7d55mJLS4vnz3wGu5eRI0d6Pau7ldPp1OrVqxUQEKC77rrLs535635qa2u1atUqDRo0SDt27PC68f+KFSs84W7FihWSeucccgSum1u0aJGWLVum7du365lnnlFRUZHXETi3263hw4fr1KlT+uc//6mwsDDPa0+fPq2rrrpK/fv31/vvv99Fe9A7jRgxQpK0f//+Dmtbn/NbUVGhmJgYz3a3261Ro0bp+PHjOnjwoNdFPehcZ86c0Xe+8x1FRUWpvLxcAQHn/7cun0HzKCkpUWZmpqZOnaq1a9dKYv66q927d+vmm29WWlqaVq9e7dXmcDiUkJCgKVOmaN26db12DnkSQzf2/vvv6+mnn9ajjz6qYcOG+axxOBz69NNPNXbsWK+/tJIUFhamsWPHqrq6us16HXQ+p9OptWvX6umnn9bzzz+v8vLyNjWNjY0qLy9XfHy8V3iTJIvFookTJ+r06dPau3fvxRo29OXtik6ePKmpU6fq7NmzKi0t1dKlS5Wfn9/m/pV8Bs2joKBAknTPPfd4tjF/3dPQoUMVFBSk//mf/2lzE/9NmzZJ+vKuGFLvnUNOoXZTTU1NnlOnres0fGm9ifH5bp48ZMgQ2e12ORwOXX755Z0yVvhWW1urBx980Gvb6NGj9cILL2jw4MGSpKqqKrlcrnbnT/pynseNG9e5A4ZH67/U+/Tpo8TERB08eNDTZrVa9dOf/lQLFy6UxGfQLA4fPqzt27fr29/+tiZPnuzZzvx1T//2b/+mX/3qV8rNzdV1113ntQaurKxM9913n+6//35JvXcOCXDd1G9+8xs5HA698847bdbenKv1Xybnrg84V0ePPkPnyMjI0A033KDhw4crLCxMBw8e1IoVK7Ru3TpNmzZN7777riIiIpi/burEiROSvlxrc80112jr1q268sorVVFRoV/84hd67rnnNHjwYN17773MoUmsXbtWLpdLd911l9d3KvPXfT344IOKjo7Wz3/+c+Xn53u233DDDUpNTfUsbeitc8gp1G7ob3/7m5599lk98sgjnitsYC45OTmaMGGCBgwYoEsuuUQjR47UH//4R915552qqanRSy+91NVDRDtcLpckKSgoSGvXrtXo0aMVHh6ucePG6cUXX5TVatVzzz3XxaOEv1wul9auXSuLxaK77767q4cDPy1evFj333+/srOz9eGHH+rIkSN688031djYqNtuu01vvPFGVw+xSxHgupmWlhbNnj1bV199tR566KEO61v/ZfHVJ120av0XR2sdulZmZqYkadeuXZKYv+6q9f/3tddeq8suu8yrbfjw4YqLi1NVVZVOnjzJHJrAO++8oyNHjmj8+PGKi4vzamP+uqd33nlHixYt0k9+8hM99NBD+va3v63w8HDdcMMN+tOf/qTAwEDl5uZK6r1zyCnUbqahocFzPn/AgAE+a26++WZJ0iuvvOK5uOGrC6tbtW4fOnToNz1UXIBLL71UkvTFF19IkuLi4mS1Wpm/biY+Pl7S+U/JtG5vbGz0zA1z2H35unihFfPXPW3evFmSdNNNN7Vps9lsio+PV0VFhRoaGnrtHBLgupng4GDNnDnTZ9u7774rh8OhW2+9Vf3791dMTIyGDh2qyy67TLt27dLp06fbXD69a9cuxcbG9qiFm2bWeiVq6xWnoaGhSkhI0O7du3X48OE2txHZtm2bwsLCNGrUqC4Zb2/V+kPjwIEDbdqam5tVWVmpsLAw9e/fXzabjc9gN/b555/rjTfeUL9+/XTbbbe1aec7tHtyOp2S/m896ld99tlnslqtCgwM7LVzyCnUbiY0NFTPPvusz1/XXXedJCk7O1vPPvusRo4cKYvFopkzZ6qhoUG/+93vvPr63e9+p4aGBs2aNasrdqXXOnDggOcI21e3t95gMjU11bO9dX4WLFggt/v/bsu4Zs0aVVdXKy0tjXvAXWSDBw/WpEmTVFlZ6Tl602rp0qWqq6vT1KlTFRAQwGewm/vTn/4kp9OpO+64Q8HBwW3amb/uqfXxmCtXrmxzajQ/P1+ffPKJrrvuOgUHB/faOeRGviYye/bsNjfylb78F8aUKVP0wQcfaNKkSbrmmmu0b98+zyNENm7cSAC4iBYtWqSVK1dq3LhxGjRokC655BIdPHhQmzdvVnNzs7KzszVv3jxPva9HaVVWVmrDhg2KiYmR3W7nUVpdoKqqSt///vd1/PhxTZkyxXPKpqysTIMGDdKWLVtks9kk8RnszsaNG6ePPvpIO3fu1NVXX+2zhvnrfs6ePasf/vCHevfddzVgwADdeuutioyM1L59+1RWVqbQ0FC9/vrrSkhIkNQ755AAZyLnC3DSl4s3n3rqKW3YsEG1tbWy2Wy6/fbb9eijjyoiIqKLRtw77dixQy+88IIqKip0/PhxffHFF7r00kuVkJCg++67T5MmTWrzmqamJi1dulTr1q3TJ598on79+mnKlCnKzc3VwIEDu2AvIElHjhzRb37zG9ntdn3++eey2Wy69dZbNXfu3DZrVPkMdj/vvfeekpKSlJCQILvd3m4t89f9NDU1aeXKlSopKdHBgwfldDo1cOBA3XjjjXr44Yd11VVXedX3tjkkwAEAAJgMa+AAAABMhgAHAABgMgQ4AAAAkyHAAQAAmAwBDgAAwGQIcAAAACZDgAMAADAZAhwAAIDJEOAAAABMhgAHAABgMgQ4AAAAkyHAAQAAmMz/A6vJj6ORtSYRAAAAAElFTkSuQmCC"/>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- END QUESTION -->
<!-- BEGIN QUESTION -->
<h3 id="Question-3d">Question 3d<a class="anchor-link" href="#Question-3d">¶</a></h3><p>As you know, the count of Roosevelt voters in a sample of 100 people drawn at random from the eligible population is expected to be 61. Just by looking at the histogram in Part <strong>c</strong>, and <strong>no other calculation</strong>, pick the correct option and <strong>explain your choice</strong>. You might want to refer to the <a href="https://www.inferentialthinking.com/chapters/14/3/SD_and_the_Normal_Curve.html">Data 8 textbook</a> again.</p>
<p>The SD of the distribution of the number of Roosevelt voters in a random sample of 100 people drawn from the eligible population is closest to</p>
<p>(i) 1.9</p>
<p>(ii) 4.9</p>
<p>(iii) 10.9</p>
<p>(iv) 15.9</p>
<!--
    BEGIN QUESTION
    name: q3d
    manual: true
-->
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><em>Type your answer here.</em></p>
<p>假设上面图片是对的，用3σ原则粗略估计5左右。</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- END QUESTION -->
<!-- BEGIN QUESTION -->
<h3 id="Question-3e">Question 3e<a class="anchor-link" href="#Question-3e">¶</a></h3><p>The <em>normal curve with mean $\mu$ and SD $\sigma$</em> is defined by</p>
<p>$$
f(x) ~ = ~ \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{1}{2}\frac{(x-\mu)^2}{\sigma^2}}, ~~~ -\infty &lt; x &lt; \infty
$$</p>
<p>Redraw your histogram from Part <strong>c</strong> and overlay the normal curve with $\mu = 61$ and $\sigma$ equal to the choice you made in Part <strong>d</strong>. You just have to call <code>plt.plot</code> after <code>integer_distribution</code>. Use <code>np.e</code> for $e$. For the curve, use 2 as the line width, and any color that is easy to see over the blue histogram. It's fine to just let Python use its default color.</p>
<p>Now you can see why centering the histogram bars over the integers was a good idea. The normal curve peaks at 26, which is the center of the corresponding bar.</p>
<!--
    BEGIN QUESTION
    name: q3e
    manual: true
-->
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">mu</span> <span class="o">=</span> <span class="mi">61</span>
<span class="n">sigma</span> <span class="o">=</span> <span class="mf">4.9</span>
<span class="n">x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mi">40</span><span class="p">,</span> <span class="mi">80</span><span class="p">,</span> <span class="mi">200</span><span class="p">)</span>
<span class="n">f_x</span> <span class="o">=</span> <span class="mi">1</span><span class="o">/</span><span class="p">(</span><span class="n">sigma</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="mi">2</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">pi</span><span class="p">))</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="o">-</span><span class="p">(</span><span class="n">x</span> <span class="o">-</span> <span class="n">mu</span><span class="p">)</span><span class="o">**</span><span class="mi">2</span> <span class="o">/</span> <span class="p">(</span><span class="mi">2</span> <span class="o">*</span> <span class="n">sigma</span><span class="o">**</span><span class="mi">2</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">f_x</span><span class="p">)</span>
<span class="n">integer_distribution</span><span class="p">(</span><span class="n">simulated_counts</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnAAAAGwCAYAAAApE1iKAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/TGe4hAAAACXBIWXMAAA9hAAAPYQGoP6dpAABwHklEQVR4nO3de1xUdf4/8NcMMNwGwURRSkDNNCtLWbMkNW9Z2WoqZGlm/Gq3dduipf0WW+bmZZe1i5qm7q4blhZk0rJplql4ISzNO5rlhYuAyMh1uM19zu8PF3I8Z4SDMNfX8/HgoX3OZz68p+PMvOdzVdTW1gogIiIiIrehdHYARERERCQPEzgiIiIiN8MEjoiIiMjNMIEjIiIicjNM4IiIiIjcDBM4IiIiIjfDBI6IiIjIzTCBIyIiInIzTOCIiIiI3AwTOCIiIiI3wwTOjen1ehQUFECv1zs7FGoH3j/3x3vo3nj/3Ju33z8mcG7OYrE4OwS6Drx/7o/30L3x/rk3b75/TOCIiIiI3AwTOCIiIiI3wwSOiIiIyM0wgSMiIiJyM0zgiIiIiNwMEzgiIiIiN8MEjoiIiMjNMIEjIiIicjNM4IiIiIjcDBM4IiIiIjfDBI6IiIjIzTCBIyIiInIzTOCIiIiI3AwTOCIiIiI3wwSOiIiIyM34OjsAIiJPE2TUQalrhCAI+KlewHGtAKP18jfmcH9gRDclQv0UAABrYDCaVIHODZiI3A4TOCKiDpZfUY93P/wGW7sNwUX/rqLrPoIF92rP4klNLua8+nuACRwRycQEjoiog1ToLPj7sXp8eNoES+RYu/UsCh/khg1EbthALN1pwvy7dZgUFQCFQuHAaInInXEOHBFRB/heY0DcF5fwwc+NsAhtf9yZBgFP7qrGc9/WQGeW8UAi8mrtSuCOHDmChIQEREVFITIyEuPHj0dWVpasNgwGA5YsWYKhQ4ciIiICAwcORFJSEioqKiTr63Q6vP/++xg1ahSio6MRFRWFuLg4vPPOO9Bqte15GkREHeKjs3r8+utKXNJZ293GZ/k6TPq6AmWNlg6MjIg8lewELicnBxMnTsT+/fsxdepUJCYmQqPRIDExEStXrmxTG1arFTNnzkRqaiq6deuGuXPnYtiwYVi/fj0mTJiAyspKm/omkwm//vWvMW/ePAiCgJkzZ2LWrFlQKBRYvHgxHnzwQTQ1Ncl9KkRE1+2jUl+8eqgRHdF5dqTShAe2VqC0wXz9jRGRR5M1B85sNiMpKQlKpRJbt27F4MGDAQCvvPIKxo0bh0WLFmHKlCmIioq6Zjvp6enIzs5GfHw81q5d2zLvIy0tDcnJyVi8eDGWL1/eUv/LL7/EoUOH8Mgjj+Djjz+2aWvmzJn46quv8MUXX+CJJ56Q83SIiK7LR2f1eL9IZfe6n9WMJzW5mFR1FH11GuiUKuwLHYD1PUfhpLq35GNKGy2Ytr0KXz8cjm4BPp0VOhG5OVk9cDk5OSgsLER8fHxL8gYAoaGhSE5OhtFoREZGRqvtrF+/HgAwf/58m0m7iYmJiImJwaZNm6DT6VrKi4qKAAATJkwQtTVx4kQAEPXaERF1pv8W6pByqNHu9QerjuHkD/+HtafX4tHKQxjcWILh9flILv0Khw/9Gf/++Z8INUs//ozWjBnbNEBVBdTaSpufIKNO8jFE5F1kJXC5ubkAgLFjxaurxo0bBwDYt2/fNdvQ6/U4dOgQ+vfvL+qpUygUGDNmDBobG3H06NGW8ltvvRUAsGPHDlF733zzDRQKBUaOHCnnqRARtdvPtSb8PrcG9kZNXx/og80n3kE//SXJ6z4Q8HR5Dr47/BfcopZu41CNgN+v3QXhxXjgih+lzn7SSETeQ9YQan5+PgCgX79+omsRERFQq9UoKCi4ZhuFhYWwWq3o27ev5PXm8vz8fIwYMQLA5V62SZMm4csvv8TIkSNx3333AQC+/fZbFBcX47333sNdd93Vpueg1+vbVM8dGI1Gmz/JvfD+uadGs4CnsrVosjPp7a1hwfjjTSYo7KZ3vxigu4jvRvlg3GcFOK6OFl1Pj7gP99ecwv8r39tSJgiCR72PORNfg+7NE+9fQEBAm+vKSuDq6uoAAF26dJG8HhIS0lKntTZCQ0Mlrze3fWU7CoUCGzZswMKFC/Hee+/hxIkTLdeeeOIJ3H///W1+DmVlZbBYPGuVl0ajcXYIdB14/9zLgjMqnKmTfut8PtqIMf5NMJnVsD8zzlZXPwW25i3B6CHzkR/YU3Q9qf8c3F2fj9sbSwEAJrMZJSUl7Q2fJPA16N485f75+PjY7dyS4hYb+TY1NeGZZ57B4cOH8cEHH7QkbHv27EFKSgp27tyJnTt3Ijpa/A32apGRkZ0creMYjUZoNBpERERApWrrxwW5Ct4/97P9ghFfXqqXvPbUzf54Y1g3AICfXrqOJAXQ06jFV8eXYHjsYtT6Bdtc1vn44+mBc7H/yBvwFazw8/VF797SCyBIHr4G3Zu33z9ZCZxU79iV6uvrERYW1qY27O3dJtXLt3TpUnz99ddIT0/Hww8/3FI+bdo0+Pv7Y9asWXj33XexYsWKVp+DnO5Jd6FSqTzyeXkL3j/3UG+y4rXDtZLXBnf1wdsjusHf5/KiLIWhoc3tKgAIAPrpL+GD0//E9NuTRXWOhcRgxY0PIrn0KygUCv576WB8Dbo3b71/shYxNM99a54LdyWNRoOGhoZWu/9iYmKgVCrtzpVrLr9ynl3z4gWphQrNZXl5eW14BkRE7fPWwWqUSmyy28UX+OxXSnRrqGpZKaoQ2rcp3JTKw3ihdJvktTf7TMd5//B2tUtEnkdWAhcXFwcA2LVrl+hadna2TR17AgMDERsbi7Nnz6K4uNjmmiAI2L17N4KDgzFkyJCWcpPJBACoqqoStddc5u/vL+OZEBG13bFKI1adNkhee/fkv3Dz64/ZrBRVCO0/keHv+Rm4tfGCqLzJJwAv3PJ0u9slIs8iK4EbPXo0YmJikJmZadPjpdVqsXTpUqhUKjz++OMt5eXl5Thz5oxouHTOnDkAgIULF0K44pvqunXrUFRUhISEBAQGBraUDx8+HADw97//HVbrL2+MFosFqampAKR754iIrpcgCHj9oBZSKdnomlN4+ooVoh3BXzBjzZl/S177qtsQbNe0PzkkIs8haw6cr68vVqxYgenTp2PSpEmYNm0a1Go1Nm/ejJKSEixatMhmIcGCBQuQkZGBVatWYdasWS3lM2fORFZWFjIzM3H+/HnExcWhoKAAW7ZsQXR0NObNm2fze5OTk/HVV1/h008/xfHjx1uStZycHPz888/o168f/vCHP1zP/wciIknflOqxr1y8TYG/1Yg1Zz6AQuIx1+s+7Rk8W7YL/44U77n52kkzdvUXoFR0xm8mInch+yzUUaNGYdu2bRg+fDiysrKQlpaGHj16IC0tDS+88ELbfqlSifT0dKSkpKCyshKrV6/GgQMHMHv2bOzYsQPh4bbzPHr37o09e/bgN7/5DQwGAz788EN89NFHsFgsePHFF5Gdnd3q4gkiIrnMVgF/OSi9aOuV4i24RVfeab87tSAD3Y3ixV7HtAI2FfA0BiJvp6itre2AI5jJGfR6PUpKStC7d2+vXIHj7nj/XN9HpxuR9F2tqLynoQanD7yMYKv0vDjl8o2wvjSjTb/jWnVX3TgBSf2fFpX3Vvvg4NQIBPiyF+568DXo3rz9/snugSMi8gZGi4C3j0vv5/Zm0ed2k7eO9JuyXegn0ctX0mDBR2d4pBaRN2MCR0Qk4dP8JsltQwY1lnb4wgV7VIIFfy3YKHltxYkGGC0cQCHyVkzgiIiuYrYKWJYn3fu2uOAz+F7HNiFyTa/4Ab+qE++9eaHJgk/zmxwWBxG5FiZwRERX+bxQh8J6ce/bnQ3n8euqww6NRQFg3vksyWvL8uphtrIXjsgbMYEjIrqCVRCw1M7ctz+f/2+nbBvSmklVR3Fnw3lReWG9BVmFXJFK5I2YwBERXWFnqQGntWZR+cAQBaZWHHRCRJd74VLOfyF5beXJBpsN0YnIOzCBIyK6wppT0gfRpwzwgQ+clyhNq/gBA5rKROV51SZ8pxFvNExEno0JHBHR//xUY8LuMvH2IDcG+WDGTc59u/SBgD8Vfyl57R92kk4i8lxM4IiI/uefdhKh39waDD+l8zfNfeLSdwhXicu3Futxvl487EtEnosJHBERgGq99LYcgT4KzBkQ7ISIxAKsJvy2j4+o3CoAa3/ixr5E3oQJHBERgE/ONkEv3jkET9wchK7+rvNW+bu+PpA6QWvD2UY0mR23Px0ROZfrvCsRETmJIAj40M7RVM8Nco3et2aRgQpM7RMoKtcaBXxRpHdCRETkDEzgiMjr5ZYbkV8n7n4b3csfA8L8nBCRfQo/FV6Kkp7vtuFULdTaypafICP3iCPyVEzgiMjr2TsYPtFF5r5dSWHQ4e43H8NgiY19v6sS8GNKEvBiPPBiPJQ6zosj8lRM4IjIawUZddBfqsDmInFPVXd/4LGwhpbeLIULbZarAPBs2W7Jax9EjnFsMETkFEzgiMhrKXWN2PBeGowSc//nnN0C1UsJLb1ZCgceYN8WMy/tQ6BFvGfdhoiR0Clda9iXiDoeEzgi8lqCIOCjnqMlrz17UbqHy1WEmZuQUHFAVF7jp8bm8FgnREREjsQEjoi81jGtgJPq3qLyMTUncbNO44SI5Hm2bJdk+YaIkQ6OhIgcjQkcEXmtDeelh0WfLs9xcCTtc2/dWcnzUbffMBjlqlAnREREjsIEjoi8kskqIKNUvHWI2qzDoxWHnBCRfAoAT5Z/Kyq3KpTI6DHC8QERkcMwgSMir7SzVI8K8RoATK/4AcFWiQsuapZmn2T5xz05jErkyZjAEZFXkjr3FACe1OQ6OJLrE2Wowv01P4rKj6ujcbzWtVbOElHHYQJHRF5Ha7RiW4n42KkofSVG1/7khIiuz1MSw6gA8EkJEzgiT8UEjoi8ztbzOhgkDq6fpcmFEq6zYW9bTa08iCCLOCHdVGqB1YU2ICaijsMEjoi8zn8Kpc8IfULznYMj6RghFj0mVx4RlZfogB8uGZ0QERF1NiZwRORVqvQW7C4TL1IY3HAeg5ouOCGijjHjknTy+bmdZJWI3BsTOCLyKl8U6WGRGFV87NJ+xwfTgR6oPoEwk/jw+v8W6mC2chiVyNMwgSMir/J5ofTq08cufe/gSDqWv2DG1MqDovIKvRW55e6zLQoRtQ0TOCLyGhebLPiuXDwn7O66c+irr3BCRB3LXhL6eQGHUYk8DRM4IvIaW4p0kmtMZ7h571uzMbWn0MOoFZV/WayDicOoRB6lXQnckSNHkJCQgKioKERGRmL8+PHIysqS1YbBYMCSJUswdOhQREREYODAgUhKSkJFhfhb8Ny5cxEWFnbNn7feeqs9T4WIvMjm89I9UfGXDjg4ks7hK1gxveIHUXmNQcA+DqMSeRRfuQ/IycnB9OnTERAQgGnTpkGtVmPz5s1ITExEaWkpXnjhhVbbsFqtmDlzJrKzszFs2DBMnjwZ+fn5WL9+Pfbu3YudO3ciPDy8pf6kSZMQFRUl2db777+PxsZGjBs3Tu5TISIvUqGz4DuNePj0Hu0Z3GiscUJEnWNqxQ9Yc+MEUfnmIj3ujwxwQkRE1BlkJXBmsxlJSUlQKpXYunUrBg8eDAB45ZVXMG7cOCxatAhTpkyxm2w1S09PR3Z2NuLj47F27VooFAoAQFpaGpKTk7F48WIsX768pf4jjzyCRx55RNTOsWPHsGTJEgwaNAixsbFyngoReZmtxXpIjSJOqxBP/Hdno7Q/o5upHlV+ITblXxbr8PY9ofBRKpwUGRF1JFlDqDk5OSgsLER8fHxL8gYAoaGhSE5OhtFoREZGRqvtrF+/HgAwf/78luQNABITExETE4NNmzZBp2t90u2GDRsAALNnz5bzNIjIC20ukn5PmVYpHnJ0Z76CFVMqD4nKL+msOMBNfYk8hqwELjf38iHPY8eOFV1rHsLct2/fNdvQ6/U4dOgQ+vfvL+qpUygUGDNmDBobG3H06NFrtqPT6bBp0yb4+/vj8ccfl/M0iMjL1BisyLkongMWG6ZAjL7SCRF1rql2ehXtzQEkIvcjawg1Pz8fANCvXz/RtYiICKjVahQUFFyzjcLCQlitVvTt21fyenN5fn4+RowYYbedL774AnV1dZg+fTq6du3a1qcAvV58XqC7MhqNNn+Se+H9c5zNBXqYJYZPp0a2/TusnDWcnVW3rfXH1ZxEqLkRWt9gm/ItRTr8ZbC/zciHN+Nr0L154v0LCGj7PFVZCVxdXR0AoEuXLpLXQ0JCWuq01kZoaKjk9ea2W2unefj0qaeeuma9q5WVlcFikTjF2o1pNBpnh0DXgfev8/33nApSb3e/jrC2vRE3yuBUggWPVB7FJz3vsym/0GTFzp8vYKCaW4pcia9B9+Yp98/Hx8du55YU2atQXUFBQQG+++47REdHY9SoUbIeGxkZ2UlROZ7RaIRGo0FERARUKpWzwyGZeP8cQ28RcGB/taj8li4+GBTm0/aG5HRadVZdGfWnVf4gSuAA4Lj5BkzoHSTzl3omvgbdm7ffP1kJXGu9Y/X19QgLC2tTG1qteLPJK9u218sHAB9//DEEQcCTTz4peyhATveku1CpVB75vLwF71/nyi3Vo8ksLp8UHQiFwtTmdhRoe2dZZ9WVU3989Un4KwHDVZ2MOy+aMW8Y/71dia9B9+at90/WIobmuW/Nc+GupNFo0NDQ0Gr3X0xMDJRKpd25cs3lUvPsAMBisSAjIwM+Pj6YNWuWnPCJyAttK5Ge9/pgb89+ww+2GjC2u/gt/liVCWWNnjWNhMgbyUrg4uLiAAC7du0SXcvOzrapY09gYCBiY2Nx9uxZFBcX21wTBAG7d+9GcHAwhgwZIvn47du34+LFixg/frxHDYcSUccTBEEygQsPUOJX3T1/yOWRXtJv8d/YSWqJyH3ISuBGjx6NmJgYZGZmIi8vr6Vcq9Vi6dKlUKlUNlt6lJeX48yZM6Lh0jlz5gAAFi5cCEH4ZTBg3bp1KCoqQkJCAgIDAyVjaF688OSTT8oJnYi80IlqE0olepseuCnAKza0ndRT+i1+Wwm3EyFyd7LmwPn6+mLFihWYPn06Jk2aZHOUVklJCRYtWoTo6OiW+gsWLEBGRgZWrVplM9w5c+ZMZGVlITMzE+fPn0dcXBwKCgqwZcsWREdHY968eZK//9KlS9i+fTt69OiBhx56qJ1PmYi8hbcOnzbrHeqPIaEmHNXazprbW2aAsqoCQb6/JLHWwGA0qaS/OBOR65F9mP2oUaOwbds2DB8+HFlZWUhLS0OPHj2QlpbWpnNQAUCpVCI9PR0pKSmorKzE6tWrceDAAcyePRs7duywOQf1ShkZGTCbzXjiiSfg6+uWC2iJyIGkEjiVEhh7o78TonE8hUGHR45nisr1ViD7r6nAi/EtP0pdoxMiJKL2alcWFBsbi8xM8ZvC1dasWYM1a9ZIXvP390dKSgpSUlLa/HuTkpKQlJTU5vpE5L3Kmyw4UileZTqqlz/UfrK/u7qtRyqPYFHMdFH5l92G4tdVR5wQERF1BO95FyMir2Jvor63DJ82G9pQhEiDeB+8L8OHwCp7EzoichVM4IjII33NBA7A5X3jJlWJz5bWqMJwKKSP4wMiog7BBI6IPE6T2Yq9ZeLD6++4wQ83qb1v/qxUAgdcHkYlIvfEBI6IPM7eMgN0FvF5Bd7W+9ZsXM1JBFrECe3WcCZwRO6KCRwReRx724c8HOWdCVyg1YTxNSdF5cfV0Sj27+aEiIjoejGBIyKPIgiC5AKGnoFK3NnNzwkRuYZH7Kw45TAqkXtiAkdEHuVkjRnlOquofGLvACgV3rvq8uGqY5Ll27rd6dhAiKhDeN9sXiLyWEFGHXLztZLXJt9ghFpbaVOmEASIZ8p5pl7GWvyqLh+HuvSzKd8TdisMCl94x9bGRJ6DCRwReQylrhHffJsHdL3NptzXasbYt58BLLZDq4rlG70mgQOAidXHRQlck08AckMHYJyTYiKi9uEQKhF5jAazgH2hA0TlI+rOIMQivbDBmzxQfUKyfPsNgx0cCRFdLyZwROQxdldYYVKKBxYmVuc5IRrXM7z+HELN4jNPd9xwhxOiIaLrwQSOiDzGdo148QIAPMAEDgDgK1gxruZHUXmeOhplOm8aTCZyf0zgiMhjSCVwEcZa3NlQ7IRoXJO9ZHbHJenkl4hcExM4IvIIBXVm5ItHBzGh+gSUXrVU4drsJXDf2Om9JCLXxASOiDzCzlLpRQqc/2YrylCFWxsviMp3XrLCYmWiS+QumMARkUfIlji8XiFYMb5GeuWlN3ug+riorNoIHKsyOSEaImoPJnBE5PYMFgHfXhQncLH1hehuqndCRK7N3jDqzgvcaoXIXTCBIyK3t19jRJNZPPzH1afSRml/RoDFKCrfdUGcBBORa2ICR0RuL9tOz9EDNUzgpARaTRil/UlUfrDCiFoDFzMQuQMmcETk9qQSuFBzI+6pO+eEaNyDVO+kVQD2SgxFE5HrYQJHRG7tYpMFP9aYReVja36Er8DeJHvsrc6115tJRK6FCRwRuTV7CQe3D7m2gU1l6K2vFJVnlxogCNxOhMjVMYEjIre2V2L7EIALGFqjgHSSe6HJgjNacY8mEbkWJnBE5LYEQZCcszWgqQxRhionRORe7CW59pJiInIdTOCIyG39VGvGJZ14ntu4mpNOiMb93F97CgqJeYJ7uJCByOUxgSMit7XHTk8RE7i2ucHciNj6QlF57kUDzDxWi8ilMYEjIre1t0y8gEEpWDG6VrzHGUkbU/ujqKzOJPBYLSIXxwSOiNySySpgX7n4NIFf1RcgzNzkhIjck73eSnu9m0TkGpjAEZFbOlxhRIPE8VljOXwqS5z2DPyt4kRYqneTiFwHEzgickv257+JhwTJvkCrCSO0Z0TlBy4Z0WTmRshErqpdCdyRI0eQkJCAqKgoREZGYvz48cjKypLVhsFgwJIlSzB06FBERERg4MCBSEpKQkVFhd3HGI1GvP/++7j//vtx00034aabbsK9996LP/3pT+15GkTkxqS2Dwn0Ae6tO+uEaNybVNJrtAL7NeKeOSJyDbITuJycHEycOBH79+/H1KlTkZiYCI1Gg8TERKxcubJNbVitVsycOROpqano1q0b5s6di2HDhmH9+vWYMGECKivFu4PX1tbioYcewrx58+Dv74+nn34aTz/9NG6++Wb85z//kfs0iMiNNZisOHhJnFzEdVMgwMrJ93LZG3bmfnBErstXTmWz2YykpCQolUps3boVgwcPBgC88sorGDduHBYtWoQpU6YgKirqmu2kp6cjOzsb8fHxWLt2LRQKBQAgLS0NycnJWLx4MZYvX27zmOeffx5HjhzB2rVrkZCQIIqLiLzHd+VGSEx/w7genBXSHrH1hQj1A7RX5b7cD47Idcl6t8vJyUFhYSHi4+NbkjcACA0NRXJyMoxGIzIyMlptZ/369QCA+fPntyRvAJCYmIiYmBhs2rQJOp2upfzgwYPYunUrHnvsMVHyBgC+vrLyUCJyc1LDpwAwrjsTuPbwgYD7w8X/7/KqTKjWW5wQERG1Rta7XW5uLgBg7Nixomvjxo0DAOzbt++abej1ehw6dAj9+/cX9dQpFAqMGTMGjY2NOHr0aEt58xDpo48+iqqqKmzYsAFLly7Fxo0bUV1dLecpEJEH2COxQrKrvwJ3hSkkalNbSPVeCgC+ldiqhYicT1bXVX5+PgCgX79+omsRERFQq9UoKCi4ZhuFhYWwWq3o27ev5PXm8vz8fIwYMQIAcOzYsZay5557DnV1dS311Wo1VqxYgWnTprXpOej1nrM03mg02vxJ7oX3r30qdFb8WCOeNhHXww9y0zc5Zw24Qt3ObHtMd+n/e9kljZjY0zMTY74G3Zsn3r+AgIA215WVwDUnTl26dJG8HhISYpNcXauN0NBQyevNbV/ZTvOihr/85S9ISEhASkoKwsLCsH37dvzpT3/Cc889h1tuuQW33357q8+hrKwMFotnDQloNBpnh0DXgfdPnm8qfAD4i8rvUDXAZA6ASk5jrpCVuUgG1zfAjB4qKy4ZbXvidpfqUNKzRsYvdT98Dbo3T7l/Pj4+dju3pLjF5DGr9fJeRIMGDcKaNWta5s099thjqK+vx8svv4x//vOfbVoFGxkZ2amxOpLRaIRGo0FERARUKlkfW+QCeP/a51RZAwDxHLjJt/aAn6/MExjkdCy5Qt1ObFvl54f7b1Ths0Lb/7cleiWErpGIUvvI+MXuga9B9+bt909WAifVO3al+vp6hIWFtakNrVYreV2ql6/57w8++KDNogcAeOihh/Dyyy/bzJm7Fjndk+5CpVJ55PPyFrx/bScIAr69VCsq7632wcDwICjqdOIHXYMCbe+kcoW6ndm2UuWPh7vp8Jn4bHscu9CAoTG/JHDWwGA0qQLb2LLr42vQvXnr/ZO1iKF57lvzXLgraTQaNDQ0tNr9FxMTA6VSaXeuXHP5lfPs+vfvD0B62LW5zJPmthGRtPMNFpQ0iKdA3N/LX/TljuRRGHQYu+I5yWu7vs4BXoxv+VHqGh0cHRFdTVYCFxcXBwDYtWuX6Fp2drZNHXsCAwMRGxuLs2fPori42OaaIAjYvXs3goODMWTIkJbykSNHAgBOnz4taq+5rLW954jI/eWWS28fMqqXeE4cyRdprMWtjRdE5XvCbpU9VY+IOpesBG706NGIiYlBZmYm8vLyWsq1Wi2WLl0KlUqFxx9/vKW8vLwcZ86cEQ2XzpkzBwCwcOFCCMIvbwvr1q1DUVEREhISEBj4S/f8lClT0K1bN2zatAk//vjLkS9GoxGpqakALm8xQkSeLdfO/m/3MYHrMGNqxcdqlft3xZnAXk6IhojskTUHztfXFytWrMD06dMxadIkTJs2DWq1Gps3b0ZJSQkWLVqE6OjolvoLFixARkYGVq1ahVmzZrWUz5w5E1lZWcjMzMT58+cRFxeHgoICbNmyBdHR0Zg3b57N7+3SpQvee+89zJkzBxMmTMDkyZMRFhaGvXv34qeffsIDDzxg0z4ReR5BEJArsSfZzV180SvI8ybYO8vo2p+w+sYHROV7ug7CAN1FJ0RERFJkb1s+atQobNu2DcOHD0dWVhbS0tLQo0cPpKWl4YUXXmjbL1UqkZ6ejpSUFFRWVmL16tU4cOAAZs+ejR07diA8PFz0mEceeQRbt27FiBEj8PXXXyMtLQ3A5SQxPT0dPj58AyfyZOcbLChtFM9/u6+n960+60yja09Jlu8JG+TgSIjoWtq1jUhsbCwyMzNbrbdmzRqsWbNG8pq/vz9SUlKQkpLS5t97zz33tOn3EpHn+ZbDpw4RbmrAHQ3FOKG2nVe893/z4LhUhMg18OBAInIL9hYwxPVkAtfR7pfohbukCsWpoBudEA0RSWECR0QuTxAE7OP8N4eRSuAAYC+HUYlcBhM4InJ5nP/mWCNrf4ZCsIrK93RlAkfkKpjAEZHL4/w3x7rB3Ig7G4pF5XvDboWVs+CIXAITOCJyaUFGHQ6USB/f90BQI9TaypYfhcDtZjuK1GrUKr8QnAy+yQnRENHV3OIweyLyXoqmBuw9cwkIsN1e6JamMkS++n+2dZdv5IkBHeT+2lN4r/fDovI9YYMw2AnxEJEt9sARkUsrbAJKAsR7Q95f+5MTovEeI7WnoZSYB7c37FYnRENEV2MCR0QubW+FOIkA7G84Sx0jzNyEIQ1FovKcsFth5VA1kdMxgSMil7a3UjqBG1X7s4Mj8T6ja8RJco2fGse1TOCInI0JHBG5LEEQJHvgBjSVoZex1vEBeRl7+8HtsdMrSkSOwwSOiFxWUb0FJTpx+WjOf3OIkdrT8BHE++/tqWAPHJGzMYEjIpf1rZ3jszj/zTFCLHrE1heKyr+ttMJiZRJH5ExM4IjIZdk7/5Tz3xznfol5cHVmIK/a5IRoiKgZEzgickmCIGDfRfH5p5z/5lj2ejvtnY5BRI7BBI6IXFJRvQUXmsTzrzj/zbHi6s7A12oWlTOBI3IuJnBE5JI4/801qC0GDKsvEJV/rzHCxHlwRE7DBI6IXJK9+W/sgXM8qe1EGswCjldxHhyRszCBIyKXY2/+28DGC+hp1DohIu8mtZAB4DAqkTMxgSMil8P5b67l3rqz8OM8OCKXwgSOiFyO/flvTOCcIchqxPC6c6Ly/ZeMMFo4D47IGZjAEZHLsbv/m5YJnLNIzYNrMgs4Uike6iaizscEjohciiAIyJUYmuP8N+eydy4qh1GJnIMJHBG5lMJ6C8qaxIelc/jUue6pOwd/q7i3bZ+GPXBEzsAEjohcCrcPcU0BVhPu0Yrnwf3AeXBETsEEjohcitTwKcD5b65AKoluMgs4ynlwRA7HBI6IXIYgCJI9cJz/5hrsJdG55UzgiByNCRwRuQx789/sTaAnx7I3D87esDcRdR4mcETkMjj/zbUFWE2S+8Ed4Dw4IodjAkdELsPu/DcmcC6D8+CIXAMTOCJyCfbmv90aokCEqc4JEZEUe72hnAdH5FjtSuCOHDmChIQEREVFITIyEuPHj0dWVpasNgwGA5YsWYKhQ4ciIiICAwcORFJSEioqKkR1z58/j7CwMLs/qamp7XkaRORC7O7/Fq5wQjRkzz115+Av8cnBeXBEjuUr9wE5OTmYPn06AgICMG3aNKjVamzevBmJiYkoLS3FCy+80GobVqsVM2fORHZ2NoYNG4bJkycjPz8f69evx969e7Fz506Eh4eLHnf77bdj0qRJovL77rtP7tMgIhdjd/5bdw4UuJIAqwnDb1Agp9J2ztuBS0aYrAL8lEy4iRxBVgJnNpuRlJQEpVKJrVu3YvDgwQCAV155BePGjcOiRYswZcoUREVFXbOd9PR0ZGdnIz4+HmvXroVCcfkFn5aWhuTkZCxevBjLly8XPe6OO+7An//8ZzkhE5GbsHck06hwJnCuZlS4EjmVFpuy5nlwd/fwd1JURN5F1jtjTk4OCgsLER8f35K8AUBoaCiSk5NhNBqRkZHRajvr168HAMyfP78leQOAxMRExMTEYNOmTdDpdHJCIyI3Zm/+24BQX0QEsEfH1djrFeU8OCLHkZXA5ebmAgDGjh0rujZu3DgAwL59+67Zhl6vx6FDh9C/f39RT51CocCYMWPQ2NiIo0ePih5bXl6OtWvX4t1338X69etRWFgoJ3wiclEFdRZclJj/dl8v9ua4ontuUEAlNQ+OB9sTOYysIdT8/HwAQL9+/UTXIiIioFarUVBQcM02CgsLYbVa0bdvX8nrzeX5+fkYMWKEzbXdu3dj9+7dLf+tUCiQkJCAZcuWITg4uE3PQa/Xt6meOzAajTZ/knvh/fvF7hLp1+Xd3RQQBAFt7YOTuxOZnPquUNdV4ghQAkO7+WJ/hdmm/HuNAfVNOreZB8fXoHvzxPsXEBDQ5rqyEri6ustL+bt06SJ5PSQkpKVOa22EhoZKXm9u+8p2goKC8H//93+YNGkS+vTpA0EQcPz4cSxatAifffYZdDodNmzY0KbnUFZWBovF0npFN6LRaJwdAl0H3j9gR5EKUm9HMaZLMJnVULW1IXfMhtwwZpPZjNsDDNgPP5tynQXY/lMZBncR96a6Mr4G3Zun3D8fHx+7nVtSZK9CdYbu3bvj9ddftykbPXo0hg0bhtGjR2PLli04duwY7rrrrlbbioyM7KQoHc9oNEKj0SAiIgIqVZs/4shF8P5dJggCjh+uwdUZxC1dfHBnv97w09e3vTG5HT9y6rtCXReJwz8oGDOilfh3iVl07aI5GE9197EpswQEoRY+orrOxtege/P2+ycrgZPqHbtSfX09wsLC2tSGVit9MHVrvXxXCgoKwowZM7B48WIcOHCgTQmcnO5Jd6FSqTzyeXkLb79/+VozynXi7p9RkQEICAiAwtDQ5rYUkNfpJKe+K9R1lTgUBh3u+duTUN23FkalbS9cTu4RpKx+y6bMd0UmAkLFW0O5Cm9/Dbo7b71/shYxNM99a54LdyWNRoOGhoZWu/9iYmKgVCrtzpVrLpeaZyelW7duAICmpqY21Sci12Jv/7f7enIBgysLtHMu6r7QATApXK+3jcjTyErg4uLiAAC7du0SXcvOzrapY09gYCBiY2Nx9uxZFBcX21wTBAG7d+9GcHAwhgwZ0qaYDh06BACt7j1HRK7JXgIX19P7hkTcjdQZtY0+ATgc0scJ0RB5F1kJ3OjRoxETE4PMzEzk5eW1lGu1WixduhQqlQqPP/54S3l5eTnOnDkjGi6dM2cOAGDhwoUQhF867detW4eioiIkJCQgMDCwpfz48eM29Zpt3rwZGRkZCAsLw/jx4+U8FSJyAfb2fxsY5ovugezFcXX2zkXdG3argyMh8j6y5sD5+vpixYoVmD59OiZNmmRzlFZJSQkWLVqE6OjolvoLFixARkYGVq1ahVmzZrWUz5w5E1lZWcjMzMT58+cRFxeHgoICbNmyBdHR0Zg3b57N733ttddQVFSEYcOGITIyEhaLBXl5efj+++/h7++P1atX213VSkSuy+7+bxw+dQv31J2DymoSzYPbG3YrXi3e4qSoiLyD7FWoo0aNwrZt25CamoqsrCyYTCYMGjQICxYswLRp09rUhlKpRHp6OpYtW4aNGzdi9erV6Nq1K2bPno158+aJzkGdMWMGNm/ejEOHDqGqqgpWqxW9evXCU089hT/84Q+45ZZb5D4NInIB33L+m1sLshpxd10+csMG2pQ3z4PzEzxryyYiV9KubURiY2ORmZnZar01a9ZgzZo1ktf8/f2RkpKClJSUVtt56qmn8NRTT8mOk4hcG+e/ub/RtadECVzzPLh7JBY5EFHH4CnRROQUgiBIHr3E+W/uxd48uJzQgZLlRNQxmMARkVPk15lRrhPPfxvJ4VO30jwP7mp7wwY5IRoi78EEjoicIrdc+vxCHmDvXoKsRgyrE+8Nui/0Fu4HR9SJmMARkVPYm/82IoLz39yN1DBqg28gjqhjHB8MkZdgAkdEDhdoaMK+Mp2o/LYuCvQx1kCtrWz5UUjsAUmuhfvBETmeWxxmT0SeJb+iARf14vLRP30DbP7IpkyxfKOss0LJ8e6tOws/qxkmpe1Hyt6wQXil5EsnRUXk2dgDR0QOt7dSOiWz15NDru3yfnBS56LeArOCHzNEnYGvLCJyuL0V4tWnADCy9mcHR0IdhfPgiByLCRwROZQgCNhbKU7gbmssQQ9TnRMioo5gfx4ctxMh6gxM4IjIoc7VmaXnv3H41K01z4O7GhcyEHUOJnBE5FC5F6X3f2MC596CrEYMqxfvB5cbOgBmK5ehEHU0JnBE5FD29n8bxQTO7dmdB1fLBI6oozGBIyKHEQRBMoG7vaEE3U31ToiIOtLo2lOS5VJzHono+jCBIyKHOas1QyNx/ukoLXvfPMG92nPS8+Aq2ANH1NGYwBGRw3xrZ/j0/hrpnhtyL8FWg/Q8uCor58ERdTAmcETkMN/aWcDAHjjPITkPzgzkVZmcEA2R52ICR0QOYW/+2+CG8wg3NTghIuoM9haj2Fu8QkTtwwSOiBzip1ozKvXi+W/325n4Tu5phPYsfCXmwTGBI+pYTOCIyCG+vWhv/huHTz3J5XlwBaLy7zVGzoMj6kBM4IjIIaR6YBSCFSM5/83jSG0nUm8SOA+OqAMxgSOiTme1M//trobz6GpuckJE1JnsnarBYVSijsMEjog63Y81ZtQYxMNnPD7LM3EeHFHnYwJHRJ3O7vw3LmDwSPbmwe3nPDiiDsMEjog6nVQCpxSsGFn7sxOiIUeQ2k6kziTgRDXnwRF1BCZwRNSpLFYB+zTiBG5ofSFCLTonRESOYHcenJ3eWCKShwkcEXWqE9Um1BnFw2YcPvVsI+rOcB4cUSdiAkdEncre+adcwODZ1BYDfsX94Ig6DRM4IupUUkNmPgrgPu1pJ0RDjiSVpHMeHFHHYAJHRJ3GbBXwnUZ8gP2vuioQYtE7ISJyJM6DI+o8TOCIqNMcrzKh3iQx/y2cbz3ewO48OImknojkade76JEjR5CQkICoqChERkZi/PjxyMrKktWGwWDAkiVLMHToUERERGDgwIFISkpCRUVFmx6fkJCAsLAwREREtOcpEJED2N3/rTsTOG+gthgQW18oKv++3MB5cETXSfa7aE5ODiZOnIj9+/dj6tSpSExMhEajQWJiIlauXNmmNqxWK2bOnInU1FR069YNc+fOxbBhw7B+/XpMmDABlZWV13z8Rx99hOzsbAQEBMgNn4gcSGrFoZ8SiOumcEI05Az25sEd57moRNdFVgJnNpuRlJQEpVKJrVu34r333sNf//pX5Obm4uabb8aiRYtQXFzcajvp6enIzs5GfHw8tm/fjjfffBMbNmzAu+++i6KiIixevNjuY8+fP4958+bh+eefR/fu3eWET0QOZLIK+F5q/lt3FYJ8mcB5izG1P0qW7+U8OKLrIiuBy8nJQWFhIeLj4zF48OCW8tDQUCQnJ8NoNCIjI6PVdtavXw8AmD9/PhSKX97IExMTERMTg02bNkGnE2/wKQgC/vCHPyAiIgKvvfaanNCJyMGOVhrRaBYPk8X19HdCNOQscdozUEl80uwtYwJHdD1kJXC5ubkAgLFjx4qujRs3DgCwb9++a7ah1+tx6NAh9O/fH1FRUTbXFAoFxowZg8bGRhw9elT02H/+85/Yt28f3n//fQQGBsoJnYgc7NuL0hPVRzKB8ypBViPuuUHc43rgkgF6iQSfiNrGV07l/Px8AEC/fv1E1yIiIqBWq1FQIN648UqFhYWwWq3o27ev5PXm8vz8fIwYMcLmdy9cuBDPPfcc7rnnHjlh29DrPWfrAqPRaPMnuRdPv397L4h70VVKYHAXKwSzgLYOosr5iJebDnRW24zZ1tjuSuRUWmzK9BYgt7QB9/X0k9lax/H016Cn88T7J2duv6wErq6uDgDQpUsXyeshISEtdVprIzQ0VPJ6c9tXtmO1WjF37lxERETgjTfekBOySFlZGSwWS+sV3YhGo3F2CHQdPPH+Ga3AgYpA4Ko07Xa1BZUXSxHRVQ1VWxtjNuR6cciMedQNVsnyr85VIdrk/MUMnvga9Caecv98fHzsdm5JkZXAOcuKFStw8OBBbNmyBUFBQdfVVmRkZAdF5XxGoxEajQYRERFQqdr8cUguwpPv3/5LJhis4i9zY6PU6N27B/z09W1vTM56B7lrIzqrbcZs455wXwT7mtB41ZZwx5sC0bt3T3mNdSBPfg16A2+/f7ISOKnesSvV19cjLCysTW1otVrJ61f38p07dw6pqal49tlncd9998kJV5Inbj2iUqk88nl5C0+8fz9USw9pjOkdjIAAfygMDW1uS4G2d/jIqduZbTNmWyofJeJ6+mN7qe3ChWPVZhiVKnSRWuXgQJ74GvQm3nr/ZL1qmue+Nc+Fu5JGo0FDQ0Or3X8xMTFQKpV258o1lzf/rp9//hkGgwFr165FWFiYzU9JSQkMBkPLf9fW1sp5OkTUSaS2iAjwubyFCHmnUb3Ei1csArBPYq9AImqdrB64uLg4LF26FLt27cL06dNtrmVnZ7fUuZbAwEDExsbi4MGDKC4utlmJKggCdu/ejeDgYAwZMgQAEBUVhdmzZ0u2lZWVBZ1Oh5kzZwIA/P25uo3I2ZrMVhy8JO6BG97DH/4+3P/NW42ODAAgHr3JuWjAQ1HcVYBILlkJ3OjRoxETE4PMzEw899xzLXvBabVaLF26FCqVCo8//nhL/fLyctTV1SEiIsJm0cKcOXNw8OBBLFy4EGvXrm3ZC27dunUoKirC008/3bJNyODBg+2e8LBnzx6YTKY2nwBBRJ1vv8YIo8Sc9dGR/ILlrRR+KgwXahGuAiqvyu1zS5ugHvjL5DhrYDCaVEzoiFojK4Hz9fXFihUrMH36dEyaNAnTpk2DWq3G5s2bUVJSgkWLFiE6Orql/oIFC5CRkYFVq1Zh1qxZLeUzZ85EVlYWMjMzcf78ecTFxaGgoABbtmxBdHQ05s2b13HPkIgcao+dDVpHSwyhkXdQGHRQvjQD9w96AZk9bLeBOlEn4NLL/w89TJd755QrMgEmcEStkj1zdNSoUdi2bRuGDx+OrKwspKWloUePHkhLS8MLL7zQtl+qVCI9PR0pKSmorKzE6tWrceDAAcyePRs7duxAeHi47CdCRK5Bav5bF5UCd3Vz3n5f5BrG1kgfq7W76yAHR0Lk/tq1jUhsbCwyMzNbrbdmzRqsWbNG8pq/vz9SUlKQkpLSnhAAACdOnGj3Y4moYwUZdajVNiBP4pDyMd0UCK2vavlvhSDI3v6M3N9YO+ei7g67DTMu7XdwNETuzS32gSMi16fUNWLP39+CcPtLomtjc9KAjB0t/61YvpEJnBfqp9MgSl+J4gDbUZZdXW9zUkRE7su5m+8QkUfJ7nq7ZLm9oTPyLgoAYyT+LRQERqAogFNniORgAkdEHUaqJ6WXoQYDm8qcEA25InvDqLvC2AtHJAcTOCLqECVNAs4G9RKVj6s5Kfu0KPJcUj1wALCbw6hEsjCBI6IOsatC+sByDp/SlSKNtbi18YKofHfYIM6LJJKBCRwRdYjsS3YSODtDZuS9xkj8myj374pTQTc6IRoi98QEjoiumyAI2CWRwN3SVIabDNVOiIhcmb1e2V12FsEQkRgTOCK6bqe1ZkidST6Ow6ckYXTtKSgFccLPDX2J2o4JHBFdt712js8aW3PSwZGQO+hqbsLQ+kJR+d6wW2G2ciYcUVswgSOi6yZ1/qlCsGJ07U9OiIbcwf21p0RlWt9gHKllAkfUFkzgiOi6mK0C9kmMnw5tKMIN5kYnRETuwO48ODurmYnIFhM4IrouRytNqDOJe03s7fdFBAD3aU9DZRWfmyu1GIaIxJjAEdF12VWmlywfx/lvdA1BViPuqTsrKt9XJaDJzCSOqDVM4Ijouuy+IB4+DbAYcZ/2tBOiIXcyvlqc5BuswHflRidEQ+RemMARUbtpjVYcrBB/2I7U/oxAieExoitNqDkhWW6vV5eIfsEEjojaLeeiARaJRYMPVOc5PhhyO0PrC3GDqV5ULtWrS0S2mMARUbvZ+6C117NCdCUfCBgvMVfyp1ozLjRanBARkftgAkdE7SY11NXLUIPbGkudEA25ownV0sn+bg6jEl0TEzgiapfCOjOK6sW9JBNqTkDhhHjIPY2301vLYVSia2MCR0TtYm+iub0eFSIpvQ3VuLXxgqh8d5kBVoGnMhDZwwSOiNpll50eEu7/RnJNqBEveqk2WJFXxZXMRPYwgSMi2UxWAd9elDg+q74QPUx1ToiI3Jm9XttsDqMS2cUEjohkO1RhlDw+azyHT6kdRtX+LH2sFhcyENnFBI6IZLM3fMrtQ6g9gq0GxEmc3PHDJSPqTTxWi0gKEzgikk1qi4cgH2CE9owToiFPIDWMarIC+8o5jEokhQkcEclSY7DiSKV4uGt0uBL+gtkJEZEneMBO7y3nwRFJYwJHRLLkXDTAKrG7w4QI7v5G7Te4oRg9jFpROfeDI5LGBI6IZMm+ID2x/IEIvp1Q+ykhSG7qe67OjPP17NkluhrfcYmozQRBwM5ScQJ3U7APBqjZA0fXx952IvYWzRB5MyZwRNRmP9aYUdYkXhU49kZ/KBRM4Oj62FvF/I3ElwYib9euBO7IkSNISEhAVFQUIiMjMX78eGRlZclqw2AwYMmSJRg6dCgiIiIwcOBAJCUloaKiQlT37NmzePHFFzFy5Ej069cPPXr0wB133IEZM2Zg79697XkKRNQOO+x8kD5wU4CDIyFP1NOoxdAw8ReBnIsG6M08VovoSrITuJycHEycOBH79+/H1KlTkZiYCI1Gg8TERKxcubJNbVitVsycOROpqano1q0b5s6di2HDhmH9+vWYMGECKisrbeqfOnUKW7ZsQWRkJKZOnYrnn38eI0aMwPfff48pU6bgnXfekfs0iKgdtkskcH5KYHSkvxOiIU/0oMRcyiazgO80HEYlupKvnMpmsxlJSUlQKpXYunUrBg8eDAB45ZVXMG7cOCxatAhTpkxBVFTUNdtJT09HdnY24uPjsXbt2pahl7S0NCQnJ2Px4sVYvnx5S/0HH3wQBQUFoiGaixcvYtSoUViyZAmeffZZhIWFyXk6RCRDrcGKHy4ZReUjIvwR4sfZGNQxHuqpxN9OW0Tl20v1GHsje3qJmsl6183JyUFhYSHi4+NbkjcACA0NRXJyMoxGIzIyMlptZ/369QCA+fPn2yRliYmJiImJwaZNm6DT6VrK/f2l59f06tULw4cPh8lkQklJiZynQkQy7S7TwyK1fchN7H2jjnP3DQp09Re/39sbvifyVrISuNzcXADA2LFjRdfGjRsHANi3b98129Dr9Th06BD69+8v6qlTKBQYM2YMGhsbcfTo0Vbjqa6uxuHDhxEUFISYmJg2Pgsiao/tpdJDWJz/Rh3JR6HAeImetvw6C/K13E6EqJmsIdT8/HwAQL9+/UTXIiIioFarUVBQcM02CgsLYbVa0bdvX8nrzeX5+fkYMWKEzbVz585h06ZNsFgsKC8vx9dffw2tVoulS5ciJCSkTc9Br/ecb3FGo9HmT3Iv7nT/rIKAHSU6UXlUsBK9/c3Q6y0IFgS0dR2qnOnonVXXVeJgzFfVFQTcH+GDTRIfJV8X1ePZAYEyf7t97vQaJDFPvH8BAW3/QiwrgaurqwMAdOnSRfJ6SEhIS53W2ggNDZW83ty2VDvnzp3DkiVLWv5brVZj1apVmDFjRuvB/09ZWRksFvH8Cnem0WicHQJdB3e4f6fqlag0iN9YhncxoLS0FADQpasaqrY26JaZhQvUdZU4OjFmk9mMW6y1UCAQV38l+LKgHhODKu08sv3c4TVI9nnK/fPx8bHbuSVFVgLnbA8++CBqa2thNBpRXFyMjz76CL/73e9w+PBhvPXWW21qIzIyspOjdByj0QiNRoOIiAioVG3+6CQX4U7377OTTQDEPXBTbrkBvSMvx+6nr297g3K2jOusuq4SB2O24efrizv69sbQc1ocrrIdMj1a54Mbet2EYN+O2XPQnV6DJObt909WAnet3jEAqK+vb3UlaHMbWq34zLsr27bXywcAKpUKN998MxYtWgSdTod//etfmDBhAiZMmNDaU5DVPekuVCqVRz4vb+EO9293ufg1H+ADjI1SI8D38lRahaGhze0p0PZOmc6q6ypxMOar6ioUCAgIwINRRhyusv1SYLACB2uAB3t37OvFHV6DZJ+33j9Zixia5741z4W7kkajQUNDQ6vdfzExMVAqlXbnyjWXS82zkzJmzBgAvyywIKKOVam34HCFSVQ+sqc/gny5fQh1LIWfCmptJaaESc9X3pNfC7W2EmptJYKM4l5hIm8h6903Li4OALBr1y7RtezsbJs69gQGBiI2NhZnz55FcXGxzTVBELB7924EBwdjyJAhbYqpvLwcAODn59em+kQkT/YFg2TvyQSuPqVOoDDogBfjcddfZqCnoUZ0/eufNBBejAdejIdS1+iECIlcg6wEbvTo0YiJiUFmZiby8vJayptXgqpUKjz++OMt5eXl5Thz5oxouHTOnDkAgIULF0IQfvloWLduHYqKipCQkIDAwF9WGh07dsymXrPi4mIsW7YMADB+/Hg5T4WIWhFk1EGtrcTuAunpDo+G6Vp6QtTaSigkXqNE7aWEgAerj4vKzwd0x09BNzohIiLXImsOnK+vL1asWIHp06dj0qRJmDZtGtRqNTZv3oySkhIsWrQI0dHRLfUXLFiAjIwMrFq1CrNmzWopnzlzJrKyspCZmYnz588jLi4OBQUF2LJlC6KjozFv3jyb3/v666+jsLAQsbGxuOmmm6BUKlFYWIidO3fCaDTihRdewD333HOd/yuI6EpKXSMsLybgm7h/AH5qm2sDmsrQ98//Z1OmWL5R9uJEomt5qPo4Pux1v6j86253YlDTBccHRORCZK9CHTVqFLZt24bU1FRkZWXBZDJh0KBBWLBgAaZNm9amNpRKJdLT07Fs2TJs3LgRq1evRteuXTF79mzMmzcP4eHhNvV/+9vfIisrC8eOHcOuXbtgNBrRvXt3TJw4EU8//XTLJsJE1LG+D+2PmquSNwCYWCXuGSHqaOOrT8BHsMCi8LEp39ptCF4u+cpJURG5hnZtIxIbG4vMzMxW661ZswZr1qyRvObv74+UlBSkpKS02s6UKVMwZcoU2XES0fXZ3C1Wsvyh6mOODYS8UqhFhzjtaeSEDbIp3xc6ANW+wbjBSXERuQIuISMiu74MHyoq62Juwujan5wQDXmjSVXiYxUtCh983e0uxwdD5EKYwBGRpJ/rrTgTJN74+sHq41AJnnWaCbmuyZWHJcvt9Q4TeQsmcEQkaXOZVbL815VHHBwJebP+Og0GNooXLHxzw2AYLFw2Q96LCRwRSdp8UZzA+VrNnP9GDvfrKnEvXINvIHZVSH/JIPIGTOCISOSSzoID1eLejVHanxFmbnJCROTN7A2jbpH4kkHkLZjAEZHIthK95J5uHD4lZxhel48IY62ofMtFK6zcQJq8FBM4IhL5qlj6HEp7PSFEnUkJAY9UilejXtQDRyvF5/QSeQMmcERko9FkxZ4ycQJ3Z8N5RBsqnRARkfQ8OAD4qpgH2pN3YgJHRDb2lBmgl9gl5NfsfSMnGldzEkEW8RcLe73FRJ6OCRwR2fiqRPoDkQkcOVOg1YQJ1SdE5T/VmlFYZ3ZCRETOxQSOiFpYrAK+kUjgbtJXYWhDkeMDIrrC5CrpRTT2vnQQeTImcETU4ocKIyr14q0Zfl11BAonxEN0pUlVR6AUxP8+OQ+OvBETOCJqYW8+0SMcPiUXEG5qwAjtGVH59xojqqUmbhJ5MCZwRAQAEAQBW86LezJCzDrcz8PryUVIbWVjFTiMSt6HCRwRAQCOV5lQVC/uxXiw+jj8BU4SJ9cw2c52IpuLOIxK3oUJHBEBAL6w8wEYX3HAwZEQ2XezToPBDedF5bvLDKg18Ggt8h5M4IgIgiDgvxIJXJBFj4eqjjk+IKJrmF7xg6jMZOViBvIuTOCICCeqTSiUGD59qOo4gqxGJ0REZN90O73C9nqRiTwREzgi4vApuZWBTRdxe0OJqHwXh1HJizCBI/Jy9oZPA32Ah6qPOT4gojaQ6oUzWYFtXI1KXoIJHJGXO1ljRn6dxOrTCCXUFoMTIiJqnb3eYakvI0SeiAkckZezO3x6I98eyHXd2lSG27qIzwfZdUEPrZHDqOT5+A5N5MUEQZBM4AJ8gEm9+PZArm26xJcMI4dRyUvwHZrIi52qMeOsVrxJ7/gbA6D25emn5Nrs9RL/t5DDqOT5mMAReTF784WmxAQ6OBIi+QZ1UWJgmK+ofFeZHnUcRiUPxwSOyItJHT/k7wM8GBXghGiI5Jss8WXDYOEwKnk+JnBEXupUjQmnJYZPx90YgBA/vjWQe3jUTm8xV6OSp+O7NJGX+iy/SbLc3gcikSu6NcwXt4SKh1F3lupRw019yYMxgSPyQlZBwKZ86dWnD/bm8Cm5D4VCgUf7iL90GK08Wos8GxM4Ii8SZNRBra3EkfwKXGgSb947uZcSkbpqqLWVUAiCEyIkajuFnwpqbSWe7i493y3ztBZqbSXU2koEGZnMkWdpVwJ35MgRJCQkICoqCpGRkRg/fjyysrJktWEwGLBkyRIMHToUERERGDhwIJKSklBRUSGqm5eXh8WLF2P8+PG4+eab0aNHD9x55514+eWXUVZW1p6nQOSVlLpG4MV4ZGzcLnl95tdvAS/GAy/GQyFw+Ilcm8KgA16Mxy2vP4ZhdedE13OrBBS98jvgxfjL//aJPIjsBC4nJwcTJ07E/v37MXXqVCQmJkKj0SAxMRErV65sUxtWqxUzZ85EamoqunXrhrlz52LYsGFYv349JkyYgMrKSpv6ycnJeOeddyAIAqZNm4bnnnsOkZGR+OCDDzBy5EicOXNG7tMg8lp6pR8ye9wtKu9mqsfE6jwnRER0/WZp9kmWZ/QY4eBIiBxDVgJnNpuRlJQEpVKJrVu34r333sNf//pX5Obm4uabb8aiRYtQXFzcajvp6enIzs5GfHw8tm/fjjfffBMbNmzAu+++i6KiIixevNimfkJCAo4cOYLs7Gy89dZbWLRoEb7++mu8+eabqKqqwrx58+Q9ayIv9tUNd0HrGywqn3Hpe/gJ4mFVInfw2KX98JH49/tJxH3gZADyRLISuJycHBQWFiI+Ph6DBw9uKQ8NDUVycjKMRiMyMjJabWf9+vUAgPnz50Oh+GW398TERMTExGDTpk3Q6X6Zr/Dcc8+hb9++onZeeOEFBAYGYt8+6W9eRCT2SUScZPkTdnowiNxBD1MdHpDoQf45+EYcUcc4PiCiTiYrgcvNzQUAjB07VnRt3LhxANBqMqXX63Ho0CH0798fUVFRNtcUCgXGjBmDxsZGHD16tNV4FAoF/Pz84OPj09anQOTVqo0Cvuo2RFTeV6fBPRJziIjcib1h1E8i7nNwJESdT7x5zjXk5+cDAPr16ye6FhERAbVajYKCgmu2UVhYCKvVKtmjBqClPD8/HyNGXHvuwhdffIG6ujo8+uijbYj+Mr3ec3bnNhqNNn+Se3HG/fu81AKTUvyyn6nZh6tPPpU77CSnvivUdZU4GHPH1Z1ceRhqsw4NvrbbimyMuBdLLFbR+z/fQ92bJ96/gIC2b+MkK4Grq6sDAHTp0kXyekhISEud1toIDQ2VvN7cdmvtlJaW4tVXX0VgYCBef/31a9a9UllZGSwWz5rno9FonB0CXQdH3r9PioMky2dK9Vy46qd0R9V1lTgYc4fVDbIaMbXyIDb0HGVTrlGFYXu5GZF+JZLN8D3UvXnK/fPx8bHbuSVFVgLnKqqrq/HYY4+hoqIC//jHP9C/f/82PzYyMrITI3Mso9EIjUaDiIgIqFQqZ4dDMjn6/p1vsOC76lpR+bC6c7hFVy5+wNVdcq2RU98V6rpKHIy5Q+vO0uSKEjgA2FimwNL7etuU8T3UvXn7/ZOVwLXWO1ZfX4+wsLA2taHVaiWvt9bLV11djcmTJ+Onn37C0qVLMWPGjLaE3kJO96S7UKlUHvm8vIWj7t+W0/WS5fbmDSkgrzNETn1XqOsqcTDmjq07puYUehlqcNG/q035f8useMtHBbXEOb98D3Vv3nr/ZC1iaJ771jwX7koajQYNDQ2tdv/FxMRAqVTanSvXXC41z645eTt58iTefvttJCYmygmfyGtZBQGfnBVvZOojWPDYpf1OiIioc/hAwOOXvhOVN1mAzTxaizyIrAQuLu7y9gO7du0SXcvOzrapY09gYCBiY2Nx9uxZ0Z5xgiBg9+7dCA4OxpAhtivlrkze3nrrLTz77LNyQifyarnlRhTWi+d+Tqg+gR6ma883JXI39nqV159pcnAkRJ1HVgI3evRoxMTEIDMzE3l5v+y3o9VqsXTpUqhUKjz++OMt5eXl5Thz5oxouHTOnDkAgIULF0K44rzFdevWoaioCAkJCQgM/GUVUU1NDaZMmYKTJ0/i73//O37729/Ke5ZEXm79GeljhBLL9zo4EqLOd2fDeQxuOC8q33/JiJ9rTU6IiKjjyZoD5+vrixUrVmD69OmYNGkSpk2bBrVajc2bN6OkpASLFi1CdHR0S/0FCxYgIyMDq1atwqxZs1rKZ86ciaysLGRmZuL8+fOIi4tDQUEBtmzZgujoaNHJCk8++SROnDiBW265BTU1NUhNTRXFNnfu3Fbn3xF5o2q9RXLoqLtRi19XHnZCRESdSwHgmYu7kdT/adG19Wca8be7wxwdElGHk70KddSoUdi2bRtSU1ORlZUFk8mEQYMGYcGCBZg2bVqb2lAqlUhPT8eyZcuwceNGrF69Gl27dsXs2bMxb948hIeH29RvHmo9c+YMlixZItnmzJkzmcARSfg0XwejxLn0T5V/CxWPziIPNVOzD6/2nQm9j+3qxE/P6fCX2FD4+8hdSkvkWtq1jUhsbCwyMzNbrbdmzRqsWbNG8pq/vz9SUlKQkpLSajsnTpyQHSMRXZ5XusHO8On/u7jHscEQOVBXcxPiKw7g454jbcqrDVZ8eV6H6X2l90Qkchey5sARkXs5WGHET7VmUfnI2p8wQHfRCREROc4zF3dLln/ExQzkAZjAEXkwex9Uz9r5YCPyJPdpT2Ng4wVRec5FAwrrxF9siNwJEzgiD1VntCKrULx4IczUiGkVPzghIiLHUsD+VIENEvsiErkTJnBEHurzAh2azOI97WdpchFo5VYK5B1ma76Fn1Xc2/bJ2SaYrHIPdSVyHUzgiDzUR3YWL9ibF0Tkibqb6vFo5SFRuUZnxc4yfpEh98UEjsgDHas04liV+MNpWFcFBjeWOCEiIud55qL49CAA2HBO7+BIiDoOEzgiD7TmVINk+bMxPg6OhMj5xtacQh+JXUN2XTThvI77wZF7YgJH5GHKmyz4j8TihRA/BR67iS958j5KCHimj/SXl41l7doOlcjp+G5O5GHSTjfCJHHywpP9gxDix94G8k7PxPjAXyKH+1LjC63UUSVELo4JHJEHMVgErPtZvHhBAeC5QWrHB0TkIrr7K/CYxOkLOqsC6fkGJ0REdH2YwBF5kM8LmlChF/cmPNg7ADEhHCoi7/Y7O19iPjirh5lbipCbYQJH5CEEQcCaU9Jbh8y9jb1vRLfd4IdRvfxF5aWNVmwt5opUci9M4Ig8xHcaI05Ui7cOua2rL0b2VDkhIiLXM3dQsGT5P+ys3CZyVUzgiDyEvQ+g5wapoVBw8QIRAEzsHYA+IeLVDN9rjDhWaXRCRETtwwSOyM0FGXWoLKvA1vPiIaBwFZAY3gS1thJqbSUUAuf5kHdS+Kmg1laiS10VXuwjXeffx6taXitBRvFWPESuhLOaidycUteI9//9X1h7Pyy69puz/0Xg9k0t/61YvhFM4cgbKQw6CC/NAAA87ROAv9y7EnW+tqtSNxYZkfrpXPQy1kK5IhNQBTojVKI2YQ8ckZurMQr4d68xonJfqxlzL+xwQkREri3EosfTF/eKyk1KX6y4aaITIiKSjwkckZtblW9Bg6+4pyC+4gdEGmsdHxCRG/jDhW+gEMRb7vwjcgJqfSXO3SJyMUzgiNxYo8mKlfkWyWsvl3zp4GiI3EdffQWmVh4Sldf7BmJN5HgnREQkDxM4Ije24WwTqiQWzk2sOo4hDecdHxCRG3mleLNk+YqbHoTOwtmi5NqYwBG5KYNFwPsnpbcOsffBRES/+FV9IcZVnxCVV6hCkVbE81HJtTGBI3JTn5xtQmmjePj0Hu0ZjNL+7ISIiNzPq3a+7Lx12gy9mb1w5LqYwBG5IaNFwNK8eslrrxZvBrftJWqbMbWnMKzunKj8gh7YcFb6aDoiV8AEjsgN2et9u6u+CI9UHXVCRETuSQHg9fP/lby2LK+evXDkspjAEbkZo0XAu3Z63+YX/Ye9b0QyTao6il/V5YvKy5qs7IUjl8UEjsjNfHi60W7v26+rDjshIiL3pgDwRtF/JK+9e7weTWYuaCDXwwSOyI00mKx4+7h079sb59n7RtReD1cfk+yFK9dZ8a9T7IUj18MEjsiN/ONUIyr04t6AIfWFmFzJ3jei9lIAmF/0ueS1ZSfqUWtgLxy5FiZwRG6iWm/BihPSvW+LCzay943oOj1UfRxxteIteLRGAe/Zee0ROQsTOCI38U5ePepM4hVxo8IVeKBGvBkpEcmjAPDXwo2S1/5xqhEXJOaeEjlLuxK4I0eOICEhAVFRUYiMjMT48eORlZUlqw2DwYAlS5Zg6NChiIiIwMCBA5GUlISKigpR3aamJqxcuRLPPvsshg0bhq5duyIsLAznz/OoIPIO+Voz1v4kPQ/nr7f5sveNqIPcpz2DSZVHROU6i4CFh7VOiIhImuwELicnBxMnTsT+/fsxdepUJCYmQqPRIDExEStXrmxTG1arFTNnzkRqaiq6deuGuXPnYtiwYVi/fj0mTJiAyspKm/oVFRV44403kJmZCb1ej7CwMLlhE7m1vxzSwiQxBeeh3gG4txs70ok60qLCzyS/FG3M1+FIhcThw0ROIOud32w2IykpCUqlElu3bsV7772Hv/71r8jNzcXNN9+MRYsWobi4uNV20tPTkZ2djfj4eGzfvh1vvvkmNmzYgHfffRdFRUVYvHixTf1u3bohKysLhYWFOHHiBIYOHSrvWRK5sdxyA74s1ovKfRXAwmFdnBARkWcb3FiCOdHSH4+vH9RCELi5LzmfrAQuJycHhYWFiI+Px+DBg1vKQ0NDkZycDKPRiIyMjFbbWb9+PQBg/vz5UCh++Z6TmJiImJgYbNq0CTqdrqVcrVZjzJgx6Nq1q5xwidye2Sog5YD0sM0zA4PRP9TPwREReYeFg3wR7Cvuh/teY8R/CnUSjyByLFkJXG5uLgBg7Nixomvjxo0DAOzbt++abej1ehw6dAj9+/dHVFSUzTWFQoExY8agsbERR4/yOCCiD35uxMlqk6g8VKXAq3eFOCEiIu8QGahA0h1qyWvzDmpRLzWngciBfOVUzs+/vMlhv379RNciIiKgVqtRUFBwzTYKCwthtVrRt29fyevN5fn5+RgxYoSc8NpErxcPRbkro9Fo8ye5l2vdvzBYUFnbiL8dESdvADBvgBK99dWA/vLKubYO6MgZ+JE7SNRZbTNm14vDK2IWBDx7sx8+PK1EWZNtsnaxyYq/HarBX4YEy2yVOpInfgYGBAS0ua6sBK6urg4A0KWL9LybkJCQljqttREaGip5vbnt1tppr7KyMlgsnrUUXKPRODsEug5S969LVzVeW7cH2p6jRNcGNZbiD++/BoXwv3/Hy6S3PZDET2nXi4Mxu15dACazGVUVpXi+tw9eP+0vuv6v0zqMDqxBv2DOh3M2T/kM9PHxsdu5JUVWAucJIiMjnR1ChzEajdBoNIiIiIBKpXJ2OCTTte7ft8VarJdI3gDgvbMfwU+44kuInD1EOquuq8TBmF2vrqvEITNm/6Bg3OXrizvDBeysNmN3hW2iZhEUWHE+CLtH+UIIDEYtfOT9Arpu3v4ZKCuBa613rL6+vtUtPprb0GqlJ2a31st3veR0T7oLlUrlkc/LW1x9/5rMVvz+mFmy7mOXvseY2lM2ZXKGUDurrqvEwZhdr66rxCE7ZoMOipdmQAFgRVAkhvwqFWal7Ufmd9UC1r71T/wuZS4CQsNltE4dyVs/A2UtYmie+9Y8F+5KGo0GDQ0NrXb/xcTEQKlU2p0r11wuNc+OyBssOVqPfIk9e9VmHd7KT3d8QERe7tamMrxU+rXktT/3fQKlTRxGJceTlcDFxcUBAHbt2iW6lp2dbVPHnsDAQMTGxuLs2bOiPeMEQcDu3bsRHByMIUOGyAmNyCMcqTDi/R8bJK/9rWAjbjJUOzgiIgKAN4qy0Ed3SVRe7xuIuUfN3BuOHE5WAjd69GjExMQgMzMTeXl5LeVarRZLly6FSqXC448/3lJeXl6OM2fOiIZL58yZAwBYuHChzT/6devWoaioCAkJCQgMDGzXEyJyVzqzgN99WwOLxOfAPdoz+F3ZTscHRUQAgGCrAavPfCB57WuNFRvONjk4IvJ2subA+fr6YsWKFZg+fTomTZqEadOmQa1WY/PmzSgpKcGiRYsQHR3dUn/BggXIyMjAqlWrMGvWrJbymTNnIisrC5mZmTh//jzi4uJQUFCALVu2IDo6GvPmzRP97nnz5qGqqgoAcOrU5TlAb7zxBoKDLy/jfuqpp3DvvffK/z9A5CIWHtbijFY8983PasY/T/8bStlL/4ioI02oOYmnynMkFxi9dkCLUb38ERPidWsDyUlk/0sbNWoUtm3bhtTUVGRlZcFkMmHQoEFYsGABpk2b1qY2lEol0tPTsWzZMmzcuBGrV69G165dMXv2bMybNw/h4eLJoF988QVKSkpsyjZv3tzy9/vuu48JHLmtvWV6rDklfVj96+ezcFvTBQdHRERS3j73CbZ3vQPl/rYnAzWYBcz9tgZfPhgOH6XcZbpE8rXrq0JsbCwyMzNbrbdmzRqsWbNG8pq/vz9SUlKQkpLSpt954sQJWTESuYsKvRW/zZFelf2runykFG+WvEZEjtfN3IB/nv43pgz+P9G17zVGvH28HilDeEYxdT5Zc+CIqGNZBSBpfwM0OvGxPAEWIz78eQ18BR7ZQ+RKJlUfwzNluyWvvXW8HrnlBgdHRN6ICRyRE318wRe7Lkofl5Va8CkGNl10cERE1Bbv5H8suSrVKgC/2VuNCp1nnfhDrocJHJGTfFtuwqoiP8lrD/dU4g8XvnFwRETUViEWPT4+9T58reKFRxebrEjcUw2zlQuPqPMwgSNygpIGM577rh5WifN9egYq8UGsr+zTiojIsYbX52Nx4WeS13LLjXjzUOec6U0EMIEjcrhGkxWzd1Wj2iD+dq5UAP8afQO6+zN9I3IHySVf4YEe0q/X939swGf53B+OOgcTOCIHsgqXN+s9ViU97+2NoV0wqpe/g6MiovZSQsD6YX64KVj6MPs/5NZgv4aLGqjjMYEjcoAgow5qbSVSv9Ngy3m9ZJ1HI5WYF6WHWlsJBY/lIXIb4f4KfDz2BvhL5HBGKzAruxqFdeK5ckTXgwkckQModY1YuWQt3j4jvTJtYOMFpG38f1AkJQAvxkPBrUOI3Mpd4SosvTdM8lqVwYrp2ytxiStTqQMxgSNygIwSC/7Y/ynJa91M9fjixDvoYtE5OCoi6kiz+gfjhdvVktcK6i2I314FrZFfzqhjMIEj6mRfF+uQeEh6+MTPakbmyWXopxfvJ0VE7kHhp4JaWwm1thLv9Ddici/pj9a8ahOe3FGBRhOTOLp+PHWXqBN9U6LHU7urYbYzpe3fp/+FkdrTjg2KiDqUwqCD8NIMAIAPgA1Kf4wdMg+HQ/qK6n57yYwZO6vw2YRuCPJlHwq1H//1EHWSbSU6zN5VBXtftt8+9zFmafY5Nigi6nTBVgO25L2NW5rKJK/nlhsxY0cVGtgTR9eBCRxRJ9iU34Qns6thb7rLn4q34I+lXzs2KCJymB6mOnx9fAkiDdWS178tN+LRbypRY2ASR+3DBI6og/3rVAN+m1Njd9h07oUdSC341LFBEZHDRRsq8c3xVPQwaiWvH6owYdJXFbjQyNWpJB8TOKIOYrEK+POBWrxyQAt7u7g9d2EnVpz9kMdkEXmJW5vKsPPYX9HdThJ3qtaM8V9ewvEqo4MjI3fHBI6oA2iNVjy5qxprTjXarfO7CzuwkskbkdcZ1HQBO47/DRHGWsnrF5usePirSnx5nlsJUdsxgSO6TqdrTRi3pQJfl0ifsAAAr97ig5VnP4TSbt8cEXmy2xtLsffoQkQHSV9vNAt4clc1Fh3WwmLl+wS1jgkcUTsFGpqQlafB2M2XcO4ax+T8/XYf/O02H/a8EXm5m3Ua5IxW4dYw+zt4vZvXgCnfVKKkgUdv0bUxgSNqh1qDFYnfapF42Ax784/9rUak/7gSf3r/cR6NRUQAgBsDFfj64e64P9Lfbp3cciPi/nsJG/ObIPBcZLKDCRyRDIIg4L+FOtydpUF6if2krIdRi+3HUvFYxX4HRkdErk7hp8JN+mp8fbeAZ2PsfwTXmQQ8l1ODxD013GqEJPEkBqI2utBowZ++r73mXDcAGFZ3Dpt+fA832dn/iYi8V/OpDX4A1gAY2msskvrPgUkp/XH83yIdvtMY8GZsFzx+cxCUCk7GoMvYA0fUCr1ZwMqT9bgnS9Nq8vbchZ3Yc3QRkzciapUCwG8v7sKuY4sQpa+0W++Szorf59ZiwpcVOFTB7UboMiZwRHaYrQI+PtuIX/1HgzcO1qHeZH8uSpipEZ+dXI5VZ9fBX+DkYyJqu3vrzuHowRTMLs+5Zr3DlSaM/7ICv8upRjEXOXg9JnBEV7FYBXxRpEPcfy/hD7m1KG1ll/SHqo7iyKE/Y1rlQQdFSESeJtSiw7qf/4mNP76HG0z116z7ab4OQzM1+ENuDfK1TOS8FefAEf1PvcmKz37SYtXPTSiwvx9vi+4qYNmx9zHj0vfcIoSIOsT0ih8wQnsGvxnwG2zrdpfdemYB+PhsE9LPNWFan0C8eLsag7upHBcoOR174Mjrna414fUftLhtYzlePty25O2p8hycGu+Dx5m8EVEH62WsxZYTb+Pze3wRrfa5Zl2rAGQW6DBqcwXGf3kJG840otHEVavegD1w5JUu6SzILNBhY34TjleZ2vy4u+vOITU/A6O1P0OpGgu+TRJRZ1AAeDQ6EBMjDFh2Fkg9bUFTK2feH6ow4VBFLV77QYv4voGY2icIIyJU8FXya6YnYgJHXiNfa8Y3pXp8U6JHbrkBFhn7Yw5svIDFhZ9hSuUh9rgRkUMoDDoE/HEG/gzgKVVXzO+TgI973geL4tq9cvUmAetON2Hd6SZ081fikegATI4JxMie/lD58B3MUzCBI49Vpbfge40R+8oN2FFquOZxV/b00V3Ca+f/i9mab+HL0xSIyEluNNbgg9P/wrzzWXin9yNY12s0jEq/Vh9XZbDiozNN+OhME4J8Fbg3QoX7e/ljdKQ/br/Bj/vKuTEmcOQRjBYBP9WakFdlwuEKI77XGHH6OlZn3Vf7M14q/Rq/rjwMHx5AT0Quoo++AqvOrsNr5/+Lpb0fxr8jx6LRJ6BNj20yC8i+YED2BQMAIEylQGx31eWfcBViu/shPODavXvkOtqdwB05cgSpqak4cOAAzGYzBg0ahOeffx5Tp05tcxsGgwHLly/Hxo0bceHCBXTt2hUTJ07EvHnz0L17d8nHfPbZZ/jHP/6Bn3/+GX5+frjnnnvw5z//GXfddVd7nwq5kQaTFfl1ZhTUmVFQZ8G5OjNOVpvwc60J1ztv199qxPQof7z4xev4VX1hxwRMRNQJbjTW4N38T/Bm0ef4tMe9+HfkWBwO6SurjVqjbUIHAL2ClBgY5odbu/piYJgfbgn1RXSILyICleytczHtSuBycnIwffp0BAQEYNq0aVCr1di8eTMSExNRWlqKF154odU2rFYrZs6ciezsbAwbNgyTJ09Gfn4+1q9fj71792Lnzp0IDw+3ecw777yDxYsXo3fv3khMTERDQwP+85//YOLEifjiiy9wzz33tOfpkIswWASUN1ku/+isuPi/v19ssqC4wYKCOjM0uo4fxhxVewpPludiWuVB3DBtHawfM3kjIvcQYtHjNxd34zcXd+PIX7Pw73N6ZF6worqdBzZcbLLiYpMBu8sMNuX+PkDvYF9EqX0QHeKDKLUvegX5oHugEt0DlOgR6IPwACUXTDiQ7ATObDYjKSkJSqUSW7duxeDBgwEAr7zyCsaNG4dFixZhypQpiIqKumY76enpyM7ORnx8PNauXQvF/zL7tLQ0JCcnY/HixVi+fHlL/fz8fPz973/HzTffjOzsbISGhgIAnnnmGUyYMAFJSUn4/vvvoVRyZ5TOZhUEGC2A0SrAZBVgsv7v7xag0WxFk1lAk1lAo1lAo6n575fLG00Cao1W1BqsqDFYUWMUUGu4/N8NZscMVSoFK+6pO4tJVUfxuOZ7RBvsH2FDROQufhVsxNB1T+A9hQ/2hg1EVvjdyOo+DJdUodfdtsECnKsztzqX+Ab/ywldqEqJLioFQvwu/9lFpUSI3y9/hvgpEeirQIDP5R9/H/zvT8VVf6IlPyBbshO4nJwcFBYWYtasWS3JGwCEhoYiOTkZv//975GRkYFXX331mu2sX78eADB//nybm5OYmIgVK1Zg06ZNSE1NRWBgIADgk08+gdlsxssvv9ySvAHA4MGDMX36dKSnp+P7779HXFyc3Kfktr65YMTn5wKgKq0HlE0QBAGCcHlfICtw+e8QWv7+y7X/1cPl/75c/kuZIABm4XJiZhYAkwX/S9QEGK2XH9PR/P/3gu0sXc0NGF1zCuNrT2JMzY/oZv7fZm8qQLjizc2qUEAIadubXWfVdZU4GLPr1XWVOBiz69W9sr4vgHHmUowrL8V75Vk42KUvvg29FTlhA3EkpA8s6Nx5bpV6Kyr1HTda4u8D+CkV8FH8708l4KsAfAA8HK7CC7077Fe5FUVtba2sj+OFCxdi6dKl+OCDDzB9+nSbaxqNBgMGDMCoUaOwefNmu23o9XpERkaiX79+OHhQfPzQH//4R6xbtw5fffUVRowYAQB44IEH8MMPP+D06dOIiIiwqf/555/jmWeewWuvvYZXXnlFztMhIiIicjuyxxvz8/MBAP369RNdi4iIgFqtRkFBwTXbKCwshNVqRd++0hMum8ubf1fz39VqtSh5uzKWK+sTEREReSrZCVxdXR0AoEuXLpLXQ0JCWuq01saVQ6FXam77ynbq6uqu+Tuvrk9ERETkqTjjn4iIiMjNyE7gpHrHrlRfX2+3p+zqNrRareR1qV6+Ll26XPN3Xl2fiIiIyFPJTuCuNd9Mo9GgoaHB7ty2ZjExMVAqlXbnyjWXXznPrl+/fmhoaIBGoxHVv9a8PCIiIiJPIzuBa96mY9euXaJr2dnZNnXsCQwMRGxsLM6ePYvi4mKba4IgYPfu3QgODsaQIUM69PcSEREReQLZCdzo0aMRExODzMxM5OXltZRrtVosXboUKpUKjz/+eEt5eXk5zpw5IxounTNnDoDL25IIwi87maxbtw5FRUVISEho2QMOAGbNmgVfX1+8++67Nm3l5eXh888/x4ABA3DvvffKfTpEREREbkf2PnCA/aO0SkpKsGjRIpujtObOnYuMjAysWrUKs2bNaim3Wq1ISEhoOUorLi4OBQUF2LJlC6KiopCdnX3No7QmT57ccpSW0WjkUVpERETkNdq1CnXUqFHYtm0bhg8fjqysLKSlpaFHjx5IS0tr0zmoAKBUKpGeno6UlBRUVlZi9erVOHDgAGbPno0dO3aIkjcA+NOf/oR//etfCA8PR1paGrKysnDvvffim2++8fjkbfny5QgLC0NYWJjk5sd1dXV47bXXcPvtt6NHjx6444478MYbb6ChocEJ0dIdd9zRcr+u/pk0aZKovsFgwJIlSzB06FBERERg4MCBSEpKQkVFhROipytt2bIFjz76KPr06YOIiAgMHjwYzzzzDEpLS23q8TXoOj755BO7r7/mn8mTJ9s8hvfP9QiCgM2bN+ORRx7BgAED0KtXL/zqV7/CSy+9hKKiIlF9b7uH7eqBI8c6deoUxowZA19fXzQ2NmLHjh0YNmxYy/XGxkY8+OCDOHHiBMaOHYvBgwcjLy8Pu3btwtChQ/HVV18hICDAic/A+9xxxx3QarWYO3eu6FpUVFSrvdH5+fn48ssvER0djZ07d0p+oaHOJQgC/vjHP+LDDz9Enz59MG7cOKjValy8eBH79u3D2rVrW6Zt8DXoWvLy8rB161bJa5s3b8ZPP/2EBQsWICkpCQDvn6t6/fXXsWrVKvTs2RMPP/wwQkJCcPLkSezatQtqtRrffPMNBg0aBMA776Hss1DJsUwmE+bOnYs77rgDffv2xWeffSaq89577+HEiRN46aWX8Oabb7aUv/nmm1i+fDlWr16N5ORkB0ZNwOWNqv/85z+3Wi89PR3Z2dmIj4/H2rVrW84GTktLQ3JyMhYvXozly5d3crR0tX/84x/48MMP8eyzz2LJkiXw8bE9P9Js/uVQb74GXcvgwYNtzupuZjQasXbtWvj6+uKJJ55oKef9cz0ajQZr1qxB7969kZuba7Px/6pVq1qSu1WrVgHwznvIHjgXl5qaiuXLl2Pv3r147733kJGRYdMDJwgCBg0ahPr6epw+fRrBwcEtj21sbMSAAQMQHh6OY8eOOekZeKc77rgDAHDixIlW6zaf85uXl4eoqKiWckEQMGTIEFRUVODcuXM2i3qoc+l0Otx6660ICwvDoUOH4Otr/7suX4PuIysrC4mJiZg0aRI++eQTALx/rurgwYOYMGECEhISsHbtWptr+fn5iI2NxcSJE7Fx40avvYc8icGFHTt2DO+++y5effVVDBw4ULJOfn4+Ll68iOHDh9v8owWA4OBgDB8+HEVFRaL5OtT5jEYjPvnkE7z77rv417/+hUOHDonq6PV6HDp0CP3797dJ3gBAoVBgzJgxaGxsxNGjRx0VNuHydkW1tbWYNGkSLBYLNm/ejGXLliEtLU20fyVfg+5j/fr1AICnnnqqpYz3zzX169cPKpUK+/fvF23iv23bNgCXd8UAvPcecgjVRRkMhpah0+Z5GlKaNzG2t3ly3759kZ2djfz8fNx0002dEitJ02g0eP75523Khg4dig8++AB9+vQBABQWFsJqtV7z/gGX7/OIESM6N2Bq0fxN3cfHB3FxcTh37lzLNaVSid///vdYvHgxAL4G3UVxcTH27t2LG2+8EePHj28p5/1zTTfccAP+8pe/YN68ebj77rtt5sDl5OTg2WefxW9/+1sA3nsPmcC5qL/97W/Iz8/Hnj17RHNvrtT8zeTK+QFXau3oM+ocs2bNwr333otBgwYhODgY586dw6pVq7Bx40ZMnjwZ3333HUJCQnj/XFRlZSWAy3Nt7rzzTuzatQu33HIL8vLy8NJLL+H9999Hnz598Mwzz/AeuolPPvkEVqsVTzzxhM17Ku+f63r++ecRGRmJF198EWlpaS3l9957L+Lj41umNnjrPeQQqgv64YcfsHLlSvzpT39qWWFD7iUlJQWjR49G9+7dERQUhMGDB+Of//wnZsyYgZKSEnz00UfODpGuwWq1AgBUKhU++eQTDB06FGq1GiNGjMCHH34IpVKJ999/38lRUltZrVZ88sknUCgUePLJJ50dDrXRkiVL8Nvf/hbJycn48ccfUVpaiq+//hp6vR6PPPIIvvrqK2eH6FRM4FyM2WzG3Llzcdttt+GPf/xjq/Wbv1lcfdJFs+ZvHM31yLkSExMBAAcOHADA++eqmv9/33XXXejVq5fNtUGDBiEmJgaFhYWora3lPXQDe/bsQWlpKUaNGoWYmBiba7x/rmnPnj1ITU3Fb37zG/zxj3/EjTfeCLVajXvvvReffvop/Pz8MG/ePADeew85hOpiGhoaWsbzu3fvLllnwoQJAICPP/64ZXHD1ROrmzWX9+vXr6NDpXbo1q0bAKCpqQkAEBMTA6VSyfvnYvr37w/A/pBMc7ler2+5N7yHrktq8UIz3j/XtGPHDgDAyJEjRdciIiLQv39/5OXloaGhwWvvIRM4F+Pv74/Zs2dLXvvuu++Qn5+Phx56COHh4YiKikK/fv3Qq1cvHDhwAI2NjaLl0wcOHEB0dLRHTdx0Z80rUZtXnAYGBiI2NhYHDx5EcXGxaBuR3bt3Izg4GEOGDHFKvN6q+UPjzJkzomsmkwkFBQUIDg5GeHg4IiIi+Bp0YdXV1fjqq6/QtWtXPPLII6LrfA91TUajEcAv81GvVlVVBaVSCT8/P6+9hxxCdTGBgYFYuXKl5M/dd98NAEhOTsbKlSsxePBgKBQKzJ49Gw0NDXj77bdt2nr77bfR0NCAOXPmOOOpeK0zZ8609LBdXd68wWR8fHxLefP9WbhwIQThl20Z161bh6KiIiQkJHAPOAfr06cPxo4di4KCgpbem2bLli2DVqvFpEmT4Ovry9egi/v0009hNBrx2GOPwd/fX3Sd9881NR+PuXr1atHQaFpaGi5cuIC7774b/v7+XnsPuZGvG5k7d65oI1/g8jeMiRMn4uTJkxg7dizuvPNOHD9+vOUIka1btzIBcKDU1FSsXr0aI0aMQO/evREUFIRz585hx44dMJlMSE5Oxvz581vqSx2lVVBQgC1btiAqKgrZ2dk8SssJCgsL8cADD6CiogITJ05sGbLJyclB7969sXPnTkRERADga9CVjRgxAqdOncK+fftw2223Sdbh/XM9FosFv/71r/Hdd9+he/fueOihhxAaGorjx48jJycHgYGB+PLLLxEbGwvAO+8hEzg3Yi+BAy5P3vz73/+OLVu2QKPRICIiAo8++iheffVVhISEOCli75Sbm4sPPvgAeXl5qKioQFNTE7p164bY2Fg8++yzGDt2rOgxBoMBy5Ytw8aNG3HhwgV07doVEydOxLx589CjRw8nPAsCgNLSUvztb39DdnY2qqurERERgYceegivvPKKaI4qX4Ou5/Dhwxg3bhxiY2ORnZ19zbq8f67HYDBg9erVyMrKwrlz52A0GtGjRw/cd999ePnllzFgwACb+t52D5nAEREREbkZzoEjIiIicjNM4IiIiIjcDBM4IiIiIjfDBI6IiIjIzTCBIyIiInIzTOCIiIiI3AwTOCIiIiI3wwSOiIiIyM0wgSMiIiJyM0zgiIiIiNwMEzgiIiIiN8MEjoiIiMjN/H/9a7MPwovYLgAAAABJRU5ErkJggg=="/>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- END QUESTION -->
<h2 id="Question-4:-Linear-Algebra">Question 4: Linear Algebra<a class="anchor-link" href="#Question-4:-Linear-Algebra">¶</a></h2><p>A common representation of data uses matrices and vectors, so it is helpful to familiarize ourselves with linear algebra notation, as well as some simple operations.</p>
<p>Define a vector $\vec{v}$ to be a column vector. Then, the following properties hold:</p>
<ul>
<li><p>$c\vec{v}$ with $c$ some constant $c \in \mathbb{R}$, is equal to a new vector where every element in $c\vec{v}$ is equal to the corresponding element in $\vec{v}$ multiplied by $c$. For example, $2 \begin{bmatrix}
   1 \\
   2 \\
\end{bmatrix} = \begin{bmatrix}
   2 \\
   4 \\
\end{bmatrix}$</p>
</li>
<li><p>$\vec{v}_1 + \vec{v}_2$ is equal to a new vector with elements equal to the elementwise addition of $\vec{v}_1$ and $\vec{v}_2$. For example, $\begin{bmatrix}
   1 \\
   2 \\
\end{bmatrix} + \begin{bmatrix}
   -3 \\
   4 \\
\end{bmatrix} = \begin{bmatrix}
  -2 \\
   6 \\
\end{bmatrix}$.</p>
</li>
</ul>
<p>The above properties form our definition for a <strong>linear combination</strong> of vectors. $\vec{v}_3$ is a linear combination of $\vec{v}_1$ and $\vec{v}_2$ if $\vec{v}_3 = a\vec{v}_1 + b\vec{v}_2$, where $a$ and $b$ are some constants.</p>
<p>Oftentimes, we stack column vectors to form a matrix. Define the <strong>rank</strong> of a matrix $A$ to be equal to the maximal number of linearly independent columns in $A$. A set of columns is <strong>linearly independent</strong> if no column can be written as a linear combination of any other column(s) within the set.</p>
<p>For example, let $A$ be a matrix with 4 columns. If three of these columns are linearly independent, but the fourth can be written as a linear combination of the other three, then $\text{rank}(A) = 3$.</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- BEGIN QUESTION -->
<p><strong>For each part below</strong>, you will be presented with a set of vectors, and a matrix consisting of those vectors stacked in columns.</p>
<ol>
<li>State the rank of the matrix, and whether or not the matrix is full rank.</li>
<li>If the matrix is <em>not</em> full rank, state a linear relationship among the vectors—for example: $\vec{v}_1 = 2\vec{v}_2$.</li>
</ol>
<!--
BEGIN QUESTION
name: q4a
manual: true
-->
<h3 id="Question-4a">Question 4a<a class="anchor-link" href="#Question-4a">¶</a></h3><p>$$
\vec{v}_1 = \begin{bmatrix}
     1 \\
     0 \\
\end{bmatrix}
, 
\vec{v}_2 = \begin{bmatrix}
     1 \\
     1 \\
\end{bmatrix}
, A = \begin{bmatrix}
    \vert &amp; \vert \\
    \vec{v}_1 &amp; \vec{v}_2   \\
    \vert &amp; \vert
\end{bmatrix}$$</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><em>Type your answer here, replacing this text.</em></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- END QUESTION -->
<!-- BEGIN QUESTION -->
<!--
BEGIN QUESTION
name: q4b
manual: true
-->
<h3 id="Question-4b">Question 4b<a class="anchor-link" href="#Question-4b">¶</a></h3><p>$$
\vec{v}_1 = \begin{bmatrix}
     3 \\
     -4 \\
\end{bmatrix}
,
\vec{v}_2 = \begin{bmatrix}
     0 \\
     0 \\
\end{bmatrix}
,
B = \begin{bmatrix}
    \vert &amp; \vert \\
    \vec{v}_1 &amp; \vec{v}_2   \\
    \vert &amp; \vert
\end{bmatrix}
$$</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><em>Type your answer here, replacing this text.</em></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- END QUESTION -->
<!-- BEGIN QUESTION -->
<!--
BEGIN QUESTION
name: q4c
manual: true
-->
<h3 id="Question-4c">Question 4c<a class="anchor-link" href="#Question-4c">¶</a></h3><p>$$
\vec{v}_1 = \begin{bmatrix}
     0 \\
     1 \\
\end{bmatrix}
,
\vec{v}_2 = \begin{bmatrix}
     5 \\
    0 \\
\end{bmatrix}
,
\vec{v}_3 = \begin{bmatrix}
     10 \\
     10 \\
\end{bmatrix}
,
C = \begin{bmatrix}
    \vert &amp; \vert &amp; \vert \\
    \vec{v}_1 &amp; \vec{v}_2 &amp; \vec{v}_3    \\
    \vert &amp; \vert &amp; \vert
\end{bmatrix}
$$</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><em>Type your answer here, replacing this text.</em></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- END QUESTION -->
<!-- BEGIN QUESTION -->
<!--
BEGIN QUESTION
name: q4d
manual: true
-->
<h3 id="Question-4d">Question 4d<a class="anchor-link" href="#Question-4d">¶</a></h3><p>$$
\vec{v}_1 = \begin{bmatrix}
     0 \\
     2 \\
     3 \\
\end{bmatrix}
, 
\vec{v}_2 = \begin{bmatrix}
     -2 \\
    -2 \\
     5 \\
\end{bmatrix}
,
\vec{v}_3 = \begin{bmatrix}
     2 \\
     4 \\
     -2 \\
\end{bmatrix}
,
D = \begin{bmatrix}
    \vert &amp; \vert &amp; \vert \\
    \vec{v}_1 &amp; \vec{v}_2 &amp; \vec{v}_3    \\
    \vert &amp; \vert &amp; \vert
\end{bmatrix}
$$</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><em>Type your answer here, replacing this text.</em></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- END QUESTION -->
<!-- BEGIN QUESTION -->
<h2 id="Question-5:-A-Least-Squares-Predictor">Question 5: A Least Squares Predictor<a class="anchor-link" href="#Question-5:-A-Least-Squares-Predictor">¶</a></h2><p>Let the list of numbers $(x_1, x_2, \ldots, x_n)$ be data. You can think of each index $i$ as the label of a household, and the entry $x_i$ as the annual income of Household $i$. Define the <strong>mean</strong> or <strong>average</strong> $\mu$ of the list to be
$$\mu ~ = ~ \frac{1}{n}\sum_{i=1}^n x_i.$$</p>
<!--
BEGIN QUESTION
name: q5a
manual: true
-->
<h3 id="Question-5a">Question 5a<a class="anchor-link" href="#Question-5a">¶</a></h3><p>The $i$ th <em>deviation from average</em> is the difference $x_i - \mu$. In Data 8 you saw in numerical examples that the <a href="https://www.inferentialthinking.com/chapters/14/2/Variability.html#The-Rough-Size-of-Deviations-from-Average">sum of all these deviations is 0</a>. Now prove that fact. That is, show that $\sum_{i=1}^n (x_i - \mu) = 0$.</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><em>Type your answer here, replacing this text.</em></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- END QUESTION -->
<!-- BEGIN QUESTION -->
<h3 id="Question-5b">Question 5b<a class="anchor-link" href="#Question-5b">¶</a></h3><p><a href="https://www.inferentialthinking.com/chapters/14/2/Variability.html#The-Rough-Size-of-Deviations-from-Average">Recall</a> that the <strong>variance</strong> of a list is defined as the <em>mean squared deviation from average</em>, and that the <a href="https://www.inferentialthinking.com/chapters/14/2/Variability.html#Standard-Deviation"><strong>standard deviation</strong></a> (SD) of the list is the square root of the variance. The SD is in the same units as the data and measures the rough size of the deviations from average.</p>
<p>Denote the variance of the list by $\sigma^2$. Write a math expression for $\sigma^2$ in terms of the data ($x_{1} \dots x_{n}$) and $\mu$. We recommend building your expression by reading the definition of variance from right to left. That is, start by writing the notation for "average", then "deviation from average", and so on.</p>
<!--
BEGIN QUESTION
name: q5b
manual: true
-->
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><em>Type your answer here, replacing this text.</em></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- END QUESTION -->
<h3 id="Mean-Squared-Error">Mean Squared Error<a class="anchor-link" href="#Mean-Squared-Error">¶</a></h3><p>Suppose you have to predict the value of $x_i$ for some $i$, but you don't get to see $i$ and you certainly don't get to see $x_i$. You decide that whatever $x_i$ is, you're just going to use some number $c$ as your <em>predictor</em>.</p>
<p>The <em>error</em> in your prediction is $x_i - c$. Thus the <strong>mean squared error</strong> (MSE) of your predictor $c$ over the entire list of $n$ data points can be written as:</p>
<p>$$MSE(c) = \frac{1}{n}\sum_{i=1}^n (x_i - c)^2.$$</p>
<p>You may already see some similarities to your definition of variance from above! You then start to wonder—if you picked your favorite number $c = \mu$ as the predictor, would it be "better" than other choices $c \neq \mu$?</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- BEGIN QUESTION -->
<!--
BEGIN QUESTION
name: q5c
manual: true
-->
<h3 id="Question-5c">Question 5c<a class="anchor-link" href="#Question-5c">¶</a></h3><p>One common approach to defining a "best" predictor is as predictor that <em>minimizes</em> the MSE on the data $(x_1, \dots, x_n)$.</p>
<p>In this course, we commonly use calculus to find the predictor $c$ as follows:</p>
<ol>
<li>Define $MSE$ to be a function of $c$, i.e., $MSE(c)$ as above. Assume that the data points $x_1, x_2, ..., x_n$ are fixed, and that $c$ is the only variable.</li>
<li>Determine the value of $c$ that minimizes $MSE(c)$.</li>
<li>Justify that this is indeed a minimum, not a maximum.</li>
</ol>
<p>Step 1 is done for you in the problem statement; follow steps 2 and 3 to show that $\mu$ is the value of $c$ that minimizes $MSE(c)$. You must do both steps.</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p><em>Type your answer here, replacing this text.</em></p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!-- END QUESTION -->
<p>Your proof above shows that $\mu$ is the <strong>least squares</strong> <em>constant</em> predictor.</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h2 id="Question-6:-A-More-Familiar-Least-Squares-Predictor">Question 6: A More Familiar Least Squares Predictor<a class="anchor-link" href="#Question-6:-A-More-Familiar-Least-Squares-Predictor">¶</a></h2><p>In Data 8 you found (numerically) the <a href="https://www.inferentialthinking.com/chapters/15/3/Method_of_Least_Squares.html">least squares <em>linear</em> predictor</a> of a variable $y$ based on a related variable $x$. In this course, we will prove your findings using a generalization of your calculation in the previous question.</p>
<p>When we get to this proof later in this course, you will need to be comfortable with vector operations. For now, you will get familiar with this notation by rewriting your least squares findings from Data 8 (and the previous question) using vector notation. <strong>This question won't require you to write LaTeX</strong>, so just focus on the mathematical notation we're presenting.</p>
<h3 id="The-Dot-Product">The Dot Product<a class="anchor-link" href="#The-Dot-Product">¶</a></h3><p>(1) We start by defining the <strong>dot product</strong> of two <em>real</em> vectors
$x = \begin{bmatrix}
     x_1 \\
     x_2 \\
     \dots \\
     x_n
     \end{bmatrix}$
and
$y = \begin{bmatrix}
     y_1 \\
     y_2 \\
     \dots \\
     y_n
\end{bmatrix}$ as follows:</p>
<p>$$x^T y = \sum_{i=1}^n x_i y_i $$</p>
<ul>
<li>Given the above definition, the dot product is (1) a <strong>scalar</strong>, not another vector; and (2) only defined for two vectors of the same length.</li>
<li><strong>Note</strong>: In this course we often opt for $x$ instead of $\vec{x}$ to simplify notation; $x$ as a vector is inferred from its use in the dot product. Then $x_i$ is the $i$-th element of the vector $x$.</li>
<li><em>Detail</em>: In this course, we prefer the notation $x^Ty$ to illustrate a dot product, defined as matrix multiplication of $x^T$ and $y$. In the literature you may also see $x \cdot y$, but we avoid this notation since the dot ($\cdot$) notation is occasionally used for scalar values.</li>
<li><em>Detail</em>: The dot product is a special case of an inner product, where $x, y \in \mathbb{R}^n$.</li>
</ul>
<p>(2) We introduce a special vector, $\mathbb{1}$, to write the <a href="https://inferentialthinking.com/chapters/14/1/Properties_of_the_Mean.html"><strong>mean</strong></a> $\bar{x}$ of data $(x_1, x_2, \dots, x_n)$ as a dot product:
\begin{align}
\bar{x} &amp;= \frac{1}{n}\sum_{i=1}^n x_i = \frac{1}{n}\sum_{i=1}^n 1x_i \\
        &amp;= \frac{1}{n}(x^T\mathbb{1}).
\end{align}</p>
<ul>
<li>The data $(x_1, \dots, x_n)$ have been defined as an $n$-dimensional column vector $x$, where $x = \begin{bmatrix}
   x_1 \\
   x_2 \\
   \dots \\
   x_n
   \end{bmatrix}$.</li>
<li>The special vector $\mathbb{1}$ is a <strong>vector of ones</strong>, whose length is defined by the vector operation in which it is used. So with $n$-dimensional column vector $x$, the dot product $x^T\mathbb{1}$ implies that $\mathbb{1}$ is an $n$-dimensional column vector where every element is $1$.</li>
<li>Because dot products produce scalars, the multiplication of two scalars $\frac{1}{n}$ and $x^T\mathbb{1}$ produces another scalar, $\bar{x}$.</li>
<li><strong>Note</strong>: We use bar notation for the mean ($\bar{x}$ instead of $\mu$) in this problem to differentiate $\bar{x}$ from $\bar{y}$, the latter of which is the mean of data $(y_1, \dots, y_n)$.</li>
</ul>
<p>(3) We can further use this definition of $\bar{x}$ to additionally write the <a href="https://www.inferentialthinking.com/chapters/14/2/Variability.html#The-Rough-Size-of-Deviations-from-Average"><strong>variance</strong></a> $\sigma_x^2$ of the data $(x_1, \dots, x_n)$ as a dot product. Verify for yourself that the below operation defines $\sigma_x^2$ as a scalar:
\begin{align}
\sigma_x^2 &amp;= \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2 \\
        &amp;= \frac{1}{n}(x - \bar{x})^T(x - \bar{x}).
\end{align}</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!--
BEGIN QUESTION
name: q6a
manual: false
-->
<h3 id="Question-6a">Question 6a<a class="anchor-link" href="#Question-6a">¶</a></h3><p>To verify your understanding of the dot product as defined above, suppose you are working with $n$ datapoints $\{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\}$.</p>
<ul>
<li>Define the $x$ data as $(x_1, \dots, x_n)$ and the $y$ data as $(y_1, \dots, y_n)$, and define $x$ and $y$ as two $n$-dimensional column vectors, where the $i$-th elements of $x$ and $y$ are $x_i$ and $y_i$, respectively.</li>
<li>Define $\bar{x}$ and $\bar{y}$ as the means of the $x$ data and $y$ data, respectively.</li>
<li>Define $\sigma_x^2$ and $\sigma_y^2$ as the variances of the $x$ data and $y$ data, respectively. Therefore $\sigma_x = \sqrt{\sigma_x^2}$ and $\sigma_y = \sqrt{\sigma_y^2}$ are the <a href="https://inferentialthinking.com/chapters/14/2/Variability.html?highlight=standard%20deviation#standard-deviation"><strong>standard deviations</strong></a> of the $x$ data and $y$ data, respectively.</li>
</ul>
<p><strong>Suppose</strong> $n = 32$. What is the <strong>dimension</strong> of each of the following expressions?</p>
<p>Expression (i). Note there are two ways it is written in the literature.
$$\dfrac{1}{\sigma_x} (x - \bar{x}) = \dfrac{x - \bar{x}}{\sigma_x} $$</p>
<p>Expression (ii).
$$\dfrac{1}{n} \left( \dfrac{x - \bar{x}}{\sigma^x}\right)^T \left( \dfrac{x - \bar{x}}{\sigma^x}\right)$$</p>
<p>Assign the variables <code>q6a_i</code> and <code>q6a_ii</code> to an integer representing the dimension of the above expressions (i) and (ii), respectively.</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">q6a_i</span> <span class="o">=</span> <span class="o">...</span>
<span class="n">q6a_ii</span> <span class="o">=</span> <span class="o">...</span>

<span class="c1"># do not modify these lines</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Q6a(i) is </span><span class="si">{</span><span class="n">q6a_i</span><span class="si">}</span><span class="s2">-dimensional"</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Q6a(ii) is </span><span class="si">{</span><span class="n">q6a_ii</span><span class="si">}</span><span class="s2">-dimensional"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">grader</span><span class="o">.</span><span class="n">check</span><span class="p">(</span><span class="s2">"q6a"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h3 id="Dot-Products-in-NumPy">Dot Products in NumPy<a class="anchor-link" href="#Dot-Products-in-NumPy">¶</a></h3><p>Next, we'll use NumPy's matrix multiplication operators to compute expressions for the <strong>regression line</strong>, which you learned in Data 8 was the unique line that minimizes the mean squared error of estimation among all straight lines. At this time, it may be helpful to review the <a href="https://inferentialthinking.com/chapters/15/2/Regression_Line.html#the-equation-of-the-regression-line">Data 8 section</a>.</p>
<p>Before we continue, let's contextualize our computation by loading in a <a href="https://inferentialthinking.com/chapters/15/4/Least_Squares_Regression.html">dataset</a> you saw in Data 8: the relation between weight lifted and shot put distance among surveyed female collegiate athletes. We've plotted the point using matplotlib's <a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html">scatter</a> function, which you will see in more detail in two weeks.</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Run this cell to plot the data.</span>
<span class="n">weight_lifted</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span> <span class="mf">37.5</span><span class="p">,</span>  <span class="mf">51.5</span><span class="p">,</span>  <span class="mf">61.3</span><span class="p">,</span>  <span class="mf">61.3</span><span class="p">,</span>  <span class="mf">63.6</span><span class="p">,</span>  <span class="mf">66.1</span><span class="p">,</span>  <span class="mf">70.</span> <span class="p">,</span>  <span class="mf">92.7</span><span class="p">,</span>  <span class="mf">90.5</span><span class="p">,</span>
        <span class="mf">90.5</span><span class="p">,</span>  <span class="mf">94.8</span><span class="p">,</span>  <span class="mf">97.</span> <span class="p">,</span>  <span class="mf">97.</span> <span class="p">,</span>  <span class="mf">97.</span> <span class="p">,</span> <span class="mf">102.</span> <span class="p">,</span> <span class="mf">102.</span> <span class="p">,</span> <span class="mf">103.6</span><span class="p">,</span> <span class="mf">100.4</span><span class="p">,</span>
       <span class="mf">108.4</span><span class="p">,</span> <span class="mf">114.</span> <span class="p">,</span> <span class="mf">115.3</span><span class="p">,</span> <span class="mf">114.9</span><span class="p">,</span> <span class="mf">114.7</span><span class="p">,</span> <span class="mf">123.6</span><span class="p">,</span> <span class="mf">125.8</span><span class="p">,</span> <span class="mf">119.1</span><span class="p">,</span> <span class="mf">118.9</span><span class="p">,</span>
       <span class="mf">141.1</span><span class="p">])</span>
<span class="n">shot_put_distance</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span> <span class="mf">6.4</span><span class="p">,</span> <span class="mf">10.2</span><span class="p">,</span> <span class="mf">12.4</span><span class="p">,</span> <span class="mf">13.</span> <span class="p">,</span> <span class="mf">13.2</span><span class="p">,</span> <span class="mf">13.</span> <span class="p">,</span> <span class="mf">12.7</span><span class="p">,</span> <span class="mf">13.9</span><span class="p">,</span> <span class="mf">15.5</span><span class="p">,</span> <span class="mf">15.8</span><span class="p">,</span> <span class="mf">15.8</span><span class="p">,</span>
       <span class="mf">16.8</span><span class="p">,</span> <span class="mf">17.1</span><span class="p">,</span> <span class="mf">17.8</span><span class="p">,</span> <span class="mf">14.8</span><span class="p">,</span> <span class="mf">15.5</span><span class="p">,</span> <span class="mf">16.1</span><span class="p">,</span> <span class="mf">16.2</span><span class="p">,</span> <span class="mf">17.9</span><span class="p">,</span> <span class="mf">15.9</span><span class="p">,</span> <span class="mf">15.8</span><span class="p">,</span> <span class="mf">16.7</span><span class="p">,</span>
       <span class="mf">17.6</span><span class="p">,</span> <span class="mf">16.8</span><span class="p">,</span> <span class="mf">17.</span> <span class="p">,</span> <span class="mf">18.2</span><span class="p">,</span> <span class="mf">19.2</span><span class="p">,</span> <span class="mf">18.6</span><span class="p">])</span>

<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">weight_lifted</span><span class="p">,</span> <span class="n">shot_put_distance</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">"Weight Lifted"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">"Shot Put Distance"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>Looks pretty linear! Let's try to fit a regression line to this data.</p>
<p>Define the vectors $x$ as the weight lifted data vector and $y$ as the shot put distance data vector, respectively, of the college athletes. Then the regression line uses the weight lifted $x$ to predict $\hat{y}$, which is the <strong>linear estimate</strong> of the actual value shot put distance $y$ as follows:</p>
<p>\begin{align}
\hat{y} &amp;= \hat{a} + \hat{b}{x}\text{, where} \\
\hat{a} &amp;= \bar{y} - \hat{b}\bar{x} \\
\hat{b} &amp;= r \dfrac{\sigma_y}{\sigma_x}
\end{align}</p>
<ul>
<li>$\bar{x}, \bar{y}$ and $\sigma_x, \sigma_y$ are the means and standard deviations, respectively of the data $x$ and $y$, respectively. Here, $r$ is the correlation coefficient as defined in Data 8!</li>
<li><strong>Note</strong>: We use the hat $\hat{}$ notation to indicate values we are <em>estimating</em>: $\hat{y}$, the predicted shot put distance, as well as $\hat{a}$ and $\hat{b}$, the respective estimated intercept and slope parameters we are using to model the "best" linear predictor of $y$ from $x$. We'll dive into this later in the course.</li>
<li><strong>Note</strong>: Remember how we dropped the $\vec{}$ vector notation? These linear regression equations therefore represent both the scalar case (predict a single value $\hat{y}$ from a single $x$) <em>and</em> the vector case (predict a vector $\hat{y}$ element-wise from a vector $x$). How convenient!!</li>
</ul>
<p>In this part, instead of using NumPy's built-in statistical functions like <code>np.mean()</code> and <code>np.std()</code>, you are going to use NumPy's matrix operations to create the components of the regression line from first principles.</p>
<p>The <code>@</code> operator multiplies NumPy matrices or arrays together (<a href="https://numpy.org/doc/stable/reference/generated/numpy.matmul.html#numpy.matmul">documentation</a>). We can use this operator to write functions to compute statistics on data, using the expressions that we defined in part (a). Check it out:</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Just run this cell.</span>
<span class="k">def</span> <span class="nf">dot_mean</span><span class="p">(</span><span class="n">arr</span><span class="p">):</span>
    <span class="n">n</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">arr</span><span class="p">)</span>
    <span class="n">all_ones</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">ones</span><span class="p">(</span><span class="n">n</span><span class="p">)</span>   <span class="c1"># creates n-dimensional vector of ones</span>
    <span class="k">return</span> <span class="p">(</span><span class="n">arr</span><span class="o">.</span><span class="n">T</span> <span class="o">@</span> <span class="n">all_ones</span><span class="p">)</span><span class="o">/</span><span class="n">n</span>

<span class="k">def</span> <span class="nf">dot_var</span><span class="p">(</span><span class="n">arr</span><span class="p">):</span>
    <span class="n">n</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">arr</span><span class="p">)</span>
    <span class="n">mean</span> <span class="o">=</span> <span class="n">dot_mean</span><span class="p">(</span><span class="n">arr</span><span class="p">)</span>
    <span class="n">zero_mean_arr</span> <span class="o">=</span> <span class="n">arr</span> <span class="o">-</span> <span class="n">mean</span>
    <span class="k">return</span> <span class="p">(</span><span class="n">zero_mean_arr</span><span class="o">.</span><span class="n">T</span> <span class="o">@</span> <span class="n">zero_mean_arr</span><span class="p">)</span><span class="o">/</span><span class="n">n</span>

<span class="k">def</span> <span class="nf">dot_std</span><span class="p">(</span><span class="n">arr</span><span class="p">):</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">dot_var</span><span class="p">(</span><span class="n">arr</span><span class="p">))</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">"np.mean(weight_lifted)  ="</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">weight_lifted</span><span class="p">),</span>
      <span class="s2">"</span><span class="se">\t</span><span class="s2">dot_mean(weight_lifted) ="</span><span class="p">,</span> <span class="n">dot_mean</span><span class="p">(</span><span class="n">weight_lifted</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"np.var(weight_lifted)   ="</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">std</span><span class="p">(</span><span class="n">weight_lifted</span><span class="p">),</span>
      <span class="s2">"</span><span class="se">\t</span><span class="s2">dot_var(weight_lifted   ="</span><span class="p">,</span> <span class="n">dot_var</span><span class="p">(</span><span class="n">weight_lifted</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"np.std(weight_lifted)   ="</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">std</span><span class="p">(</span><span class="n">weight_lifted</span><span class="p">),</span>
      <span class="s2">"</span><span class="se">\t</span><span class="s2">dot_std(weight_lifted   ="</span><span class="p">,</span> <span class="n">dot_std</span><span class="p">(</span><span class="n">weight_lifted</span><span class="p">))</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!--
BEGIN QUESTION
name: q6bi
manual: false
-->
<p>Now, you will write code to define the expressions you explored in part (a) of this question.</p>
<h3 id="Question-6b-(i)">Question 6b (i)<a class="anchor-link" href="#Question-6b-(i)">¶</a></h3><p>Use the NumPy <code>@</code> operator to compute expression (i) from part (a). For convenience, we've rewritten the expression below.
Note that this expression is also referred to as $x$ in <strong>standard units</strong> (<a href="https://inferentialthinking.com/chapters/14/2/Variability.html#standard-units">Data 8 textbook section</a>).</p>
<p>$$\dfrac{x - \bar{x}}{\sigma_x} $$</p>
<p>Write the body of the function <code>dot_su</code> which takes in a 1-D NumPy array <code>arr</code> and returns <code>arr</code> in standard units.</p>
<ul>
<li><strong>Do not use <code>np.mean(), np.std(), np.var(), np.sum()</code> nor any Python loops.</strong></li>
<li>You should only use a <em>subset</em> of <code>@, /, +, -, len()</code>, the <code>dot_mean(), dot_var(), and dot_std()</code> functions defined above.</li>
</ul>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">dot_su</span><span class="p">(</span><span class="n">arr</span><span class="p">):</span>
    <span class="o">...</span>

<span class="c1"># do not edit below this line</span>
<span class="n">q6bi_su</span> <span class="o">=</span> <span class="n">dot_su</span><span class="p">(</span><span class="n">weight_lifted</span><span class="p">)</span>
<span class="n">q6bi_su</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">grader</span><span class="o">.</span><span class="n">check</span><span class="p">(</span><span class="s2">"q6bi"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!--
BEGIN QUESTION
name: q6bii
manual: false
-->
<h3 id="Question-6b-(ii)">Question 6b (ii)<a class="anchor-link" href="#Question-6b-(ii)">¶</a></h3><p>Next use the NumPy <code>@</code> operator to compute the correlation coefficient $r$, which is expression (ii) from part (a). For convenience, we've rewritten the expression below.</p>
<p>$$r = \dfrac{1}{n} \left( \dfrac{x - \bar{x}}{\sigma^x}\right)^T \left( \dfrac{x - \bar{x}}{\sigma^x}\right)$$</p>
<p>Write the body of the function <code>dot_corr_coeff</code> which takes in two 1-D NumPy arrays <code>x</code> and <code>y</code> and returns the correlation coefficient of <code>x</code> and <code>y</code>.</p>
<ul>
<li>As before, <strong>Do not use <code>np.mean(), np.std(), np.var(), np.sum()</code> nor any Python loops.</strong></li>
<li>As before, you should only use a <em>subset</em> of <code>@, /, +, -, len()</code>, the <code>dot_mean(), dot_var(), and dot_std()</code> functions defined above.</li>
<li>You may also use the <code>dot_su()</code> function that you defined in the previous part.</li>
</ul>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">dot_corr_coeff</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">):</span>
    <span class="o">...</span>

<span class="c1"># do not edit below this line</span>
<span class="n">q6bii_r</span> <span class="o">=</span> <span class="n">dot_corr_coeff</span><span class="p">(</span><span class="n">weight_lifted</span><span class="p">,</span> <span class="n">shot_put_distance</span><span class="p">)</span>
<span class="n">q6bii_r</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">grader</span><span class="o">.</span><span class="n">check</span><span class="p">(</span><span class="s2">"q6bii"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<!--
BEGIN QUESTION
name: q6c
manual: false
-->
<h3 id="Question-6c">Question 6c<a class="anchor-link" href="#Question-6c">¶</a></h3><p>We're ready to put everything together! Finally, use the <code>dot_</code>-prefixed functions in this question to compute the regression line. For convenience, we've rewritten the expressions below. $\hat{y}$ is the linear estimate of the value $y$ based on $x$.</p>
<p>\begin{align}
\hat{y} &amp;= \hat{a} + \hat{b}{x}\text{, where} \\
\hat{a} &amp;= \bar{y} - \hat{b}\bar{x} \\
\hat{b} &amp;= r \dfrac{\sigma_y}{\sigma_x}
\end{align}</p>
<p>Define the functions <code>compute_a_hat</code> and <code>compute_b_hat</code> which return the intercept and slope, respectively, of the regression line defind above for a linear estimator of <code>y</code> using <code>x</code>. Verify how the functions are used to plot the linear regression line (implemented for you).</p>
<ul>
<li>As before, <strong>Do not use <code>np.mean(), np.std(), np.var(), np.sum()</code>, or any for loops.</strong></li>
<li>You may use a <em>subset</em> of <code>@, /, +, -, len(), dot_mean(), dot_var(), dot_std(), dot_su(), dot_corr_coeff()</code>.</li>
<li><strong>Hint:</strong> You may want to define a_hat in terms of b_hat.</li>
</ul>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">compute_a_hat</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">):</span>
    <span class="o">...</span>

<span class="k">def</span> <span class="nf">compute_b_hat</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">):</span>
    <span class="o">...</span>

<span class="c1"># do not edit below this line</span>
<span class="n">a_hat</span> <span class="o">=</span> <span class="n">compute_a_hat</span><span class="p">(</span><span class="n">weight_lifted</span><span class="p">,</span> <span class="n">shot_put_distance</span><span class="p">)</span>
<span class="n">b_hat</span> <span class="o">=</span> <span class="n">compute_b_hat</span><span class="p">(</span><span class="n">weight_lifted</span><span class="p">,</span> <span class="n">shot_put_distance</span><span class="p">)</span>
<span class="n">shot_put_hats</span> <span class="o">=</span> <span class="n">a_hat</span> <span class="o">+</span> <span class="n">b_hat</span> <span class="o">*</span> <span class="n">weight_lifted</span>
<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">weight_lifted</span><span class="p">,</span> <span class="n">shot_put_distance</span><span class="p">)</span> <span class="c1"># the actual data</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">weight_lifted</span><span class="p">,</span> <span class="n">shot_put_hats</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s1">'g'</span><span class="p">,</span> <span class="n">alpha</span><span class="o">=</span><span class="mf">0.5</span><span class="p">)</span> <span class="c1"># the prediction line, transparent green</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">"Weight Lifted"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">"Shot Put Distance"</span><span class="p">)</span>
<span class="n">display</span><span class="p">(</span><span class="n">compute_a_hat</span><span class="p">(</span><span class="n">weight_lifted</span><span class="p">,</span> <span class="n">shot_put_distance</span><span class="p">))</span>
<span class="n">display</span><span class="p">(</span><span class="n">compute_b_hat</span><span class="p">(</span><span class="n">weight_lifted</span><span class="p">,</span> <span class="n">shot_put_distance</span><span class="p">))</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">grader</span><span class="o">.</span><span class="n">check</span><span class="p">(</span><span class="s2">"q6c"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<hr/>
<p>To double-check your work, the cell below will rerun all of the autograder tests.</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">grader</span><span class="o">.</span><span class="n">check_all</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h2 id="Submission">Submission<a class="anchor-link" href="#Submission">¶</a></h2><p>Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output. The cell below will generate a zip file for you to submit. <strong>Please save before exporting!</strong></p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Save your notebook first, then run this cell to export your submission.</span>
<span class="n">grader</span><span class="o">.</span><span class="n">export</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
</div>
</div>
</div>
</div>
</main>
</body>
</html>
