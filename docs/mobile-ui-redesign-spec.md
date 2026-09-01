# Podcast ASR UI Redesign Specification

Source: `claude_kimi` using `claude-fable-5`. This document is the canonical
implementation contract for the redesign. Existing APIs, routes, polling,
downloads, localStorage keys, task restoration, and publish redirects must not
change.

## Product shell

- Use one shared visual language for podcast and meeting workflows.
- Page background: `#f6f7f8`; surface: `#fff`; sunken surface: `#eef0f2`;
  primary text: `#1b2126`; secondary text: `#55606a`; border: `#dde1e5`;
  strong border: `#c3c9cf`; accent: `#0b6bcb`; success: `#157f3d`;
  warning: `#9a6a00`; danger: `#c22828`.
- Add matching `prefers-color-scheme: dark` values: background `#101315`,
  surface `#171b1e`, sunken `#0c0e10`, text `#e8ebee`, secondary `#9aa5ae`,
  border `#2b3238`, strong border `#3d454d`, accent `#6ea8fe`, success
  `#4ecb7f`, warning `#e3b341`, danger `#f08080`.
- No gradients, decorative artwork, marketing hero, nested cards, oversized
  typography, negative letter spacing, or pill-shaped controls.
- Font stack: system UI with PingFang SC and Microsoft YaHei fallbacks. Body is
  14/21px. Scale: 12/16, 13/18, 14/21, 16/24, 18/26, 20/28, 24/30px. The
  24px page title is the maximum. Use 600 as the strongest weight.
- Spacing scale: 4, 8, 12, 16, 24, 32, 48px. Radius is 6px; table/log radius
  is 4px. Controls are 40px high; compact controls are 32px.
- Content is centered at max-width 1120px with 24px desktop and 12px mobile
  gutters. Breakpoints: mobile 0-599px, medium 600-959px, desktop 960px+.
- Every control has a visible 2px accent focus ring with 2px offset. Maintain
  WCAG AA contrast, semantic headings, labels, live task status, and keyboard
  operation. Touch targets are at least 40px high on mobile.

Use a 56px sticky surface app bar with a bottom border. Left side is
`Podcast ASR Studio` plus the current zone (`播客转写` or `会议转写`). Right
side contains compact navigation between the two workflows and the existing
legacy/health destinations. It may wrap on viewports below 360px but must not
clip or horizontally overflow. Footer is separated by a top border and uses
12px secondary text.

## Podcast index

Replace the current hero with a compact unframed page header: `播客 ASR 索引`
at 24px and one functional sentence below. Preserve `#import` and `#library`.

The import workspace is one flat bordered surface with 16px padding:

- Header row: `导入播客` and a secondary `URL / 音频文件` label.
- A 36px segmented tab control switches between `贴入链接` and `上传音频`.
  Implement tablist/tab semantics and keyboard arrow navigation.
- URL mode is a flex row with the URL input and `创建任务`; below 600px it is
  a column and the button is full width.
- Audio mode is a dashed bordered message with an `打开会议转写` button. It
  routes to the meeting page; do not duplicate meeting upload in this page.
- Show the four pipeline steps as divider-separated text on desktop, 2x2 at
  medium width, and a vertical list on mobile. Do not create mini cards.

Keep the task status within this panel. It is hidden before use, then appears
as a sunken bordered surface. Show status and job metadata, current phase,
transcription/summary/TLDR capability checklist, an indeterminate 4px progress
bar for queued/running tasks, a message that background work continues after
leaving, report artifact buttons when available, and a collapsible 240px log
tail. Preserve 5-second polling, all current phase strings, one-time publish
redirects, stale-task handling, and submit-button states. Errors use inline
danger styling and return focus to the invalid input. Toasts are dismissible,
bottom-right on desktop, and inset 12px across the bottom on mobile.

Render the aggregate library metrics in one bordered strip with divider-only
cells: four columns on desktop and 2x2 below 960px. The library toolbar contains
the 20px heading, search field, and a live `{shown} / {total} 集` count. Search
is full width on mobile.

Episode cards use a 2-column grid at 960px+ and one column below. Each card is
a flat bordered surface with 16px padding and no shadow. Order content as:
metadata line, 18px linked title, 3-line summary, plain divider-separated metric
line, then 32px action buttons. Replace badge and metric mini-cards with text.
The task-completion hash target gets a temporary 2px accent outline and
`scroll-margin-top: 72px`. Provide explicit no-episodes and no-search-results
states; the latter includes `清除搜索`.

## Podcast report and full transcript

The report header is unframed: title, 3-line summary, metadata, a divider-style
four-cell metric strip, and compact actions. At 960px+ the body uses a 240px
sticky section navigation and flexible content column. Below 960px the section
navigation is a horizontally scrollable sticky row below the app bar.

Keep all report anchors and artifacts. Each major report section is one flat
bordered panel with 24px desktop or 16px mobile padding. Within a panel, use
dividers and plain rows rather than nested cards. Tables live in horizontal
scroll containers and retain a 640px minimum table width. Outline timestamps
use a fixed 72px column. Summary topic blocks are divider-separated. Quotes use
a 3px accent left border and sunken background. Downloads wrap as compact
buttons. Preserve all empty states and conditional artifacts.

On the full transcript page, put `返回报告` in the app bar. Use an unframed
title and compact download row. The search toolbar sticks below the app bar;
on mobile the match count moves below the input/buttons. Transcript chunks are
plain articles separated by horizontal rules, not cards. Preserve chunk IDs,
search filtering/highlighting, whitespace, clear/refocus behavior, and use a
76ch maximum text measure.

## Meeting transcription

Use the shared app bar and a compact unframed header titled `会议录音转写`.
Desktop layout at 960px+ is `minmax(0, 1fr) 320px`: upload/current-task workspace
on the left and recent tasks on the right. Below 960px use one column with the
upload workspace first and recent tasks second.

The upload surface contains:

- A normal 40px primary `选择录音` button backed by the existing mobile-safe
  file picker behavior and a secondary folder-picker action where supported.
- A visible drop zone for desktop drag/drop, but do not make the drop zone the
  only way to invoke the picker. Mobile must open the operating system file
  browser, not a camera/photo capture flow.
- Accepted formats and 256MiB limit as concise secondary text.
- Selected filename, size, and remove/change actions before upload.
- A determinate upload progress bar from existing XHR progress. After creation,
  transition to background task progress without resizing the workspace.

The current-task surface presents state, task ID, filename, phase/message,
timestamps, and progress in a stable layout. Queued/running tasks poll every
3 seconds and explicitly say the user may leave and return. Completed tasks
show transcript preview plus download actions; failed tasks show the backend
error and retry/reselect controls. Preserve active-task localStorage restore.

Recent tasks are dense rows, not individual decorative cards. Each row contains
filename, status, created time, progress or completion metadata, and an open
action. Use status text plus color, never color alone. On mobile rows may wrap
but actions remain reachable and no text is clipped.

Transcript output uses speaker blocks separated by dividers. Speaker name and
`H:MM:SS` timestamp form a compact header; utterance text is 16/24px with
preserved line breaks. Long transcripts scroll naturally with no fixed-height
trap. Keep TXT/JSON and any existing result downloads visible in the completed
state.

## State and responsive acceptance

Both workflows must visibly support initial, loading, uploading, queued,
running, completed, failed, restored, empty, and network-error states without
layout shifts that move the primary control. Disabled controls remain readable
and use `not-allowed`; loading indicators honor `prefers-reduced-motion`.

Acceptance viewports: 320x568, 390x844, 768x1024, 1024x768, and 1440x900.
At each viewport:

- no horizontal page overflow, overlapping controls, clipped Chinese text, or
  inaccessible actions;
- sticky app/navigation bars do not cover anchor targets or transcript chunks;
- file selection works from a clearly labeled button on mobile;
- URL submission, task restoration/polling, publish redirect, search/filter,
  task selection, uploads, progress, result rendering, and downloads retain
  their existing behavior;
- keyboard focus order is logical and every interactive item has a visible
  focus state;
- light and dark themes have readable borders, status colors, and controls.
