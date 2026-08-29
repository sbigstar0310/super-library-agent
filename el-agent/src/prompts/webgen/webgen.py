"""WebGen-Bench bench-level rules (app + library scaffold conventions).

Two constants:
- APP_RULES : what an app submission must look like (stack, file layout,
              styling discipline, import discipline, vite.config
              template). Inject in agents that EDIT app code
              (coding / apply / local_extract).
- LIB_RULES : what the shared `ui-lib` package must look like, including
              the MANDATORY self-contained styling rule and the
              caller-facing override interface. Inject in agents that
              EDIT lib code (global_extract / library).
"""

from __future__ import annotations


__all__ = ["APP_RULES", "LIB_RULES"]


APP_RULES = """\
[Stack]
- React 18 + Vite 4 + plain CSS. No Tailwind, no UI kit.
- All state lives in-memory or in `localStorage`. No backend, no `fetch()`
  unless the instruction explicitly requires it.

[Error UI]
- Do NOT use `alert()` for validation or auth errors. Render messages
  as in-page elements. The browser modal dialog blocks the UI-test
  runner (Selenium aborts on unexpected alert).

[Auth seed]
If the app has login/registration, seed at least one demo account that
accepts username `admin`, email `admin@example.com`, and password
`admin123456`. Any of the three identifier strings (username OR email)
must work depending on which the login form requests. These are the
exact credentials the UI-test runner enters.

[Build]
Before ending the turn, the production build at `<workspace_dir>` MUST
exit 0. Actually run it — reading files and declaring "looks correct"
does NOT satisfy this. Fix any error and re-run before ending.

If you swapped symbols or removed files during this turn, also clean up
the resulting stale references and dead code (orphaned imports,
abandoned helper files, unused colocated assets) before the final
build. Skip this clean-up step when nothing was swapped or removed.

[Required files at submission root]
  package.json   (name, type=module, scripts: dev/build/preview,
                  deps: react ^18 + react-dom ^18,
                  devDeps: @vitejs/plugin-react ^4 + vite ^4)
  vite.config.js (defineConfig + react plugin; template below)
  index.html     (single <div id="root"></div>)
  src/main.jsx   (ReactDOM.createRoot → <App />)
  src/App.jsx    (top-level component)

[Styling discipline]
- Each app-owned component co-locates its `.css` next to the `.jsx`
  file and imports it from the top of the JSX (`import './Foo.css'`).
- Do NOT define styles for class names that belong to a library
  component's internals. Library-internal class names belong in the
  library's own CSS file.
- Override library styling only through documented props
  (`style` / `className` / `variant`).

[Library import discipline]
- Apps import library symbols via a RELATIVE PATH to the library's
  `src/index.js` barrel (named imports only). Never via an alias, a
  package specifier, or a deep path that bypasses the barrel.

[vite.config.js template] — rewrite the existing file if it diverges
(no `resolve.alias`; the shared library is imported via relative paths,
not aliases or npm workspaces)

```js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
});
```
"""


LIB_RULES = """\
[Directory layout] — plain JS/JSX directory at `<library_dir>`:
  src/index.js                      barrel re-export of every public symbol
  src/components/<Name>.jsx         + co-located <Name>.css
  src/hooks/<useX>.js
  src/utils/<x>.js
  src/services/<x>.js
JSX in `.jsx` only. Plain CSS only — no Tailwind / Bootstrap wrappers.
The library is NOT an npm package — no `package.json`, no install step.

[Module conventions]
- Components: `export default function Foo(...)`.
- Hooks / utils / services: named exports.

[Barrel exports — `src/index.js`]
Re-export every public symbol as a NAMED export. The barrel is the
ONLY supported entry point — apps must not import library internals
via subpaths. Example:

```js
export { default as LoginForm }     from './components/LoginForm.jsx';
export { default as SearchBar }     from './components/SearchBar.jsx';
export { useFormValidation }        from './hooks/useFormValidation.js';
export { loadStorage, saveStorage } from './utils/localStorage.js';
```

[Self-contained styling — MANDATORY]
Every component MUST own its styles. Concretely:
- For each `components/<Name>.jsx`, ship a sibling `components/<Name>.css`
  containing every class name the component renders (including `:hover`,
  `:focus`, state classes, and `@media`).
- The component MUST `import './<Name>.css'` at the top of the JSX
  file. Do not rely on the importing app to define any of the
  component's class names.

[Style override interface]
Components SHOULD expose at minimum:
- `className?: string` — appended to the root element so the caller can
  scope overrides.
- `style?: React.CSSProperties` — merged into the root element's inline
  style.
For multi-part components, prefer a small `variant` enum prop over
leaking internals.

[Reuse discipline]
Do not duplicate symbols already present in the seeded library. Before
adding a new symbol, scan the existing barrel for an equivalent — extend
or parameterize the existing one if possible.
"""
